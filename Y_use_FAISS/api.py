from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, field_validator, Field, model_validator
from typing import List, Optional, Any
import numpy as np
import os
import secrets
import json
import functools
import inspect
from .core import VectorDB
import structlog
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response

app = FastAPI()
db = None

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # Security: Defense-in-depth by adding security headers
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; object-src 'none';"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
    return response
# Security: Load API Key from environment variable, default for dev
API_KEY = os.getenv("API_KEY", "supersecretkey")

logger = structlog.get_logger()

if API_KEY == "supersecretkey":
    logger.warning("Using default insecure API_KEY. Please set API_KEY environment variable in production.")

REQUEST_COUNT = Counter('vectordb_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('vectordb_request_latency_seconds', 'API request latency', ['endpoint', 'method'])

def api_key_auth(x_api_key: Optional[str] = Header(None)):
    # Security: Return 401 instead of 422 if key is missing, and use compare_digest to prevent timing attacks
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

class InitRequest(BaseModel):
    dim: int = Field(..., gt=0, le=10000)  # Security: Limit dimension to prevent excessive memory usage
    storage_path: str
    ef_construction: int = Field(400, gt=0, le=1000)  # Security: Limit ef_construction to prevent DoS
    M: int = Field(32, gt=0, le=128)  # Security: Limit M to prevent DoS
    ef_search: int = Field(128, gt=0, le=1000)  # Security: Limit ef_search to prevent DoS

    @field_validator('storage_path')
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        # Security: Prevent path traversal and restrict file extensions
        if ".." in v or os.path.isabs(v) or v.startswith("/") or v.startswith("\\"):
            raise ValueError("storage_path must be a relative path and cannot contain '..'")
        if not (v.endswith(".h5") or v.endswith(".hdf5")):
            raise ValueError("storage_path must have a .h5 or .hdf5 extension")
        return v

class AddRequest(BaseModel):
    vectors: List[List[float]] = Field(..., min_length=1, max_length=10000)  # Security: Batch size limit
    ids: Optional[List[int]] = Field(None, max_length=10000)

    @field_validator('vectors')
    @classmethod
    def validate_vectors_dim(cls, v: List[List[float]]) -> List[List[float]]:
        # Security: Limit individual vector dimension to prevent memory exhaustion
        for vector in v:
            if len(vector) > 10000:
                raise ValueError("Vector dimension exceeds limit of 10000")
        return v

    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v: Optional[List[int]]) -> Optional[List[int]]:
        # Security: Prevent negative IDs which are not supported by hnswlib (uint64)
        if v is not None:
            for vector_id in v:
                if vector_id < 0:
                    raise ValueError("IDs must be non-negative")
        return v
    metadata: Optional[List[dict]] = None

    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Optional[List[dict]]) -> Optional[List[dict]]:
        # Security: Limit metadata size and key count to prevent DoS
        if v is None:
            return v
        for entry in v:
            if len(entry) > 100:
                raise ValueError("Metadata entry exceeds limit of 100 keys")
            if len(json.dumps(entry)) > 10240:
                raise ValueError("Metadata entry size exceeds limit of 10 KB")
        return v

    @model_validator(mode='after')
    def validate_add_request(self) -> 'AddRequest':
        n_vectors = len(self.vectors)
        if self.ids is not None and len(self.ids) != n_vectors:
            raise ValueError(f"Number of ids ({len(self.ids)}) must match number of vectors ({n_vectors})")
        if self.metadata is not None and len(self.metadata) != n_vectors:
            raise ValueError(f"Number of metadata entries ({len(self.metadata)}) must match number of vectors ({n_vectors})")

        # Security: Limit total elements to prevent memory exhaustion (DoS)
        total_elements = sum(len(v) for v in self.vectors)
        if total_elements > 2000000:
            raise ValueError(f"Total elements in vectors ({total_elements}) exceeds limit of 2,000,000")
        return self

class SearchRequest(BaseModel):
    queries: List[List[float]] = Field(..., min_length=1, max_length=10000)  # Security: Batch size limit

    @field_validator('queries')
    @classmethod
    def validate_queries_dim(cls, v: List[List[float]]) -> List[List[float]]:
        # Security: Limit individual query dimension to prevent memory exhaustion
        for query in v:
            if len(query) > 10000:
                raise ValueError("Query dimension exceeds limit of 10000")
        return v

    @model_validator(mode='after')
    def validate_search_request(self) -> 'SearchRequest':
        # Security: Limit total elements to prevent memory exhaustion (DoS)
        total_elements = sum(len(q) for q in self.queries)
        if total_elements > 2000000:
            raise ValueError(f"Total elements in queries ({total_elements}) exceeds limit of 2,000,000")
        return self
    k: int = Field(10, gt=0, le=1000)  # Security: Limit k to prevent DoS
    filter_metadata: Optional[dict] = None

    @field_validator('filter_metadata')
    @classmethod
    def validate_filter_metadata(cls, v: Optional[dict]) -> Optional[dict]:
        # Security: Limit filter metadata size and key count to prevent DoS
        if v is None:
            return v
        if len(v) > 100:
            raise ValueError("Filter metadata exceeds limit of 100 keys")
        if len(json.dumps(v)) > 10240:
            raise ValueError("Filter metadata size exceeds limit of 10 KB")
        return v

class DeleteRequest(BaseModel):
    ids: List[int] = Field(..., max_length=10000)  # Security: Batch size limit

    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v: List[int]) -> List[int]:
        # Security: Prevent negative IDs which are not supported by hnswlib (uint64)
        for vector_id in v:
            if vector_id < 0:
                raise ValueError("IDs must be non-negative")
        return v

class UpdateRequest(BaseModel):
    ids: List[int] = Field(..., min_length=1, max_length=10000)  # Security: Batch size limit
    vectors: List[List[float]] = Field(..., min_length=1, max_length=10000)

    @field_validator('vectors')
    @classmethod
    def validate_vectors_dim(cls, v: List[List[float]]) -> List[List[float]]:
        # Security: Limit individual vector dimension to prevent memory exhaustion
        for vector in v:
            if len(vector) > 10000:
                raise ValueError("Vector dimension exceeds limit of 10000")
        return v

    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v: List[int]) -> List[int]:
        # Security: Prevent negative IDs which are not supported by hnswlib (uint64)
        for vector_id in v:
            if vector_id < 0:
                raise ValueError("IDs must be non-negative")
        return v

    @model_validator(mode='after')
    def validate_update_request(self) -> 'UpdateRequest':
        if len(self.ids) != len(self.vectors):
            raise ValueError(f"Number of ids ({len(self.ids)}) must match number of vectors ({len(self.vectors)})")

        # Security: Limit total elements to prevent memory exhaustion (DoS)
        total_elements = sum(len(v) for v in self.vectors)
        if total_elements > 2000000:
            raise ValueError(f"Total elements in vectors ({total_elements}) exceeds limit of 2,000,000")
        return self

@app.get("/metrics")
def metrics(x_api_key: str = Depends(api_key_auth)):
    return Response(generate_latest(), media_type="text/plain")

def instrument(endpoint):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            method = func.__name__
            REQUEST_COUNT.labels(endpoint, method).inc()
            with REQUEST_LATENCY.labels(endpoint, method).time():
                logger.info("api_call", endpoint=endpoint, method=method)
                # Correctly handle both sync and async endpoints
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return await run_in_threadpool(func, *args, **kwargs)
        return wrapper
    return decorator

@app.post("/init")
@instrument("/init")
def init_db(req: InitRequest, x_api_key: str = Depends(api_key_auth)):
    global db
    db = VectorDB(req.dim, req.storage_path, ef_construction=req.ef_construction, M=req.M, ef_search=req.ef_search)
    return {"status": "initialized", "dim": req.dim, "storage_path": req.storage_path, "ef_construction": req.ef_construction, "M": req.M, "ef_search": req.ef_search}

@app.post("/add")
@instrument("/add")
def add_vectors(req: AddRequest, x_api_key: str = Depends(api_key_auth)):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    try:
        vectors = np.array(req.vectors, dtype=np.float32)
    except ValueError as e:
        # Security: Return 400 instead of 500 when input vectors are inhomogeneous
        raise HTTPException(status_code=400, detail=f"Invalid input vectors. Ensure all vectors have the same dimension. Error: {str(e)}")

    # Security: Reject non-finite values (NaN, Inf) to prevent 500 Internal Server Error during JSON serialization
    if not np.isfinite(vectors).all():
        raise HTTPException(status_code=400, detail="Vectors contain non-finite values (NaN or Infinity)")

    # Security: Validate vector dimensions
    if vectors.shape[1] != db.dim:
        raise HTTPException(status_code=400, detail=f"Invalid vector dimension. Expected {db.dim}, got {vectors.shape[1]}")
    db.add(vectors, req.ids, req.metadata)
    return {"status": "ok", "count": len(vectors)}

@app.post("/search")
@instrument("/search")
def search_vectors(req: SearchRequest, x_api_key: str = Depends(api_key_auth)):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    try:
        queries = np.array(req.queries, dtype=np.float32)
    except ValueError as e:
        # Security: Return 400 instead of 500 when queries are inhomogeneous
        raise HTTPException(status_code=400, detail=f"Invalid queries. Ensure all queries have the same dimension. Error: {str(e)}")

    # Security: Reject non-finite values (NaN, Inf) to prevent 500 Internal Server Error during JSON serialization
    if not np.isfinite(queries).all():
        raise HTTPException(status_code=400, detail="Queries contain non-finite values (NaN or Infinity)")

    # Security: Validate query dimensions
    if queries.shape[1] != db.dim:
        raise HTTPException(status_code=400, detail=f"Invalid query dimension. Expected {db.dim}, got {queries.shape[1]}")
    labels, distances = db.search(queries, req.k, filter_metadata=req.filter_metadata)
    # Convert numpy types to native Python types for JSON serialization
    return {"labels": labels.tolist(), "distances": distances.tolist()}

@app.post("/delete")
@instrument("/delete")
def delete_vectors(req: DeleteRequest, x_api_key: str = Depends(api_key_auth)):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    db.delete(req.ids)
    return {"status": "ok", "deleted": len(req.ids)}

@app.post("/update")
@instrument("/update")
def update_vectors(req: UpdateRequest, x_api_key: str = Depends(api_key_auth)):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    try:
        vectors = np.array(req.vectors, dtype=np.float32)
    except ValueError as e:
        # Security: Return 400 instead of 500 when input vectors are inhomogeneous
        raise HTTPException(status_code=400, detail=f"Invalid input vectors. Ensure all vectors have the same dimension. Error: {str(e)}")

    # Security: Reject non-finite values (NaN, Inf) to prevent 500 Internal Server Error during JSON serialization
    if not np.isfinite(vectors).all():
        raise HTTPException(status_code=400, detail="Vectors contain non-finite values (NaN or Infinity)")

    # Security: Validate vector dimensions
    if vectors.shape[1] != db.dim:
        raise HTTPException(status_code=400, detail=f"Invalid vector dimension. Expected {db.dim}, got {vectors.shape[1]}")
    db.update(req.ids, vectors)
    return {"status": "ok", "updated": len(req.ids)}

@app.get("/status")
@instrument("/status")
def status(x_api_key: str = Depends(api_key_auth)):
    return {"status": "ok", "db_initialized": db is not None}
