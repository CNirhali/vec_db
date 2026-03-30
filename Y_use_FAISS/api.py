from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.concurrency import run_in_threadpool
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel, field_validator, Field, model_validator
from typing import List, Optional, Any
import numpy as np
import os
import secrets
import json
import re
import functools
import inspect
from .core import VectorDB
import structlog
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response, JSONResponse

# Security: Load API Key from environment variable, default for dev
API_KEY = os.getenv("API_KEY", "supersecretkey")

logger = structlog.get_logger()

if API_KEY == "supersecretkey":
    logger.warning("Using default insecure API_KEY. Please set API_KEY environment variable in production.")

def api_key_auth(request: Request, x_api_key: Optional[str] = Header(None)):
    # Security: Return 401 instead of 422 if key is missing, and use compare_digest to prevent timing attacks
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        # Security: Log failed authentication attempts for auditing and threat detection
        client_ip = request.client.host if request.client else "unknown"
        logger.warning("auth_failed", client_ip=client_ip, reason="missing_or_invalid_key")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")

# Security: Enforce authentication globally to prevent information leakage about the API schema to unauthenticated users
app = FastAPI(dependencies=[Depends(api_key_auth)])
db = None

# Security: Limit request body size to 150MB to prevent memory-based DoS
MAX_REQUEST_BODY_SIZE = 150 * 1024 * 1024

# Security: Initialize rate limiter to protect against DoS and brute-force attacks
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.exception_handler(OSError)
async def os_error_handler(request: Request, exc: OSError):
    # Security: Catch system-level errors and return a generic 500 error to prevent information leakage
    logger.error("os_error", exc_info=exc, path=request.url.path)
    return JSONResponse(
        content={"detail": "A storage-related error occurred. Please ensure the database is properly initialized and the storage path is accessible."},
        status_code=500
    )

@app.exception_handler(RuntimeError)
async def runtime_error_handler(request: Request, exc: RuntimeError):
    # Security: Catch indexing-level errors and return a generic 500 error to prevent internal detail leakage
    logger.error("runtime_error", exc_info=exc, path=request.url.path)
    return JSONResponse(
        content={"detail": "An internal indexing error occurred. The operation could not be completed securely."},
        status_code=500
    )

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    # Security: Enforce request body size limit to prevent memory-based DoS
    content_length = request.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_REQUEST_BODY_SIZE:
        return JSONResponse(
            content={"detail": "Payload Too Large. Maximum allowed size is 150MB."},
            status_code=413
        )

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

REQUEST_COUNT = Counter('vectordb_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('vectordb_request_latency_seconds', 'API request latency', ['endpoint', 'method'])

class InitRequest(BaseModel):
    dim: int = Field(..., gt=0, le=10000)  # Security: Limit dimension to prevent excessive memory usage
    storage_path: str = Field(..., max_length=255)  # Security: Limit path length
    ef_construction: int = Field(400, gt=0, le=1000)  # Security: Limit ef_construction to prevent DoS
    M: int = Field(32, gt=0, le=128)  # Security: Limit M to prevent DoS
    ef_search: int = Field(128, gt=0, le=1000)  # Security: Limit ef_search to prevent DoS

    @field_validator('storage_path')
    @classmethod
    def validate_storage_path(cls, v: str) -> str:
        # Security: Prevent path traversal and restrict file extensions
        if ".." in v or os.path.isabs(v) or v.startswith("/") or v.startswith("\\"):
            raise ValueError("storage_path must be a relative path and cannot contain '..'")
        # Security: Defense-in-depth via regex validation for safe characters, forbidding directory separators
        if not re.match(r"^[a-zA-Z0-9_\-\.]+$", v):
            raise ValueError("storage_path contains invalid characters or directory separators")
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
        # Security: Prevent negative IDs, ensure uniqueness, and cap at uint64 max to avoid index inconsistencies and OverflowError
        if v is not None:
            if len(v) != len(set(v)):
                raise ValueError("IDs in a batch must be unique")
            for vector_id in v:
                if vector_id < 0:
                    raise ValueError("IDs must be non-negative")
                if vector_id > 18446744073709551615:
                    raise ValueError("IDs must not exceed 64-bit unsigned integer limit (18446744073709551615)")
        return v
    metadata: Optional[List[dict]] = Field(None, max_length=10000)

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

        # Security: Limit total results (n_queries * k) to prevent result-set DoS
        total_results = len(self.queries) * self.k
        if total_results > 100000:
            raise ValueError(f"Total requested results ({total_results}) exceeds limit of 100,000")
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
    ids: List[int] = Field(..., min_length=1, max_length=10000)  # Security: Batch size limit

    @field_validator('ids')
    @classmethod
    def validate_ids(cls, v: List[int]) -> List[int]:
        # Security: Prevent negative IDs, ensure uniqueness, and cap at uint64 max to avoid index inconsistencies and OverflowError
        if len(v) != len(set(v)):
            raise ValueError("IDs in a batch must be unique")
        for vector_id in v:
            if vector_id < 0:
                raise ValueError("IDs must be non-negative")
            if vector_id > 18446744073709551615:
                raise ValueError("IDs must not exceed 64-bit unsigned integer limit (18446744073709551615)")
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
        # Security: Prevent negative IDs, ensure uniqueness, and cap at uint64 max to avoid index inconsistencies and OverflowError
        if len(v) != len(set(v)):
            raise ValueError("IDs in a batch must be unique")
        for vector_id in v:
            if vector_id < 0:
                raise ValueError("IDs must be non-negative")
            if vector_id > 18446744073709551615:
                raise ValueError("IDs must not exceed 64-bit unsigned integer limit (18446744073709551615)")
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
@limiter.limit("200/minute")
def metrics(request: Request):
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
@limiter.limit("5/minute")
@instrument("/init")
def init_db(request: Request, req: InitRequest):
    global db
    try:
        db = VectorDB(req.dim, req.storage_path, ef_construction=req.ef_construction, M=req.M, ef_search=req.ef_search)
    except ValueError as e:
        # Security: Specific 400 for dimension mismatch or invalid parameters to avoid 500 error
        raise HTTPException(status_code=400, detail=str(e))
    # Security: OSError and RuntimeError are handled by global exception handlers
    return {"status": "initialized", "dim": req.dim, "storage_path": req.storage_path, "ef_construction": req.ef_construction, "M": req.M, "ef_search": req.ef_search}

@app.post("/add")
@limiter.limit("100/minute")
@instrument("/add")
def add_vectors(request: Request, req: AddRequest):
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
@limiter.limit("100/minute")
@instrument("/search")
def search_vectors(request: Request, req: SearchRequest):
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
@limiter.limit("100/minute")
@instrument("/delete")
def delete_vectors(request: Request, req: DeleteRequest):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    db.delete(req.ids)
    return {"status": "ok", "deleted": len(req.ids)}

@app.post("/update")
@limiter.limit("100/minute")
@instrument("/update")
def update_vectors(request: Request, req: UpdateRequest):
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

@app.post("/save")
@limiter.limit("100/minute")
@instrument("/save")
def save_db(request: Request):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    db.save()
    return {"status": "ok", "message": "Index saved to disk"}

@app.get("/status")
@limiter.limit("200/minute")
@instrument("/status")
def status(request: Request):
    return {"status": "ok", "db_initialized": db is not None}
