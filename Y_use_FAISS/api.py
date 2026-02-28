from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from .core import VectorDB
import structlog
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import os
import secrets
import functools
import inspect

app = FastAPI()
db = None
# Security: Load API Key from environment variable, default for dev
API_KEY = os.getenv("API_KEY", "supersecretkey")

logger = structlog.get_logger()

REQUEST_COUNT = Counter('vectordb_requests_total', 'Total API requests', ['endpoint', 'method'])
REQUEST_LATENCY = Histogram('vectordb_request_latency_seconds', 'API request latency', ['endpoint', 'method'])

def api_key_auth(x_api_key: str = Header(...)):
    # Security: Use compare_digest to prevent timing attacks
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Invalid API Key")

class InitRequest(BaseModel):
    dim: int
    storage_path: str
    ef_construction: int = 400
    M: int = 32
    ef_search: int = 128

class AddRequest(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[int]] = None
    metadata: Optional[List[dict]] = None

class SearchRequest(BaseModel):
    queries: List[List[float]]
    k: int = 10
    filter_metadata: Optional[dict] = None

class DeleteRequest(BaseModel):
    ids: List[int]

class UpdateRequest(BaseModel):
    ids: List[int]
    vectors: List[List[float]]

@app.get("/metrics")
def metrics():
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
    # Security: Prevent path traversal
    if ".." in req.storage_path or req.storage_path.startswith("/") or req.storage_path.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid storage path")
    db = VectorDB(req.dim, req.storage_path, ef_construction=req.ef_construction, M=req.M, ef_search=req.ef_search)
    return {"status": "initialized", "dim": req.dim, "storage_path": req.storage_path, "ef_construction": req.ef_construction, "M": req.M, "ef_search": req.ef_search}

@app.post("/add")
@instrument("/add")
def add_vectors(req: AddRequest, x_api_key: str = Depends(api_key_auth)):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    vectors = np.array(req.vectors, dtype=np.float32)
    db.add(vectors, req.ids, req.metadata)
    return {"status": "ok", "count": len(vectors)}

@app.post("/search")
@instrument("/search")
def search_vectors(req: SearchRequest, x_api_key: str = Depends(api_key_auth)):
    if db is None:
        raise HTTPException(status_code=400, detail="DB not initialized")
    queries = np.array(req.queries, dtype=np.float32)
    labels, distances = db.search(queries, req.k, filter_metadata=req.filter_metadata)
    return {"labels": labels, "distances": distances}

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
    vectors = np.array(req.vectors, dtype=np.float32)
    db.update(req.ids, vectors)
    return {"status": "ok", "updated": len(req.ids)}

@app.get("/status")
@instrument("/status")
def status():
    return {"status": "ok", "db_initialized": db is not None} 