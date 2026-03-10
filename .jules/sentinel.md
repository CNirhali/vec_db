## 2026-02-28 - [FastAPI Decorator & Security Enhancements]
**Vulnerability:** Hardcoded API key, timing attack vulnerability, path traversal risk, and event-loop-blocking decorator bug.
**Learning:** In FastAPI, custom decorators wrapping endpoint functions must use `@functools.wraps` to preserve function signatures for dependency injection (like `Depends(api_key_auth)`). Furthermore, if a decorator is `async def`, it must explicitly handle synchronous endpoint functions using `run_in_threadpool` to avoid blocking the event loop or causing errors when `await` is called on a non-coroutine.
**Prevention:** Always use `secrets.compare_digest` for secret comparisons. Validate user-provided file paths for common traversal patterns. Use `@functools.wraps` and `inspect.iscoroutinefunction` in decorators that wrap web endpoints.

## 2026-02-28 - [HNSW Parameter Validation]
**Vulnerability:** Denial-of-Service (DoS) via unconstrained HNSW index parameters.
**Learning:** Parameters like `ef_construction` and `M` in HNSW index construction directly impact CPU and memory usage. Without upper bounds, an attacker can provide extremely large values that exhaust server resources during index initialization.
**Prevention:** Enforce strict upper limits on resource-intensive algorithm parameters using Pydantic's `Field(le=...)` validation in API request models.

## 2026-03-07 - [Core Thread-Safety & Dynamic Index Resizing]
**Vulnerability:** Service instability and potential data corruption due to non-thread-safe access to HNSW index and HDF5 storage.
**Learning:** Concurrent write/search operations in a stateful FastAPI application can lead to `RuntimeError` or `Internal Server Error` if the underlying C++ libraries (like `hnswlib`) or file handles (like `h5py`) are not synchronized. Furthermore, `hnswlib` requires explicit `resize_index` calls if the number of elements exceeds the initial `max_elements`, which can happen during concurrent bursts.
**Prevention:** Implement a global `threading.Lock` within the core database class to wrap all state-modifying operations (add, update, delete) and sensitive reads (metadata filtering). Ensure the index dynamically resizes based on current occupancy to avoid out-of-capacity errors.

## 2026-03-08 - [HDF5 Metadata Truncation & JSON Bypass]
**Vulnerability:** Data corruption and potential security filter bypass due to silent truncation of HDF5 string datasets.
**Learning:** Using fixed-length byte strings (`dtype='S'`) in HDF5/h5py causes subsequent data appends to be truncated if they exceed the initial maximum length. This leads to invalid JSON storage, causing `JSONDecodeError` (DoS) and incorrect metadata filtering.
**Prevention:** Always use variable-length UTF-8 strings (`h5py.string_dtype(encoding='utf-8')`) for HDF5 datasets storing unstructured text or JSON data. Robustly handle both `bytes` and `str` during decoding to ensure compatibility across storage states.

## 2026-03-09 - [HNSW Deletion Resilience]
**Vulnerability:** Application-level Denial-of-Service (DoS) and crash via non-existent vector IDs and uninitialized storage access.
**Learning:** `hnswlib` throws a `RuntimeError` ("Label not found") if `mark_deleted` is called on an ID that doesn't exist, and can segmentation fault if called on an uninitialized index. Furthermore, `h5py` operations on non-existent files or datasets can lead to `FileNotFoundError` or `KeyError`, crashing the API.
**Prevention:** Explicitly catch and ignore "Label not found" errors in `index.py`. Implement guard checks for `self.initialized` in index operations and `os.path.exists` in storage operations. Ensure `load_vectors` returns empty numpy arrays of correct types (e.g., `np.int64`) instead of `None` to prevent `TypeError` in downstream vectorized operations.

## 2026-03-10 - [Non-finite Value Serialization Error]
**Vulnerability:** Information leakage and Denial-of-Service (DoS) via 500 Internal Server Errors caused by non-finite float values (NaN, Inf).
**Learning:** Standard JSON encoders used by FastAPI/Starlette (like Python's `json` module) do not support `NaN` or `Infinity` by default according to the JSON spec (RFC 7159). When these values are returned in an API response (e.g., in distances), the serialization fails with a `ValueError`, resulting in a 500 error that can leak internals or disrupt service.
**Prevention:** Always validate float-based input arrays for finiteness using `np.isfinite(arr).all()` before processing. Return a `400 Bad Request` if non-finite values are detected to ensure the API fails gracefully and securely.
