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

## 2026-03-11 - [Data Integrity & Multi-Layer Consistency]
**Vulnerability:** Denial-of-Service (DoS) crash and data loss due to inconsistent state between indexing and storage layers.
**Learning:** In a multi-layered database (Index + Persistent Storage), any auto-generated state (like implicit IDs) must be explicitly returned to the coordinating layer to ensure it is correctly persisted. If the storage layer ends up with fewer IDs than vectors, vectorized operations (like numpy boolean masking during deletion) will crash with a `ValueError` or `IndexError`, leading to a permanent DoS for maintenance operations.
**Prevention:** Ensure all state-generating methods in sub-components return the new state to the caller. Implement explicit length-validation safeguards in the storage layer before performing destructive operations or multi-array indexing.

## 2026-03-12 - [Metadata Alignment & Data Integrity]
**Vulnerability:** Metadata misalignment and cross-contamination due to inconsistent dataset sizes in HDF5.
**Learning:** When using multiple parallel datasets in HDF5 (e.g., vectors, IDs, and metadata), they must be kept in perfect sync. If metadata is only added for some batches, the datasets will drift in size, causing IDs to point to the wrong metadata entries or causing `zip()` operations to truncate data. This can lead to sensitive metadata being associated with the wrong vector, potentially bypassing security filters.
**Prevention:** Always ensure that all parallel datasets are updated together. If a batch is missing metadata, use explicit placeholders (like empty JSON objects). If a dataset is created after other data has already been stored, backfill it with placeholders for the existing records.

## 2026-03-14 - [HNSW Uninitialized Index DoS]
**Vulnerability:** Denial-of-Service (DoS) crash when querying or saving an uninitialized `hnswlib` index.
**Learning:** Calling `knn_query` or `save_index` on a `hnswlib.Index` object that hasn't been initialized via `init_index` or `load_index` can cause the entire Python process to crash (segmentation fault or unhandled C++ exception). This allows an attacker to easily crash the service by simply initializing a database and then performing a search before any data is added.
**Prevention:** Always track the initialization state of the index. Implement guard checks in all methods that interact with the underlying C++ index (search, save, delete) and handle the uninitialized state gracefully (e.g., by returning empty results or returning early).

## 2026-03-15 - [Batch ID Uniqueness]
**Vulnerability:** Potential data corruption and index inconsistency via duplicate IDs in batch requests.
**Learning:** Underlying C++ libraries like `hnswlib` may exhibit undefined behavior or state drift when processing duplicate IDs within a single batch operation (e.g., adding the same ID twice in one call). This can lead to the persistent storage and the in-memory index becoming out of sync.
**Prevention:** Enforce ID uniqueness at the API boundary using Pydantic model validators for all batch endpoints (`/add`, `/update`, `/delete`).
