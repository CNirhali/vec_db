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
## 2026-03-17 - [ID Collision & Index Recovery]
**Vulnerability:** Deliberate ID collisions in batch requests and data loss after server restarts due to lack of automatic index restoration.
**Learning:** In a vector database using separate indexing and storage layers, allowing duplicate IDs in a single request can lead to inconsistent state or internal crashes in the indexing library. Furthermore, if the in-memory index is not automatically restored from persistent storage upon restart, the database becomes unsearchable even if data exists on disk.
**Prevention:** Implement strict uniqueness validation for IDs in all batch endpoints. Enhance the initialization process to automatically load the binary index or rebuild it from the underlying storage layer if the index file is missing. Avoid synchronous full-index writes on every small update to mitigate Disk I/O Denial-of-Service (DoS) risks.

## 2026-03-18 - [Explicit Persistence & Thread-Safe Index Saving]
**Vulnerability:** Data loss or corruption due to missing manual persistence endpoint and non-thread-safe index saving.
**Learning:** In high-concurrency environments, calling `save_index` on a shared state without proper synchronization can lead to race conditions where the index file is written in an inconsistent state if updates are happening simultaneously. Furthermore, relying purely on implicit saves or lacking a manual save trigger can lead to significant data loss if the process terminates unexpectedly between automatic save intervals.
**Prevention:** Implement an explicit `/save` API endpoint for manual persistence. Ensure all persistence operations are wrapped in the same global database lock used by modification methods (`add`, `update`, `delete`) to guarantee state consistency during the write-to-disk process.

## 2026-03-19 - [Robust Search Parameters & Safe Metadata Comparison]
**Vulnerability:** Denial-of-Service (DoS) via 500 Internal Server Errors caused by unvalidated search parameters (k > count) and unhashable metadata types.
**Learning:** In `hnswlib`, requesting more neighbors (`k`) than available items can lead to a `RuntimeError` that crashes the request. Additionally, performing set-based metadata comparisons (like `items() <= items()`) raises a `TypeError` if metadata values are unhashable (e.g., lists), leading to further 500 errors.
**Prevention:** Always cap the requested `k` to the current number of elements in the index. Use safer comparison patterns like `all()` with `.get()` for metadata filtering to robustly handle diverse JSON-compatible data types without crashing the server.

## 2026-03-20 - [Integer Overflow & Strict Path Whitelisting]
**Vulnerability:** Service-level Denial-of-Service (DoS) via `OverflowError` and potential injection risks in file paths.
**Learning:** Large Python integers exceeding the 64-bit unsigned limit ($2^{64}-1$) cause `OverflowError` when passed to C-extension libraries like `hnswlib`, resulting in a 500 Internal Server Error. Furthermore, relying solely on blacklisting (like `..`) for path traversal is less robust than whitelisting allowed characters.
**Prevention:** Explicitly validate that all user-provided IDs do not exceed $18,446,744,073,709,551,615$. Use a strict regex whitelist (e.g., `^[a-zA-Z0-9_\-\./]+$`) for storage paths to prevent the use of shell metacharacters or other unexpected input.

## 2026-03-22 - [Automatic ID Collision Prevention]
**Vulnerability:** ID collisions and data corruption during automatic ID generation.
**Learning:** Relying on 'get_current_count()' from the indexing library to generate new IDs is unreliable when vectors have been deleted or when the index is rebuilt from a subset of data. This leads to new vectors being assigned IDs that are already in use in the persistent storage, causing metadata misalignment and data corruption.
**Prevention:** Implement a 'max_id' tracker within the index class that persists across the application's lifecycle. Always generate new IDs by incrementing the highest ID ever used, and re-calculate this value from the existing dataset upon index initialization or loading.

## 2026-03-24 - [Dimension Consistency & Secure Exception Handling]
**Vulnerability:** Data corruption via dimension mismatch and information leakage via unhandled OSErrors.
**Learning:** Re-initializing a database with a different dimension than the existing HDF5 storage leads to silent data corruption (truncation) during subsequent operations. Furthermore, narrow exception handling (only `ValueError`) in API endpoints results in 500 Internal Server Errors when file system issues (`OSError`) occur, which can leak stack traces or system details.
**Prevention:** Verify requested dimensions against existing storage during initialization and raise a `ValueError` on mismatch. Use broader exception handling (catching both `ValueError` and `OSError`) in API endpoints and return generic error messages for system-level errors to prevent information leakage.

## 2026-03-25 - [Result-Set Denial-of-Service (DoS)]
**Vulnerability:** Memory exhaustion and service degradation via extremely large search result sets.
**Learning:** Even with strict input validation on query batch sizes and vector dimensions, an attacker can trigger a Denial-of-Service by requesting a high number of neighbors (`k`) across many queries. The combined result set can lead to massive memory consumption and prolonged JSON serialization times.
**Prevention:** Enforce a strict upper limit on the *total* number of results (e.g., `len(queries) * k`) in the API validation layer. Additionally, optimize the backend to avoid loading unnecessary datasets (like full vectors) when performing metadata-only filtered searches.

## 2026-03-26 - [Memory-Optimized Metadata Filtering & Input Validation]
**Vulnerability:** Denial-of-Service (OOM) during filtered searches and unnecessary processing of empty requests.
**Learning:** Loading the entire metadata dataset into memory for post-filtering search results creates a significant memory-based DoS vector as the database grows. Furthermore, accepting empty batch requests for deletions leads to unnecessary Disk I/O and potential resource waste.
**Prevention:** Implement indexed metadata loading by mapping search result labels to their positional indices in the HDF5 storage. Use Pydantic's `min_length=1` to enforce non-empty batches for destructive or resource-intensive API endpoints.

## 2026-03-27 - [Global Exception Handling & Information Leakage]
**Vulnerability:** Information leakage and fragile error state via unhandled system-level exceptions (OSError, RuntimeError).
**Learning:** Relying on per-endpoint exception handling is prone to omissions as the API grows. Unhandled system-level exceptions from underlying libraries like `h5py` or `hnswlib` can result in 500 Internal Server Errors that leak internal details (stack traces, file paths) to the client. Using a global exception handler ensures a consistent, secure failure mode across the entire application.
**Prevention:** Implement global exception handlers for `OSError` and `RuntimeError` that log the full error context internally but return a generic, secure `JSONResponse` with a 500 status code to the client. This maintains security without sacrificing observability.

## 2026-03-29 - [Global Auth & Non-Root Container Security]
**Vulnerability:** Information leakage of API schema to unauthenticated users and root-level container execution risk.
**Learning:** In FastAPI, placing `Depends(api_key_auth)` in individual endpoint signatures allows Pydantic to perform body validation *before* the authentication dependency is resolved. This can leak schema details (via 422 errors) to unauthenticated attackers. Furthermore, running the container as root provides an unnecessarily large attack surface.
**Prevention:** Enforce authentication globally using `app = FastAPI(dependencies=[Depends(api_key_auth)])` to ensure it runs before body parsing. Implement a non-privileged system user in the `Dockerfile` and use `COPY --chown` to maintain the Principle of Least Privilege.

## 2026-03-31 - [Subdirectory Injection via Path Whitelist]
**Vulnerability:** File creation/overwrite in arbitrary subdirectories via allowed directory separators in path whitelist.
**Learning:** Whitelisting characters like `/` in a `storage_path` allows attackers to escape the intended "flat" storage directory and interact with other parts of the filesystem (e.g., application code or configuration directories) if they exist. Even if `..` is blocked, subdirectory injection can still lead to unauthorized file manipulation.
**Prevention:** For stateful applications managing their own storage, enforce a flat file structure by strictly forbidding directory separators (`/`, `\`) in user-provided filenames.

## 2026-04-02 - [FastAPI Schema Discovery & Information Leakage]
**Vulnerability:** Exposure of API schema and documentation to unauthenticated users.
**Learning:** In FastAPI, default documentation endpoints (`/docs`, `/redoc`) and the OpenAPI specification (`/openapi.json`) are enabled by default and are not automatically covered by global dependencies that require authentication if they are mounted before the dependency is evaluated. This allows unauthenticated attackers to discover the full API structure, models, and validation rules, facilitating the crafting of more targeted attacks.
**Prevention:** Explicitly disable documentation endpoints in production environments by setting `docs_url=None`, `redoc_url=None`, and `openapi_url=None` in the `FastAPI` constructor unless they are specifically required and protected by an additional authentication layer.

## 2026-04-05 - [API Key Brute-Force Protection]
**Vulnerability:** Lack of rate limiting on failed authentication attempts allowed for potential API key brute-force attacks.
**Learning:** Global authentication dependencies in FastAPI run before route-specific rate limit decorators. If authentication fails, the request is rejected before the standard rate limiter can track it, leaving the authentication mechanism unprotected from brute-force attempts.
**Prevention:** Implement manual rate limiting within the authentication dependency using `limiter.limiter.hit()`. This ensures that even failed attempts are tracked and throttled, providing defense-in-depth for the API's entry point.

## 2026-05-15 - [ID Collision & Metadata Alignment]
**Vulnerability:** Cross-tenant data leakage and Denial-of-Service (DoS) via ID collisions and O(N) disk scans.
**Learning:** Allowing duplicate IDs in the `/add` endpoint causes the indexing layer (HNSW) to store multiple vectors for the same ID, while the storage layer (HDF5) might only associate that ID with a single (latest) metadata entry. This leads to data leakage where vectors from different tenants sharing the same ID are returned together. Furthermore, performing metadata filtering by scanning the entire ID dataset from disk is an O(N) operation that creates a major DoS vector.
**Prevention:** Enforce ID uniqueness at the application layer by maintaining an in-memory `id_to_pos` hash map. This allows for O(1) collision checks during additions and efficient indexed retrieval of metadata during searches, significantly improving both security and performance.

## 2026-04-10 - [Auth Token Depletion & Middleware Robustness]
**Vulnerability:** Double consumption of rate-limit tokens and 500 errors via malformed HTTP headers.
**Learning:** Sequential calls to `limiter.limiter.hit()` in a custom FastAPI dependency consume multiple tokens for a single request, causing rate limits to trigger twice as fast as intended. Furthermore, performing direct `int()` casting on user-controlled headers like `Content-Length` in global middleware can crash the request with a 500 error if the header is malformed (e.g., non-integer), which can be exploited for DoS or probing.
**Prevention:** Consolidate manual rate-limiting calls into a single check. Always wrap header parsing logic in `try-except` blocks within middleware to ensure the application fails gracefully with a 400 Bad Request instead of a 500 Internal Server Error.
