## 2026-02-28 - [FastAPI Decorator & Security Enhancements]
**Vulnerability:** Hardcoded API key, timing attack vulnerability, path traversal risk, and event-loop-blocking decorator bug.
**Learning:** In FastAPI, custom decorators wrapping endpoint functions must use `@functools.wraps` to preserve function signatures for dependency injection (like `Depends(api_key_auth)`). Furthermore, if a decorator is `async def`, it must explicitly handle synchronous endpoint functions using `run_in_threadpool` to avoid blocking the event loop or causing errors when `await` is called on a non-coroutine.
**Prevention:** Always use `secrets.compare_digest` for secret comparisons. Validate user-provided file paths for common traversal patterns. Use `@functools.wraps` and `inspect.iscoroutinefunction` in decorators that wrap web endpoints.

## 2026-02-28 - [HNSW Parameter Validation]
**Vulnerability:** Denial-of-Service (DoS) via unconstrained HNSW index parameters.
**Learning:** Parameters like `ef_construction` and `M` in HNSW index construction directly impact CPU and memory usage. Without upper bounds, an attacker can provide extremely large values that exhaust server resources during index initialization.
**Prevention:** Enforce strict upper limits on resource-intensive algorithm parameters using Pydantic's `Field(le=...)` validation in API request models.
