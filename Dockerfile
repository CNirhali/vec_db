FROM python:3.10-slim

# Security: Create a non-root user for defense-in-depth
RUN groupadd -r vectordb && useradd -r -g vectordb vectordb

WORKDIR /app

# Copy application and set ownership
COPY --chown=vectordb:vectordb . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Security: Switch to non-root user
USER vectordb

EXPOSE 8000

# Security: Use the correct FastAPI application entrypoint
CMD ["uvicorn", "Y_use_FAISS.api:app", "--host", "0.0.0.0", "--port", "8000"]
