#!/bin/bash
# Waits for Weaviate to be healthy, then starts the FastAPI app.

echo "Waiting for Weaviate to be ready..."
until wget --no-verbose --tries=1 --spider http://localhost:8080/v1/.well-known/ready 2>/dev/null; do
    sleep 2
done
echo "Weaviate is ready. Starting API..."

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
