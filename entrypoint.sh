#!/bin/bash
set -e

CHECKPOINT_PATH="checkpoints/sam2/sam2.1_hiera_small.pt"
R2_URL="${R2_CHECKPOINT_URL}"  # e.g. https://pub-xxx.r2.dev/checkpoints/sam2/sam2.1_hiera_small.pt

# Download SAM2 checkpoint if not already cached
if [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "[entrypoint] Downloading SAM2 checkpoint from R2..."
  curl -L -o "$CHECKPOINT_PATH" "$R2_URL"
  echo "[entrypoint] Checkpoint downloaded."
else
  echo "[entrypoint] Checkpoint already exists, skipping download."
fi

# Start Celery worker in background
echo "[entrypoint] Starting Celery worker..."
celery -A app.tasks.celery_app worker --loglevel=info &

# Start FastAPI server
echo "[entrypoint] Starting FastAPI server..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000