#!/bin/bash
set -e

# Start FastAPI backend in the background
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Wait for the backend to be ready before starting the frontend
echo "Waiting for backend to start..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null; then
    echo "Backend is up."
    break
  fi
  sleep 1
done

# Start Streamlit in the foreground (this is what keeps the container alive).
# Render (and similar platforms) injects $PORT — fall back to 8501 locally.
cd /app/frontend
streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}