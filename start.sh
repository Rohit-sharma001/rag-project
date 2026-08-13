#!/bin/bash
set -e

export PYTHONUNBUFFERED=1

# Start FastAPI backend in the background, capturing its output so we can
# inspect it if startup fails (backgrounded processes can otherwise have
# their logs swallowed/delayed).
cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to start..."
BACKEND_UP=false
for i in $(seq 1 40); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "Backend is up."
    BACKEND_UP=true
    break
  fi
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend process exited early. Log output:"
    cat /tmp/backend.log
    exit 1
  fi
  sleep 1
done

if [ "$BACKEND_UP" = false ]; then
  echo "ERROR: Backend failed to start within timeout. Log output:"
  cat /tmp/backend.log
  exit 1
fi

# Start Streamlit in the foreground (this is what keeps the container alive).
# Render (and similar platforms) injects $PORT — fall back to 8501 locally.
cd /app/frontend
streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}
