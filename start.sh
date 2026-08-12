#!/bin/bash
set -e

cd /app/backend
uvicorn main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
BACKEND_PID=$!

echo "Waiting for backend to start..."
BACKEND_UP=false
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null; then
    echo "Backend is up."
    BACKEND_UP=true
    break
  fi
  # if uvicorn already died, no point waiting the full 30s
  if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Backend process exited early."
    break
  fi
  sleep 1
done

if [ "$BACKEND_UP" = false ]; then
  echo "ERROR: Backend failed to start within 30s. Backend log:"
  cat /tmp/backend.log
  exit 1
fi

cd /app/frontend
streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}