FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/requirements.txt
COPY frontend/requirements.txt frontend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && pip install --no-cache-dir -r frontend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY data/ data/

# Streamlit will bind to the port HF Spaces expects (7860); FastAPI runs
# internally on 8000 and is only reached by Streamlit inside the container.
ENV BACKEND_URL=http://localhost:8000

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Render (and most platforms) assign a dynamic port via $PORT — default to
# 7860 only as a local-testing fallback.
EXPOSE 8000

CMD ["/app/start.sh"]