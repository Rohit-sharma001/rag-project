"""
main.py
--------
FastAPI backend exposing:
  GET  /health   – liveness check
  POST /ingest    – load + chunk + embed documents from a directory into Chroma
  POST /query     – retrieve relevant chunks and generate a cited answer

Run locally:  uvicorn main:app --reload --port 8000
"""

import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag_engine import get_engine
from evaluation import evaluate_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="Company Filings RAG API",
    description="Retrieval-Augmented Generation over real company 10-K / annual report filings.",
    version="1.0.0",
)

# Allow the Streamlit frontend (different origin/port) to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class IngestRequest(BaseModel):
    directory: Optional[str] = Field(
        default=None, description="Path to a directory of .pdf/.txt/.html filings to ingest"
    )


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(default=4, ge=1, le=15)
    evaluate: bool = Field(default=True, description="Attach lightweight quality scores")


class SourceChunk(BaseModel):
    source: str
    chunk_id: str | int
    preview: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    evaluation: Optional[dict] = None


# ----------------------------------------------------------------------
# Endpoints
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    try:
        engine = get_engine()
        result = engine.ingest_documents(directory=req.directory)
        return result
    except Exception as e:  # noqa: BLE001
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    try:
        engine = get_engine()
        result = engine.query(req.question, top_k=req.top_k)

        evaluation = None
        if req.evaluate:
            evaluation = evaluate_response(
                question=req.question,
                answer=result["answer"],
                retrieved_chunks=result["retrieved_chunks"],
                embeddings=engine.embeddings,
            )

        return QueryResponse(
            answer=result["answer"],
            sources=[SourceChunk(**s) for s in result["sources"]],
            evaluation=evaluation,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
