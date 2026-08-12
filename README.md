# Company Filings RAG Assistant

A production-shaped, containerized Retrieval-Augmented Generation (RAG) app
that answers questions about **real public company filings** (10-Ks from
Apple, Tesla, Microsoft, pulled live from SEC EDGAR) with cited, grounded
answers — and refuses to answer when the context doesn't support it.

```
rag-project/
├── backend/
│   ├── main.py            # FastAPI app — /ingest, /query, /health
│   ├── rag_engine.py       # loading, chunking, embeddings, vector store, LLM chain
│   ├── evaluation.py       # context precision/relevance + faithfulness proxy scoring
│   ├── download_data.py    # pulls real 10-K filings from SEC EDGAR (no API key)
│   ├── requirements.txt
│   ├── .env.example
│   └── Dockerfile
├── frontend/
│   ├── app.py               # Streamlit chat UI, source attribution, retrieval settings
│   ├── requirements.txt
│   └── Dockerfile
├── data/
│   ├── raw/                 # SEC filings land here after download_data.py
│   └── sample/               # 2 tiny offline sample docs so you can test without SEC access
└── docker-compose.yml
```

## Architecture

```
User question ──▶ Streamlit UI ──▶ FastAPI /query
                                        │
                                        ▼
                          Chroma vector search (top_k chunks)
                                        │
                                        ▼
                    Prompt with citation + "not found" instruction
                                        │
                                        ▼
                              OpenAI LLM ──▶ cited answer
                                        │
                                        ▼
                    evaluation.py scores retrieval quality
                                        │
                                        ▼
                   Answer + sources + scores returned to UI
```

**Chunking:** `RecursiveCharacterTextSplitter`, `chunk_size=500`, `chunk_overlap=50`.
**Embeddings:** local `sentence-transformers/all-MiniLM-L6-v2` by default (free,
no API key) — swap to OpenAI embeddings via `.env` if you prefer.
**Vector store:** ChromaDB, persisted to disk (survives container restarts via
a Docker volume).
**Grounding:** the prompt forces per-claim citations like
`[Source: Apple_10K.html, chunk 12]` and a hard fallback —
`"I cannot find this in the ingested company filings."` — when retrieval
comes up empty or off-topic.

## Quickstart

### 1. Get real data (optional — sample docs are included for offline testing)
```bash
cd backend
pip install requests
python download_data.py   # downloads Apple/Tesla/Microsoft 10-Ks into ../data/raw
```

### 2. Configure environment
```bash
cp backend/.env.example backend/.env
# edit backend/.env and add your OPENAI_API_KEY
```

### 3. Run with Docker Compose
```bash
docker compose up --build
```
- Backend: http://localhost:8000/docs (FastAPI auto-generated Swagger UI)
- Frontend: http://localhost:8501

### 4. Ingest and query
In the Streamlit sidebar, click **"Ingest documents"** (points at `../data` by
default, which includes the sample docs even before you run the SEC
downloader). Then just start chatting — e.g. *"What are Tesla's manufacturing
risk factors?"*

## Running without Docker (local dev)
```bash
# Terminal 1
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Terminal 2
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

## Evaluation

Every `/query` response includes three lightweight, zero-extra-cost metrics
(no additional LLM calls needed):
- **context_precision** — fraction of retrieved chunks above a relevance threshold
- **context_relevance** — mean cosine similarity between question and retrieved chunks
- **faithfulness_proxy** — lexical grounding of the answer in retrieved context

For a more rigorous, LLM-judged evaluation report (recommended before you cite
numbers on your resume), run `evaluate_with_ragas()` in `evaluation.py`
offline against a small hand-labeled Q&A set using the
[RAGAS](https://github.com/explodinggym/ragas) library.

## Extending this for your resume

- Swap the corpus for any domain you care about — earnings call transcripts,
  research papers, legal contracts — the pipeline doesn't care.
- Add re-ranking (e.g. Cohere Rerank or a cross-encoder) between retrieval and
  generation to boost precision.
- Add a `/feedback` endpoint + thumbs up/down in the UI, log it, and use it to
  build a small offline eval set — this is the kind of detail that separates
  a toy project from something that reads as "I understand how RAG systems
  are actually operated in production."
- Deploy: push backend to Render/Railway, frontend to Streamlit Community
  Cloud or Hugging Face Spaces, for a live demo link on your resume.

**Suggested resume bullet:**
> Built and containerized a RAG system (FastAPI + Streamlit + ChromaDB) over
> real SEC 10-K filings, with per-claim source citation, a hallucination
> refusal policy, and automated context-precision/faithfulness scoring on
> every query.
