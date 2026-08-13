# Company Filings RAG Assistant

A production-oriented **Retrieval-Augmented Generation (RAG)** application for querying SEC EDGAR company filings using semantic search and citation-grounded LLM responses.

The system ingests company filings, transforms them into searchable chunks, generates semantic embeddings, stores them in a persistent vector database, and retrieves relevant context to generate grounded answers.

## 🚀 Features

* **SEC EDGAR Filing Search** — Query company 10-K filings using natural language.
* **Retrieval-Augmented Generation** — Retrieves relevant document context before generating an answer.
* **Semantic Search** — Uses Sentence Transformers to generate embeddings and ChromaDB for vector retrieval.
* **Citation-Grounded Responses** — Answers are supported by retrieved document sources.
* **Unsupported Query Handling** — The system avoids generating unsupported answers when relevant context cannot be retrieved.
* **Configurable Retrieval** — Control the number of retrieved documents using Top-K search.
* **Interactive UI** — Streamlit-based interface for document ingestion and querying.
* **REST API** — FastAPI backend exposing the core application functionality.
* **Persistent Vector Store** — ChromaDB maintains the indexed document embeddings between sessions.
* **Retrieval Monitoring** — Provides retrieval-quality information for evaluating search results.
* **Docker Support** — Backend, frontend, and vector storage can be run using Docker Compose.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │   SEC EDGAR Filings  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  Document Ingestion  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Chunking        │
                    │ 500 tokens / 50      │
                    │ token overlap        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sentence Transformers│
                    │ Embeddings            │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     ChromaDB         │
                    │ Persistent Vector DB │
                    └──────────┬───────────┘
                               │
                    User Query │
                               ▼
                    ┌──────────────────────┐
                    │ Semantic Retrieval   │
                    │      Top-K Search    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ LLM + Retrieved      │
                    │ Context              │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Grounded Answer +    │
                    │ Citations            │
                    └──────────────────────┘
```

---

## 🛠️ Tech Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* Streamlit

### RAG / AI

* LangChain
* Sentence Transformers
* Hugging Face
* Groq LLM

### Data & Storage

* ChromaDB
* SEC EDGAR filings

### DevOps

* Docker
* Docker Compose
* Git / GitHub

---

## 📁 Project Structure

```text
rag-company-filings/
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   ├── .env
│   ├── .venv/
│   └── chroma_store/
│
├── frontend/
│   ├── app.py
│   ├── requirements.txt
│   └── .venv/
│
├── data/
│   ├── sample/
│   └── raw/
│
├── download_data.py
├── docker-compose.yml
├── .gitignore
└── README.md
```

> `.env`, virtual environments, ChromaDB data, and raw downloaded filings are intentionally excluded from Git.

---

# ⚙️ Local Setup

## 1. Clone the Repository

```powershell
cd Downloads
git clone https://github.com/Rohit-sharma001/rag-project.git
cd rag-project
```

---

## 2. Set Up the Backend

Open a PowerShell terminal:

```powershell
cd backend

python -m venv .venv

.\.venv\Scripts\Activate.ps1

pip install --upgrade pip

pip install -r requirements.txt
```

---

## 3. Configure Environment Variables

The `.env` file is intentionally excluded from the repository because it contains API credentials.

Create it inside the `backend` directory:

```powershell
notepad .env
```

Add:

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your_key_here
GROQ_MODEL=llama-3.1-8b-instant

EMBEDDING_PROVIDER=huggingface
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

CHUNK_SIZE=500
CHUNK_OVERLAP=50

CHROMA_PERSIST_DIR=./chroma_store
DATA_DIR=../data

CORPUS_LABEL=the ingested company filings
```

Replace:

```text
gsk_your_key_here
```

with your own Groq API key.

**Never commit ****`.env`**** or expose your API key publicly.**

---

## 4. Verify Sample Data

Sample documents are included in the repository.

```powershell
dir ..\data\sample
```

If you want to download the complete set of raw SEC filings:

```powershell
pip install requests
python download_data.py
```

Raw downloaded files are stored locally and are excluded from Git.

---

# ▶️ Running the Application

The application consists of two components:

```text
Frontend (Streamlit)
        │
        ▼
Backend (FastAPI)
        │
        ▼
RAG Pipeline
        │
        ▼
ChromaDB + LLM
```

## Terminal 1 — Start the Backend

```powershell
cd Downloads\rag-company-filings\backend

.\.venv\Scripts\Activate.ps1

uvicorn main:app --reload --port 8000 --reload-exclude ".venv/*"
```

The FastAPI server will be available at:

```text
http://localhost:8000
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

---

## Terminal 2 — Start the Frontend

Open another PowerShell terminal:

```powershell
cd Downloads\rag-company-filings\frontend

.\.venv\Scripts\Activate.ps1

streamlit run app.py
```

Streamlit will provide a local URL in the terminal, typically:

```text
http://localhost:8501
```

---

# 🔄 Running the Project Again

After closing VS Code or shutting down your computer, **you do not need to reinstall the dependencies or recreate the virtual environments**.

The virtual environments remain stored locally.

Every time you return to the project, simply start the two servers again.

### Backend

```powershell
cd Downloads\rag-company-filings\backend
.\.venv\Scripts\Activate.ps1
uvicorn main:app --reload --port 8000 --reload-exclude ".venv/*"
```

### Frontend

Open a second terminal:

```powershell
cd Downloads\rag-company-filings\frontend
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

That's all.

You **do not need to run ****`pip install`**** again** unless the project's `requirements.txt` has changed.

You also **do not need to re-ingest the documents** if your existing `chroma_store` is still present, because the vector database persists the previously indexed documents locally.

### Important

Both terminals must remain open while using the application.

Start the **backend first**, wait until it reports that the application has started, and then launch Streamlit.

---

# 🔎 Using the Application

1. Start the FastAPI backend.
2. Start the Streamlit frontend.
3. Open the Streamlit URL in your browser.
4. Use **Ingest Documents** to index the available filings.
5. Enter a natural-language question.
6. The system retrieves the most relevant document chunks.
7. The LLM generates an answer using the retrieved context.
8. Relevant sources are displayed alongside the response.

Example queries:

```text
What are Apple's major business risks?

How did Tesla's revenue change?

What were Microsoft's major operating expenses?

What risks did the company report in its latest 10-K?
```

---

# 🧠 RAG Pipeline

The application follows a standard retrieval-augmented generation workflow:

### 1. Document Ingestion

SEC EDGAR company filings are loaded into the application.

### 2. Text Chunking

Documents are divided into smaller chunks using:

```text
Chunk size: 500 tokens
Chunk overlap: 50 tokens
```

This allows relevant sections to be retrieved without passing entire filings to the LLM.

### 3. Embedding Generation

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert text chunks into dense vector representations.

### 4. Vector Storage

Embeddings and document metadata are stored persistently in **ChromaDB**.

### 5. Semantic Retrieval

When a user submits a question, the system performs similarity search and retrieves the most relevant chunks using configurable **Top-K retrieval**.

### 6. Grounded Generation

The retrieved context is provided to the LLM, which generates an answer based on the available evidence.

### 7. Citation & Reliability

Retrieved sources are associated with the generated answer, while unsupported queries are handled explicitly instead of blindly generating a response.

---

# 📊 Retrieval Evaluation

The application includes retrieval-quality monitoring to help evaluate the RAG pipeline.

The system tracks signals related to:

* Context precision
* Context relevance
* Faithfulness

These metrics help identify whether the retrieval pipeline is returning useful context and whether generated responses remain grounded in the retrieved information.

---

# 🐳 Docker

The application also supports Docker Compose for running the project in a reproducible environment.

Build and start the services:

```powershell
docker compose up --build
```

Stop the services:

```powershell
docker compose down
```

Persistent application data can be retained through the configured storage volumes.

---

# 🔐 Environment & Security

The following files/directories are intentionally excluded from Git:

```text
.env
.venv/
chroma_store/
data/raw/
```

This prevents:

* API keys from being committed
* Local Python environments from being uploaded
* Generated vector databases from bloating the repository
* Large raw filing datasets from being committed

If you clone the repository on a new machine, recreate `.env` and the virtual environments locally.

---

# 🚧 Future Improvements

* Add RAGAS-based evaluation for more comprehensive RAG benchmarking
* Support additional SEC filing types
* Add more companies and historical filings
* Introduce hybrid keyword + semantic retrieval
* Add reranking for improved retrieval accuracy
* Deploy the complete application to a cloud platform
* Add automated CI/CD testing and deployment

---

# 👨‍💻 Author

**Rohit Sharma**

B.Tech Software Engineering
Delhi Technological University

* GitHub: [Rohit-sharma001](https://github.com/Rohit-sharma001)
