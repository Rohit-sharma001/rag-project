"""
rag_engine.py
--------------
Core RAG logic: document loading, recursive chunking, embedding generation,
vector store management, and the retrieval-augmented answer chain.

Domain: public SEC 10-K / annual filings for real companies (Apple, Tesla,
Microsoft, etc.) — see backend/download_data.py to pull real filings from
SEC EDGAR (public, no API key required).
"""

import os
import glob
import logging
from typing import List, Dict, Any

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
# NOTE: langchain_chroma, langchain_huggingface, langchain_openai, langchain_groq
# are intentionally NOT imported here at module level. Each pulls in a heavy
# dependency (torch, chromadb's native deps, etc.) — importing them eagerly
# at app boot is what was causing an out-of-memory crash on Render's free
# 512MB instance before the server even finished starting. They're imported
# lazily inside the functions that actually use them instead.

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_engine")

# ----------------------------------------------------------------------
# Config (all overridable via .env)
# ----------------------------------------------------------------------
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "huggingface")  # "huggingface" | "huggingface_api" | "openai"
HF_EMBEDDING_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_TOKEN = os.getenv("HF_TOKEN", "")  # required only for "huggingface_api" (free at hf.co/settings/tokens)
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")  # "groq" | "ollama" | "openai"
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OPENAI_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
DATA_DIR = os.getenv("DATA_DIR", "../data")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
COMPANY_LABEL = os.getenv("CORPUS_LABEL", "the ingested company filings")

CITATION_PROMPT = PromptTemplate(
    input_variables=["context", "question", "company_label"],
    template="""You are a precise financial/corporate research assistant.
Answer the question using ONLY the CONTEXT below, which is drawn from real
company filings (10-Ks, annual reports, etc.).

Rules:
1. Every factual claim must be followed by a citation in the form [Source: <file>, chunk <id>].
2. If the answer is not present in the context, respond exactly with:
   "I cannot find this in {company_label}."
   Do not guess or use outside knowledge.
3. Be concise and quote specific figures/terms where relevant.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with citations):""",
)


def _load_html_as_document(path: str) -> List[Document]:
    """Lightweight HTML -> plain text loader (avoids the heavy `unstructured` dependency)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return [Document(page_content=text, metadata={"source": os.path.basename(path)})]


class RAGEngine:
    """Encapsulates the full ingest -> embed -> store -> retrieve -> answer pipeline."""

    def __init__(self):
        self.embeddings = self._load_embeddings()
        self.vectorstore = self._load_or_init_vectorstore()
        self.llm = self._load_llm()

    def _load_llm(self):
        if LLM_PROVIDER == "ollama":
            from langchain_ollama import ChatOllama
            logger.info("Using local Ollama model: %s", OLLAMA_MODEL)
            return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)
        elif LLM_PROVIDER == "openai":
            from langchain_openai import ChatOpenAI
            logger.info("Using OpenAI model: %s", OPENAI_LLM_MODEL)
            return ChatOpenAI(model=OPENAI_LLM_MODEL, temperature=0)
        else:
            from langchain_groq import ChatGroq
            logger.info("Using Groq model: %s", GROQ_MODEL)
            return ChatGroq(model=GROQ_MODEL, temperature=0)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------
    def _load_embeddings(self):
        if EMBEDDING_PROVIDER == "openai":
            from langchain_openai import OpenAIEmbeddings
            logger.info("Using OpenAI embeddings: %s", OPENAI_EMBEDDING_MODEL)
            return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        elif EMBEDDING_PROVIDER == "huggingface_api":
            # Lightweight: calls HuggingFace's free hosted Inference API over
            # HTTP instead of loading the model (and torch) locally. Ideal
            # for low-memory deployments like Render's free tier.
            from langchain_huggingface import HuggingFaceEndpointEmbeddings
            logger.info("Using HuggingFace Inference API embeddings: %s", HF_EMBEDDING_MODEL)
            return HuggingFaceEndpointEmbeddings(
                model=HF_EMBEDDING_MODEL,
                huggingfacehub_api_token=HF_TOKEN,
            )
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Using local HuggingFace embeddings: %s", HF_EMBEDDING_MODEL)
        return HuggingFaceEmbeddings(model_name=HF_EMBEDDING_MODEL)

    def _load_or_init_vectorstore(self):
        from langchain_chroma import Chroma
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        return Chroma(
            collection_name="company_filings",
            embedding_function=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------
    def _load_documents(self, directory: str) -> List[Document]:
        docs: List[Document] = []
        patterns = {
            "*.txt": lambda p: TextLoader(p, encoding="utf-8").load(),
            "*.pdf": lambda p: PyPDFLoader(p).load(),
            "*.html": _load_html_as_document,
            "*.htm": _load_html_as_document,
        }
        for pattern, loader_fn in patterns.items():
            for path in glob.glob(os.path.join(directory, "**", pattern), recursive=True):
                try:
                    loaded = loader_fn(path)
                    for d in loaded:
                        d.metadata["source"] = os.path.basename(path)
                    docs.extend(loaded)
                    logger.info("Loaded %s (%d docs)", path, len(loaded))
                except Exception as e:  # noqa: BLE001
                    logger.warning("Failed to load %s: %s", path, e)
        return docs

    def ingest_documents(self, directory: str = None) -> Dict[str, Any]:
        directory = directory or DATA_DIR
        raw_docs = self._load_documents(directory)
        if not raw_docs:
            return {"status": "no_documents_found", "directory": directory, "chunks_added": 0}

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(raw_docs)

        # Give each chunk a stable id for citation purposes
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = i

        self.vectorstore.add_documents(chunks)
        # Note: langchain_chroma.Chroma auto-persists on write — no explicit
        # .persist() call needed (older langchain_community.vectorstores.Chroma
        # required one; this package doesn't).

        return {
            "status": "success",
            "directory": directory,
            "source_documents": len(raw_docs),
            "chunks_added": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        }

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------
    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs: List[Document] = retriever.invoke(question)

        if not retrieved_docs:
            return {
                "answer": f"I cannot find this in {COMPANY_LABEL}.",
                "sources": [],
                "retrieved_chunks": [],
            }

        context = "\n\n".join(
            f"[{d.metadata.get('source', 'unknown')}, chunk {d.metadata.get('chunk_id', '?')}]\n{d.page_content}"
            for d in retrieved_docs
        )

        prompt = CITATION_PROMPT.format(
            context=context, question=question, company_label=COMPANY_LABEL
        )
        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "chunk_id": d.metadata.get("chunk_id", "?"),
                "preview": d.page_content[:300],
            }
            for d in retrieved_docs
        ]

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": [d.page_content for d in retrieved_docs],
        }


# Singleton accessor so FastAPI doesn't reload the model on every request
_engine_instance = None


def get_engine() -> RAGEngine:
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = RAGEngine()
    return _engine_instance
