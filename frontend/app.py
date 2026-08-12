"""
app.py — Streamlit frontend for the Company Filings RAG assistant.

Run locally:  streamlit run app.py
"""

import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Company Filings RAG Assistant", page_icon="📊", layout="wide")

# ------------------------------------------------------------------
# Sidebar — retrieval settings + ingestion trigger
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Chunks to retrieve (top_k)", min_value=1, max_value=10, value=4)
    show_eval = st.checkbox("Show quality scores", value=True)

    st.divider()
    st.header("📥 Data Ingestion")
    ingest_dir = st.text_input("Directory to ingest", value="../data")
    if st.button("Ingest documents", use_container_width=True):
        with st.spinner("Chunking, embedding, and indexing documents..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ingest", json={"directory": ingest_dir}, timeout=300
                )
                resp.raise_for_status()
                result = resp.json()
                if result.get("status") == "success":
                    st.success(
                        f"Indexed {result['chunks_added']} chunks from "
                        f"{result['source_documents']} document(s)."
                    )
                else:
                    st.warning(f"No documents found in `{ingest_dir}`.")
            except Exception as e:  # noqa: BLE001
                st.error(f"Ingestion failed: {e}")

    st.divider()
    st.caption(
        "Corpus: real public company 10-K filings (Apple, Tesla, Microsoft, "
        "etc. — pulled from SEC EDGAR, see `backend/download_data.py`)."
    )

st.title("📊 Company Filings RAG Assistant")
st.caption("Ask questions grounded in real, cited excerpts from company filings.")

# ------------------------------------------------------------------
# Chat state
# ------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source chunk(s)"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}** — chunk `{s['chunk_id']}`")
                    st.text(s["preview"])
                    st.divider()
            if msg.get("evaluation"):
                ev = msg["evaluation"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Context precision", ev.get("context_precision", "—"))
                c2.metric("Context relevance", ev.get("context_relevance", "—"))
                c3.metric("Faithfulness (proxy)", ev.get("faithfulness_proxy", "—"))

# ------------------------------------------------------------------
# Chat input
# ------------------------------------------------------------------
if question := st.chat_input("Ask about revenue, risk factors, manufacturing, etc..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant chunks and generating a cited answer..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": question, "top_k": top_k, "evaluate": show_eval},
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                st.markdown(data["answer"])

                sources = data.get("sources", [])
                if sources:
                    with st.expander(f"📎 {len(sources)} source chunk(s)"):
                        for s in sources:
                            st.markdown(f"**{s['source']}** — chunk `{s['chunk_id']}`")
                            st.text(s["preview"])
                            st.divider()

                evaluation = data.get("evaluation")
                if show_eval and evaluation:
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Context precision", evaluation.get("context_precision", "—"))
                    c2.metric("Context relevance", evaluation.get("context_relevance", "—"))
                    c3.metric("Faithfulness (proxy)", evaluation.get("faithfulness_proxy", "—"))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": data["answer"],
                        "sources": sources,
                        "evaluation": evaluation,
                    }
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Query failed: {e}")
