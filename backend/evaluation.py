"""
evaluation.py
--------------
Lightweight, dependency-cheap evaluation of RAG quality:

1. context_precision  – how many retrieved chunks are actually relevant
   to the question (cosine similarity between question & chunk embeddings
   against a threshold).
2. context_relevance   – average similarity score across retrieved chunks
   (a continuous signal, not just a pass/fail count).
3. faithfulness_proxy  – lexical overlap between the generated answer and
   the retrieved context, as a cheap proxy for whether the answer is
   "grounded" in the retrieved chunks (a real implementation would use
   RAGAS's LLM-judged faithfulness metric — see note below).

These are intentionally simple (no extra LLM calls) so they're fast and
free to run on every query. For a more rigorous, paper-backed evaluation,
swap in the `ragas` library (see `evaluate_with_ragas` below) which uses
an LLM-as-judge to score faithfulness, answer relevance, and context
precision/recall against a labeled eval set.
"""

from typing import List, Dict, Any
import re
import numpy as np


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def context_scores(
    question: str,
    retrieved_chunks: List[str],
    embeddings,
    relevance_threshold: float = 0.35,
) -> Dict[str, Any]:
    """Embed the question and each retrieved chunk, then score relevance."""
    if not retrieved_chunks:
        return {"context_precision": 0.0, "context_relevance": 0.0, "per_chunk_scores": []}

    q_vec = np.array(embeddings.embed_query(question))
    chunk_vecs = [np.array(v) for v in embeddings.embed_documents(retrieved_chunks)]

    sims = [_cosine_sim(q_vec, cv) for cv in chunk_vecs]
    relevant_count = sum(1 for s in sims if s >= relevance_threshold)

    return {
        "context_precision": round(relevant_count / len(sims), 3),
        "context_relevance": round(float(np.mean(sims)), 3),
        "per_chunk_scores": [round(s, 3) for s in sims],
    }


def faithfulness_proxy(answer: str, retrieved_chunks: List[str]) -> float:
    """
    Cheap lexical-overlap heuristic: what fraction of the answer's
    'content words' (len > 3, alphabetic) also appear in the retrieved
    context. Not a substitute for LLM-judged faithfulness, but useful as
    a fast sanity check / regression guard in CI.
    """
    if not answer or not retrieved_chunks:
        return 0.0

    context_text = " ".join(retrieved_chunks).lower()
    answer_words = set(
        w for w in re.findall(r"[a-zA-Z]+", answer.lower()) if len(w) > 3
    )
    if not answer_words:
        return 0.0

    grounded = sum(1 for w in answer_words if w in context_text)
    return round(grounded / len(answer_words), 3)


def evaluate_response(
    question: str,
    answer: str,
    retrieved_chunks: List[str],
    embeddings,
) -> Dict[str, Any]:
    """Convenience wrapper combining all lightweight metrics into one payload."""
    ctx = context_scores(question, retrieved_chunks, embeddings)
    faithfulness = faithfulness_proxy(answer, retrieved_chunks)
    return {
        **ctx,
        "faithfulness_proxy": faithfulness,
    }


def evaluate_with_ragas(eval_dataset):
    """
    Optional: real RAGAS evaluation for a proper eval report (requires
    `pip install ragas datasets` and an OpenAI key, since RAGAS uses an
    LLM judge). Kept out of the hot query path — run this offline as a
    batch job over a small labeled Q&A set to produce a report for your
    resume/README, e.g.:

        from ragas import evaluate
        from ragas.metrics import faithfulness, context_precision, answer_relevancy
        result = evaluate(eval_dataset, metrics=[faithfulness, context_precision, answer_relevancy])

    `eval_dataset` should be a HuggingFace `datasets.Dataset` with columns:
    question, answer, contexts, ground_truth.
    """
    from ragas import evaluate  # noqa: E501  (imported lazily; optional dependency)
    from ragas.metrics import faithfulness, context_precision, answer_relevancy

    return evaluate(
        eval_dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
    )
