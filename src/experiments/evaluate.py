"""Retrieval evaluation: hit-rate@k and mean reciprocal rank.

Both are real, standard information-retrieval metrics, computed directly
against the eval fixture (src/experiments/eval_set.py) using only the
existing Retriever -- no external model, no MLflow, and nothing
fabricated. Named explicitly as what they measure (self-retrieval lexical
recall), not presented as a general-purpose RAG quality score.
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.rag.retriever import Retriever


def evaluate_retriever(
    retriever: Retriever, eval_set: List[Dict[str, Any]], top_k: int
) -> Dict[str, float]:
    """For each eval question, checks whether the expected article number
    appears within the top_k retrieved results.
    """
    n = len(eval_set)
    if n == 0:
        return {f"hit_rate_at_{top_k}": 0.0, "mean_reciprocal_rank": 0.0, "n_eval_questions": 0}

    hits = 0
    reciprocal_ranks = []
    for item in eval_set:
        results = retriever.retrieve(item["question"], top_k=top_k)
        rank = None
        for i, r in enumerate(results, start=1):
            if r["article_number"] == item["expected_article_number"]:
                rank = i
                break
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    return {
        f"hit_rate_at_{top_k}": hits / n,
        "mean_reciprocal_rank": sum(reciprocal_ranks) / n,
        "n_eval_questions": n,
    }
