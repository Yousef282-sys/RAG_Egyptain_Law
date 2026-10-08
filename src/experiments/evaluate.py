"""Retrieval evaluation: hit-rate@k and mean reciprocal rank.

Standard information-retrieval metrics computed directly against an eval
set using only the existing Retriever -- no external model, no MLflow,
nothing fabricated. These measure RETRIEVAL only; they are never a
substitute for the RAGAS generation metrics (src/experiments/ragas_eval.py).

An eval item is a hit at depth k if ANY of its expected articles
(`expected_article_numbers`, or the single legacy `expected_article_number`)
appears among the first k retrieved chunks.

Comparable-across-configs metrics: configs differ in `top_k`, so hit rates
are also reported at fixed depths (COMPARISON_KS) and MRR at depth
MRR_DEPTH, retrieved once at the deepest depth needed. Keys:
  hit_rate_at_{k}         for k in COMPARISON_KS and top_k
  mean_reciprocal_rank    MRR over the first `top_k` results (legacy key)
  mrr_at_10               MRR over the first 10 results (use to compare configs)
  n_eval_questions
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from src.rag.retriever import Retriever

COMPARISON_KS = (1, 3, 5, 10)
MRR_DEPTH = 10


def _expected(item: Dict[str, Any]) -> set:
    if item.get("expected_article_numbers"):
        return set(item["expected_article_numbers"])
    return {item["expected_article_number"]}


def _first_rank(results: List[Dict[str, Any]], expected: set) -> Optional[int]:
    for i, r in enumerate(results, start=1):
        if r["article_number"] in expected:
            return i
    return None


def evaluate_retriever(
    retriever: Retriever,
    eval_set: List[Dict[str, Any]],
    top_k: int,
    ks: Iterable[int] = COMPARISON_KS,
    per_question: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, float]:
    """Compute retrieval metrics. If `per_question` is a list, one dict per
    eval item (id, expected, retrieved article numbers, first hit rank) is
    appended to it -- used as an MLflow artifact.
    """
    depths = sorted(set(ks) | {top_k})
    n = len(eval_set)
    if n == 0:
        zero = {f"hit_rate_at_{k}": 0.0 for k in depths}
        return {**zero, "mean_reciprocal_rank": 0.0, "mrr_at_10": 0.0, "n_eval_questions": 0}

    fetch = max(max(depths), top_k, MRR_DEPTH)
    ranks: List[Optional[int]] = []
    for item in eval_set:
        results = retriever.retrieve(item["question"], top_k=fetch)
        rank = _first_rank(results, _expected(item))
        ranks.append(rank)
        if per_question is not None:
            per_question.append(
                {
                    "id": item.get("id"),
                    "question": item["question"],
                    "expected_article_numbers": sorted(_expected(item)),
                    "retrieved_article_numbers": [r["article_number"] for r in results[:10]],
                    "first_hit_rank": rank,
                }
            )

    def hit_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / n

    def mrr_at(depth: int) -> float:
        return sum(1.0 / r for r in ranks if r is not None and r <= depth) / n

    metrics: Dict[str, float] = {f"hit_rate_at_{k}": hit_at(k) for k in depths}
    metrics["mean_reciprocal_rank"] = mrr_at(top_k)
    metrics["mrr_at_10"] = mrr_at(MRR_DEPTH)
    metrics["n_eval_questions"] = n
    return metrics
