"""Reusable interface for connecting a real generation backend + RAGAS
evaluation to this project's existing retrieval pipeline.

This module does NOT compute a real faithfulness score in this sandbox
(no `ragas` package, no generation backend -- see scripts/ragas_gate.py
for the full reasoning). It exists so that once both exist, wiring them
in is a small, mechanical step rather than a rewrite:
`build_ragas_records` already reuses the existing `Retriever`
(src/rag/retriever.py) and the existing Day 4 eval set
(src/experiments/eval_set.py) -- no retrieval logic is duplicated here.
`evaluate_faithfulness` is a thin wrapper around `ragas.evaluate` for
whenever that package is actually available.
"""
from __future__ import annotations

from typing import Callable, Dict, List

from src.rag.retriever import Retriever

# A real generation backend, once one exists, is any callable with this
# shape: (question, retrieved_context_texts) -> generated_answer_string.
GenerateFn = Callable[[str, List[str]], str]


def build_ragas_records(
    eval_set: List[Dict],
    retriever: Retriever,
    generate_fn: GenerateFn,
    top_k: int = 5,
) -> List[Dict]:
    """For each eval question: retrieve real context via the existing
    Retriever, call `generate_fn` for a real generated answer, and
    package into RAGAS's expected record shape:
    `{question, contexts, answer}`. Never fabricates contexts or
    answers.
    """
    records = []
    for item in eval_set:
        results = retriever.retrieve(item["question"], top_k=top_k)
        contexts = [r["text_ar"] or r["text_en"] or "" for r in results]
        answer = generate_fn(item["question"], contexts)
        records.append(
            {
                "question": item["question"],
                "contexts": contexts,
                "answer": answer,
            }
        )
    return records


def evaluate_faithfulness(records: List[Dict]) -> Dict[str, float]:
    """Thin wrapper around `ragas.evaluate`'s faithfulness metric.

    Raises ImportError if `ragas` is not installed. Callers MUST catch
    this and fail closed -- never substitute a fake or estimated score.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness

    dataset = Dataset.from_list(records)
    result = evaluate(dataset, metrics=[faithfulness])
    return dict(result)
