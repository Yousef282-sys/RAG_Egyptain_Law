"""RAGAS evaluation of the real RAG pipeline: the four required metrics --
faithfulness, answer relevancy, context precision, context recall.

Pipeline: for each eval question run the SAME service the API serves
(src/rag/service.py: retrieve -> vLLM generate), collect
{question, contexts, answer, reference}, then score with RAGAS using a
judge LLM and an embedding model:

    records = build_ragas_records_from_service(eval_set, service)
    llm, embeddings = build_judges()          # env-configured, see below
    summary = evaluate_ragas(records, llm, embeddings)

Nothing here ever invents a score: a missing/unconfigured piece raises
RagasUnavailable; per-row failures come back from RAGAS as NaN and are
counted as 0.0 in the gating score (conservative, fail-closed) while the
valid-row count is reported separately.

STATUS: UNVERIFIED at runtime. `ragas`, the judge LLM, and a generation
backend are not available in the development sandbox. The integration is
written against ragas 0.2.x (`EvaluationDataset`, `Faithfulness`,
`ResponseRelevancy`, `LLMContextPrecisionWithReference`,
`LLMContextRecall`; pinned `ragas>=0.2.6,<0.3` in pyproject). Everything
that does not need ragas (record building, row mapping, score
normalisation/aggregation) is unit-tested offline with stand-ins.

Judge configuration (environment):
  RAGAS_JUDGE_MODEL      judge LLM name (required)
  RAGAS_JUDGE_BASE_URL   OpenAI-compatible endpoint (e.g. a vLLM server);
                         unset -> api.openai.com
  RAGAS_JUDGE_API_KEY    API key (falls back to OPENAI_API_KEY; not needed
                         for a keyless local endpoint)
  RAGAS_EMBEDDING_MODEL  sentence-transformers model for answer relevancy
                         (default: the retrieval embedding model)
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.rag.chunking import full_text
from src.rag.retriever import Retriever

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")

# A real generation backend is any callable with this shape.
GenerateFn = Callable[[str, List[str]], str]


class RagasUnavailable(RuntimeError):
    """ragas / judge LLM / embeddings are not installed or not configured."""


# ----------------------------- records -----------------------------------


def build_ragas_records(
    eval_set: List[Dict],
    retriever: Retriever,
    generate_fn: GenerateFn,
    top_k: int = 5,
) -> List[Dict]:
    """Retrieve real context via the existing Retriever, call `generate_fn`
    for a real generated answer, package as RAGAS records. Never fabricates
    contexts or answers. `ground_truth` is the eval item's corpus-derived
    `reference` (needed for context precision/recall) when present.
    """
    records = []
    for item in eval_set:
        results = retriever.retrieve(item["question"], top_k=top_k)
        contexts = [full_text(r["text_ar"], r["text_en"]) for r in results]
        answer = generate_fn(item["question"], contexts)
        records.append(
            {
                "question": item["question"],
                "contexts": contexts,
                "answer": answer,
                "ground_truth": item.get("reference"),
                "retrieved_article_numbers": [r["article_number"] for r in results],
            }
        )
    return records


def build_ragas_records_from_service(eval_set: List[Dict], service) -> List[Dict]:
    """Same, but through RagService.ask -- i.e. exactly what /ask serves,
    including the article-numbered prompt, so the generator can cite.
    """
    records = []
    for item in eval_set:
        out = service.ask(item["question"])
        records.append(
            {
                "id": item.get("id"),
                "question": item["question"],
                "contexts": [s["text"] for s in out["sources"]],
                "answer": out["answer"],
                "ground_truth": item.get("reference"),
                "retrieved_article_numbers": [s["article_number"] for s in out["sources"]],
            }
        )
    return records


def to_ragas_rows(records: List[Dict]) -> List[Dict]:
    """Map our records to the RAGAS 0.2 single-turn sample schema. Only
    these four keys are passed (extra keys are rejected by RAGAS).
    """
    rows = []
    for r in records:
        if not r.get("ground_truth"):
            raise ValueError(
                "every record needs a ground_truth/reference for context "
                "precision and recall (use the curated eval set)"
            )
        rows.append(
            {
                "user_input": r["question"],
                "retrieved_contexts": list(r["contexts"]),
                "response": r["answer"],
                "reference": r["ground_truth"],
            }
        )
    return rows


# ---------------------------- aggregation --------------------------------


def _is_nan(x: Any) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def normalize_metric_names(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map whatever column names the installed ragas version emitted
    (e.g. `llm_context_precision_with_reference`) onto our four names.
    """
    out: Dict[str, Any] = {}
    for key, value in row.items():
        k = key.lower()
        if "faithful" in k:
            out["faithfulness"] = value
        elif "relevan" in k:
            out["answer_relevancy"] = value
        elif "precision" in k:
            out["context_precision"] = value
        elif "recall" in k:
            out["context_recall"] = value
    return out


def summarize_scores(rows: List[Dict[str, Any]], expected_n: Optional[int] = None) -> Dict[str, Any]:
    """Aggregate per-question RAGAS rows.

    `scores[m]` is the GATING score: the mean over ALL questions with NaN
    (failed/unscorable rows) counted as 0.0 -- so judge failures can only
    lower, never inflate, a score. `n_valid[m]` and `mean_valid[m]` report
    the scorable rows separately. Raises ValueError if any of the four
    metrics is absent from the rows.
    """
    n = expected_n if expected_n is not None else len(rows)
    if len(rows) != n:
        raise ValueError(f"expected {n} scored rows, got {len(rows)}")
    normalised = [normalize_metric_names(r) for r in rows]
    missing = [m for m in RAGAS_METRICS if not all(m in r for r in normalised)]
    if missing:
        raise ValueError(f"RAGAS result is missing metrics: {missing}")

    scores, n_valid, mean_valid = {}, {}, {}
    for m in RAGAS_METRICS:
        values = [r[m] for r in normalised]
        valid = [float(v) for v in values if not _is_nan(v)]
        scores[m] = sum(valid) / n if n else 0.0
        n_valid[m] = len(valid)
        mean_valid[m] = sum(valid) / len(valid) if valid else None
    return {"n": n, "scores": scores, "n_valid": n_valid, "mean_valid": mean_valid}


# ----------------------------- RAGAS -------------------------------------


def judge_config_problems() -> List[str]:
    """Unmet judge requirements (empty list == configured)."""
    problems = []
    if not os.environ.get("RAGAS_JUDGE_MODEL"):
        problems.append("RAGAS_JUDGE_MODEL is not set (judge LLM)")
    if not (
        os.environ.get("RAGAS_JUDGE_BASE_URL")
        or os.environ.get("RAGAS_JUDGE_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    ):
        problems.append("no judge endpoint/key: set RAGAS_JUDGE_BASE_URL or RAGAS_JUDGE_API_KEY/OPENAI_API_KEY")
    return problems


def ragas_environment_problems() -> List[str]:
    """Everything needed for a REAL RAGAS run that is currently missing
    (empty list == ready): ragas installed, a generation backend, a judge.
    """
    problems = []
    try:
        import ragas  # noqa: F401
    except ImportError:
        problems.append("`ragas` is not installed/importable (pip install -e '.[ragas]')")
    if not os.environ.get("GENERATION_MODEL"):
        problems.append(
            "no generation backend: GENERATION_MODEL is unset (RAGAS scores a "
            "generated answer; the retrieval-only fallback is not evaluated)"
        )
    problems.extend(judge_config_problems())
    return problems


def build_judges() -> Tuple[Any, Any]:
    """Judge LLM + embeddings wrapped for RAGAS, from the environment.
    Raises RagasUnavailable (never returns a stand-in) when something is
    missing.
    """
    problems = judge_config_problems()
    if problems:
        raise RagasUnavailable("; ".join(problems))
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise RagasUnavailable(f"ragas/langchain not installed: {exc} (pip install -e '.[ragas]')") from exc

    from src.rag.embeddings import DEFAULT_MODEL_NAME, SentenceTransformerEmbedder

    base_url = os.environ.get("RAGAS_JUDGE_BASE_URL") or None
    api_key = os.environ.get("RAGAS_JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    chat = ChatOpenAI(
        model=os.environ["RAGAS_JUDGE_MODEL"],
        base_url=base_url,
        api_key=api_key or "EMPTY",
        temperature=0,
    )
    st = SentenceTransformerEmbedder(os.environ.get("RAGAS_EMBEDDING_MODEL", DEFAULT_MODEL_NAME))

    class _ProjectEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return st.embed_documents(texts).tolist()

        def embed_query(self, text: str) -> List[float]:
            return st.embed_query(text).tolist()

    return LangchainLLMWrapper(chat), LangchainEmbeddingsWrapper(_ProjectEmbeddings())


def _load_metrics():
    try:
        from ragas.metrics import (
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RagasUnavailable(f"ragas>=0.2.6,<0.3 required: {exc} (pip install -e '.[ragas]')") from exc
    return [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithReference(), LLMContextRecall()]


def evaluate_ragas(records: List[Dict], llm: Any, embeddings: Any) -> Dict[str, Any]:
    """Score `records` with the four RAGAS metrics. Returns
    summarize_scores(...) plus `per_question` rows. Raises RagasUnavailable
    if ragas is missing.
    """
    try:
        from ragas import EvaluationDataset, evaluate
    except ImportError as exc:
        raise RagasUnavailable(f"ragas not installed: {exc} (pip install -e '.[ragas]')") from exc

    dataset = EvaluationDataset.from_list(to_ragas_rows(records))
    result = evaluate(
        dataset=dataset,
        metrics=_load_metrics(),
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    df = result.to_pandas()
    rows = df.to_dict("records")
    summary = summarize_scores(rows, expected_n=len(records))
    summary["per_question"] = [
        {**{k: v for k, v in normalize_metric_names(row).items()}, "question": rec["question"]}
        for row, rec in zip(rows, records)
    ]
    return summary


def evaluate_faithfulness(records: List[Dict]) -> Dict[str, float]:
    """Legacy single-metric wrapper (Day 6). Raises ImportError if ragas
    is not installed; callers must fail closed, never substitute a score.
    """
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness

    dataset = Dataset.from_list(records)
    result = evaluate(dataset, metrics=[faithfulness])
    return dict(result)
