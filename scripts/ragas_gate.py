"""RAGAS faithfulness quality gate for CI.

PASSES (exit 0) only when a REAL RAGAS evaluation of the real pipeline
(retrieve -> vLLM generate -> judge) over the curated eval set (>= 50
questions) yields faithfulness >= RAGAS_FAITHFULNESS_THRESHOLD (default
0.75). All four RAGAS metrics (faithfulness, answer relevancy, context
precision, context recall) are computed and reported; faithfulness is the
gated one.

Fail-closed, never fabricated:
  exit 1  UNAVAILABLE -- a requirement is unmet (ragas not installed, no
          generation backend (GENERATION_MODEL), no judge LLM, eval set
          missing/undersized/invalid, index cannot be built/loaded, or the
          evaluation itself errored). No score is printed or invented.
  exit 2  FAILED -- a real evaluation ran and faithfulness is below the
          threshold (or NaN).
  exit 0  PASSED.
The gating score counts rows RAGAS could not score as 0.0 (see
src/experiments/ragas_eval.summarize_scores). Hit@K / MRR are never
substituted for any RAGAS metric.

STATUS: the evaluation path is UNVERIFIED end to end (no ragas, judge LLM
or vLLM in the development sandbox). The decision logic
(`evaluate_gate`, `run_gate` with an injected pipeline) and every
fail-closed branch are unit-tested offline (tests/test_quality_gates.py).

Environment: RAGAS_FAITHFULNESS_THRESHOLD, GENERATION_MODEL + VLLM_*
(src/rag/generation.py), RAGAS_JUDGE_* (src/experiments/ragas_eval.py),
VECTOR_STORE_DIR, RAG_TOP_K, EMBEDDING_MODEL / EMBEDDING_BACKEND,
RAGAS_EVAL_SET.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_THRESHOLD = 0.75
CORPUS_PATH = "data/processed/civil_code.json"
REPORT_PATH = Path("reports/ragas_results.json")

EXIT_PASSED, EXIT_UNAVAILABLE, EXIT_FAILED = 0, 1, 2


def threshold_from_env() -> float:
    return float(os.environ.get("RAGAS_FAITHFULNESS_THRESHOLD", str(DEFAULT_THRESHOLD)))


def ragas_importable() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def generation_backend_configured() -> bool:
    return bool(os.environ.get("GENERATION_MODEL"))


def unmet_environment_requirements() -> List[str]:
    from src.experiments.ragas_eval import ragas_environment_problems

    return ragas_environment_problems()


def evaluate_gate(summary: Dict[str, Any], threshold: float) -> Tuple[bool, str]:
    """Pure decision: does this (real) RAGAS summary pass? Fails on a
    missing/NaN faithfulness or a score below `threshold`.
    """
    score = summary.get("scores", {}).get("faithfulness")
    if score is None or score != score:  # None or NaN
        return False, "faithfulness missing/NaN"
    if score < threshold:
        return False, f"faithfulness={score:.4f} < threshold {threshold}"
    return True, f"faithfulness={score:.4f} >= threshold {threshold}"


def default_pipeline(eval_set: List[Dict]) -> Dict[str, Any]:
    """The real evaluation: build/open the index, serve through RagService,
    judge with RAGAS. Needs the full environment (see module docstring).
    """
    from src.experiments.ragas_eval import (
        build_judges,
        build_ragas_records_from_service,
        evaluate_ragas,
    )
    from src.rag.generation import get_generator
    from src.rag.retriever import Retriever
    from src.rag.service import RagService
    from src.rag.vector_store import build_index_from_json, read_manifest

    store_dir = os.environ.get("VECTOR_STORE_DIR", "data/vector_store")
    if read_manifest(store_dir) is None:
        print(f"no index at {store_dir}; building from {CORPUS_PATH}")
        build_index_from_json(CORPUS_PATH, store_dir)
    manifest = read_manifest(store_dir)
    retriever = Retriever(store_dir=store_dir, top_k=int(os.environ.get("RAG_TOP_K", "5")))
    generator = get_generator()
    service = RagService(retriever, generator)

    records = build_ragas_records_from_service(eval_set, service)
    llm, embeddings = build_judges()
    summary = evaluate_ragas(records, llm, embeddings)
    summary["context"] = {
        "generator": generator.name,
        "judge_model": os.environ.get("RAGAS_JUDGE_MODEL"),
        "embedding_kind": manifest.get("embedding_kind"),
        "embedder_model": manifest.get("embedder_model"),
        "vector_backend": manifest.get("backend"),
        "top_k": retriever.top_k,
    }
    return summary


def run_gate(
    threshold: Optional[float] = None,
    eval_set_path: Optional[str] = None,
    pipeline: Optional[Callable[[List[Dict]], Dict[str, Any]]] = None,
) -> int:
    threshold = threshold_from_env() if threshold is None else threshold
    eval_set_path = eval_set_path or os.environ.get("RAGAS_EVAL_SET", "data/eval/retrieval_eval.json")

    unmet: List[str] = []
    eval_set: List[Dict] = []
    try:
        from src.experiments.eval_set import load_curated_eval_set

        eval_set = load_curated_eval_set(eval_set_path, corpus_json_path=CORPUS_PATH)
    except (OSError, ValueError) as exc:
        unmet.append(f"eval set unusable ({type(exc).__name__}: {exc})")
    if pipeline is None:
        unmet.extend(unmet_environment_requirements())

    if unmet:
        print("RAGAS faithfulness gate: UNAVAILABLE -- failing closed.")
        for reason in unmet:
            print(f"  - {reason}")
        print(f"Threshold: {threshold}. No score was computed.")
        return EXIT_UNAVAILABLE

    try:
        summary = (pipeline or default_pipeline)(eval_set)
    except Exception as exc:  # evaluation itself failed: never pass
        print("RAGAS faithfulness gate: UNAVAILABLE -- failing closed.")
        print(f"  - evaluation errored: {type(exc).__name__}: {exc}")
        return EXIT_UNAVAILABLE

    if summary.get("n") != len(eval_set):
        print("RAGAS faithfulness gate: UNAVAILABLE -- failing closed.")
        print(f"  - evaluated {summary.get('n')} rows for {len(eval_set)} questions")
        return EXIT_UNAVAILABLE

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"threshold": threshold, **summary}, f, ensure_ascii=False, indent=2)

    for name, value in summary["scores"].items():
        print(f"  {name}={value:.4f} (valid rows: {summary['n_valid'][name]}/{summary['n']})")
    passed, message = evaluate_gate(summary, threshold)
    if passed:
        print(f"RAGAS faithfulness gate: PASSED ({message})")
        return EXIT_PASSED
    print(f"RAGAS faithfulness gate: FAILED ({message})")
    return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(run_gate())
