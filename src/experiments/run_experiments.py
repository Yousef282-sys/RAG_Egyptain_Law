"""Reproducible RAG experiment runner (MLflow).

For each configuration: build a real chunked index from the real corpus
(chunk_size / overlap / embedding model), evaluate retrieval on the curated
eval set (data/eval/retrieval_eval.json, >= 50 questions), optionally run
the four RAGAS metrics through the real pipeline (retrieve -> vLLM generate
-> judge), and log one MLflow run per configuration:

  params     chunk_size, overlap, embedding_model, embedding_kind,
             embedding_backend, top_k, vector_backend, prefixes, generator,
             judge model, ragas_status, eval-set/corpus hashes
  metrics    faithfulness, answer_relevancy, context_precision,
             context_recall (when RAGAS ran), hit_rate_at_{1,3,5,10},
             mrr_at_10, chunk_count, build/eval seconds
  artifacts  config.json, index manifest.json, per-question retrieval
             results, per-question RAGAS rows, the eval set file

HONESTY RULES
* A "sentence-transformers" config uses the strict loader: if the model
  cannot be loaded the run is recorded as status "skipped" with the reason
  -- it never silently becomes TF-IDF. The two `tfidf_*` configs are
  explicit lexical CONTROLS (embedding_kind "tfidf-lexical"), not semantic.
* If `mlflow` is not installed, runs are computed and written to the local
  JSON report with `mlflow_logged: false` and no run id. Nothing is
  claimed to be in MLflow unless log_experiment_run returned a run id.
* RAGAS metrics are absent (never 0, never estimated) unless RAGAS really
  ran; `ragas_status` says why not.
* The best configuration is chosen by the documented rule in
  src/experiments/tracking.select_best_run.

Deterministic command (after `pip install -e ".[rag,api,experiments,ragas]"`
and starting a vLLM server; see README "Experiments"):

  MLFLOW_TRACKING_URI=sqlite:///mlflow.db EMBEDDING_BACKEND=sentence-transformers \
  GENERATION_MODEL=<model> VLLM_BASE_URL=<url> RAGAS_JUDGE_MODEL=<judge> \
  RAGAS_JUDGE_BASE_URL=<url> \
  python -m src.experiments.run_experiments --with-ragas yes --require-mlflow --register

UNVERIFIED in the development sandbox: semantic runs, RAGAS, MLflow
logging and registry (none of those packages/services are available
here). The two TF-IDF control runs execute for real offline.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.experiments.eval_set import DEFAULT_EVAL_SET_PATH, load_curated_eval_set
from src.experiments.evaluate import evaluate_retriever
from src.experiments.index_builder import CORPUS_PATH, build_config_index
from src.experiments.ragas_eval import (
    RagasUnavailable,
    build_judges,
    build_ragas_records_from_service,
    evaluate_ragas,
    ragas_environment_problems,
)
from src.experiments.tracking import (
    MLflowUnavailable,
    get_mlflow,
    log_experiment_run,
    register_best_configuration,
    select_best_run,
)
from src.rag.embeddings import EmbeddingBackendUnavailable
from src.rag.generation import get_generator
from src.rag.retriever import Retriever
from src.rag.service import RagService
from src.rag.vector_store import file_sha256, read_manifest

RESULTS_PATH = Path("reports/phase2_experiment_results.json")

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
E5_SMALL = "intfloat/multilingual-e5-small"

# 7 genuinely different configurations: 2 lexical TF-IDF controls (runnable
# offline) + 5 semantic runs varying embedding model, chunk_size, overlap
# and top_k. NOTE: only ~12 corpus articles exceed 1000 chars, so chunk_size
# 1000 vs 500 changes the chunking of relatively few articles; the overlap
# only matters where an article is actually split.
CONFIGURATIONS: List[Dict[str, Any]] = [
    {"run_name": "tfidf_control_c1000", "embedding_backend": "tfidf",
     "embedding_model": "tfidf-fallback", "chunk_size": 1000, "overlap": 0, "top_k": 5,
     "tfidf_max_features": 4096, "tfidf_ngram_range": (1, 1)},
    {"run_name": "tfidf_control_bigram_c500_o100", "embedding_backend": "tfidf",
     "embedding_model": "tfidf-fallback", "chunk_size": 500, "overlap": 100, "top_k": 5,
     "tfidf_max_features": 8192, "tfidf_ngram_range": (1, 2)},
    {"run_name": "minilm_c1000_o0_k5", "embedding_backend": "sentence-transformers",
     "embedding_model": MINILM, "chunk_size": 1000, "overlap": 0, "top_k": 5},
    {"run_name": "minilm_c500_o100_k5", "embedding_backend": "sentence-transformers",
     "embedding_model": MINILM, "chunk_size": 500, "overlap": 100, "top_k": 5},
    {"run_name": "minilm_c1000_o0_k3", "embedding_backend": "sentence-transformers",
     "embedding_model": MINILM, "chunk_size": 1000, "overlap": 0, "top_k": 3},
    {"run_name": "e5small_c1000_o0_k5", "embedding_backend": "sentence-transformers",
     "embedding_model": E5_SMALL, "chunk_size": 1000, "overlap": 0, "top_k": 5},
    {"run_name": "e5small_c500_o100_k5", "embedding_backend": "sentence-transformers",
     "embedding_model": E5_SMALL, "chunk_size": 500, "overlap": 100, "top_k": 5},
]


def config_key(config: Dict[str, Any]) -> tuple:
    """Identity of a configuration (everything except its name)."""
    return (
        config["embedding_backend"], config["embedding_model"], config["chunk_size"],
        config.get("overlap", 0), config["top_k"],
        config.get("tfidf_max_features"), str(config.get("tfidf_ngram_range")),
    )


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
    except Exception:
        return None


def run_one_configuration(
    config: Dict[str, Any],
    eval_set: List[Dict[str, Any]],
    eval_set_path: str = str(DEFAULT_EVAL_SET_PATH),
    judges: Optional[tuple] = None,
    log_to_mlflow: bool = True,
) -> Dict[str, Any]:
    """Run one configuration. `judges` = (llm, embeddings) enables RAGAS;
    None means RAGAS is not run for this configuration.
    """
    base_config = dict(config)
    if "tfidf_ngram_range" in base_config:  # MLflow params must be scalar
        base_config["tfidf_ngram_range"] = str(base_config["tfidf_ngram_range"])
    result: Dict[str, Any] = {
        "config": base_config,
        "status": "completed",
        "metrics": {},
        "mlflow_logged": False,
        "mlflow_run_id": None,
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        store_dir = tmp_path / "index"
        t0 = time.time()
        try:
            manifest = build_config_index(config, store_dir)
        except EmbeddingBackendUnavailable as exc:
            result["status"] = "skipped"
            result["reason"] = f"embedding backend unavailable: {exc}"
            print(f"  [skipped] {config['run_name']}: {result['reason']}")
            return result
        build_seconds = time.time() - t0

        retriever = Retriever(store_dir=store_dir, top_k=config["top_k"])
        per_question: List[Dict[str, Any]] = []
        t0 = time.time()
        metrics = evaluate_retriever(retriever, eval_set, config["top_k"], per_question=per_question)
        metrics["chunk_count"] = manifest["chunk_count"]
        metrics["build_seconds"] = round(build_seconds, 3)
        metrics["eval_seconds"] = round(time.time() - t0, 3)

        full_config = dict(result["config"])
        for key in ("embedder_class", "embedding_kind", "backend", "query_prefix",
                    "passage_prefix", "model_revision", "corpus_sha256"):
            if manifest.get(key) is not None:
                full_config["vector_backend" if key == "backend" else key] = manifest[key]
        full_config["eval_set_size"] = len(eval_set)
        full_config["eval_set_sha256"] = file_sha256(eval_set_path)

        artifacts: Dict[str, str] = {}
        ragas_rows = None
        generator = get_generator()
        full_config["generator"] = generator.name
        if judges is None:
            full_config["ragas_status"] = "not_run"
        else:
            service = RagService(retriever, generator)
            records = build_ragas_records_from_service(eval_set, service)
            summary = evaluate_ragas(records, *judges)
            metrics.update(summary["scores"])
            for name, count in summary["n_valid"].items():
                metrics[f"{name}_n_valid"] = count
            ragas_rows = summary["per_question"]
            full_config["ragas_status"] = "completed"
            full_config["judge_model"] = os.environ.get("RAGAS_JUDGE_MODEL")

        result["config"] = full_config
        result["metrics"] = metrics

        # artifacts needed to reproduce / audit the run
        def _dump(name: str, payload: Any) -> None:
            path = tmp_path / name
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            artifacts[name] = str(path)

        _dump("config.json", full_config)
        _dump("index_manifest.json", read_manifest(store_dir))
        _dump("retrieval_per_question.json", per_question)
        if ragas_rows is not None:
            _dump("ragas_per_question.json", ragas_rows)
        artifacts["eval_set"] = eval_set_path

        if log_to_mlflow:
            try:
                result["mlflow_run_id"] = log_experiment_run(
                    config=full_config,
                    metrics=metrics,
                    artifact_paths=artifacts,
                    run_name=config["run_name"],
                    tags={
                        "embedding_kind": str(full_config.get("embedding_kind")),
                        "git_commit": _git_commit() or "unknown",
                        "ragas_status": full_config["ragas_status"],
                    },
                )
                result["mlflow_logged"] = True
            except MLflowUnavailable as exc:
                print(f"  [mlflow] not logged for '{config['run_name']}': {exc}")
    return result


def run_all(
    configurations: List[Dict[str, Any]] = CONFIGURATIONS,
    eval_set_path: str = str(DEFAULT_EVAL_SET_PATH),
    with_ragas: str = "auto",
    require_mlflow: bool = False,
    register: bool = False,
    results_path: Path = RESULTS_PATH,
) -> Dict[str, Any]:
    """with_ragas: "yes" (required; raises RagasUnavailable if not ready),
    "auto" (run RAGAS only if fully ready), "no".
    """
    if require_mlflow and get_mlflow() is None:
        raise MLflowUnavailable("--require-mlflow given but mlflow is not installed")

    eval_set = load_curated_eval_set(eval_set_path, corpus_json_path=CORPUS_PATH)

    judges = None
    ragas_note = "not requested"
    if with_ragas != "no":
        problems = ragas_environment_problems()
        if problems and with_ragas == "yes":
            raise RagasUnavailable("; ".join(problems))
        if problems:
            ragas_note = "NOT RUN: " + "; ".join(problems)
        else:
            judges = build_judges()
            ragas_note = "ran"
    print(f"RAGAS: {ragas_note}")

    results = []
    for config in configurations:
        print(f"Running configuration: {config['run_name']} ...")
        result = run_one_configuration(config, eval_set, eval_set_path, judges)
        results.append(result)
        if result["status"] == "completed":
            print(f"  metrics: {result['metrics']}")

    selection = select_best_run(results)
    registry = None
    if register:
        registry = register_best_configuration(results)  # raises MLflowUnavailable if missing
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "eval_set_path": eval_set_path,
        "eval_set_size": len(eval_set),
        "mlflow_available": get_mlflow() is not None,
        "ragas": ragas_note,
        "selection": None if selection is None else {
            "run_name": selection["result"]["config"]["run_name"],
            "metric": selection["metric"], "value": selection["value"], "rule": selection["rule"],
        },
        "registry": registry,
        "results": results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", help="comma-separated run_names to run")
    ap.add_argument("--with-ragas", choices=["auto", "yes", "no"], default="auto")
    ap.add_argument("--require-mlflow", action="store_true")
    ap.add_argument("--register", action="store_true", help="register the best config in the MLflow Registry")
    ap.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET_PATH))
    ap.add_argument("--out", default=str(RESULTS_PATH))
    args = ap.parse_args(argv)

    configs = CONFIGURATIONS
    if args.only:
        wanted = set(args.only.split(","))
        configs = [c for c in CONFIGURATIONS if c["run_name"] in wanted]
    try:
        report = run_all(configs, args.eval_set, args.with_ragas, args.require_mlflow,
                         args.register, Path(args.out))
    except (MLflowUnavailable, RagasUnavailable) as exc:
        print(f"ERROR: {exc}")
        return 1
    done = [r for r in report["results"] if r["status"] == "completed"]
    logged = [r for r in done if r["mlflow_logged"]]
    print(f"completed={len(done)} skipped={len(report['results']) - len(done)} "
          f"logged_to_mlflow={len(logged)} registry={report['registry']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
