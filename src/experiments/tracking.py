"""Thin MLflow integration for the RAG experiment runner.

If `mlflow` is not installed (as in this development sandbox --
confirmed: no network access, `pip install mlflow` fails), logging
functions raise `MLflowUnavailable` rather than faking a run record.
Callers (src/experiments/run_experiments.py) catch this and fall back to
local-only recording of the exact same config/metrics, clearly labeled
as not logged to MLflow.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

EXPERIMENT_NAME = "arabic-legal-rag"


class MLflowUnavailable(RuntimeError):
    """Raised instead of faking a run when the `mlflow` package is not
    installed / importable.
    """


def get_mlflow():
    """Return the `mlflow` module if importable, else None. Never raises."""
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


def log_experiment_run(
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    artifact_paths: Optional[Dict[str, str]] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> str:
    """Log one real run to MLflow under EXPERIMENT_NAME: all non-None
    `config` values as params, all numeric `metrics`, `tags`, and the files
    in `artifact_paths` as artifacts. Returns the real MLflow run_id.

    Raises MLflowUnavailable if `mlflow` cannot be imported -- this
    function never creates a "run" of any kind without real mlflow
    backing it.
    """
    mlflow = get_mlflow()
    if mlflow is None:
        raise MLflowUnavailable(
            "mlflow is not installed in this environment -- no run was "
            "created (never faked)."
        )

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name) as run:
        for key, value in config.items():
            if value is not None:
                mlflow.log_param(key, value)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == value:
                mlflow.log_metric(key, value)
        if tags:
            mlflow.set_tags(tags)
        if artifact_paths:
            for _, path in artifact_paths.items():
                mlflow.log_artifact(path)
        return run.info.run_id


# ------------------------- best-run selection ----------------------------

SELECTION_PRIMARY_METRIC = "faithfulness"  # the quality-gate metric (RAGAS)
SELECTION_FALLBACK_METRIC = "mrr_at_10"  # retrieval-only, comparable across top_k
SELECTION_TIEBREAK_METRIC = "mrr_at_10"
REGISTERED_MODEL_NAME = "arabic-legal-rag-retriever"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def select_best_run(
    results: list,
    primary: str = SELECTION_PRIMARY_METRIC,
    fallback: str = SELECTION_FALLBACK_METRIC,
) -> Optional[Dict[str, Any]]:
    """Pick the best configuration by ONE documented metric.

    Rule: among COMPLETED runs, rank by `primary` (RAGAS faithfulness, the
    quality-gate metric) if EVERY completed run has it; otherwise rank all
    by `fallback` (retrieval MRR@10) -- never mixing metrics across runs.
    Ties break on MRR@10, then on the earlier run in the list. Returns
    {"result", "metric", "value", "rule"} or None if nothing completed.
    """
    done = [r for r in results if r.get("status", "completed") == "completed"]
    if not done:
        return None
    metric = primary if all(_is_number(r["metrics"].get(primary)) for r in done) else fallback
    best = None
    for r in done:
        value = r["metrics"].get(metric)
        if not _is_number(value):
            continue
        key = (value, r["metrics"].get(SELECTION_TIEBREAK_METRIC) or 0.0)
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        return None
    return {
        "result": best[1],
        "metric": metric,
        "value": best[0][0],
        "rule": (
            f"highest {metric} among {len(done)} completed runs"
            + ("" if metric == primary else f" ({primary} not available for every run)")
        ),
    }


def register_best_configuration(
    results: list, model_name: str = REGISTERED_MODEL_NAME, corpus_path: str = "data/processed/civil_code.json"
) -> Dict[str, Any]:
    """Register the best configuration in the MLflow Model Registry.

    What is registered: an MLflow pyfunc model (src/experiments/
    mlflow_model.py) packaging the winning config (rag_config.json) and the
    index rebuilt from it, so the registry entry is loadable and
    reproducible. Sets the alias `champion` on the new version.

    Raises MLflowUnavailable if mlflow is missing. Returns
    {"registered": False, "reason": ...} when there is no eligible run.
    `registered` is True only after mlflow.register_model returned.
    UNVERIFIED: never executed in the development sandbox.
    """
    mlflow = get_mlflow()
    if mlflow is None:
        raise MLflowUnavailable(
            "mlflow is not installed in this environment -- nothing was "
            "registered (never faked)."
        )
    selection = select_best_run(results)
    if selection is None:
        return {"registered": False, "reason": "no completed run to register"}

    import json
    import tempfile
    from pathlib import Path


    config = selection["result"]["config"]
    with tempfile.TemporaryDirectory() as tmp:
        index_dir = Path(tmp) / "index"
        config_path = Path(tmp) / "rag_config.json"
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

        # Build Chroma in a separate process to release SQLite handles on exit.
        import subprocess
        import sys

        build_code = (
            "import json, sys; "
            "from src.experiments.index_builder import build_config_index; "
            "config=json.load(open(sys.argv[1], encoding='utf-8')); "
            "build_config_index(config, sys.argv[2], sys.argv[3])"
        )
        subprocess.run(
            [sys.executable, "-c", build_code, str(config_path), str(index_dir), str(corpus_path)],
            check=True,
        )

        # Package the index so MLflow does not access the live Chroma database.
        import shutil
        index_archive = Path(shutil.make_archive(
            str(Path(tmp) / "index_bundle"),
            "zip",
            root_dir=tmp,
            base_dir="index",
        ))

        model_code_path = Path(tmp) / "rag_model_entry.py"
        model_code_path.write_text(
            "from src.experiments.mlflow_model import RagRetrieverModel\n"
            "import mlflow\n"
            "mlflow.models.set_model(RagRetrieverModel())\n",
            encoding="utf-8",
        )

        mlflow.set_experiment(EXPERIMENT_NAME)
        with mlflow.start_run(run_name=f"register-{config['run_name']}") as run:
            mlflow.log_params({k: v for k, v in config.items() if v is not None})
            mlflow.log_metric(selection["metric"], selection["value"])
            mlflow.pyfunc.log_model(
                artifact_path="rag_model",
                python_model=str(model_code_path),
                artifacts={"index": str(index_archive), "config": str(config_path)},
                code_paths=["src"],
                pip_requirements=["numpy", "scikit-learn", "sentence-transformers", "chromadb"],
            )
            version = mlflow.register_model(f"runs:/{run.info.run_id}/rag_model", model_name)

    alias_set = False
    try:
        mlflow.tracking.MlflowClient().set_registered_model_alias(
            model_name, "champion", version.version
        )
        alias_set = True
    except Exception:  # alias API missing/unsupported by the tracking backend
        pass

    return {
        "registered": True,
        "model_name": model_name,
        "version": version.version,
        "alias_champion_set": alias_set,
        "selection_metric": selection["metric"],
        "selection_value": selection["value"],
        "selection_rule": selection["rule"],
        "source_run_name": config["run_name"],
        "registration_mlflow_run_id": run.info.run_id,
    }


def register_best_run_artifact(
    results: list,
    metric_name: str = "hit_rate_at_5",
    model_name: str = "arabic-legal-rag-tfidf-embedder",
) -> Dict[str, Any]:
    """Registry integration (Phase 5; legacy -- superseded by register_best_configuration).

    The RAG system as a whole is NOT a conventional ML model, so nothing
    here pretends otherwise. What genuinely IS a trained model artifact
    in this pipeline is the fitted TF-IDF vectorizer used by the offline
    fallback embedder (a real scikit-learn model) -- so that is what gets
    registered, tied to the run that produced it, when the best-scoring
    configuration (by `metric_name`) used that backend.

    If the best run used a real sentence-transformer instead (not
    available in this sandbox), there is no locally-trained artifact to
    register the same way -- the pretrained model name/version (e.g. from
    Hugging Face) is the artifact of record in that case, and this
    function returns a `{"registered": False, "reason": ...}` result
    rather than fabricating a registration.

    Raises MLflowUnavailable if `mlflow` is not installed (never fakes a
    registry entry).
    """
    mlflow = get_mlflow()
    if mlflow is None:
        raise MLflowUnavailable(
            "mlflow is not installed in this environment -- no artifact "
            "was registered (never faked)."
        )

    scored = [r for r in results if metric_name in r.get("metrics", {})]
    if not scored:
        return {"registered": False, "reason": f"no run had metric '{metric_name}'"}
    best = max(scored, key=lambda r: r["metrics"][metric_name])

    if best["config"].get("embedder_class") != "TfidfFallbackEmbedder":
        return {
            "registered": False,
            "reason": (
                f"best run's embedder ({best['config'].get('embedder_class')}) "
                "is a pretrained model, not a locally-trained artifact -- "
                "nothing to register as a new model version."
            ),
            "best_run_name": best["config"].get("run_name"),
        }

    # Re-fit the winning TF-IDF configuration so we have the actual fitted
    # model object to log (the earlier run's fitted vectorizer was only
    # persisted to a temp dir that no longer exists) -- same corpus, same
    # config, so this reproduces the identical artifact, not a new one.
    from src.rag.chunking import chunk_articles, load_articles
    from src.rag.embeddings import TfidfFallbackEmbedder

    config = best["config"]
    articles = load_articles("data/processed/civil_code.json")
    chunks = chunk_articles(articles, max_chars=config["chunk_size"])
    ngram_range = _parse_ngram_range(config["tfidf_ngram_range"])  # "(1, 1)" -> (1, 1)
    embedder = TfidfFallbackEmbedder(
        max_features=config["tfidf_max_features"], ngram_range=ngram_range
    )
    texts = [c.text_ar or c.text_en or "" for c in chunks]
    embedder.embed(texts)

    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=f"register-{config['run_name']}"):
        mlflow.log_params(config)
        model_info = mlflow.sklearn.log_model(embedder._vectorizer, "tfidf_vectorizer")
        registered = mlflow.register_model(model_info.model_uri, model_name)

    return {
        "registered": True,
        "model_name": model_name,
        "version": registered.version,
        "source_run_name": config["run_name"],
        f"source_{metric_name}": best["metrics"][metric_name],
    }


def _parse_ngram_range(value: str):
    import ast

    return ast.literal_eval(value)
