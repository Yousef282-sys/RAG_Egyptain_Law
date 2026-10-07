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
) -> str:
    """Log one real run to MLflow under EXPERIMENT_NAME: all of `config`
    as params, all numeric `metrics`, and any files in `artifact_paths`
    as artifacts. Returns the real MLflow run_id.

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
            mlflow.log_param(key, value)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                mlflow.log_metric(key, value)
        if artifact_paths:
            for _, path in artifact_paths.items():
                mlflow.log_artifact(path)
        return run.info.run_id


def register_best_run_artifact(
    results: list,
    metric_name: str = "hit_rate_at_5",
    model_name: str = "arabic-legal-rag-tfidf-embedder",
) -> Dict[str, Any]:
    """Registry integration (Phase 5).

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
    with mlflow.start_run(run_name=f"register-{config['run_name']}") as run:
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
