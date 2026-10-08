"""MLflow tracking + best-run selection + registry interface (Phase 2).
mlflow is replaced by a recording stand-in (tests/fakes.py): this verifies
WHAT we log/register and the selection rule -- NOT that real MLflow runs
or a real registry entry exist (UNVERIFIED until run with mlflow).
"""
import math

from src.experiments.index_builder import make_embedder
from src.experiments.tracking import (
    MLflowUnavailable,
    log_experiment_run,
    register_best_configuration,
    select_best_run,
)
from tests.fakes import fake_mlflow, installed


def _run(name, mrr, faith=None, status="completed", top_k=5):
    metrics = {"mrr_at_10": mrr}
    if faith is not None:
        metrics["faithfulness"] = faith
    return {"config": {"run_name": name, "embedding_backend": "tfidf", "embedding_model": "tfidf-fallback",
                       "chunk_size": 1000, "overlap": 0, "top_k": top_k, "tfidf_max_features": 256,
                       "tfidf_ngram_range": "(1, 1)"},
            "status": status, "metrics": metrics}


def test_log_experiment_run_logs_params_metrics_tags_and_artifacts():
    import tempfile
    from pathlib import Path

    mlflow_mod, pyfunc_mod, rec = fake_mlflow()
    with tempfile.TemporaryDirectory() as d:
        art = Path(d) / "config.json"
        art.write_text("{}", encoding="utf-8")
        with installed(mlflow_mod, pyfunc_mod):
            run_id = log_experiment_run(
                config={"chunk_size": 500, "overlap": 100, "embedding_model": "m", "skip_me": None},
                metrics={"faithfulness": 0.8, "mrr_at_10": 0.7, "nan_metric": math.nan, "flag": True},
                artifact_paths={"config": str(art)}, run_name="r1", tags={"embedding_kind": "semantic"})
    assert run_id == "fake-run-1" and rec.runs == ["r1"] and rec.experiment == "arabic-legal-rag"
    assert rec.params == {"chunk_size": 500, "overlap": 100, "embedding_model": "m"}  # None skipped
    assert rec.metrics == {"faithfulness": 0.8, "mrr_at_10": 0.7}  # NaN / bool never logged as metrics
    assert rec.tags == {"embedding_kind": "semantic"} and rec.artifacts == ["config.json"]


def test_log_experiment_run_raises_rather_than_faking_when_mlflow_missing():
    from src.experiments.tracking import get_mlflow

    if get_mlflow() is not None:
        return "SKIPPED"
    try:
        log_experiment_run({"a": 1}, {"b": 1.0})
    except MLflowUnavailable:
        return
    raise AssertionError("expected MLflowUnavailable")


def test_best_run_uses_faithfulness_when_every_run_has_it():
    best = select_best_run([_run("a", 0.9, 0.70), _run("b", 0.5, 0.82), _run("c", 0.8, 0.80)])
    assert best["metric"] == "faithfulness" and best["result"]["config"]["run_name"] == "b"


def test_best_run_falls_back_to_one_retrieval_metric_never_mixing_metrics():
    best = select_best_run([_run("a", 0.60, 0.99), _run("b", 0.75)])  # b has no faithfulness
    assert best["metric"] == "mrr_at_10" and best["result"]["config"]["run_name"] == "b"
    assert "faithfulness not available" in best["rule"]


def test_best_run_ties_break_on_mrr_then_earlier_run_and_skips_skipped_runs():
    tie = select_best_run([_run("a", 0.5, 0.8), _run("b", 0.7, 0.8), _run("c", 0.7, 0.8)])
    assert tie["result"]["config"]["run_name"] == "b"
    skipped = select_best_run([_run("a", 0.1, status="skipped"), _run("b", 0.2)])
    assert skipped["result"]["config"]["run_name"] == "b"
    assert select_best_run([_run("a", 0.1, status="skipped")]) is None
    assert select_best_run([]) is None


def test_register_best_configuration_registers_a_loadable_model_with_champion_alias():
    mlflow_mod, pyfunc_mod, rec = fake_mlflow()
    results = [_run("weak", 0.5), _run("strong", 0.9)]
    with installed(mlflow_mod, pyfunc_mod):  # builds a REAL (tf-idf) index from the real corpus
        out = register_best_configuration(results)
    assert out["registered"] is True and out["version"] == "1" and out["source_run_name"] == "strong"
    assert out["selection_metric"] == "mrr_at_10" and out["alias_champion_set"] is True
    assert rec.registered == [("runs:/fake-run-1/rag_model", "arabic-legal-rag-retriever")]
    assert rec.aliases == [("arabic-legal-rag-retriever", "champion", "1")]
    logged = rec.pyfunc_logged[0]
    assert set(logged["artifacts"]) == {"index", "config"} and logged["artifact_path"] == "rag_model"
    assert logged["code_paths"] == ["src"]


def test_register_with_no_completed_run_registers_nothing():
    mlflow_mod, pyfunc_mod, rec = fake_mlflow()
    with installed(mlflow_mod, pyfunc_mod):
        out = register_best_configuration([_run("a", 0.1, status="skipped")])
    assert out["registered"] is False and rec.registered == [] and rec.pyfunc_logged == []


def test_register_raises_mlflow_unavailable_when_missing_never_claims_registration():
    from src.experiments.tracking import get_mlflow

    if get_mlflow() is not None:
        return "SKIPPED"
    try:
        register_best_configuration([_run("a", 0.5)])
    except MLflowUnavailable:
        return
    raise AssertionError("expected MLflowUnavailable")


def test_make_embedder_is_strict_for_semantic_configs():
    from src.rag.embeddings import EmbeddingBackendUnavailable, TfidfFallbackEmbedder

    assert isinstance(make_embedder({"embedding_backend": "tfidf", "tfidf_ngram_range": "(1, 2)"}),
                      TfidfFallbackEmbedder)
    try:
        make_embedder({"embedding_backend": "sentence-transformers", "embedding_model": "m"})
    except EmbeddingBackendUnavailable:
        return  # sentence-transformers absent here; with it present this would load the model
    return "SKIPPED"
