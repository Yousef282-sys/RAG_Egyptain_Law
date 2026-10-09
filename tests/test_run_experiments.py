import json
import tempfile
from pathlib import Path

from src.experiments.eval_set import DEFAULT_EVAL_SET_PATH, load_curated_eval_set
from src.experiments.run_experiments import CONFIGURATIONS, config_key, run_all, run_one_configuration
from tests.fakes import fake_mlflow, fake_sentence_transformers, installed

CORPUS = "data/processed/civil_code.json"


def test_at_least_5_configurations_defined():
    assert len(CONFIGURATIONS) >= 5


def test_configurations_are_genuinely_different():
    keys = [config_key(c) for c in CONFIGURATIONS]
    assert len(set(keys)) == len(keys), "duplicate configuration"
    names = [c["run_name"] for c in CONFIGURATIONS]
    assert len(set(names)) == len(names)


def test_configurations_have_required_fields():
    required = {"run_name", "chunk_size", "overlap", "top_k", "embedding_backend", "embedding_model"}
    for c in CONFIGURATIONS:
        assert required.issubset(c.keys()), c["run_name"]
        assert c["embedding_backend"] in ("tfidf", "sentence-transformers")


def test_configurations_vary_chunk_size_overlap_top_k_and_embedding_model():
    assert len({c["chunk_size"] for c in CONFIGURATIONS}) > 1
    assert len({c["overlap"] for c in CONFIGURATIONS}) > 1
    assert len({c["top_k"] for c in CONFIGURATIONS}) > 1
    semantic_models = {c["embedding_model"] for c in CONFIGURATIONS
                       if c["embedding_backend"] == "sentence-transformers"}
    assert len(semantic_models) >= 2  # at least two real embedding models compared


def test_at_least_five_semantic_configurations_and_lexical_controls_are_labelled():
    semantic = [c for c in CONFIGURATIONS if c["embedding_backend"] == "sentence-transformers"]
    controls = [c for c in CONFIGURATIONS if c["embedding_backend"] == "tfidf"]
    assert len(semantic) >= 5 and controls
    assert all("tfidf" in c["run_name"] for c in controls)  # never passed off as semantic


def test_semantic_configuration_is_skipped_not_downgraded_when_model_unavailable():
    # sentence-transformers cannot load -> the run is SKIPPED with a reason;
    # it must never silently become a TF-IDF result under a semantic name.
    config = next(c for c in CONFIGURATIONS if c["embedding_backend"] == "sentence-transformers")
    eval_set = load_curated_eval_set(corpus_json_path=CORPUS)
    with installed(fake_sentence_transformers(fail_with=OSError("offline"))):
        result = run_one_configuration(config, eval_set, str(DEFAULT_EVAL_SET_PATH), log_to_mlflow=False)
    assert result["status"] == "skipped" and "unavailable" in result["reason"]
    assert result["metrics"] == {} and result["mlflow_logged"] is False


def test_real_offline_run_logs_required_params_metrics_and_artifacts_to_mlflow_stand_in():
    # A REAL TF-IDF run (real corpus, real index, real retrieval metrics),
    # logged to a stand-in MLflow: verifies WHAT we log, not MLflow itself.
    config = next(c for c in CONFIGURATIONS if c["run_name"] == "tfidf_control_c1000")
    eval_set = load_curated_eval_set(corpus_json_path=CORPUS)
    mlflow_mod, pyfunc_mod, rec = fake_mlflow()
    with installed(mlflow_mod, pyfunc_mod):
        result = run_one_configuration(config, eval_set, str(DEFAULT_EVAL_SET_PATH))
    assert result["status"] == "completed" and result["mlflow_logged"] is True
    assert result["mlflow_run_id"] == "fake-run-1"
    for param in ("chunk_size", "overlap", "embedding_model", "embedding_kind", "top_k"):
        assert param in rec.params, param
    assert rec.params["embedding_kind"] == "tfidf-lexical"
    assert {"hit_rate_at_5", "mrr_at_10", "chunk_count"} <= set(rec.metrics)
    # RAGAS was not run here -> faithfulness must be ABSENT, never 0 or estimated
    assert "faithfulness" not in rec.metrics and rec.params["ragas_status"] == "not_run"
    assert {"config.json", "index_manifest.json", "retrieval_per_question.json",
            DEFAULT_EVAL_SET_PATH.name} <= set(rec.artifacts)
    assert rec.experiment == "arabic-legal-rag"


def test_run_all_without_mlflow_records_nothing_as_logged_and_selects_by_mrr():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "results.json"
        only = [c for c in CONFIGURATIONS if c["run_name"] == "tfidf_control_c1000"]
        report = run_all(only, with_ragas="no", log_to_mlflow=False, results_path=out)
        saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["eval_set_size"] >= 50
    if report["results"][0]["mlflow_logged"]:
        assert isinstance(report["results"][0]["mlflow_run_id"], str)
    else:
        assert report["results"][0]["mlflow_run_id"] is None
    assert report["selection"]["metric"] == "mrr_at_10"
    assert report["registry"] is None


def test_run_all_require_mlflow_fails_when_mlflow_missing():
    from src.experiments.tracking import MLflowUnavailable, get_mlflow

    if get_mlflow() is not None:
        return "SKIPPED"
    try:
        run_all(CONFIGURATIONS[:1], with_ragas="no", require_mlflow=True)
    except MLflowUnavailable:
        return
    raise AssertionError("expected MLflowUnavailable")


def test_run_all_ragas_yes_fails_closed_when_environment_not_ready():
    from src.experiments.ragas_eval import RagasUnavailable, ragas_environment_problems

    if not ragas_environment_problems():
        return "SKIPPED"
    try:
        run_all(CONFIGURATIONS[:1], with_ragas="yes")
    except RagasUnavailable:
        return
    raise AssertionError("expected RagasUnavailable")
