"""Tests for src/experiments/tracking.py.

`mlflow` is not installed in this sandbox and there is no network here to
install it (confirmed before writing this module). Tests that need a real
mlflow import/run honestly SKIP (returning the literal string "SKIPPED",
counted separately by run_tests.py) rather than fabricating a pass.
`get_mlflow()`'s no-crash behavior and `log_experiment_run`'s
`MLflowUnavailable` behavior, however, are real, always-executable
requirements and are tested for real below regardless of mlflow
availability.
"""
import tempfile
from pathlib import Path

from src.experiments.tracking import (
    EXPERIMENT_NAME,
    MLflowUnavailable,
    get_mlflow,
    log_experiment_run,
)

MLFLOW_AVAILABLE = get_mlflow() is not None


def _skip(name: str) -> str:
    print(
        f"  ({name}: mlflow not installed in this sandbox -- not executed "
        "here, run in an environment with that package)"
    )
    return "SKIPPED"


def test_get_mlflow_never_raises():
    # Always runs for real: must return None cleanly when mlflow is
    # unavailable, never raise.
    result = get_mlflow()
    assert result is None or hasattr(result, "start_run")


def test_experiment_name_is_arabic_legal_rag():
    assert EXPERIMENT_NAME == "arabic-legal-rag"


def test_log_experiment_run_raises_mlflow_unavailable_when_uninstalled():
    if MLFLOW_AVAILABLE:
        return _skip("test_log_experiment_run_raises_mlflow_unavailable_when_uninstalled")
    # This sandbox's real, current state: mlflow is not installed, so this
    # must raise MLflowUnavailable -- never silently return a fake run id.
    raised = False
    try:
        log_experiment_run(config={"a": 1}, metrics={"b": 2.0})
    except MLflowUnavailable:
        raised = True
    assert raised


def test_log_experiment_run_creates_a_real_run_when_mlflow_available():
    if not MLFLOW_AVAILABLE:
        return _skip("test_log_experiment_run_creates_a_real_run_when_mlflow_available")
    with tempfile.TemporaryDirectory() as d:
        import mlflow

        mlflow.set_tracking_uri(f"file://{d}")
        run_id = log_experiment_run(
            config={"chunk_size": 1000, "top_k": 5},
            metrics={"hit_rate_at_5": 1.0},
            run_name="test_run",
        )
        assert isinstance(run_id, str) and run_id


def test_register_best_run_artifact_raises_mlflow_unavailable_when_uninstalled():
    from src.experiments.tracking import register_best_run_artifact

    if MLFLOW_AVAILABLE:
        return _skip("test_register_best_run_artifact_raises_mlflow_unavailable_when_uninstalled")
    fake_results = [
        {
            "config": {"run_name": "x", "embedder_class": "TfidfFallbackEmbedder"},
            "metrics": {"hit_rate_at_5": 1.0},
        }
    ]
    raised = False
    try:
        register_best_run_artifact(fake_results)
    except MLflowUnavailable:
        raised = True
    assert raised
