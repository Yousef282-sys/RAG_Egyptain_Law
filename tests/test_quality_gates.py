"""Tests for scripts/check_corpus_quality.py and scripts/ragas_gate.py.

Both scripts run for real in this sandbox (no external dependency
needed for the corpus gate; the RAGAS gate's whole point is to correctly
report itself as unavailable here, which is directly testable).
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

_corpus_spec = importlib.util.spec_from_file_location(
    "check_corpus_quality", Path("scripts/check_corpus_quality.py")
)
check_corpus_quality = importlib.util.module_from_spec(_corpus_spec)
_corpus_spec.loader.exec_module(check_corpus_quality)

_ragas_spec = importlib.util.spec_from_file_location("ragas_gate", Path("scripts/ragas_gate.py"))
ragas_gate = importlib.util.module_from_spec(_ragas_spec)
_ragas_spec.loader.exec_module(ragas_gate)


def _good_report():
    return {
        "total_articles": check_corpus_quality.EXPECTED_TOTAL_ARTICLES,
        "duplicate_numbers": list(check_corpus_quality.EXPECTED_DUPLICATE_NUMBERS),
        "number_gaps": list(range(check_corpus_quality.EXPECTED_GAP_COUNT)),
        "article_issues": dict(check_corpus_quality.EXPECTED_ARTICLE_ISSUES),
    }


def test_corpus_quality_gate_passes_on_known_good_baseline():
    problems = check_corpus_quality.check(_good_report())
    assert problems == []


def test_corpus_quality_gate_fails_on_new_duplicate():
    report = _good_report()
    report["duplicate_numbers"] = [1, 2, 999]  # a genuinely new duplicate
    problems = check_corpus_quality.check(report)
    assert problems  # must not silently pass


def test_corpus_quality_gate_fails_on_article_count_change():
    report = _good_report()
    report["total_articles"] = 1090  # articles disappeared
    problems = check_corpus_quality.check(report)
    assert any("total_articles" in p for p in problems)


def test_corpus_quality_gate_fails_on_new_article_issue():
    report = _good_report()
    report["article_issues"] = {"500": ["text_ar is empty"]}  # a NEW, unexplained gap
    problems = check_corpus_quality.check(report)
    assert any("article_issues" in p for p in problems)


def test_corpus_quality_gate_real_run_against_real_corpus_passes():
    # end-to-end, against the actual current corpus -- real subprocess call
    report = check_corpus_quality.run_validate()
    problems = check_corpus_quality.check(report)
    assert problems == [], f"unexpected corpus regression: {problems}"


def test_ragas_gate_reports_unavailable_in_this_sandbox():
    # real, direct check: ragas is genuinely not installed here
    assert ragas_gate.ragas_importable() is False


def test_ragas_gate_reports_no_generation_backend_by_default():
    import os

    os.environ.pop("GENERATION_MODEL", None)
    assert ragas_gate.generation_backend_configured() is False


def test_ragas_gate_fails_closed_exit_code():
    exit_code = ragas_gate.run_gate()
    assert exit_code != 0  # never silently passes when unavailable


def test_ragas_gate_never_reports_a_fabricated_numeric_score():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ragas_gate.run_gate()
    output = buf.getvalue().lower()
    # the threshold config is fine to print; an actual "faithfulness
    # score: X" line would mean a number was fabricated -- must never
    # appear here
    assert "faithfulness score" not in output
    assert "unavailable" in output


# ---------------- Phase 2: threshold + decision logic ----------------------
# The pipeline is INJECTED with hand-built summaries purely to test the
# gate's decision/fail-closed logic. These are test inputs, not RAGAS scores.

def _summary(faith, n=None, n_questions=57):
    from src.experiments.ragas_eval import summarize_scores

    rows = [{"faithfulness": f, "answer_relevancy": 0.9, "context_precision": 0.8,
             "context_recall": 0.7} for f in faith]
    return summarize_scores(rows, expected_n=n or len(rows))


def _run_gate_with(summary=None, raises=None, eval_set_path=None, threshold=None):
    import tempfile

    calls = []

    def pipeline(eval_set):
        calls.append(len(eval_set))
        if raises:
            raise raises
        return summary

    with tempfile.TemporaryDirectory() as d:
        old = ragas_gate.REPORT_PATH
        ragas_gate.REPORT_PATH = Path(d) / "ragas_results.json"
        try:
            code = ragas_gate.run_gate(threshold=threshold, eval_set_path=eval_set_path, pipeline=pipeline)
            wrote = ragas_gate.REPORT_PATH.exists()
        finally:
            ragas_gate.REPORT_PATH = old
    return code, calls, wrote


def test_gate_default_threshold_is_0_75():
    import os

    saved = os.environ.pop("RAGAS_FAITHFULNESS_THRESHOLD", None)
    try:
        assert ragas_gate.DEFAULT_THRESHOLD == 0.75 and ragas_gate.threshold_from_env() == 0.75
        os.environ["RAGAS_FAITHFULNESS_THRESHOLD"] = "0.8"
        assert ragas_gate.threshold_from_env() == 0.8
    finally:
        os.environ.pop("RAGAS_FAITHFULNESS_THRESHOLD", None)
        if saved is not None:
            os.environ["RAGAS_FAITHFULNESS_THRESHOLD"] = saved


def test_ci_workflow_runs_the_gate_at_0_75():
    text = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert 'RAGAS_FAITHFULNESS_THRESHOLD: "0.75"' in text and '"0.7"' not in text


def test_evaluate_gate_boundaries():
    s = lambda f: {"scores": {"faithfulness": f}}  # noqa: E731
    assert ragas_gate.evaluate_gate(s(0.80), 0.75)[0] is True
    assert ragas_gate.evaluate_gate(s(0.75), 0.75)[0] is True       # exactly at threshold passes
    assert ragas_gate.evaluate_gate(s(0.7499), 0.75)[0] is False    # below fails
    assert ragas_gate.evaluate_gate(s(float("nan")), 0.75)[0] is False
    assert ragas_gate.evaluate_gate({"scores": {}}, 0.75)[0] is False
    assert ragas_gate.evaluate_gate({}, 0.75)[0] is False


def test_gate_passes_only_when_faithfulness_meets_threshold():
    code, calls, wrote = _run_gate_with(_summary([0.9] * 57), threshold=0.75)
    assert code == ragas_gate.EXIT_PASSED == 0 and calls == [57] and wrote


def test_gate_fails_with_exit_2_when_faithfulness_below_threshold():
    code, _, wrote = _run_gate_with(_summary([0.7] * 57), threshold=0.75)
    assert code == ragas_gate.EXIT_FAILED == 2 and wrote  # real (bad) result is still reported


def test_gate_counts_unscorable_rows_against_faithfulness():
    # 40% of rows NaN: mean over valid rows would be 1.0, but the gate
    # score is 0.6 -> must FAIL (fail-closed against flaky judges).
    rows = [1.0] * 34 + [float("nan")] * 23
    code, _, _ = _run_gate_with(_summary(rows), threshold=0.75)
    assert code == 2


def test_gate_fails_closed_when_the_evaluation_raises():
    code, _, wrote = _run_gate_with(raises=RuntimeError("judge down"), threshold=0.75)
    assert code == ragas_gate.EXIT_UNAVAILABLE == 1 and not wrote


def test_gate_fails_closed_when_row_count_does_not_match_eval_set():
    code, _, wrote = _run_gate_with(_summary([1.0] * 10), threshold=0.75)
    assert code == 1 and not wrote


def test_gate_refuses_an_undersized_or_missing_eval_set_without_running_anything():
    import json
    import tempfile

    items = json.loads(Path("data/eval/retrieval_eval.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as d:
        small = Path(d) / "small.json"
        small.write_text(json.dumps(items[:10], ensure_ascii=False), encoding="utf-8")
        code, calls, _ = _run_gate_with(_summary([1.0] * 10), eval_set_path=str(small))
        assert code == 1 and calls == []
        code, calls, _ = _run_gate_with(_summary([1.0] * 57), eval_set_path=str(Path(d) / "missing.json"))
        assert code == 1 and calls == []


def test_gate_without_injected_pipeline_lists_every_unmet_requirement():
    import contextlib
    import io
    import os

    saved = {k: os.environ.pop(k, None) for k in ("GENERATION_MODEL", "RAGAS_JUDGE_MODEL")}
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            code = ragas_gate.run_gate()
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
    out = buf.getvalue()
    assert code == 1 and "UNAVAILABLE" in out and "GENERATION_MODEL" in out
    assert "RAGAS_JUDGE_MODEL" in out and "No score was computed" in out
