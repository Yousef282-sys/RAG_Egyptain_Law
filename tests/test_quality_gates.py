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
