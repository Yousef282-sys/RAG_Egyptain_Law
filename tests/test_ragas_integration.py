"""RAGAS integration interface (Phase 2). `ragas`, a judge LLM and a
generation backend are not available in the sandbox, so everything that
needs them is checked only up to the point of failing closed; record
building, row mapping and score aggregation are tested for real.
"""
import math
import os

from src.experiments.ragas_eval import (
    RAGAS_METRICS,
    RagasUnavailable,
    build_judges,
    build_ragas_records_from_service,
    evaluate_ragas,
    judge_config_problems,
    normalize_metric_names,
    ragas_environment_problems,
    summarize_scores,
    to_ragas_rows,
)


class _FakeService:
    def ask(self, question):
        return {"answer": f"answer to {question}",
                "sources": [{"article_number": 147, "citation": "c", "text": "ctx text"}]}


def _env(**changes):
    saved = {k: os.environ.get(k) for k in changes}
    for k, v in changes.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    return saved


def _restore(saved):
    for k, v in saved.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_the_four_required_metrics_are_defined():
    assert set(RAGAS_METRICS) == {"faithfulness", "answer_relevancy", "context_precision", "context_recall"}


def test_records_from_service_carry_generated_answer_real_contexts_and_reference():
    items = [{"id": "q1", "question": "Q?", "reference": "ref text"}]
    rec = build_ragas_records_from_service(items, _FakeService())[0]
    assert rec["answer"] == "answer to Q?" and rec["contexts"] == ["ctx text"]
    assert rec["ground_truth"] == "ref text" and rec["retrieved_article_numbers"] == [147]


def test_to_ragas_rows_uses_exactly_the_ragas_sample_schema():
    rec = {"question": "Q?", "contexts": ["c"], "answer": "a", "ground_truth": "r",
           "retrieved_article_numbers": [1], "id": "x"}
    assert to_ragas_rows([rec]) == [
        {"user_input": "Q?", "retrieved_contexts": ["c"], "response": "a", "reference": "r"}]


def test_to_ragas_rows_requires_a_reference_for_precision_and_recall():
    try:
        to_ragas_rows([{"question": "Q", "contexts": ["c"], "answer": "a", "ground_truth": None}])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_metric_names_from_different_ragas_versions_normalise():
    raw = {"user_input": "q", "faithfulness": 1, "answer_relevancy": 2,
           "llm_context_precision_with_reference": 3, "context_recall": 4}
    assert normalize_metric_names(raw) == {
        "faithfulness": 1, "answer_relevancy": 2, "context_precision": 3, "context_recall": 4}


def _rows(faith):
    return [{"faithfulness": f, "answer_relevancy": 0.9, "context_precision": 0.8,
             "context_recall": 0.7} for f in faith]


def test_summary_averages_all_four_metrics():
    s = summarize_scores(_rows([1.0, 0.5]))
    assert s["n"] == 2 and s["scores"]["faithfulness"] == 0.75
    assert set(s["scores"]) == set(RAGAS_METRICS)


def test_unscorable_rows_count_as_zero_for_the_gating_score_and_are_reported():
    s = summarize_scores(_rows([1.0, 1.0, 1.0, float("nan")]))
    assert s["scores"]["faithfulness"] == 0.75  # NaN row can only lower the score
    assert s["n_valid"]["faithfulness"] == 3 and s["mean_valid"]["faithfulness"] == 1.0
    assert not math.isnan(s["scores"]["faithfulness"])


def test_summary_fails_if_a_metric_is_missing_or_row_count_wrong():
    for bad in ([{"faithfulness": 1.0}], ):
        try:
            summarize_scores(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError for missing metrics")
    try:
        summarize_scores(_rows([1.0]), expected_n=2)
    except ValueError:
        return
    raise AssertionError("expected ValueError for row-count mismatch")


def test_judge_not_configured_by_default_and_build_judges_fails_closed():
    saved = _env(RAGAS_JUDGE_MODEL=None, RAGAS_JUDGE_BASE_URL=None,
                 RAGAS_JUDGE_API_KEY=None, OPENAI_API_KEY=None)
    try:
        assert len(judge_config_problems()) == 2
        try:
            build_judges()
        except RagasUnavailable:
            return
        raise AssertionError("expected RagasUnavailable")
    finally:
        _restore(saved)


def test_judge_config_satisfied_by_model_plus_endpoint():
    saved = _env(RAGAS_JUDGE_MODEL="judge", RAGAS_JUDGE_BASE_URL="http://x/v1",
                 RAGAS_JUDGE_API_KEY=None, OPENAI_API_KEY=None)
    try:
        assert judge_config_problems() == []
    finally:
        _restore(saved)


def test_environment_problems_list_missing_generation_backend():
    saved = _env(GENERATION_MODEL=None)
    try:
        assert any("GENERATION_MODEL" in p for p in ragas_environment_problems())
    finally:
        _restore(saved)


def test_evaluate_ragas_raises_when_ragas_missing_never_returns_scores():
    try:
        import ragas  # noqa: F401
        return "SKIPPED"  # real ragas present: exercised only in a full environment
    except ImportError:
        pass
    try:
        evaluate_ragas([{"question": "q", "contexts": ["c"], "answer": "a", "ground_truth": "r"}], None, None)
    except RagasUnavailable:
        return
    raise AssertionError("expected RagasUnavailable")
