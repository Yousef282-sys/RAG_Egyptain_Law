import tempfile
from pathlib import Path

from src.observability.logger import build_event, log_event
from src.observability.monitoring import load_events, print_summary, summarize_retrieval


def test_load_events_empty_when_no_log_file():
    events = load_events("/tmp/definitely-does-not-exist-day8.jsonl")
    assert events == []


def test_load_events_reads_back_exactly_what_was_logged():
    with tempfile.TemporaryDirectory() as d:
        log_file = Path(d) / "requests.jsonl"
        log_event(build_event(request_id="r1", route="/ask", status="ok", result_count=2), log_file)
        log_event(build_event(request_id="r2", route="/ask", status="ok", result_count=0), log_file)
        events = load_events(log_file)
    assert len(events) == 2
    assert events[0]["request_id"] == "r1"


def test_summarize_retrieval_with_no_events_reports_zero_not_fabricated_rate():
    summary = summarize_retrieval([])
    assert summary == {"n_requests": 0}


def test_summarize_retrieval_ignores_non_ask_routes():
    events = [build_event(request_id="r1", route="/health", status="ok")]
    summary = summarize_retrieval(events)
    assert summary["n_requests"] == 0


def test_summarize_retrieval_computes_correct_empty_rate():
    events = [
        build_event(request_id="r1", route="/ask", status="ok", result_count=3, latency_ms=10.0),
        build_event(request_id="r2", route="/ask", status="ok", result_count=0, latency_ms=8.0),
    ]
    summary = summarize_retrieval(events)
    assert summary["n_requests"] == 2
    assert summary["empty_result_count"] == 1
    assert summary["empty_result_rate"] == 0.5
    assert summary["avg_result_count"] == 1.5


def test_summarize_retrieval_computes_correct_latency_stats():
    events = [
        build_event(request_id="r1", route="/ask", status="ok", result_count=1, latency_ms=10.0),
        build_event(request_id="r2", route="/ask", status="ok", result_count=1, latency_ms=20.0),
    ]
    summary = summarize_retrieval(events)
    assert summary["avg_latency_ms"] == 15.0
    assert summary["min_latency_ms"] == 10.0
    assert summary["max_latency_ms"] == 20.0


def test_summarize_retrieval_counts_errors_correctly():
    events = [
        build_event(request_id="r1", route="/ask", status="ok", result_count=1),
        build_event(request_id="r2", route="/ask", status="error_500", result_count=0),
        build_event(request_id="r3", route="/ask", status="error_503", result_count=0),
    ]
    summary = summarize_retrieval(events)
    assert summary["error_count"] == 2
    assert round(summary["error_rate"], 4) == round(2 / 3, 4)


def test_summary_never_reports_answer_quality_or_faithfulness_keys():
    events = [build_event(request_id="r1", route="/ask", status="ok", result_count=1, latency_ms=5.0)]
    summary = summarize_retrieval(events)
    forbidden = ("faithfulness", "accuracy", "quality", "hit_rate", "reciprocal_rank")
    for key in summary:
        for bad in forbidden:
            assert bad not in key.lower()


def test_print_summary_runs_end_to_end_on_a_real_log_file():
    import contextlib
    import io

    with tempfile.TemporaryDirectory() as d:
        log_file = Path(d) / "requests.jsonl"
        log_event(build_event(request_id="r1", route="/ask", status="ok", result_count=2, latency_ms=9.0), log_file)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            summary = print_summary(log_file)
    assert summary["n_requests"] == 1
    assert "Retrieval monitoring summary" in buf.getvalue()
