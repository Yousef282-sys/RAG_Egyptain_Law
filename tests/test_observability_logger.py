import json
import tempfile
from pathlib import Path

from src.observability.logger import ALLOWED_EVENT_FIELDS, build_event, log_event, now_iso


def test_build_event_accepts_allowlisted_fields():
    event = build_event(request_id="r1", route="/ask", latency_ms=10.0, status="ok", result_count=3)
    assert event["request_id"] == "r1"
    assert event["result_count"] == 3


def test_build_event_rejects_question_text_field():
    raised = False
    try:
        build_event(request_id="r1", question="ما هو عقد البيع؟")
    except ValueError as e:
        raised = True
        assert "question" in str(e)
    assert raised


def test_build_event_rejects_citation_or_article_text_fields():
    raised = False
    try:
        build_event(request_id="r1", citation="المادة 1", text_ar="نص المادة")
    except ValueError:
        raised = True
    assert raised


def test_build_event_rejects_arbitrary_unknown_field():
    raised = False
    try:
        build_event(request_id="r1", api_key="secret123")
    except ValueError:
        raised = True
    assert raised


def test_allowed_fields_never_includes_content_like_names():
    safe_suffixes = ("_length", "_count", "_id", "_ms")
    forbidden_substrings = ("text", "citation", "secret", "password", "token")
    for field in ALLOWED_EVENT_FIELDS:
        if field.endswith(safe_suffixes):
            continue
        for bad in forbidden_substrings:
            assert bad not in field.lower(), f"{field!r} looks like it could hold sensitive content"
        assert field != "question"
        assert field != "key" and "_key" not in field.lower()


def test_log_event_writes_one_json_line():
    with tempfile.TemporaryDirectory() as d:
        log_file = Path(d) / "requests.jsonl"
        event = build_event(request_id="r1", route="/ask", status="ok", result_count=1)
        ok = log_event(event, log_file=log_file)
        assert ok is True
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == event


def test_log_event_appends_multiple_events():
    with tempfile.TemporaryDirectory() as d:
        log_file = Path(d) / "requests.jsonl"
        for i in range(3):
            log_event(build_event(request_id=f"r{i}", route="/ask", status="ok"), log_file=log_file)
        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3


def test_log_event_never_raises_on_bad_path():
    bad_path = "/this/path/does/not/exist/and/cannot/be/created\0invalid"
    result = log_event(build_event(request_id="r1"), log_file=bad_path)
    assert result is False


def test_log_event_default_destination_is_resolved_at_call_time():
    # log_file must NOT be an eager default-argument value (which would
    # bake in the module's DEFAULT_LOG_FILE at import time and make it
    # impossible to redirect via monkeypatching, as src/api/main.py's
    # /ask handler relies on for its own tests).
    import src.observability.logger as obs_logger

    with tempfile.TemporaryDirectory() as d:
        redirected = Path(d) / "redirected.jsonl"
        original = obs_logger.DEFAULT_LOG_FILE
        try:
            obs_logger.DEFAULT_LOG_FILE = redirected
            log_event(build_event(request_id="r1", route="/ask", status="ok"))
        finally:
            obs_logger.DEFAULT_LOG_FILE = original
        assert redirected.exists()


def test_now_iso_returns_a_parseable_timestamp():
    from datetime import datetime

    ts = now_iso()
    parsed = datetime.fromisoformat(ts)
    assert parsed is not None
