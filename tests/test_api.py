"""Tests for the FastAPI app in src/api/main.py.

This development sandbox has no network access and does not have
`fastapi`/`pydantic`(v2)/`httpx` installed, so these tests cannot
actually be executed here. Rather than skip silently (which would look
identical to a real pass in a naive runner) or fabricate a "passed"
result, each test explicitly detects the missing dependency, prints a
SKIP notice, and returns the string "SKIPPED" -- run_tests.py counts
these separately from real passes/failures.

Run these for real wherever fastapi/pydantic/httpx ARE installed (e.g.
the user's own dev environment) -- see PROGRESS.md / README.
"""
try:
    from fastapi.testclient import TestClient

    from src.api.main import app

    _client = TestClient(app)
    API_AVAILABLE = True
except Exception:
    API_AVAILABLE = False


def _skip(name: str) -> str:
    print(
        f"  ({name}: fastapi/pydantic/httpx not installed in this sandbox "
        "-- not executed here, run in an environment with those packages)"
    )
    return "SKIPPED"


def test_health_endpoint_returns_ok():
    if not API_AVAILABLE:
        return _skip("test_health_endpoint_returns_ok")
    response = _client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_valid_request_returns_answer_and_sources():
    if not API_AVAILABLE:
        return _skip("test_ask_valid_request_returns_answer_and_sources")
    response = _client.post("/ask", json={"question": "ما هو عقد البيع؟"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    for source in body["sources"]:
        assert "article_number" in source
        assert "citation" in source
        assert "text" in source


def test_ask_empty_question_returns_422():
    if not API_AVAILABLE:
        return _skip("test_ask_empty_question_returns_422")
    response = _client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_whitespace_only_question_returns_422():
    if not API_AVAILABLE:
        return _skip("test_ask_whitespace_only_question_returns_422")
    response = _client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_ask_sources_never_expose_chunk_ids_only_article_numbers():
    if not API_AVAILABLE:
        return _skip("test_ask_sources_never_expose_chunk_ids_only_article_numbers")
    response = _client.post("/ask", json={"question": "ما هو عقد الايجار؟"})
    body = response.json()
    for source in body["sources"]:
        assert isinstance(source["article_number"], int)


def test_config_env_vars_have_sane_defaults():
    # Day 3: config-through-env-vars requirement. Pure import-level check,
    # gated the same way since it imports src.api.main.
    if not API_AVAILABLE:
        return _skip("test_config_env_vars_have_sane_defaults")
    from src.api import main as api_main

    assert api_main.RAG_TOP_K >= 1
    assert api_main.VECTOR_STORE_DIR  # non-empty


def test_ask_returns_503_not_a_crash_when_index_unavailable():
    if not API_AVAILABLE:
        return _skip("test_ask_returns_503_not_a_crash_when_index_unavailable")
    from src.api import main as api_main

    # Force the lazy singleton to re-attempt loading from a path that
    # cannot possibly hold a real index, then restore state afterwards
    # so other tests in this module are unaffected.
    saved_retriever = api_main._retriever
    saved_error = api_main._retriever_load_error
    saved_dir = api_main.VECTOR_STORE_DIR
    try:
        api_main._retriever = None
        api_main._retriever_load_error = None
        api_main.VECTOR_STORE_DIR = "/nonexistent/path/for/testing/only"

        response = _client.post("/ask", json={"question": "سؤال تجريبي"})
        assert response.status_code == 503
    finally:
        api_main._retriever = saved_retriever
        api_main._retriever_load_error = saved_error
        api_main.VECTOR_STORE_DIR = saved_dir


def test_ask_missing_question_field_returns_422():
    # Day 7 audit: realistic client-error case -- payload missing the
    # required field entirely, not just empty.
    if not API_AVAILABLE:
        return _skip("test_ask_missing_question_field_returns_422")
    response = _client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_malformed_json_body_returns_422_not_500():
    # Day 7 audit: a non-JSON body must not crash the process with a
    # bare 500 -- FastAPI/Pydantic should reject it cleanly.
    if not API_AVAILABLE:
        return _skip("test_ask_malformed_json_body_returns_422_not_500")
    response = _client.post(
        "/ask", content=b"not valid json{{{", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422


def test_ask_over_max_length_question_returns_422():
    # Day 7 audit: exercises the MAX_QUESTION_LENGTH guard
    # (src/api/validation.py) at the actual API layer.
    if not API_AVAILABLE:
        return _skip("test_ask_over_max_length_question_returns_422")
    from src.api import main as api_main

    too_long = "س" * (api_main.MAX_QUESTION_LENGTH + 1)
    response = _client.post("/ask", json={"question": too_long})
    assert response.status_code == 422


def test_ask_wrong_type_for_question_returns_422():
    # Day 7 audit: a non-string question (e.g. a number) must be
    # rejected by Pydantic type validation, not crash retrieval.
    if not API_AVAILABLE:
        return _skip("test_ask_wrong_type_for_question_returns_422")
    response = _client.post("/ask", json={"question": 12345})
    assert response.status_code == 422


def test_ask_logs_a_structured_event_without_question_text():
    # Day 8: confirms /ask actually calls the observability logger, and
    # that the logged event never contains the raw question text.
    if not API_AVAILABLE:
        return _skip("test_ask_logs_a_structured_event_without_question_text")
    import tempfile
    from pathlib import Path

    from src.observability import logger as obs_logger
    from src.observability.monitoring import load_events

    with tempfile.TemporaryDirectory() as d:
        log_file = Path(d) / "requests.jsonl"
        original_default = obs_logger.DEFAULT_LOG_FILE
        try:
            obs_logger.DEFAULT_LOG_FILE = log_file
            _client.post("/ask", json={"question": "ما هو عقد البيع؟"})
        finally:
            obs_logger.DEFAULT_LOG_FILE = original_default

        events = load_events(log_file)
        assert len(events) >= 1
        for event in events:
            assert "ما هو عقد البيع" not in str(event)
            assert set(event.keys()).issubset(obs_logger.ALLOWED_EVENT_FIELDS)
