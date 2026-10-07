"""Tests for tests/load/locustfile.py.

Two things are checked, deliberately separated:
1. Syntax validity -- via py_compile, which does NOT require `locust` to
   be installed. This always runs for real.
2. Actual import/structure (HttpUser subclass, tasks) -- requires
   `locust`, which is NOT installed in this sandbox (no network here to
   install it). Honestly SKIPPED (counted separately by run_tests.py)
   when unavailable, never fabricated as passing.

No load-test benchmark numbers are produced by this file -- see
PROGRESS.md for why (no locust, no running server in this sandbox).
"""
import py_compile
from pathlib import Path

LOCUSTFILE_PATH = Path(__file__).parent / "load" / "locustfile.py"

try:
    from locust import HttpUser

    import importlib.util

    _spec = importlib.util.spec_from_file_location("locustfile", LOCUSTFILE_PATH)
    _locustfile = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_locustfile)
    LOCUST_AVAILABLE = True
except Exception:
    LOCUST_AVAILABLE = False


def _skip(name: str) -> str:
    print(
        f"  ({name}: locust not installed in this sandbox -- not executed "
        "here, run in an environment with that package)"
    )
    return "SKIPPED"


def test_locustfile_syntax_is_valid():
    # Always runs for real -- py_compile needs no locust import.
    py_compile.compile(str(LOCUSTFILE_PATH), doraise=True)


def test_locustfile_defines_an_http_user_with_ask_task():
    if not LOCUST_AVAILABLE:
        return _skip("test_locustfile_defines_an_http_user_with_ask_task")
    assert issubclass(_locustfile.AskUser, HttpUser)
    assert hasattr(_locustfile.AskUser, "ask_question")


def test_locustfile_uses_real_ask_request_body_shape():
    if not LOCUST_AVAILABLE:
        return _skip("test_locustfile_uses_real_ask_request_body_shape")
    assert len(_locustfile.SAMPLE_QUESTIONS) > 0
    for q in _locustfile.SAMPLE_QUESTIONS:
        assert isinstance(q, str) and q.strip()
