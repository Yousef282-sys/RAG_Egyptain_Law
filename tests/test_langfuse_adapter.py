import os

from src.observability.langfuse_adapter import (
    REQUIRED_ENV_VARS,
    langfuse_configured,
    langfuse_importable,
    send_trace,
)


def test_langfuse_importable_is_false_in_this_sandbox():
    assert langfuse_importable() is False


def test_langfuse_configured_is_false_without_env_vars():
    for var in REQUIRED_ENV_VARS:
        os.environ.pop(var, None)
    assert langfuse_configured() is False


def test_langfuse_configured_requires_all_vars_not_just_one():
    os.environ.pop("LANGFUSE_SECRET_KEY", None)
    os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-test-only"
    try:
        assert langfuse_configured() is False
    finally:
        os.environ.pop("LANGFUSE_PUBLIC_KEY", None)


def test_send_trace_never_claims_success_when_package_missing():
    result = send_trace({"route": "/ask", "status": "ok"})
    assert result["sent"] is False
    assert "langfuse" in result["reason"].lower()


def test_send_trace_result_never_contains_a_fabricated_trace_id():
    result = send_trace({"route": "/ask"})
    assert "trace_id" not in result
