"""Optional Langfuse tracing adapter.

`langfuse` is NOT installed in this sandbox, and no Langfuse server or
credentials are configured here (confirmed before writing this module;
this sandbox also has no network to reach a real server even if
credentials existed). This adapter is a thin, optional wrapper around the
local observability events already produced by
src/observability/logger.py -- it does not duplicate that logic.

`send_trace()` NEVER claims success unless a real `langfuse` SDK call
actually returns. If either the package or the required credentials are
missing, it returns a clear `{"sent": False, "reason": ...}` result.

Exact prerequisites to make this real:
1. `pip install langfuse`.
2. A real Langfuse server (cloud or self-hosted) reachable from wherever
   this runs.
3. `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` environment variables
   set to real credentials for that server.
None of the three exist in this sandbox.
"""
from __future__ import annotations

import os
from typing import Any, Dict

REQUIRED_ENV_VARS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def langfuse_importable() -> bool:
    try:
        import langfuse  # noqa: F401

        return True
    except ImportError:
        return False


def langfuse_configured() -> bool:
    return all(os.environ.get(var) for var in REQUIRED_ENV_VARS)


def send_trace(event: Dict[str, Any]) -> Dict[str, Any]:
    """Attempt to forward one local observability event to Langfuse as a
    trace. Returns {"sent": False, "reason": <str>} whenever the package
    isn't installed or credentials aren't configured -- both true in
    this sandbox. Only returns `sent: True` after a real Langfuse client
    call succeeds.
    """
    if not langfuse_importable():
        return {"sent": False, "reason": "langfuse package is not installed"}
    if not langfuse_configured():
        missing = [v for v in REQUIRED_ENV_VARS if not os.environ.get(v)]
        return {"sent": False, "reason": f"missing environment variable(s): {missing}"}

    # Real path -- only reachable once both requirements above are met.
    # Not implemented/exercised in this sandbox (never reachable here).
    from langfuse import Langfuse

    client = Langfuse()
    trace = client.trace(name=event.get("route", "unknown"), metadata=event)
    return {"sent": True, "trace_id": trace.id}
