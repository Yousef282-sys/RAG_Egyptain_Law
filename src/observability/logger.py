"""Structured, local, dependency-free request logging.

Writes one JSON line per event to a local file -- no external service
required, works fully offline. Deliberately does NOT use any optional
package (no `structlog`, no `langfuse`) so it always works regardless of
what's installed.

Privacy/safety guardrail: `build_event` only accepts fields in
`ALLOWED_EVENT_FIELDS` and raises on anything else. This is enforced in
code, not just by convention, specifically so a future edit can never
accidentally start logging the raw question text, retrieved article
text, secrets, or environment variables -- it would fail loudly instead.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULT_LOG_DIR = Path(os.environ.get("RAG_LOG_DIR", "logs"))
DEFAULT_LOG_FILE = DEFAULT_LOG_DIR / "requests.jsonl"

# Everything this system is allowed to log about a request. Notably
# absent: the question text itself, retrieved article text/citations,
# any API key/secret, or raw environment variables. `question_length`
# (a count, not content) is allowed as a useful, non-sensitive signal.
ALLOWED_EVENT_FIELDS = frozenset(
    {
        "request_id",
        "timestamp",
        "route",
        "latency_ms",
        "status",
        "result_count",
        "question_length",
        "error_type",
    }
)


def build_event(**fields: Any) -> Dict[str, Any]:
    """Build a log event dict, enforcing the field allowlist above.
    Raises ValueError if any field is not explicitly allowed -- this is
    the guardrail against ever logging sensitive content by accident.
    """
    unknown = set(fields) - ALLOWED_EVENT_FIELDS
    if unknown:
        raise ValueError(
            f"Refusing to log un-allowlisted field(s): {sorted(unknown)}. "
            f"Allowed fields: {sorted(ALLOWED_EVENT_FIELDS)}"
        )
    return dict(fields)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_event(event: Dict[str, Any], log_file: Optional[str | Path] = None) -> bool:
    """Append one event as a JSON line to the local log file.

    `log_file` defaults to `DEFAULT_LOG_FILE`, resolved at CALL time (not
    baked in as an eager default-argument value) -- this lets tests (and
    callers like src/api/main.py's /ask handler, which doesn't pass
    `log_file` explicitly) redirect logging by monkeypatching
    `logger.DEFAULT_LOG_FILE` before this runs.

    Never raises: a logging failure (disk full, permissions, etc.) must
    never break the actual request it's describing. Returns True if the
    write succeeded, False otherwise.
    """
    if log_file is None:
        log_file = DEFAULT_LOG_FILE
    try:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return True
    except Exception:
        return False
