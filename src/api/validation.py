"""Validation and config-parsing logic for the API, kept dependency-free
(no fastapi/pydantic import here) so it can be unit-tested in this
sandbox, which has neither package installed.

src/api/main.py imports these functions and wraps them in a pydantic
validator / env var read -- the actual logic lives here so tests don't
need fastapi/pydantic to exercise it.
"""
from __future__ import annotations

from typing import Optional

DEFAULT_MAX_QUESTION_LENGTH = 2000
DEFAULT_TOP_K = 5


def validate_top_k(raw_value: str) -> int:
    """Parse and validate RAG_TOP_K. Fails fast (raises ValueError) on a
    bad value rather than letting the service silently misbehave at
    request time -- e.g. top_k=0 would make every /ask return an empty
    sources list forever; a negative value would error deep inside numpy
    with a confusing message far from the actual misconfiguration.
    """
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        raise ValueError(f"RAG_TOP_K must be an integer, got {raw_value!r}")
    if value <= 0:
        raise ValueError(f"RAG_TOP_K must be a positive integer, got {value}")
    return value


def validate_question(value: Optional[str], max_length: int = DEFAULT_MAX_QUESTION_LENGTH) -> str:
    """Validate the /ask request's question field. Raises ValueError on
    any invalid input (empty/whitespace-only, or unreasonably long --
    guards against a pathologically large payload being embedded/
    retrieved against on every request).
    """
    if not value or not value.strip():
        raise ValueError("question must not be empty or whitespace-only")
    if len(value) > max_length:
        raise ValueError(
            f"question exceeds the maximum allowed length ({max_length} "
            f"characters); got {len(value)} characters"
        )
    return value
