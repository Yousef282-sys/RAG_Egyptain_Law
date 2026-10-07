"""FastAPI app exposing /health and /ask for the Arabic Legal RAG pipeline.

IMPORTANT SANDBOX NOTE: this module requires `fastapi`, `pydantic` (v2),
and `uvicorn` to run, none of which are installed in this development
sandbox (no network access here to install them). The code below is
standard FastAPI/Pydantic v2 and is expected to run as-is once those
packages are installed (see README) -- but it has NOT been executed or
tested in this sandbox. tests/test_api.py documents this and skips
(rather than fabricating a pass) when these imports are unavailable.

Configuration (env vars, all optional):
  VECTOR_STORE_DIR    -- path to the built index (default: data/vector_store)
  RAG_TOP_K           -- number of chunks to retrieve per question (default: 5)
  MAX_QUESTION_LENGTH -- max accepted question length, chars (default: 2000)

Day 7 audit: RAG_TOP_K is validated at import time (fail fast on a bad
value instead of misbehaving at request time), and /ask questions are
rejected if unreasonably long. Both checks' actual logic lives in
src/api/validation.py (no fastapi/pydantic import there), kept separate
so it stays unit-testable in this sandbox without either package --
see tests/test_api_validation.py.

Day 8: every /ask request logs one structured, local event (request id,
timestamp, latency, status, result count -- never the question text
itself or retrieved article text) via src/observability/logger.py. See
that module's ALLOWED_EVENT_FIELDS for the exact, enforced allowlist.
"""
from __future__ import annotations

import os
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from src.api.validation import (
    DEFAULT_MAX_QUESTION_LENGTH,
    validate_question,
    validate_top_k,
)
from src.observability.logger import build_event, log_event, now_iso
from src.rag.retriever import Retriever
from src.rag.vector_store import DEFAULT_STORE_DIR

VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR", str(DEFAULT_STORE_DIR))
MAX_QUESTION_LENGTH = int(os.environ.get("MAX_QUESTION_LENGTH", str(DEFAULT_MAX_QUESTION_LENGTH)))
RAG_TOP_K = validate_top_k(os.environ.get("RAG_TOP_K", "5"))

app = FastAPI(title="Arabic Legal RAG - Day 3 (retrieval-only)")

# Loaded once, lazily, on first request -- never rebuilt/reloaded per
# request. A load failure (e.g. index not built yet) is cached as an
# error and reported as a clean 503 on every subsequent request instead
# of retrying (and instead of crashing the process).
_retriever: Optional[Retriever] = None
_retriever_load_error: Optional[str] = None


def get_retriever() -> Retriever:
    global _retriever, _retriever_load_error
    if _retriever is not None:
        return _retriever
    if _retriever_load_error is not None:
        raise HTTPException(status_code=503, detail=_retriever_load_error)
    try:
        _retriever = Retriever(store_dir=VECTOR_STORE_DIR, top_k=RAG_TOP_K)
    except Exception as exc:  # index missing/corrupt, backend unavailable, etc.
        _retriever_load_error = (
            f"Retrieval index unavailable ({type(exc).__name__}: {exc}). "
            f"Build it first: see README 'Build / rebuild the index'."
        )
        raise HTTPException(status_code=503, detail=_retriever_load_error)
    return _retriever


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        return validate_question(value, max_length=MAX_QUESTION_LENGTH)


class Source(BaseModel):
    article_number: int
    citation: str
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


@app.get("/health")
def health() -> dict:
    # Deliberately cheap/independent of the index so it reflects "the
    # process is up", not "the index is loaded" -- Kubernetes-style
    # liveness check. Index status is reported via the 503 path on /ask.
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    def _latency_ms() -> float:
        return round((time.perf_counter() - start) * 1000, 2)

    try:
        retriever = get_retriever()  # raises HTTPException(503) if unavailable
    except HTTPException as exc:
        log_event(
            build_event(
                request_id=request_id,
                timestamp=now_iso(),
                route="/ask",
                latency_ms=_latency_ms(),
                status=f"error_{exc.status_code}",
                result_count=0,
                error_type="RetrieverUnavailable",
            )
        )
        raise

    try:
        results = retriever.retrieve(payload.question)
    except Exception as exc:
        log_event(
            build_event(
                request_id=request_id,
                timestamp=now_iso(),
                route="/ask",
                latency_ms=_latency_ms(),
                status="error_500",
                result_count=0,
                error_type=type(exc).__name__,
            )
        )
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    sources = [
        Source(
            article_number=r["article_number"],
            citation=r["citation"] or "",
            text=r["text_ar"] or r["text_en"] or "",
        )
        for r in results
    ]

    if not sources:
        answer = "No relevant articles were found for this question."
    else:
        top = sources[0]
        # Development-only, retrieval-only answer. No generation model is
        # configured, so this is explicitly NOT presented as an
        # LLM-generated answer -- it only names the top retrieved article
        # and points at the sources list.
        answer = (
            "[retrieval-only, no generation model configured] "
            f"The most relevant provision found is {top.citation}. "
            "See the sources list for the retrieved article text."
        )

    log_event(
        build_event(
            request_id=request_id,
            timestamp=now_iso(),
            route="/ask",
            latency_ms=_latency_ms(),
            status="ok",
            result_count=len(sources),
            question_length=len(payload.question),
        )
    )
    return AskResponse(answer=answer, sources=sources)
