"""FastAPI app: /health, /ask, /ask/stream for the Arabic Legal RAG pipeline.

Retrieval -> (vLLM-compatible) generation -> {answer, sources}; sources are
keyed by legal ARTICLE NUMBER. All business logic lives in
src/rag/service.py (shared with the BentoML service); this module is the
thin HTTP layer.

Requires fastapi / pydantic v2 / uvicorn. Those are NOT installed in the
development sandbox, so this module has not been executed there --
tests/test_api.py skips (never fakes a pass) when they are missing. The
logic it delegates to (service/generation) IS unit-tested offline.

Configuration (env vars, all optional):
  VECTOR_STORE_DIR    -- path to the built index (default: data/vector_store)
  RAG_TOP_K           -- chunks retrieved per question (default: 5)
  MAX_QUESTION_LENGTH -- max accepted question length, chars (default: 2000)
  GENERATION_MODEL / VLLM_BASE_URL / VLLM_API_KEY / GENERATION_MAX_TOKENS /
  GENERATION_TEMPERATURE / GENERATION_TIMEOUT_S -- vLLM backend, see
  src/rag/generation.py. With GENERATION_MODEL unset the API runs in an
  explicitly-labelled retrieval-only mode.

Every request logs one structured local event (request id, latency,
status, result count -- never question or article text), see
src/observability/logger.py.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from src.api.validation import (
    DEFAULT_MAX_QUESTION_LENGTH,
    validate_question,
    validate_top_k,
)
from src.observability.logger import build_event, log_event, now_iso
from src.rag.generation import GenerationError, get_generator
from src.rag.retriever import Retriever
from src.rag.service import RagService, index_info
from src.rag.vector_store import DEFAULT_STORE_DIR

VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR", str(DEFAULT_STORE_DIR))
MAX_QUESTION_LENGTH = int(os.environ.get("MAX_QUESTION_LENGTH", str(DEFAULT_MAX_QUESTION_LENGTH)))
RAG_TOP_K = validate_top_k(os.environ.get("RAG_TOP_K", "5"))

app = FastAPI(title="Arabic Legal RAG")

# Generator construction does not contact the model server.
_generator = get_generator()

# Loaded once, lazily, on first request. A load failure (e.g. index not
# built yet) is cached and reported as a clean 503 instead of crashing.
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
    # Cheap and independent of the vector index/model server: "process is
    # up" plus index metadata read from the manifest. Never loads the
    # embedder and never calls the generation backend.
    return {
        "status": "ok",
        "generator": _generator.name,
        "generation_is_llm": _generator.is_llm,
        **index_info(VECTOR_STORE_DIR),
    }


def _log(request_id: str, start: float, route: str, status: str, **extra) -> None:
    log_event(
        build_event(
            request_id=request_id,
            timestamp=now_iso(),
            route=route,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            status=status,
            **extra,
        )
    )


def _service() -> RagService:
    return RagService(get_retriever(), _generator)  # raises 503 if no index


@app.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        service = _service()
    except HTTPException as exc:
        _log(request_id, start, "/ask", f"error_{exc.status_code}",
             result_count=0, error_type="RetrieverUnavailable")
        raise

    try:
        result = service.ask(payload.question)
    except GenerationError as exc:
        _log(request_id, start, "/ask", "error_502",
             result_count=0, error_type="GenerationError")
        raise HTTPException(status_code=502, detail=f"Generation backend error: {exc}")
    except Exception as exc:
        _log(request_id, start, "/ask", "error_500",
             result_count=0, error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Request failed: {exc}")

    _log(request_id, start, "/ask", "ok",
         result_count=len(result["sources"]), question_length=len(payload.question))
    return AskResponse(**result)


@app.post("/ask/stream")
def ask_stream(payload: AskRequest) -> StreamingResponse:
    """Server-sent events: one `sources` event, then `token` events as the
    generation backend produces text, then `done` (or `error`).
    """
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        events = _service().ask_stream(payload.question)
        first = next(events)  # retrieval runs here -> real HTTP errors, pre-stream
    except HTTPException as exc:
        _log(request_id, start, "/ask/stream", f"error_{exc.status_code}",
             result_count=0, error_type="RetrieverUnavailable")
        raise
    except Exception as exc:
        _log(request_id, start, "/ask/stream", "error_500",
             result_count=0, error_type=type(exc).__name__)
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}")

    def _sse(event: dict) -> str:
        return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    def _body():
        yield _sse(first)
        try:
            for event in events:
                yield _sse(event)
        except GenerationError as exc:
            _log(request_id, start, "/ask/stream", "error_502",
                 result_count=len(first["sources"]), error_type="GenerationError")
            yield _sse({"type": "error", "detail": f"Generation backend error: {exc}"})
            return
        _log(request_id, start, "/ask/stream", "ok",
             result_count=len(first["sources"]), question_length=len(payload.question))

    return StreamingResponse(_body(), media_type="text/event-stream")
