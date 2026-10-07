"""BentoML service for the Arabic Legal RAG pipeline.

A thin wrapper over the SAME `RagService` (src/rag/service.py) that backs
the FastAPI app -- retrieval, article-cited sources and (vLLM) generation
are not reimplemented here. Same env configuration as the FastAPI app
(VECTOR_STORE_DIR, RAG_TOP_K, MAX_QUESTION_LENGTH, GENERATION_MODEL,
VLLM_BASE_URL, ...).

Endpoints: `ask` (-> {answer, sources}), `ask_stream` (async generator of
JSON lines: sources, token..., done), `health`.

UNVERIFIED: `bentoml` is not installed in the development sandbox, so this
module has not been imported/served/built there. Written for the
class-based API (BentoML >= 1.2):
    bentoml serve src.serving.bento_service:ArabicLegalRAGService
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator

import bentoml

from src.api.validation import (
    DEFAULT_MAX_QUESTION_LENGTH,
    validate_question,
    validate_top_k,
)
from src.rag.retriever import Retriever
from src.rag.service import RagService
from src.rag.vector_store import DEFAULT_STORE_DIR

_VECTOR_STORE_DIR = os.environ.get("VECTOR_STORE_DIR", str(DEFAULT_STORE_DIR))
_TOP_K = validate_top_k(os.environ.get("RAG_TOP_K", "5"))
_MAX_Q = int(os.environ.get("MAX_QUESTION_LENGTH", str(DEFAULT_MAX_QUESTION_LENGTH)))
_END = object()


@bentoml.service(name="arabic_legal_rag")
class ArabicLegalRAGService:
    """Loads retriever + generator once per worker process."""

    def __init__(self) -> None:
        self._service = RagService(Retriever(store_dir=_VECTOR_STORE_DIR, top_k=_TOP_K))

    @bentoml.api
    async def ask(self, question: str) -> dict:
        question = validate_question(question, max_length=_MAX_Q)  # ValueError on empty
        return await asyncio.to_thread(self._service.ask, question)

    @bentoml.api
    async def ask_stream(self, question: str) -> AsyncGenerator[str, None]:
        question = validate_question(question, max_length=_MAX_Q)
        events = self._service.ask_stream(question)
        # Backend iteration is blocking (HTTP read); keep the event loop free.
        while True:
            event = await asyncio.to_thread(next, events, _END)
            if event is _END:
                return
            yield json.dumps(event, ensure_ascii=False) + "\n"

    @bentoml.api
    def health(self) -> dict:
        return self._service.health(_VECTOR_STORE_DIR)
