"""BentoML service exposing async `ask` for the Arabic Legal RAG pipeline.

NOT EXECUTED in this sandbox: `bentoml` is not installed here and there
is no network access to install it (confirmed before writing this file).
This module is a thin serving wrapper -- it reuses the existing
`Retriever` (src/rag/retriever.py) rather than reimplementing retrieval,
per the Day 3 instructions. Written for BentoML's class-based service API
(BentoML >= 1.2). Install `bentoml` (see README) and run
`bentoml serve src.serving.bento_service:ArabicLegalRAGService` to try it
for real -- that command has not been run here.

Streaming: `ask` is already async, but there is no generation model
configured (this pipeline is retrieval-only -- see src/api/main.py's
docstring), so there is no token stream to relay. Faking a streamed
response over a single, already-complete string would misrepresent it as
token-by-token generation, which the Day 3 instructions explicitly
disallow. Once a real streaming generation backend exists, `ask` only
needs to become an async generator yielding partial output instead of
returning one dict -- no other architecture change is required here.
"""
from __future__ import annotations

import bentoml

from src.rag.retriever import Retriever


@bentoml.service(name="arabic_legal_rag")
class ArabicLegalRAGService:
    """Loads the retriever once per worker process (not per request) --
    same "load once, reuse the persistent index" requirement as the
    FastAPI app in src/api/main.py.
    """

    def __init__(self) -> None:
        self._retriever = Retriever()

    @bentoml.api
    async def ask(self, question: str) -> dict:
        if not question or not question.strip():
            raise ValueError("question must not be empty or whitespace-only")

        results = self._retriever.retrieve(question)
        sources = [
            {
                "article_number": r["article_number"],
                "citation": r["citation"] or "",
                "text": r["text_ar"] or r["text_en"] or "",
            }
            for r in results
        ]
        if not sources:
            answer = "No relevant articles were found for this question."
        else:
            top = sources[0]
            # Retrieval-only, clearly labeled -- never presented as an
            # LLM-generated answer (no generation model is configured).
            answer = (
                "[retrieval-only, no generation model configured] "
                f"The most relevant provision found is {top['citation']}. "
                "See the sources list for the retrieved article text."
            )
        return {"answer": answer, "sources": sources}

    @bentoml.api
    def health(self) -> dict:
        return {"status": "ok"}
