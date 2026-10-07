"""Single implementation of "question -> cited answer", shared by the
FastAPI app (src/api/main.py) and the BentoML service
(src/serving/bento_service.py) so the two can never drift apart.

Dependency-free (no fastapi/bentoml/pydantic) so it is unit-testable
anywhere. Sources are always keyed by legal ARTICLE NUMBER.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .generation import Generator, get_generator


def to_sources(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "article_number": r["article_number"],
            "citation": r.get("citation") or "",
            "text": r.get("text_ar") or r.get("text_en") or "",
        }
        for r in results
    ]


def index_info(store_dir: str | Path) -> Dict[str, Any]:
    """Index metadata read from the manifest only -- does not load the
    embedder or vectors, so it is cheap enough for /health.
    """
    manifest_path = Path(store_dir) / "manifest.json"
    if not manifest_path.exists():
        return {"index_available": False}
    try:
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
    except (OSError, ValueError):
        return {"index_available": False, "index_error": "unreadable manifest"}
    embedder_class = m.get("embedder_class", "unknown")
    return {
        "index_available": True,
        "vector_backend": m.get("backend"),
        "chunk_count": m.get("chunk_count"),
        "embedder_model": m.get("embedder_model"),
        # False => lexical TF-IDF fallback index, NOT semantic embeddings.
        "semantic_embeddings": embedder_class != "TfidfFallbackEmbedder",
    }


class RagService:
    def __init__(self, retriever, generator: Optional[Generator] = None):
        self.retriever = retriever
        self.generator = generator or get_generator()

    def retrieve_sources(self, question: str) -> List[Dict[str, Any]]:
        return to_sources(self.retriever.retrieve(question))

    def ask(self, question: str) -> Dict[str, Any]:
        sources = self.retrieve_sources(question)
        answer = self.generator.generate(question, sources)
        return {"answer": answer, "sources": sources}

    def ask_stream(self, question: str) -> Iterator[Dict[str, Any]]:
        """Yield events: one {"type": "sources"}, then {"type": "token"}
        events as the backend produces text, then {"type": "done"}.
        With the retrieval-only generator the "token" is one complete,
        labelled message (it is not model streaming).
        """
        sources = self.retrieve_sources(question)
        yield {"type": "sources", "sources": sources}
        for piece in self.generator.stream(question, sources):
            yield {"type": "token", "text": piece}
        yield {"type": "done"}

    def health(self, store_dir: str | Path) -> Dict[str, Any]:
        return {
            "status": "ok",
            "generator": self.generator.name,
            "generation_is_llm": self.generator.is_llm,
            **index_info(store_dir),
        }
