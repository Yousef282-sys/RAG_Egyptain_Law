"""Question -> retrieved article chunks, independent of any LLM."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .vector_store import DEFAULT_STORE_DIR, get_vector_store

DEFAULT_TOP_K = 5


class Retriever:
    """Embeds a question and returns the top-k most relevant chunks from
    a previously-built vector store, as clean dicts with article-level
    provenance (never chunk IDs or page numbers as the citation).
    """

    def __init__(
        self,
        store_dir: str | Path = DEFAULT_STORE_DIR,
        top_k: int = DEFAULT_TOP_K,
        store=None,
    ):
        if store is None:
            manifest_path = Path(store_dir) / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"Vector index manifest not found at {manifest_path}. Build the index first."
                )
            self.store = get_vector_store(store_dir)
        else:
            self.store = store
        self.embedder = self.store.load_embedder()
        self.top_k = top_k

    def close(self) -> None:
        """Release resources owned by the underlying vector store."""
        close_store = getattr(self.store, "close", None)
        if callable(close_store):
            close_store()

    def retrieve(self, question: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if not question or not question.strip():
            return []
        k = top_k if top_k is not None else self.top_k
        embed_query = getattr(self.embedder, "embed_query", None)
        query_vector = embed_query(question) if embed_query else self.embedder.embed([question])[0]
        raw_results = self.store.search(query_vector, top_k=k)
        return [
            {
                "article_number": r["article_number"],
                "text_ar": r.get("text_ar"),
                "text_en": r.get("text_en"),
                "citation": r.get("citation"),
                "book": r.get("book"),
                "chapter": r.get("chapter"),
                "section": r.get("section"),
                "topic": r.get("topic"),
                "source_page": r.get("source_page"),
                "is_repealed": r.get("is_repealed"),
                "score": r.get("score"),
            }
            for r in raw_results
        ]
