"""Local vector store for the RAG pipeline.

Primary backend: Chroma (persistent, local, on-disk), used if the
`chromadb` package is available.

Fallback backend: THIS DEVELOPMENT SANDBOX does not have `chromadb`
installed and has no network access to install it, so a simple
numpy-based store (cosine similarity over an in-memory matrix, persisted
as .npy + JSON metadata) is used instead. Same on-disk directory either
way (`data/vector_store/` by default), so `retriever.py` and callers do
not need to know which backend actually built the index -- a manifest
file records which one was used.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import Chunk, chunk_articles, load_articles
from .embeddings import (
    DEFAULT_MODEL_NAME,
    SentenceTransformerEmbedder,
    TfidfFallbackEmbedder,
    get_embedder,
)

DEFAULT_STORE_DIR = Path("data/vector_store")

_VECTORS_FILE = "vectors.npy"
_METADATA_FILE = "metadata.json"
_MANIFEST_FILE = "manifest.json"
_TFIDF_FILE = "tfidf_vectorizer.pkl"


def _chunk_text_for_embedding(chunk: Chunk) -> str:
    # Arabic is the primary language of the corpus; fall back to English
    # only for the one known Day-1 case where text_ar is empty (art. 1022).
    return chunk.text_ar or chunk.text_en or ""


class SimpleVectorStore:
    """numpy-backed cosine-similarity store (the fallback actually used
    in this sandbox, since chromadb is unavailable).
    """

    def __init__(self, directory: str | Path = DEFAULT_STORE_DIR):
        self.directory = Path(directory)

    def build(self, chunks: List[Chunk], embedder=None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        embedder = embedder or get_embedder()
        texts = [_chunk_text_for_embedding(c) for c in chunks]

        vectors = np.asarray(embedder.embed(texts), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        np.save(self.directory / _VECTORS_FILE, vectors)

        metadata = [c.to_dict() for c in chunks]
        with open(self.directory / _METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        manifest = {
            "backend": "simple-numpy",
            "embedder_model": getattr(embedder, "model_name", "unknown"),
            "embedder_class": type(embedder).__name__,
            "chunk_count": len(chunks),
        }
        with open(self.directory / _MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        if isinstance(embedder, TfidfFallbackEmbedder):
            embedder.save(self.directory / _TFIDF_FILE)

    def load_embedder(self):
        with open(self.directory / _MANIFEST_FILE, encoding="utf-8") as f:
            manifest = json.load(f)
        tfidf_path = self.directory / _TFIDF_FILE
        if tfidf_path.exists():
            return TfidfFallbackEmbedder.load(tfidf_path)
        return SentenceTransformerEmbedder(manifest.get("embedder_model", DEFAULT_MODEL_NAME))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        vectors = np.load(self.directory / _VECTORS_FILE)
        with open(self.directory / _METADATA_FILE, encoding="utf-8") as f:
            metadata = json.load(f)

        qv = np.asarray(query_vector, dtype="float32").reshape(-1)
        qnorm = np.linalg.norm(qv)
        if qnorm > 0:
            qv = qv / qnorm

        scores = vectors @ qv
        top_k = min(top_k, len(metadata))
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for idx in top_idx:
            item = dict(metadata[int(idx)])
            item["score"] = float(scores[int(idx)])
            results.append(item)
        return results


class ChromaVectorStore:
    """Chroma-backed persistent store. Requires the `chromadb` package,
    which is NOT installed in this development sandbox (no network here
    to install it), so this class is untested in this sandbox -- it is
    written to the same interface as SimpleVectorStore and should work
    as-is once chromadb is available (see README).
    """

    def __init__(self, directory: str | Path = DEFAULT_STORE_DIR):
        import chromadb  # raises ImportError if unavailable

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.directory))
        self._collection = self._client.get_or_create_collection("civil_code")

    def build(self, chunks: List[Chunk], embedder=None) -> None:
        embedder = embedder or get_embedder()
        texts = [_chunk_text_for_embedding(c) for c in chunks]
        vectors = np.asarray(embedder.embed(texts), dtype="float32")
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[c.to_dict() for c in chunks],
        )
        manifest = {
            "backend": "chromadb",
            "embedder_model": getattr(embedder, "model_name", "unknown"),
            "embedder_class": type(embedder).__name__,
            "chunk_count": len(chunks),
        }
        with open(self.directory / _MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        if isinstance(embedder, TfidfFallbackEmbedder):
            embedder.save(self.directory / _TFIDF_FILE)

    def load_embedder(self):
        with open(self.directory / _MANIFEST_FILE, encoding="utf-8") as f:
            manifest = json.load(f)
        tfidf_path = self.directory / _TFIDF_FILE
        if tfidf_path.exists():
            return TfidfFallbackEmbedder.load(tfidf_path)
        return SentenceTransformerEmbedder(manifest.get("embedder_model", DEFAULT_MODEL_NAME))

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        raw = self._collection.query(
            query_embeddings=[np.asarray(query_vector, dtype="float32").tolist()],
            n_results=top_k,
        )
        results = []
        for metadata, distance in zip(raw["metadatas"][0], raw["distances"][0]):
            item = dict(metadata)
            item["score"] = float(distance)
            results.append(item)
        return results


def get_vector_store(directory: str | Path = DEFAULT_STORE_DIR):
    """Return a Chroma-backed store if `chromadb` is installed, otherwise
    the numpy fallback actually used in this sandbox.
    """
    try:
        return ChromaVectorStore(directory)
    except ImportError:
        return SimpleVectorStore(directory)


def build_index_from_json(
    json_path: str | Path,
    store_dir: str | Path = DEFAULT_STORE_DIR,
    embedder=None,
) -> Dict[str, Any]:
    """Convenience entry point: JSON articles -> chunks -> embeddings ->
    persisted vector store. Returns the manifest dict for logging.
    """
    articles = load_articles(json_path)
    chunks = chunk_articles(articles)
    store = SimpleVectorStore(store_dir)
    store.build(chunks, embedder=embedder)
    with open(Path(store_dir) / _MANIFEST_FILE, encoding="utf-8") as f:
        return json.load(f)
