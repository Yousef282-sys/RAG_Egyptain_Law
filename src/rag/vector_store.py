"""Persistent vector store for the RAG pipeline.

Backends (same on-disk directory, `data/vector_store/` by default; a
`manifest.json` records which backend and embedder built the index):

* ChromaVectorStore -- the vector database (persistent, on-disk), used when
  `chromadb` is installed.
* SimpleVectorStore -- numpy cosine-similarity fallback (.npy + JSON), used
  when chromadb is unavailable (e.g. the offline development sandbox).

Loading is driven by the MANIFEST, not by "whatever is installed": an index
built by one backend is always reopened with that backend (Phase 1 built
with numpy but loaded with Chroma whenever chromadb was installed -- the
two never matched in a real environment). Set VECTOR_STORE_BACKEND
(`auto` | `chroma` | `simple`) to choose the backend when BUILDING.

Every chunk's metadata is preserved end to end: article_number, citation,
book, chapter, section, topic, source_page, is_repealed (plus chunk ids/
indices and both text fields), so article-based citations stay intact.
`manifest.json` additionally records the embedder configuration
(`embedding_kind` is "semantic" or "tfidf-lexical" -- never conflated) and
the chunking parameters / corpus hash the index was built from.

The Chroma code path has NOT been executed against a real chromadb in the
development sandbox (not installed, no network); it is exercised only
against an in-memory stand-in in tests/test_vector_store_backends.py.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .chunking import MAX_CHUNK_CHARS, Chunk, chunk_articles, load_articles
from .embeddings import (
    TfidfFallbackEmbedder,
    embedder_from_manifest,
    get_embedder,
)

DEFAULT_STORE_DIR = Path("data/vector_store")

_VECTORS_FILE = "vectors.npy"
_METADATA_FILE = "metadata.json"
_MANIFEST_FILE = "manifest.json"
_TFIDF_FILE = "tfidf_vectorizer.pkl"

BACKEND_CHROMA = "chromadb"
BACKEND_SIMPLE = "simple-numpy"

# Every field of a Chunk is stored and returned on search.
CHUNK_FIELDS = [f.name for f in fields(Chunk)]


def _chunk_text_for_embedding(chunk: Chunk) -> str:
    # Full article text (text_ar + text_en): the corpus fields are
    # interleaved, not language-pure -- see chunking.full_text.
    return chunk.full_text()

def _make_manifest(
    backend: str, embedder, chunks: List[Chunk], build_info: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    describe = embedder.describe() if hasattr(embedder, "describe") else {
        "embedder_class": type(embedder).__name__,
        "embedder_model": getattr(embedder, "model_name", "unknown"),
        "embedding_kind": getattr(embedder, "kind", "unknown"),
    }
    manifest: Dict[str, Any] = {
        "backend": backend,
        **describe,
        "chunk_count": len(chunks),
        "article_count": len({c.article_number for c in chunks}),
    }
    if build_info:
        manifest.update(build_info)
    return manifest


def _write_manifest(directory: Path, manifest: Dict[str, Any]) -> None:
    with open(directory / _MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def read_manifest(directory: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(directory) / _MANIFEST_FILE
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class SimpleVectorStore:
    """numpy-backed cosine-similarity store (fallback when chromadb is
    unavailable).
    """

    def __init__(self, directory: str | Path = DEFAULT_STORE_DIR):
        self.directory = Path(directory)

    def build(self, chunks: List[Chunk], embedder=None, build_info=None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        embedder = embedder or get_embedder()
        texts = [_chunk_text_for_embedding(c) for c in chunks]

        embed_documents = getattr(embedder, "embed_documents", embedder.embed)
        vectors = np.asarray(embed_documents(texts), dtype="float32")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
        np.save(self.directory / _VECTORS_FILE, vectors)

        metadata = [c.to_dict() for c in chunks]
        with open(self.directory / _METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False)

        _write_manifest(
            self.directory, _make_manifest(BACKEND_SIMPLE, embedder, chunks, build_info)
        )
        if isinstance(embedder, TfidfFallbackEmbedder):
            embedder.save(self.directory / _TFIDF_FILE)

    def load_embedder(self):
        return embedder_from_manifest(
            read_manifest(self.directory) or {}, self.directory / _TFIDF_FILE
        )

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
    """Chroma-backed persistent vector database. Requires `chromadb`.

    * cosine space, so `score` is cosine similarity (1 - distance), the same
      meaning and direction as SimpleVectorStore's score;
    * `build` replaces the collection, so rebuilding with different
      chunking never leaves stale chunks behind;
    * Chroma metadata must be flat scalars without None, so None-valued
      fields are dropped on write and restored as None on read.
    """

    COLLECTION = "civil_code"

    def __init__(self, directory: str | Path = DEFAULT_STORE_DIR):
        import chromadb  # raises ImportError if unavailable

        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self.directory))

    def close(self) -> None:
        """Release the persistent Chroma client and its file handles."""
        client = getattr(self, "_client", None)
        if client is not None:
            self._client = None
            client.close()

    def _collection(self, create: bool = False):
        if create:
            try:
                self._client.delete_collection(self.COLLECTION)
            except Exception:  # collection did not exist yet
                pass
            return self._client.create_collection(
                self.COLLECTION, metadata={"hnsw:space": "cosine"}
            )
        return self._client.get_collection(self.COLLECTION)

    def build(self, chunks: List[Chunk], embedder=None, build_info=None) -> None:
        embedder = embedder or get_embedder()
        texts = [_chunk_text_for_embedding(c) for c in chunks]
        embed_documents = getattr(embedder, "embed_documents", embedder.embed)
        vectors = np.asarray(embed_documents(texts), dtype="float32")
        collection = self._collection(create=True)
        batch = 500
        for start in range(0, len(chunks), batch):
            part = chunks[start : start + batch]
            collection.upsert(
                ids=[f"{i}-{c.chunk_id}" for i, c in enumerate(part, start=start)],
                embeddings=vectors[start : start + batch].tolist(),
                documents=texts[start : start + batch],
                metadatas=[
                    {k: v for k, v in c.to_dict().items() if v is not None} for c in part
                ],
            )
        _write_manifest(
            self.directory, _make_manifest(BACKEND_CHROMA, embedder, chunks, build_info)
        )
        if isinstance(embedder, TfidfFallbackEmbedder):
            embedder.save(self.directory / _TFIDF_FILE)

    def load_embedder(self):
        return embedder_from_manifest(
            read_manifest(self.directory) or {}, self.directory / _TFIDF_FILE
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        collection = self._collection()
        n = collection.count()
        if n == 0:
            return []
        raw = collection.query(
            query_embeddings=[np.asarray(query_vector, dtype="float32").tolist()],
            n_results=min(top_k, n),
        )
        results = []
        for metadata, distance in zip(raw["metadatas"][0], raw["distances"][0]):
            item = {name: metadata.get(name) for name in CHUNK_FIELDS}
            item["score"] = 1.0 - float(distance)  # cosine distance -> similarity
            results.append(item)
        return results


def resolve_vector_backend(backend: Optional[str] = None) -> str:
    value = (backend or os.environ.get("VECTOR_STORE_BACKEND") or "auto").strip().lower()
    if value not in ("auto", "chroma", "simple"):
        raise ValueError(f"VECTOR_STORE_BACKEND must be auto|chroma|simple, got {value!r}")
    return value


def get_vector_store(directory: str | Path = DEFAULT_STORE_DIR, backend: Optional[str] = None):
    """Open (or, with no manifest yet, choose a backend for) a store.

    An existing index is ALWAYS reopened with the backend recorded in its
    manifest. Otherwise: "chroma" is strict (ImportError if missing),
    "simple" uses numpy, "auto" prefers Chroma when installed.
    """
    manifest = read_manifest(directory)
    if manifest is not None and manifest.get("backend") == BACKEND_CHROMA:
        return ChromaVectorStore(directory)  # ImportError if chromadb missing: honest
    if manifest is not None and manifest.get("backend") == BACKEND_SIMPLE:
        return SimpleVectorStore(directory)

    choice = resolve_vector_backend(backend)
    if choice == "simple":
        return SimpleVectorStore(directory)
    if choice == "chroma":
        return ChromaVectorStore(directory)
    try:
        return ChromaVectorStore(directory)
    except ImportError:
        return SimpleVectorStore(directory)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_index_from_json(
    json_path: str | Path,
    store_dir: str | Path = DEFAULT_STORE_DIR,
    embedder=None,
    chunk_size: int = MAX_CHUNK_CHARS,
    overlap: int = 0,
    backend: Optional[str] = None,
) -> Dict[str, Any]:
    """JSON articles -> chunks -> embeddings -> persisted vector store.
    Returns the manifest dict (embedder config, chunking params, corpus
    sha256) for logging / reproducibility.
    """
    articles = load_articles(json_path)
    chunks = chunk_articles(articles, max_chars=chunk_size, overlap=overlap)
    store = get_vector_store(store_dir, backend=backend)
    try:
        store.build(
            chunks,
            embedder=embedder,
            build_info={
                "chunk_size": chunk_size,
                "overlap": overlap,
                "corpus_path": str(json_path),
                "corpus_sha256": file_sha256(json_path),
            },
        )
    finally:
        close = getattr(store, "close", None)
        if callable(close):
            close()
        import gc
        del store
        gc.collect()
    return read_manifest(store_dir)





