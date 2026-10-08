"""Embedding backends for the RAG pipeline.

Primary: a multilingual, Arabic-capable sentence-transformer model,
loaded via the `sentence-transformers` package. Model name is
configurable via the EMBEDDING_MODEL environment variable so the choice
is explicit and reproducible; default is a small, CPU-friendly
multilingual model.

Fallback: THIS DEVELOPMENT SANDBOX has no network access and does not
have `sentence-transformers` installed, so a real transformer model
cannot be downloaded or loaded here. In that case, a deterministic,
offline scikit-learn TF-IDF vectorizer is used instead so the rest of
the pipeline (chunking -> embedding -> vector store -> retrieval) stays
runnable and testable end-to-end without network access.

This is a real, documented limitation, not a substitute presented as
equivalent: TF-IDF captures lexical (word-overlap) similarity only, not
semantic meaning, so retrieval quality is expected to be materially
weaker than with a real sentence-transformer -- especially for Arabic,
where surface word forms vary a lot. See README / PROGRESS.md.
"""
from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

DEFAULT_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

# Backend selection (env EMBEDDING_BACKEND):
#   "sentence-transformers" -- STRICT: real semantic model or an error.
#   "tfidf"                 -- the offline lexical fallback, explicitly.
#   "auto" (default)        -- semantic if it loads, otherwise TF-IDF with a
#                              loud warning (never silent).
BACKEND_SEMANTIC = "sentence-transformers"
BACKEND_TFIDF = "tfidf"
BACKEND_AUTO = "auto"

KIND_SEMANTIC = "semantic"
KIND_LEXICAL = "tfidf-lexical"  # NOT semantic


class EmbeddingBackendUnavailable(RuntimeError):
    """The requested (strict) embedding backend cannot be loaded."""


def default_prefixes(model_name: str) -> Dict[str, str]:
    """E5-family models are trained with "query: " / "passage: " prefixes
    and degrade without them; other models use none.
    """
    if "e5" in model_name.lower().split("/")[-1]:
        return {"query_prefix": "query: ", "passage_prefix": "passage: "}
    return {"query_prefix": "", "passage_prefix": ""}


class SentenceTransformerEmbedder:
    """REAL SEMANTIC embedding backend (kind == "semantic"). Requires the
    `sentence-transformers` package and, on first use, network access (or
    a populated Hugging Face cache) for the model weights.

    Reproducible configuration: model name (EMBEDDING_MODEL), optional
    pinned model revision (EMBEDDING_MODEL_REVISION), query/passage
    prefixes and normalisation are all recorded by `describe()` and
    persisted in the index manifest, so the same index can be reloaded with
    the identical embedder later.
    """

    kind = KIND_SEMANTIC

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None,
        revision: Optional[str] = None,
        normalize: bool = True,
    ):
        try:
            from sentence_transformers import SentenceTransformer  # may raise ImportError
        except ImportError as exc:
            raise EmbeddingBackendUnavailable(
                "sentence-transformers is not installed (pip install -e '.[rag]')"
            ) from exc

        defaults = default_prefixes(model_name)
        self.model_name = model_name
        self.query_prefix = defaults["query_prefix"] if query_prefix is None else query_prefix
        self.passage_prefix = defaults["passage_prefix"] if passage_prefix is None else passage_prefix
        self.revision = revision or os.environ.get("EMBEDDING_MODEL_REVISION") or None
        self.normalize = normalize
        try:
            kwargs = {"revision": self.revision} if self.revision else {}
            self._model = SentenceTransformer(model_name, **kwargs)
        except Exception as exc:  # download failure, bad name, offline w/o cache...
            raise EmbeddingBackendUnavailable(
                f"could not load embedding model {model_name!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def _encode(self, texts: List[str], prefix: str, batch_size: int) -> np.ndarray:
        prefixed = [f"{prefix}{t}" for t in texts]
        return np.asarray(
            self._model.encode(
                prefixed,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=self.normalize,
            ),
            dtype="float32",
        )

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self._encode(list(texts), self.passage_prefix, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text], self.query_prefix, 1)[0]

    # Backwards-compatible alias: documents (the build path).
    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.embed_documents(texts, batch_size)

    def describe(self) -> Dict[str, Any]:
        return {
            "embedder_class": type(self).__name__,
            "embedder_model": self.model_name,
            "embedding_kind": self.kind,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "model_revision": self.revision,
            "normalize_embeddings": self.normalize,
        }

    def save(self, path: str | Path) -> None:
        # Stateless beyond the configuration recorded in the manifest.
        pass

    @classmethod
    def load(cls, path: str | Path, model_name: str) -> "SentenceTransformerEmbedder":
        return cls(model_name)

    @classmethod
    def from_manifest(cls, manifest: Dict[str, Any]) -> "SentenceTransformerEmbedder":
        """Rebuild the exact embedder an index was built with."""
        return cls(
            manifest.get("embedder_model", DEFAULT_MODEL_NAME),
            query_prefix=manifest.get("query_prefix"),
            passage_prefix=manifest.get("passage_prefix"),
            revision=manifest.get("model_revision"),
            normalize=manifest.get("normalize_embeddings", True),
        )


class TfidfFallbackEmbedder:
    """Deterministic, offline, no-download fallback. NOT a semantic
    embedding model (kind == "tfidf-lexical") -- lexical (bag-of-words/
    TF-IDF) similarity only. Used only when a real sentence-transformer
    cannot be loaded or is not requested. The fitted vocabulary must be
    persisted and reused at query time (a query embedded against a
    freshly-refit vectorizer would live in a different, incomparable
    vector space).
    """

    kind = KIND_LEXICAL
    model_name = "tfidf-fallback (offline, lexical only -- not semantic)"

    def __init__(self, max_features: int = 4096, ngram_range: tuple = (1, 1)):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range
        )
        self._fitted = False
        self.max_features = max_features
        self.ngram_range = ngram_range

    def fit(self, texts: List[str]) -> None:
        self._vectorizer.fit(texts)
        self._fitted = True

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        if not self._fitted:
            # Fit-on-first-use (corpus build call) keeps the call signature
            # symmetric with the sentence-transformer embedder.
            self.fit(list(texts))
        matrix = self._vectorizer.transform(list(texts))
        return matrix.toarray().astype("float32")

    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return self.embed(texts, batch_size)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    def describe(self) -> Dict[str, Any]:
        return {
            "embedder_class": type(self).__name__,
            "embedder_model": self.model_name,
            "embedding_kind": self.kind,
            "tfidf_max_features": self.max_features,
            "tfidf_ngram_range": str(self.ngram_range),
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self._vectorizer, f)

    @classmethod
    def load(cls, path: str | Path, model_name: Optional[str] = None) -> "TfidfFallbackEmbedder":
        obj = cls()
        with open(path, "rb") as f:
            obj._vectorizer = pickle.load(f)
        obj._fitted = True
        return obj


def resolve_backend(backend: Optional[str] = None) -> str:
    value = (backend or os.environ.get("EMBEDDING_BACKEND") or BACKEND_AUTO).strip().lower()
    if value not in (BACKEND_SEMANTIC, BACKEND_TFIDF, BACKEND_AUTO):
        raise ValueError(
            f"EMBEDDING_BACKEND must be one of "
            f"{BACKEND_SEMANTIC!r}, {BACKEND_TFIDF!r}, {BACKEND_AUTO!r}; got {value!r}"
        )
    return value


def get_embedder(model_name: Optional[str] = None, backend: Optional[str] = None, **tfidf_kwargs):
    """Return an embedder per `backend` / env EMBEDDING_BACKEND.

    * "sentence-transformers": strict -- raises EmbeddingBackendUnavailable
      rather than ever returning TF-IDF.
    * "tfidf": the lexical fallback (tfidf_kwargs forwarded).
    * "auto" (default): semantic if loadable, else TF-IDF **with a warning**
      so a lexical index is never mistaken for a semantic one; the manifest
      and /health also record which one was used.
    """
    name = model_name or DEFAULT_MODEL_NAME
    choice = resolve_backend(backend)
    if choice == BACKEND_TFIDF:
        return TfidfFallbackEmbedder(**tfidf_kwargs)
    try:
        return SentenceTransformerEmbedder(name)
    except EmbeddingBackendUnavailable as exc:
        if choice == BACKEND_SEMANTIC:
            raise
        warnings.warn(
            f"falling back to the TF-IDF LEXICAL embedder (not semantic): {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return TfidfFallbackEmbedder(**tfidf_kwargs)


def embedder_from_manifest(manifest: Dict[str, Any], tfidf_path: Optional[Path] = None):
    """Recreate the embedder an index was built with, from its manifest."""
    if manifest.get("embedding_kind") == KIND_LEXICAL or manifest.get(
        "embedder_class"
    ) == "TfidfFallbackEmbedder":
        if tfidf_path is None or not Path(tfidf_path).exists():
            raise EmbeddingBackendUnavailable("TF-IDF index is missing its fitted vectorizer file")
        return TfidfFallbackEmbedder.load(tfidf_path)
    return SentenceTransformerEmbedder.from_manifest(manifest)
