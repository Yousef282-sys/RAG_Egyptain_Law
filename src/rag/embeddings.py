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
from pathlib import Path
from typing import List, Optional

import numpy as np

DEFAULT_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)


class SentenceTransformerEmbedder:
    """Real embedding backend. Requires the `sentence-transformers`
    package and (for most models, on first use) network access to
    download the model weights.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        from sentence_transformers import SentenceTransformer  # may raise ImportError

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts), batch_size=batch_size, show_progress_bar=False
            ),
            dtype="float32",
        )

    def save(self, path: str | Path) -> None:
        # Stateless beyond the model name; nothing to persist.
        pass

    @classmethod
    def load(cls, path: str | Path, model_name: str) -> "SentenceTransformerEmbedder":
        return cls(model_name)


class TfidfFallbackEmbedder:
    """Deterministic, offline, no-download fallback. NOT a semantic
    embedding model -- lexical (bag-of-words/TF-IDF) similarity only.
    Used only when a real sentence-transformer cannot be loaded (e.g. in
    this sandbox). The fitted vocabulary must be persisted and reused at
    query time (a query embedded against a freshly-refit vectorizer would
    live in a different, incomparable vector space).
    """

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


def get_embedder(model_name: Optional[str] = None):
    """Return the best available embedder: a real sentence-transformer if
    the package and model can be loaded, otherwise the TF-IDF fallback.
    Never silently retries network calls beyond what the underlying
    library does on its own.
    """
    name = model_name or DEFAULT_MODEL_NAME
    try:
        return SentenceTransformerEmbedder(name)
    except Exception:
        return TfidfFallbackEmbedder()
