"""Build the vector index for one experiment configuration. Shared by the
experiment runner and the MLflow registration step so both construct the
identical index from the same config.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Dict

from src.rag.embeddings import (
    BACKEND_SEMANTIC,
    BACKEND_TFIDF,
    TfidfFallbackEmbedder,
    get_embedder,
)
from src.rag.vector_store import build_index_from_json

CORPUS_PATH = "data/processed/civil_code.json"


def make_embedder(config: Dict[str, Any]):
    """STRICT: a "sentence-transformers" config either yields the real
    semantic model or raises EmbeddingBackendUnavailable -- it can never
    silently become TF-IDF.
    """
    backend = config.get("embedding_backend", BACKEND_SEMANTIC)
    if backend == BACKEND_TFIDF:
        ngram = config.get("tfidf_ngram_range", (1, 1))
        if isinstance(ngram, str):
            ngram = ast.literal_eval(ngram)
        return TfidfFallbackEmbedder(
            max_features=config.get("tfidf_max_features", 4096), ngram_range=tuple(ngram)
        )
    return get_embedder(config["embedding_model"], backend=BACKEND_SEMANTIC)


def build_config_index(
    config: Dict[str, Any], store_dir: str | Path, corpus_path: str = CORPUS_PATH
) -> Dict[str, Any]:
    return build_index_from_json(
        corpus_path,
        store_dir,
        embedder=make_embedder(config),
        chunk_size=config["chunk_size"],
        overlap=config.get("overlap", 0),
        backend=config.get("vector_backend"),
    )
