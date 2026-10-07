"""Reproducible RAG experiment runner.

For each configuration: builds real chunks from the real corpus with that
chunk_size, builds a real (persisted) vector store with that embedder
config, evaluates retrieval against the fixed eval set
(src/experiments/eval_set.py), and records the exact configuration +
measured metrics. Logs each run to MLflow under EXPERIMENT_NAME if
`mlflow` is installed; otherwise falls back to local-only recording
(same config/metrics, clearly labeled `mlflow_logged: false`) rather than
faking a run. Every result written to reports/day4_experiment_results.json
is something that was actually computed in this process, never invented.

Every config in this sandbox necessarily uses the TF-IDF fallback
embedder (no `sentence-transformers`/network here) -- each result
explicitly records `embedder_class` and `embedder_model` so this is never
misrepresented as a semantic-model result.
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List

from src.experiments.eval_set import load_or_build_eval_set
from src.experiments.evaluate import evaluate_retriever
from src.experiments.tracking import MLflowUnavailable, log_experiment_run
from src.rag.chunking import chunk_articles, load_articles
from src.rag.embeddings import TfidfFallbackEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import SimpleVectorStore

CORPUS_PATH = "data/processed/civil_code.json"
RESULTS_PATH = Path("reports/day4_experiment_results.json")

# 5 genuinely different configurations. All vary something real:
# chunk_size (article-splitting threshold, src/rag/chunking.py),
# top_k (retrieval depth), and the TF-IDF backend's own configuration
# (max_features, ngram_range) -- the one embedding/backend axis this
# sandbox can actually vary without network access.
CONFIGURATIONS: List[Dict[str, Any]] = [
    {
        "run_name": "baseline",
        "chunk_size": 1000,  # same default as Day 2
        "top_k": 5,
        "tfidf_max_features": 4096,
        "tfidf_ngram_range": (1, 1),
    },
    {
        "run_name": "smaller_chunks",
        "chunk_size": 500,
        "top_k": 5,
        "tfidf_max_features": 4096,
        "tfidf_ngram_range": (1, 1),
    },
    {
        "run_name": "larger_chunks",
        "chunk_size": 2000,
        "top_k": 5,
        "tfidf_max_features": 4096,
        "tfidf_ngram_range": (1, 1),
    },
    {
        "run_name": "smaller_top_k",
        "chunk_size": 1000,
        "top_k": 3,
        "tfidf_max_features": 4096,
        "tfidf_ngram_range": (1, 1),
    },
    {
        "run_name": "bigram_tfidf",
        "chunk_size": 1000,
        "top_k": 5,
        "tfidf_max_features": 4096,
        "tfidf_ngram_range": (1, 2),
    },
]


def run_one_configuration(config: Dict[str, Any], eval_set: List[Dict[str, Any]]) -> Dict[str, Any]:
    articles = load_articles(CORPUS_PATH)
    chunks = chunk_articles(articles, max_chars=config["chunk_size"])

    embedder = TfidfFallbackEmbedder(
        max_features=config["tfidf_max_features"],
        ngram_range=config["tfidf_ngram_range"],
    )

    with tempfile.TemporaryDirectory() as store_dir:
        store = SimpleVectorStore(store_dir)
        t0 = time.time()
        store.build(chunks, embedder=embedder)
        build_seconds = time.time() - t0

        retriever = Retriever(store_dir=store_dir, top_k=config["top_k"], store=store)
        t0 = time.time()
        metrics = evaluate_retriever(retriever, eval_set, top_k=config["top_k"])
        eval_seconds = time.time() - t0

        manifest_path = Path(store_dir) / "manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    metrics["chunk_count"] = len(chunks)
    metrics["build_seconds"] = round(build_seconds, 3)
    metrics["eval_seconds"] = round(eval_seconds, 3)

    full_config = dict(config)
    full_config["tfidf_ngram_range"] = str(config["tfidf_ngram_range"])  # MLflow params must be scalar
    full_config["embedder_class"] = manifest["embedder_class"]
    full_config["embedder_model"] = manifest["embedder_model"]
    full_config["backend"] = manifest["backend"]

    mlflow_run_id = None
    mlflow_logged = False
    try:
        mlflow_run_id = log_experiment_run(
            config=full_config, metrics=metrics, run_name=config["run_name"]
        )
        mlflow_logged = True
    except MLflowUnavailable as exc:
        mlflow_run_id = None
        mlflow_logged = False
        print(f"  [mlflow] not logged for '{config['run_name']}': {exc}")

    return {
        "config": full_config,
        "metrics": metrics,
        "mlflow_logged": mlflow_logged,
        "mlflow_run_id": mlflow_run_id,
    }


def run_all(configurations: List[Dict[str, Any]] = CONFIGURATIONS) -> List[Dict[str, Any]]:
    eval_set = load_or_build_eval_set(CORPUS_PATH)
    results = []
    for config in configurations:
        print(f"Running configuration: {config['run_name']} ...")
        result = run_one_configuration(config, eval_set)
        results.append(result)
        print(f"  metrics: {result['metrics']}")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "eval_set_size": len(eval_set),
                "eval_set_path": "data/eval/retrieval_eval.json",
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return results


if __name__ == "__main__":
    all_results = run_all()

    from src.experiments.tracking import MLflowUnavailable, register_best_run_artifact

    try:
        registry_result = register_best_run_artifact(all_results)
        print("Registry result:", registry_result)
    except MLflowUnavailable as exc:
        print(f"[mlflow] registry step skipped: {exc}")
