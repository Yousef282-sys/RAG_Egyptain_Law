"""End-to-end integration test for the real, non-API part of the
pipeline: corpus JSON -> chunking -> vector store -> retrieval ->
observability logging -> monitoring summary.

No mocks anywhere in this file -- every component genuinely runs
against the real corpus (data/processed/civil_code.json). The only
piece NOT exercised here is the FastAPI HTTP layer itself (fastapi is
unavailable in this sandbox; see tests/test_api.py for those,
honestly skipped there).
"""
import tempfile
import time
from pathlib import Path

from src.observability.logger import build_event, log_event, now_iso
from src.observability.monitoring import load_events, summarize_retrieval
from src.rag.chunking import chunk_articles, load_articles
from src.rag.embeddings import TfidfFallbackEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import SimpleVectorStore

CORPUS_PATH = "data/processed/civil_code.json"


def test_full_pipeline_runs_end_to_end_against_the_real_corpus():
    articles = load_articles(CORPUS_PATH)
    assert len(articles) == 1095  # known-good baseline, re-asserted here

    chunks = chunk_articles(articles)
    assert len(chunks) >= len(articles)  # chunking never drops an article

    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(d)
        store.build(chunks, embedder=TfidfFallbackEmbedder(max_features=4096))
        retriever = Retriever(store_dir=d, top_k=3, store=store)

        log_file = Path(d) / "requests.jsonl"
        questions = ["ما هو عقد البيع؟", "متى ينتهي عقد الايجار؟"]
        for q in questions:
            start = time.perf_counter()
            results = retriever.retrieve(q)
            latency = round((time.perf_counter() - start) * 1000, 2)
            assert len(results) > 0
            for r in results:
                assert "article_number" in r
                assert "citation" in r
            log_event(
                build_event(
                    request_id="integration-test",
                    timestamp=now_iso(),
                    route="/ask",
                    latency_ms=latency,
                    status="ok",
                    result_count=len(results),
                    question_length=len(q),
                ),
                log_file=log_file,
            )

        events = load_events(log_file)
        assert len(events) == len(questions)
        summary = summarize_retrieval(events)
        assert summary["n_requests"] == len(questions)
        assert summary["error_count"] == 0


def test_empty_question_through_the_full_chain_returns_no_results_not_an_error():
    articles = load_articles(CORPUS_PATH)
    chunks = chunk_articles(articles)
    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(d)
        store.build(chunks, embedder=TfidfFallbackEmbedder(max_features=1024))
        retriever = Retriever(store_dir=d, top_k=3, store=store)
        results = retriever.retrieve("")
        assert results == []


def test_dockerfile_copies_every_src_subpackage_via_wildcard():
    # Day 9 audit: the Dockerfile must not enumerate src/ subpackages
    # individually (which would silently miss a new one like
    # src/observability/) -- it must copy the whole src/ tree.
    content = Path("Dockerfile").read_text(encoding="utf-8")
    assert "COPY src ./src" in content


def test_bentofile_includes_all_python_files_via_wildcard():
    content = Path("bentofile.yaml").read_text(encoding="utf-8")
    assert "src/**/*.py" in content


def test_dvc_yaml_paths_all_exist_after_day7_day8_rebuild():
    import yaml

    config = yaml.safe_load(Path("dvc.yaml").read_text(encoding="utf-8"))
    stage = config["stages"]["extract_corpus"]
    for path in stage["deps"] + stage["outs"]:
        assert Path(path).exists(), f"dvc.yaml references missing path: {path}"
