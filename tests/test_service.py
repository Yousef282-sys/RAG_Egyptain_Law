"""Tests for src/rag/service.py (shared by FastAPI and BentoML) using a
fake retriever and fake generators -- no model, no network, no fastapi."""
import json
import tempfile
from pathlib import Path

from src.rag.generation import GenerationError, RetrievalOnlyGenerator
from src.rag.service import RagService, index_info, to_sources


class FakeRetriever:
    def retrieve(self, question, top_k=None):
        return [
            {"article_number": 147, "text_ar": "نص عربي", "text_en": "en", "citation": "المادة 147", "score": 0.9},
            {"article_number": 1022, "text_ar": "", "text_en": "english only", "citation": None, "score": 0.5},
        ]


class EmptyRetriever:
    def retrieve(self, question, top_k=None):
        return []


class TokenGen:
    name, is_llm = "fake", True

    def generate(self, q, s):
        return "full"

    def stream(self, q, s):
        yield "a"
        yield "b"


class FailingStreamGen(TokenGen):
    def stream(self, q, s):
        yield "a"
        raise GenerationError("down")


def test_to_sources_uses_article_numbers_and_text_fallback():
    s = to_sources(FakeRetriever().retrieve("q"))
    assert [x["article_number"] for x in s] == [147, 1022]
    assert s[0]["text"] == "نص عربي" and s[1]["text"] == "english only"
    assert s[1]["citation"] == ""
    assert all(set(x) == {"article_number", "citation", "text"} for x in s)  # no chunk ids/pages


def test_ask_returns_answer_and_sources():
    out = RagService(FakeRetriever(), TokenGen()).ask("q")
    assert out["answer"] == "full" and len(out["sources"]) == 2


def test_ask_with_no_results_is_handled():
    out = RagService(EmptyRetriever(), RetrievalOnlyGenerator()).ask("q")
    assert out["sources"] == [] and "No relevant" in out["answer"]


def test_ask_stream_event_order():
    ev = list(RagService(FakeRetriever(), TokenGen()).ask_stream("q"))
    assert [e["type"] for e in ev] == ["sources", "token", "token", "done"]
    assert ev[0]["sources"][0]["article_number"] == 147
    assert "".join(e["text"] for e in ev if e["type"] == "token") == "ab"


def test_ask_stream_propagates_generation_error_after_partial_output():
    it = RagService(FakeRetriever(), FailingStreamGen()).ask_stream("q")
    seen = [next(it), next(it)]
    assert seen[1]["text"] == "a"
    try:
        next(it)
    except GenerationError:
        return
    raise AssertionError("expected GenerationError")


def test_index_info_missing_and_present_and_tfidf_flag():
    with tempfile.TemporaryDirectory() as d:
        assert index_info(d) == {"index_available": False}
        manifest = {"backend": "simple-numpy", "chunk_count": 5,
                    "embedder_model": "m", "embedder_class": "TfidfFallbackEmbedder"}
        Path(d, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        info = index_info(d)
        assert info["index_available"] and info["chunk_count"] == 5
        assert info["semantic_embeddings"] is False
        manifest["embedder_class"] = "SentenceTransformerEmbedder"
        Path(d, "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        assert index_info(d)["semantic_embeddings"] is True
        Path(d, "manifest.json").write_text("{bad", encoding="utf-8")
        assert index_info(d)["index_available"] is False


def test_health_reports_generator_mode():
    with tempfile.TemporaryDirectory() as d:
        h = RagService(FakeRetriever(), RetrievalOnlyGenerator()).health(d)
        assert h["status"] == "ok" and h["generation_is_llm"] is False
        assert h["generator"] == "retrieval-only" and h["index_available"] is False
