"""Index build/load with metadata preservation, for both backends.

ChromaVectorStore is exercised against an in-memory chromadb STAND-IN
(tests/fakes.py, which enforces Chroma's no-None-metadata rule) -- this
verifies our wiring (cosine space, metadata round trip, rebuild, backend
choice from the manifest), NOT the real chromadb.
"""
import json
import os
import tempfile
from pathlib import Path

from src.rag.chunking import chunk_articles
from src.rag.embeddings import SentenceTransformerEmbedder, TfidfFallbackEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import (
    CHUNK_FIELDS,
    ChromaVectorStore,
    SimpleVectorStore,
    build_index_from_json,
    get_vector_store,
    read_manifest,
)
from tests.fakes import fake_chromadb, fake_sentence_transformers, installed

ARTICLES = [
    dict(article_number=7, book="BOOK I", chapter="CH 1", section="SEC 2", topic=None,
         text_ar="عقد البيع ينعقد بمجرد اتفاق الطرفين", text_en="A sale is concluded by agreement",
         is_repealed=False, source_page=4, citation="BOOK I - المادة 7"),
    dict(article_number=8, book=None, chapter=None, section=None, topic=None,
         text_ar="الايجار عقد يمكن المستأجر من الانتفاع", text_en="A lease allows use of a thing",
         is_repealed=True, source_page=5, citation="المادة 8"),
    dict(article_number=9, book=None, chapter=None, section=None, topic="sports",
         text_ar="قواعد كرة القدم", text_en="Football rules", is_repealed=False,
         source_page=6, citation="المادة 9"),
]


def _tfidf():
    return TfidfFallbackEmbedder(max_features=64)


def _assert_metadata_preserved(results, expect_article):
    top = results[0]
    assert top["article_number"] == expect_article
    for field in ("article_number", "citation", "book", "chapter", "section", "topic",
                  "source_page", "is_repealed"):
        assert field in top, field
    assert set(CHUNK_FIELDS) <= set(top)


def test_simple_store_preserves_all_metadata_and_article_citation():
    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(d)
        store.build(chunk_articles(ARTICLES), embedder=_tfidf())
        emb = store.load_embedder()
        results = store.search(emb.embed_query("عقد البيع"), top_k=3)
    _assert_metadata_preserved(results, 7)
    assert results[0]["citation"] == "BOOK I - المادة 7" and results[0]["book"] == "BOOK I"
    assert results[0]["source_page"] == 4 and results[0]["is_repealed"] is False


def test_chroma_store_round_trips_metadata_including_none_fields():
    with installed(fake_chromadb()):
        with tempfile.TemporaryDirectory() as d:
            store = ChromaVectorStore(d)
            store.build(chunk_articles(ARTICLES), embedder=_tfidf())  # None metadata must not crash
            emb = store.load_embedder()
            results = store.search(emb.embed_query("عقد البيع"), top_k=3)
    _assert_metadata_preserved(results, 7)
    by_article = {r["article_number"]: r for r in results}
    assert by_article[7]["topic"] is None and by_article[7]["book"] == "BOOK I"  # None restored
    assert by_article[8]["is_repealed"] is True and by_article[9]["topic"] == "sports"
    assert all(-1.0001 <= r["score"] <= 1.0001 for r in results)  # cosine similarity, not distance
    assert results[0]["score"] >= results[-1]["score"]  # best first


def test_chroma_rebuild_replaces_collection_no_stale_chunks():
    with installed(fake_chromadb()):
        with tempfile.TemporaryDirectory() as d:
            store = ChromaVectorStore(d)
            store.build(chunk_articles(ARTICLES), embedder=_tfidf())
            store.build(chunk_articles(ARTICLES[:1]), embedder=_tfidf())
            assert store._collection().count() == 1


def test_chroma_search_handles_top_k_larger_than_collection():
    with installed(fake_chromadb()):
        with tempfile.TemporaryDirectory() as d:
            store = ChromaVectorStore(d)
            store.build(chunk_articles(ARTICLES), embedder=_tfidf())
            assert len(store.search(store.load_embedder().embed_query("عقد"), top_k=50)) == 3


def test_index_is_reopened_with_the_backend_recorded_in_its_manifest():
    # Phase-1 bug: built with numpy, but reopened with Chroma whenever
    # chromadb was installed -> empty results. The manifest must win.
    with installed(fake_chromadb()):
        with tempfile.TemporaryDirectory() as d:
            SimpleVectorStore(d).build(chunk_articles(ARTICLES), embedder=_tfidf())
            assert isinstance(get_vector_store(d), SimpleVectorStore)
            assert Retriever(store_dir=d, top_k=1).retrieve("عقد البيع")[0]["article_number"] == 7
        with tempfile.TemporaryDirectory() as d:
            ChromaVectorStore(d).build(chunk_articles(ARTICLES), embedder=_tfidf())
            assert isinstance(get_vector_store(d), ChromaVectorStore)
            assert Retriever(store_dir=d, top_k=1).retrieve("عقد البيع")[0]["article_number"] == 7


def test_new_index_prefers_chroma_when_installed_and_simple_when_not_or_requested():
    with installed(fake_chromadb()):
        with tempfile.TemporaryDirectory() as d:
            assert isinstance(get_vector_store(d), ChromaVectorStore)
            assert isinstance(get_vector_store(d, backend="simple"), SimpleVectorStore)

    # Simulate chromadb being unavailable, even if it is installed globally.
    import sys
    old_chromadb = sys.modules.get("chromadb")
    sys.modules["chromadb"] = None
    try:
        with tempfile.TemporaryDirectory() as d:
            assert isinstance(get_vector_store(d), SimpleVectorStore)
    finally:
        if old_chromadb is None:
            sys.modules.pop("chromadb", None)
        else:
            sys.modules["chromadb"] = old_chromadb


def test_chroma_index_without_chromadb_raises_instead_of_returning_wrong_store():
    with installed(fake_chromadb()):
        with tempfile.TemporaryDirectory() as d:
            ChromaVectorStore(d).build(chunk_articles(ARTICLES), embedder=_tfidf())
            manifest_dir = d
            import sys

            sys.modules.pop("chromadb")  # simulate a runtime without chromadb
            sys.modules["chromadb"] = None  # makes `import chromadb` raise ImportError
            try:
                get_vector_store(manifest_dir)
            except ImportError:
                return
            finally:
                sys.modules.pop("chromadb", None)
    raise AssertionError("expected ImportError")


def test_manifest_records_semantic_embedder_config_and_reload_gives_same_embedder():
    st = fake_sentence_transformers()
    with installed(st):
        embedder = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        with tempfile.TemporaryDirectory() as d:
            store = SimpleVectorStore(d)
            store.build(chunk_articles(ARTICLES), embedder=embedder)
            manifest = read_manifest(d)
            reloaded = store.load_embedder()
            doc_calls = st.instances[0].calls[0]["texts"]
    assert manifest["embedding_kind"] == "semantic"
    assert manifest["embedder_model"] == "intfloat/multilingual-e5-small"
    assert manifest["passage_prefix"] == "passage: " and manifest["query_prefix"] == "query: "
    assert all(t.startswith("passage: ") for t in doc_calls)  # passages embedded with prefix
    assert reloaded.describe() == embedder.describe()  # "loading the same index later"


def test_manifest_marks_tfidf_index_as_lexical_not_semantic():
    with tempfile.TemporaryDirectory() as d:
        SimpleVectorStore(d).build(chunk_articles(ARTICLES), embedder=_tfidf())
        manifest = read_manifest(d)
    assert manifest["embedding_kind"] == "tfidf-lexical"
    assert "not semantic" in manifest["embedder_model"]


def test_embedding_text_is_the_full_article_text_both_fields():
    from src.rag.vector_store import _chunk_text_for_embedding

    chunk = chunk_articles(ARTICLES)[0]
    text = _chunk_text_for_embedding(chunk)
    assert chunk.text_ar in text and chunk.text_en in text


def test_build_index_from_json_records_chunking_params_and_corpus_hash():
    with tempfile.TemporaryDirectory() as d:
        corpus = Path(d) / "c.json"
        corpus.write_text(json.dumps(ARTICLES, ensure_ascii=False), encoding="utf-8")
        manifest = build_index_from_json(
            corpus, Path(d) / "idx", embedder=_tfidf(), chunk_size=500, overlap=50, backend="simple"
        )
    assert manifest["chunk_size"] == 500 and manifest["overlap"] == 50
    assert len(manifest["corpus_sha256"]) == 64 and manifest["article_count"] == 3


def test_overlap_repeats_boundary_sentence_in_next_chunk():
    from src.rag.chunking import _split_long_text

    text = "Alpha one. Bravo two two. Charlie three three three. Delta four. Echo five five."
    plain = _split_long_text(text, 30)
    over = _split_long_text(text, 30, overlap=20)
    assert plain[0] == "Alpha one. Bravo two two." and "Bravo two two." not in plain[1]
    assert over[1].startswith("Bravo two two.")  # carried-over sentence
    assert len(plain) == len(over)


def test_retriever_returns_all_required_metadata_fields():
    with tempfile.TemporaryDirectory() as d:
        SimpleVectorStore(d).build(chunk_articles(ARTICLES), embedder=_tfidf())
        result = Retriever(store_dir=d, top_k=1).retrieve("عقد البيع")[0]
    for field in ("article_number", "citation", "book", "chapter", "section", "topic",
                  "source_page", "is_repealed"):
        assert field in result
