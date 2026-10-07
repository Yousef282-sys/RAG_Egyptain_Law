import tempfile
from pathlib import Path

from src.rag.chunking import chunk_articles
from src.rag.embeddings import TfidfFallbackEmbedder
from src.rag.vector_store import SimpleVectorStore

# Tiny synthetic fixture -- not real Civil Code content.
_FIXTURE_ARTICLES = [
    dict(
        article_number=1,
        book="BOOK I",
        chapter=None,
        section=None,
        topic=None,
        text_ar="عقد البيع ينعقد بمجرد اتفاق الطرفين على الثمن والمبيع",
        text_en="A sale contract is concluded by mere agreement on price and object",
        is_repealed=False,
        source_page=1,
        citation="المادة 1",
    ),
    dict(
        article_number=2,
        book="BOOK I",
        chapter=None,
        section=None,
        topic=None,
        text_ar="الايجار عقد يلتزم به المؤجر بتمكين المستأجر من الانتفاع بالعين",
        text_en="A lease is a contract by which the lessor enables use of a thing",
        is_repealed=False,
        source_page=2,
        citation="المادة 2",
    ),
    dict(
        article_number=3,
        book="BOOK I",
        chapter=None,
        section=None,
        topic=None,
        text_ar="قواعد كرة القدم تحدد عدد اللاعبين والحكم",
        text_en="Football rules define player counts and the referee",
        is_repealed=False,
        source_page=3,
        citation="المادة 3",
    ),
]


def test_build_and_search_returns_expected_article():
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(Path(d))
        embedder = TfidfFallbackEmbedder(max_features=64)
        store.build(chunks, embedder=embedder)

        query_vec = embedder.embed(["عقد البيع والثمن"])[0]
        results = store.search(query_vec, top_k=2)

        assert len(results) == 2
        assert results[0]["article_number"] == 1  # most lexically similar
        assert "score" in results[0]


def test_manifest_and_metadata_persisted():
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(Path(d))
        store.build(chunks, embedder=TfidfFallbackEmbedder(max_features=32))

        assert (Path(d) / "manifest.json").exists()
        assert (Path(d) / "metadata.json").exists()
        assert (Path(d) / "vectors.npy").exists()
        assert (Path(d) / "tfidf_vectorizer.pkl").exists()


def test_load_embedder_round_trip_matches_build_embedder():
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(Path(d))
        build_embedder = TfidfFallbackEmbedder(max_features=64)
        store.build(chunks, embedder=build_embedder)

        loaded_embedder = store.load_embedder()
        v1 = build_embedder.embed(["عقد البيع"])
        v2 = loaded_embedder.embed(["عقد البيع"])
        assert (v1 == v2).all()


def test_search_never_returns_article_number_not_in_fixture():
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    valid_numbers = {a["article_number"] for a in _FIXTURE_ARTICLES}
    with tempfile.TemporaryDirectory() as d:
        store = SimpleVectorStore(Path(d))
        embedder = TfidfFallbackEmbedder(max_features=64)
        store.build(chunks, embedder=embedder)

        query_vec = embedder.embed(["نص عشوائي غير مرتبط"])[0]
        results = store.search(query_vec, top_k=3)
        for r in results:
            assert r["article_number"] in valid_numbers
