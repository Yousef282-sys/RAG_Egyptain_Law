import tempfile
from pathlib import Path

from src.rag.chunking import chunk_articles
from src.rag.embeddings import TfidfFallbackEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import SimpleVectorStore

_FIXTURE_ARTICLES = [
    dict(
        article_number=10,
        book=None, chapter=None, section=None, topic=None,
        text_ar="عقد البيع ينعقد بمجرد اتفاق الطرفين على الثمن والمبيع",
        text_en="A sale contract is concluded by mere agreement on price and object",
        is_repealed=False, source_page=1, citation="المادة 10",
    ),
    dict(
        article_number=20,
        book=None, chapter=None, section=None, topic=None,
        text_ar="الايجار عقد يلتزم به المؤجر بتمكين المستأجر من الانتفاع بالعين",
        text_en="A lease is a contract by which the lessor enables use of a thing",
        is_repealed=False, source_page=2, citation="المادة 20",
    ),
]


def _build_fixture_store(directory):
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    store = SimpleVectorStore(directory)
    store.build(chunks, embedder=TfidfFallbackEmbedder(max_features=64))
    return store


def test_retrieve_returns_required_keys():
    with tempfile.TemporaryDirectory() as d:
        _build_fixture_store(Path(d))
        retriever = Retriever(store_dir=Path(d), top_k=2)
        results = retriever.retrieve("عقد البيع والثمن")
        assert len(results) == 2
        for r in results:
            for key in ("article_number", "text_ar", "text_en", "citation", "score"):
                assert key in r


def test_retrieve_respects_top_k_override():
    with tempfile.TemporaryDirectory() as d:
        _build_fixture_store(Path(d))
        retriever = Retriever(store_dir=Path(d), top_k=2)
        results = retriever.retrieve("عقد البيع", top_k=1)
        assert len(results) == 1


def test_retrieve_empty_question_returns_no_results():
    with tempfile.TemporaryDirectory() as d:
        _build_fixture_store(Path(d))
        retriever = Retriever(store_dir=Path(d))
        assert retriever.retrieve("") == []
        assert retriever.retrieve("   ") == []


def test_retrieve_never_fabricates_article_numbers():
    valid_numbers = {a["article_number"] for a in _FIXTURE_ARTICLES}
    with tempfile.TemporaryDirectory() as d:
        _build_fixture_store(Path(d))
        retriever = Retriever(store_dir=Path(d), top_k=2)
        results = retriever.retrieve("عقد الايجار والانتفاع")
        for r in results:
            assert r["article_number"] in valid_numbers


def test_retriever_raises_clear_error_on_missing_index():
    # Day-3 test: a missing/never-built index must raise (so the API
    # layer can turn it into a clean 503), not silently return empty
    # results or crash with an obscure traceback.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        empty_dir = Path(d) / "does-not-exist"
        try:
            Retriever(store_dir=empty_dir)
            raised = False
        except Exception:
            raised = True
        assert raised
