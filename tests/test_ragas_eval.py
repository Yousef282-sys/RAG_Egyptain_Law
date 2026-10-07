import tempfile
from pathlib import Path

from src.experiments.ragas_eval import build_ragas_records, evaluate_faithfulness
from src.rag.chunking import chunk_articles
from src.rag.embeddings import TfidfFallbackEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import SimpleVectorStore

try:
    import ragas  # noqa: F401

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def _skip(name: str) -> str:
    print(
        f"  ({name}: ragas not installed in this sandbox -- not executed "
        "here, run in an environment with that package)"
    )
    return "SKIPPED"


_FIXTURE_ARTICLES = [
    dict(
        article_number=1, book=None, chapter=None, section=None, topic=None,
        text_ar="عقد البيع ينعقد بمجرد اتفاق الطرفين على الثمن والمبيع",
        text_en="sale contract", is_repealed=False, source_page=1, citation="المادة 1",
    ),
]


def _build_fixture_retriever(directory):
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    store = SimpleVectorStore(directory)
    store.build(chunks, embedder=TfidfFallbackEmbedder(max_features=64))
    return Retriever(store_dir=directory, top_k=1, store=store)


def _mock_generate_fn(question: str, contexts: list) -> str:
    joined = " | ".join(contexts)
    return f"[mock generation for testing only] Based on: {joined[:30]}"


def test_build_ragas_records_reuses_real_retriever_and_generate_fn():
    eval_set = [{"question": "ما هو عقد البيع؟", "expected_article_number": 1}]
    with tempfile.TemporaryDirectory() as d:
        retriever = _build_fixture_retriever(Path(d))
        records = build_ragas_records(eval_set, retriever, _mock_generate_fn, top_k=1)

    assert len(records) == 1
    record = records[0]
    assert record["question"] == "ما هو عقد البيع؟"
    assert isinstance(record["contexts"], list)
    assert len(record["contexts"]) == 1
    assert "mock generation for testing only" in record["answer"]


def test_build_ragas_records_contexts_come_from_real_corpus_text():
    eval_set = [{"question": "عقد البيع", "expected_article_number": 1}]
    with tempfile.TemporaryDirectory() as d:
        retriever = _build_fixture_retriever(Path(d))
        records = build_ragas_records(eval_set, retriever, _mock_generate_fn, top_k=1)

    assert "عقد البيع" in records[0]["contexts"][0]


def test_build_ragas_records_empty_eval_set_returns_empty_list():
    with tempfile.TemporaryDirectory() as d:
        retriever = _build_fixture_retriever(Path(d))
        records = build_ragas_records([], retriever, _mock_generate_fn)
    assert records == []


def test_evaluate_faithfulness_raises_import_error_when_ragas_unavailable():
    if RAGAS_AVAILABLE:
        return _skip("test_evaluate_faithfulness_raises_import_error_when_ragas_unavailable")
    raised = False
    try:
        evaluate_faithfulness([{"question": "q", "contexts": ["c"], "answer": "a"}])
    except ImportError:
        raised = True
    assert raised
