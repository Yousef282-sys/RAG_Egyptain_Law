import tempfile
from pathlib import Path

from src.experiments.evaluate import evaluate_retriever
from src.rag.chunking import chunk_articles
from src.rag.embeddings import TfidfFallbackEmbedder
from src.rag.retriever import Retriever
from src.rag.vector_store import SimpleVectorStore

_FIXTURE_ARTICLES = [
    dict(
        article_number=1, book=None, chapter=None, section=None, topic=None,
        text_ar="عقد البيع ينعقد بمجرد اتفاق الطرفين على الثمن والمبيع بينهما",
        text_en="sale", is_repealed=False, source_page=1, citation="المادة 1",
    ),
    dict(
        article_number=2, book=None, chapter=None, section=None, topic=None,
        text_ar="الايجار عقد يلتزم به المؤجر بتمكين المستأجر من الانتفاع بالعين",
        text_en="lease", is_repealed=False, source_page=2, citation="المادة 2",
    ),
]


def _build_fixture_retriever(directory):
    chunks = chunk_articles(_FIXTURE_ARTICLES)
    store = SimpleVectorStore(directory)
    store.build(chunks, embedder=TfidfFallbackEmbedder(max_features=64))
    return Retriever(store_dir=directory, top_k=2, store=store)


def test_evaluate_perfect_self_retrieval_gets_hit_rate_1():
    eval_set = [
        {"question": "عقد البيع ينعقد بمجرد اتفاق الطرفين", "expected_article_number": 1},
        {"question": "الايجار عقد يلتزم به المؤجر", "expected_article_number": 2},
    ]
    with tempfile.TemporaryDirectory() as d:
        retriever = _build_fixture_retriever(Path(d))
        metrics = evaluate_retriever(retriever, eval_set, top_k=2)
    assert metrics["hit_rate_at_2"] == 1.0
    assert metrics["mean_reciprocal_rank"] == 1.0
    assert metrics["n_eval_questions"] == 2


def test_evaluate_miss_lowers_hit_rate():
    eval_set = [
        {"question": "عقد البيع ينعقد بمجرد اتفاق الطرفين", "expected_article_number": 1},
        {"question": "نص عشوائي لا علاقة له بأي مادة", "expected_article_number": 999},
    ]
    with tempfile.TemporaryDirectory() as d:
        retriever = _build_fixture_retriever(Path(d))
        metrics = evaluate_retriever(retriever, eval_set, top_k=2)
    assert metrics["hit_rate_at_2"] == 0.5
    assert metrics["n_eval_questions"] == 2


def test_evaluate_empty_eval_set_returns_zero_not_crash():
    with tempfile.TemporaryDirectory() as d:
        retriever = _build_fixture_retriever(Path(d))
        metrics = evaluate_retriever(retriever, [], top_k=5)
    assert metrics["n_eval_questions"] == 0
    assert metrics["hit_rate_at_5"] == 0.0
