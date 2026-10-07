from src.rag.chunking import chunk_article, chunk_articles

# Synthetic article fixtures only -- not real Civil Code content.


def _make_article(**overrides) -> dict:
    base = dict(
        article_number=1,
        book="BOOK I",
        chapter="Chapter I",
        section="Section I",
        topic=None,
        text_ar="نص تجريبي قصير لهذه المادة.",
        text_en="Short synthetic test text for this article.",
        is_repealed=False,
        source_page=1,
        citation="المادة 1",
    )
    base.update(overrides)
    return base


def test_short_article_becomes_one_chunk():
    chunks = chunk_article(_make_article())
    assert len(chunks) == 1
    c = chunks[0]
    assert c.article_number == 1
    assert c.chunk_index == 0
    assert c.chunk_count == 1
    assert c.text_ar == "نص تجريبي قصير لهذه المادة."


def test_chunk_preserves_all_metadata_fields():
    article = _make_article(
        book="BOOK II", chapter="Chapter III", section="Section IV", is_repealed=True
    )
    chunks = chunk_article(article)
    c = chunks[0]
    assert c.book == "BOOK II"
    assert c.chapter == "Chapter III"
    assert c.section == "Section IV"
    assert c.is_repealed is True
    assert c.citation == "المادة 1"
    assert c.source_page == 1
    assert c.topic is None  # never invented


def test_long_article_splits_but_keeps_article_number():
    long_text_ar = " ".join([f"جملة تجريبية رقم {i}." for i in range(200)])
    long_text_en = " ".join([f"Synthetic sentence number {i}." for i in range(200)])
    article = _make_article(
        article_number=42, text_ar=long_text_ar, text_en=long_text_en
    )
    chunks = chunk_article(article, max_chars=200)
    assert len(chunks) > 1
    for c in chunks:
        assert c.article_number == 42
        assert c.citation == "المادة 1"
        assert c.chunk_count == len(chunks)
        assert len(c.text_ar) <= 250  # sentence-boundary split, small slack


def test_long_article_chunks_have_distinct_indices():
    long_text_ar = " ".join([f"جملة رقم {i}." for i in range(100)])
    article = _make_article(article_number=5, text_ar=long_text_ar, text_en="")
    chunks = chunk_article(article, max_chars=100)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_chunk_articles_processes_a_list():
    articles = [_make_article(article_number=1), _make_article(article_number=2)]
    chunks = chunk_articles(articles)
    assert [c.article_number for c in chunks] == [1, 2]


def test_no_content_fabricated_when_splitting():
    # every word in the split chunks must have come from the original text
    long_text_ar = " ".join([f"كلمة{i}" for i in range(150)])
    article = _make_article(article_number=9, text_ar=long_text_ar, text_en="")
    chunks = chunk_article(article, max_chars=150)
    reconstructed = " ".join(c.text_ar for c in chunks)
    for word in long_text_ar.split():
        assert word in reconstructed
