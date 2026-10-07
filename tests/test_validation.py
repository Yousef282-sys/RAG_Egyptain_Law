from src.ingestion.schema import Article
from src.ingestion.validation import (
    find_duplicate_numbers,
    find_number_gaps,
    validate_article,
    validate_corpus,
)


def _make_article(**overrides) -> Article:
    base = dict(
        article_number=1,
        book=None,
        chapter=None,
        section=None,
        topic=None,
        text_ar="نص تجريبي كافي الطول لهذا الاختبار.",
        text_en=None,
        is_repealed=False,
        source_page=1,
        citation="المادة 1",
    )
    base.update(overrides)
    return Article(**base)


def test_valid_article_has_no_issues():
    assert validate_article(_make_article()) == []


def test_empty_text_ar_is_flagged():
    issues = validate_article(_make_article(text_ar=""))
    assert any("empty" in i for i in issues)


def test_non_integer_article_number_is_flagged():
    issues = validate_article(_make_article(article_number="1"))
    assert any("integer" in i for i in issues)


def test_suspiciously_short_text_flagged_when_not_repealed():
    issues = validate_article(_make_article(text_ar="قصير", is_repealed=False))
    assert any("short" in i for i in issues)


def test_short_text_allowed_when_repealed():
    issues = validate_article(_make_article(text_ar="ملغاة", is_repealed=True))
    assert not any("short" in i for i in issues)


def test_find_number_gaps_detects_missing():
    assert find_number_gaps([1, 2, 4, 5]) == [3]


def test_find_number_gaps_empty_when_continuous():
    assert find_number_gaps([1, 2, 3]) == []


def test_find_duplicate_numbers():
    assert find_duplicate_numbers([1, 2, 2, 3, 3, 3]) == [2, 3]


def test_validate_corpus_reports_gaps_and_issues():
    articles = [
        _make_article(article_number=1, citation="المادة 1"),
        _make_article(article_number=3, citation="المادة 3"),
        _make_article(article_number=5, text_ar="", citation="المادة 5"),
    ]
    report = validate_corpus(articles)
    assert report.total_articles == 3
    assert report.number_gaps == [2, 4]
    assert 5 in report.article_issues
    assert report.is_clean is False
