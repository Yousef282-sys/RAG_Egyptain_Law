from src.ingestion.arabic_utils import (
    extract_article_number,
    is_repealed,
    normalize_arabic_indic_digits,
    normalize_arabic_text,
    to_western_int,
)


def test_normalize_arabic_indic_digits_basic():
    assert normalize_arabic_indic_digits("١٢٣") == "123"


def test_normalize_arabic_indic_digits_mixed_with_text():
    assert normalize_arabic_indic_digits("المادة ١٤٥") == "المادة 145"


def test_to_western_int_from_arabic_indic():
    assert to_western_int("١٤٥") == 145


def test_to_western_int_from_western():
    assert to_western_int("145") == 145


def test_to_western_int_invalid_returns_none():
    assert to_western_int("abc") is None
    assert to_western_int("") is None


def test_extract_article_number_simple():
    # synthetic example, not real Civil Code text
    assert extract_article_number("المادة 10 - نص تجريبي") == 10


def test_extract_article_number_arabic_indic():
    assert extract_article_number("مادة رقم ١٠ نص تجريبي") == 10


def test_extract_article_number_none_when_absent():
    assert extract_article_number("نص عادي بدون رقم مادة") is None


def test_is_repealed_detects_marker():
    assert is_repealed("هذه المادة ملغاة بموجب القانون") is True


def test_is_repealed_false_for_normal_text():
    assert is_repealed("نص تجريبي عادي") is False


def test_normalize_arabic_text_strips_diacritics_and_tatweel():
    raw = "نَـــص تجريبي"
    normalized = normalize_arabic_text(raw)
    assert "َ" not in normalized
    assert "ـ" not in normalized


def test_normalize_arabic_text_collapses_whitespace():
    assert normalize_arabic_text("نص    تجريبي  ") == "نص تجريبي"
