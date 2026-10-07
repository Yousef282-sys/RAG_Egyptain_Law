from src.ingestion.bidi_fix import fix_rtl_line, fix_rtl_text

# Synthetic examples only, mimicking pdfplumber's visual-order Arabic output.


def test_fix_rtl_line_simple_word():
    # "١ ةدام" visual-order -> "مادة ١" logical order (synthetic snippet)
    assert fix_rtl_line("١ ةدام") == "مادة ١"


def test_fix_rtl_line_leaves_latin_untouched():
    assert fix_rtl_line("Article 1") == "Article 1"


def test_fix_rtl_line_mixed_bilingual():
    result = fix_rtl_line("SECTION I لولأا لصفلا")
    assert result.startswith("SECTION I")
    assert "الفصل" in result


def test_fix_rtl_line_keeps_digit_group_order():
    # digits are weak-LTR: already correct in the raw extraction and must
    # NOT be internally reversed when the surrounding Arabic run is fixed
    assert fix_rtl_line("Article 105 ١٠٥ ةدام") == "Article 105 مادة ١٠٥"


def test_fix_rtl_text_multiline():
    text = "١ ةدام\nArticle 1"
    fixed = fix_rtl_text(text)
    assert fixed.splitlines()[0] == "مادة ١"
    assert fixed.splitlines()[1] == "Article 1"


def test_fix_rtl_text_empty():
    assert fix_rtl_text("") == ""
