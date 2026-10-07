"""Utilities for handling Arabic legal text extracted from PDFs.

Covers:
- Arabic-Indic digit -> Western integer conversion
- light-touch Arabic text normalization (whitespace/tatweel only,
  never touching letters that change legal meaning)
- extraction of the article number from a line such as "المادة ١٤٥"
- detection of explicit "repealed" markers
"""
from __future__ import annotations

import re
from typing import Optional

# Arabic-Indic (٠-٩) and Persian/Extended Arabic-Indic (۰-۹) digit maps.
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_EXTENDED_ARABIC_INDIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_DIGIT_MAP = {
    **{d: str(i) for i, d in enumerate(_ARABIC_INDIC_DIGITS)},
    **{d: str(i) for i, d in enumerate(_EXTENDED_ARABIC_INDIC_DIGITS)},
}

# Arabic diacritics (tashkeel) unicode range, safe to strip for search/matching
# purposes without changing the legal meaning of the word.
_TASHKEEL_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u06D6-\u06ED]")
_TATWEEL_RE = re.compile(r"\u0640")  # ـ elongation character, purely cosmetic
_WHITESPACE_RE = re.compile(r"[ \t\u00A0]+")

# Words that introduce an article, in the forms seen in the Egyptian Civil Code.
_ARTICLE_WORD = r"(?:مادة|المادة)"

# Word-then-number: "المادة 145", "مادة رقم ١٤٥".
_ARTICLE_NUMBER_AFTER_RE = re.compile(
    rf"{_ARTICLE_WORD}\s*(?:رقم)?\s*[:\-]?\s*([0-9٠-٩۰-۹]+)\s*(مكرر)?",
)
# Number-then-word: "13 ( مادة" -- seen in this PDF's bilingual heading
# lines where the Arabic clause "مادة (13)" comes out with the number
# ahead of the word (parenthesis handling is a separate, documented
# cosmetic limitation; the digit value itself is unaffected here since
# digit groups are never reordered by the bidi fix).
_ARTICLE_NUMBER_BEFORE_RE = re.compile(
    rf"([0-9٠-٩۰-۹]+)\s*[\(\)\s]*\s*{_ARTICLE_WORD}",
)

# Explicit repeal markers used in the Egyptian Civil Code and its amendments.
_REPEALED_PATTERNS = [
    re.compile(r"ملغا?ة"),   # ملغاة / ملغا
    re.compile(r"ملغى"),
    re.compile(r"محذوفة"),
    re.compile(r"\(?\s*ملغ[اى]ة?\s*\)?"),
]


def normalize_arabic_indic_digits(text: str) -> str:
    """Convert Arabic-Indic / extended Arabic-Indic digits to Western digits.

    Non-digit characters are left untouched.
    """
    return "".join(_DIGIT_MAP.get(ch, ch) for ch in text)


def to_western_int(number_text: str) -> Optional[int]:
    """Parse a digit string (Arabic-Indic, extended, or Western) to an int.

    Returns None if no valid integer can be parsed.
    """
    if not number_text:
        return None
    converted = normalize_arabic_indic_digits(number_text.strip())
    converted = re.sub(r"[^0-9]", "", converted)
    if not converted:
        return None
    try:
        return int(converted)
    except ValueError:
        return None


def normalize_arabic_text(text: str, strip_diacritics: bool = True) -> str:
    """Light-touch normalization that never changes legal meaning.

    - collapses runs of spaces/tabs into one space
    - strips the cosmetic tatweel elongation character
    - optionally strips tashkeel (diacritics) since the Civil Code source
      text is not diacritic-dependent for meaning
    - trims leading/trailing whitespace
    Does NOT alter letters (e.g. does not normalize alef/hamza variants),
    since that could change the meaning of a legal term.
    """
    if text is None:
        return text
    out = _TATWEEL_RE.sub("", text)
    if strip_diacritics:
        out = _TASHKEEL_RE.sub("", out)
    out = _WHITESPACE_RE.sub(" ", out)
    return out.strip()


def find_article_marker(line: str) -> Optional[tuple]:
    """Find an article marker in a line and return (number, start, end),
    where start/end is the marker's character span in the *original* line
    (so callers can tell a genuine heading from a mid-sentence citation
    such as "... in accordance with Article 1119." by checking how much
    of the line is left over outside the matched span).

    Returns None if no article marker is found.
    """
    if not line:
        return None
    match = _ARTICLE_NUMBER_AFTER_RE.search(line)
    if match:
        number = to_western_int(match.group(1))
        if number is not None:
            return (number, match.start(), match.end())
    match = _ARTICLE_NUMBER_BEFORE_RE.search(line)
    if match:
        number = to_western_int(match.group(1))
        if number is not None:
            return (number, match.start(), match.end())
    return None


def extract_article_number(line: str) -> Optional[int]:
    """Find and parse an article number from a line of text.

    Matches "المادة 145" / "مادة رقم ١٤٥" / "المادة: 12" (word-then-number)
    and "١٤٥ ( مادة" (number-then-word). Returns None if no article marker
    is found.
    """
    marker = find_article_marker(line)
    return marker[0] if marker else None


def is_repealed(text: str) -> bool:
    """Detect whether a block of article text carries an explicit repeal marker."""
    if not text:
        return False
    return any(p.search(text) for p in _REPEALED_PATTERNS)
