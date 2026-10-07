"""Parse extracted PDF page text into structured Article records.

This module is deliberately independent of the PDF backend: it takes
plain (page_number, page_text) pairs so it can be unit-tested with
small synthetic text blocks, and reused unchanged once a real PDF
reader feeds it real pages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Tuple

from .arabic_utils import (
    find_article_marker,
    is_repealed,
    normalize_arabic_text,
)
from .schema import Article

# A genuine article-heading line is (almost) nothing but the marker itself
# (optionally with an English "Article N" duplicate of the same heading,
# which in this PDF can carry a trailing sentence fragment before the
# Arabic marker arrives -- see _looks_like_heading). A mid-paragraph
# citation, e.g. "...in accordance with Article 1119.", has substantial
# other sentence text and no matching companion marker in the other
# language on the same line. This distinguishes the two so a stray
# cross-reference never gets mistaken for the start of a new article.
# "A?" tolerates a rare extraction glitch that drops the leading "A"
# (observed once, on the real PDF's "Article 452" heading).
_ENGLISH_ARTICLE_RE = re.compile(r"A?rticle\s*[0-9]+")

# Hierarchy header markers, in the order the Egyptian Civil Code nests them.
# Lines are frequently bilingual ("BOOK II الكتاب الثاني"), so the Arabic
# keyword is searched anywhere in the line, not just at position 0.
_HIERARCHY_MARKERS = (
    ("book", re.compile(r"الكتاب\b")),
    ("chapter", re.compile(r"الباب\b")),
    ("section", re.compile(r"الفصل\b")),
)

# A range-repeal notice, e.g. "Articles 389-417 repealed المواد من ٣٨٩ إلى
# ٤١٧ ملغاة". These articles have no text to extract (removed from later
# reprints of the code) -- validation's gap report already surfaces the
# missing numbers, so this notice is recorded as a skipped/no-op line: it
# must never be absorbed into whichever article is currently open (that
# would both pollute its text and wrongly flip its is_repealed flag).
_RANGE_REPEAL_RE = re.compile(r"المواد\s+من.*(?:ملغاة|محذوفة)")
_HIERARCHY_LINE_MAX_CHARS = 40

# A line is treated as "mostly English" if the ratio of Latin letters to
# total letters exceeds this threshold. Used for best-effort bilingual
# splitting; exact column-based splitting needs the real PDF layout.
_ENGLISH_RATIO_THRESHOLD = 0.6
_LATIN_RE = re.compile(r"[A-Za-z]")
_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06FF]")


@dataclass
class _HierarchyState:
    book: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None


@dataclass
class _ArticleBuilder:
    article_number: int
    source_page: int
    hierarchy: _HierarchyState
    ar_lines: List[str] = field(default_factory=list)
    en_lines: List[str] = field(default_factory=list)

    def add_line(self, line: str) -> None:
        arabic_chars = len(_ARABIC_LETTER_RE.findall(line))
        latin_chars = len(_LATIN_RE.findall(line))
        total_letters = arabic_chars + latin_chars
        if total_letters == 0:
            return
        if latin_chars / total_letters > _ENGLISH_RATIO_THRESHOLD:
            self.en_lines.append(line)
        else:
            self.ar_lines.append(line)

    def build(self) -> Article:
        text_ar = normalize_arabic_text(" ".join(self.ar_lines))
        text_en = " ".join(self.en_lines).strip() or None
        combined_for_repeal_check = text_ar + " " + " ".join(self.ar_lines)
        citation = _make_citation(self.article_number, self.hierarchy)
        return Article(
            article_number=self.article_number,
            book=self.hierarchy.book,
            chapter=self.hierarchy.chapter,
            section=self.hierarchy.section,
            topic=None,
            text_ar=text_ar,
            text_en=text_en,
            is_repealed=is_repealed(combined_for_repeal_check),
            source_page=self.source_page,
            citation=citation,
        )


def _make_citation(article_number: int, hierarchy: _HierarchyState) -> str:
    parts = [p for p in (hierarchy.book, hierarchy.chapter, hierarchy.section) if p]
    prefix = " - ".join(parts)
    base = f"المادة {article_number}"
    return f"{prefix} - {base}" if prefix else base


def _looks_like_heading(line: str, marker: tuple) -> bool:
    """True if `line` (which contains an Arabic article marker) is a
    genuine heading rather than a mid-sentence/wrapped citation.

    Two cases, matching what's actually seen in this PDF:
    (a) An English "Article N" duplicate is ALSO present with the SAME
        number, AND nothing precedes it on the line -- real headings here
        always OPEN the line with "Article N" (optionally followed by a
        trailing English sentence fragment before the Arabic marker
        arrives, e.g. "Article 277 If the option belongs to the debtor
        and not مادة ٢٧٧", which is tolerated). A mid-sentence citation
        that happens to name the same article in both languages (e.g.
        "with Article 298 claim from the co-debtor ... للمادة ٢٩٨") always
        has real sentence text BEFORE the English marker too -- checking
        only that the numbers match let those slip through, so the
        "nothing before it" check is required.
    (b) No English marker -- accepted only if the Arabic marker is the
        ENTIRE line (whitespace-trimmed, no punctuation stripped). This
        is what correctly rejects short wrapped citations like
        ".عليه فى المادة 914" or ".المادة 1148" (which have a leftover
        period), while still accepting real bare headings like "مادة 1".
    """
    number, start, end = marker
    leftover = line[:start] + line[end:]
    en_match = _ENGLISH_ARTICLE_RE.search(line)

    if en_match:
        en_number_match = re.search(r"[0-9]+", en_match.group())
        en_number = int(en_number_match.group()) if en_number_match else None
        return en_number == number and line[: en_match.start()].strip() == ""

    return leftover.strip() == ""


def _match_hierarchy(line: str) -> Optional[Tuple[str, str]]:
    stripped = line.strip()
    if len(stripped) > _HIERARCHY_LINE_MAX_CHARS:
        # Real headers ("BOOK II الكتاب الثاني") are short. Long sentences
        # that merely mention the word "chapter"/"book" in passing (e.g.
        # "قلم الكتاب" = court clerk's office; "في هذا الباب" = "in this
        # Chapter" as a cross-reference) must not reset hierarchy state.
        return None
    for field_name, pattern in _HIERARCHY_MARKERS:
        if pattern.search(stripped):
            return field_name, stripped
    return None


def _find_english_only_heading(line: str) -> Optional[tuple]:
    """Fallback for the rare case where the Arabic marker is missing from
    an otherwise-genuine heading line entirely (observed once in the real
    PDF: "Article1022" with no Arabic pairing at all). Only accepted when
    the English marker is the ENTIRE line -- nothing else at all -- so
    this can never swallow an ordinary English sentence that happens to
    mention another article's number in passing.
    """
    match = _ENGLISH_ARTICLE_RE.search(line)
    if not match:
        return None
    leftover = line[: match.start()] + line[match.end() :]
    if leftover.strip() != "":
        return None
    number_match = re.search(r"[0-9]+", match.group())
    if not number_match:
        return None
    return (int(number_match.group()), match.start(), match.end())


def _strip_markers(line: str) -> str:
    """Remove the English "Article N" and Arabic "مادة N"/"المادة N"
    marker text from a confirmed heading line, leaving only any genuine
    body-text fragment that shared the line (kept verbatim, not
    fabricated) plus leftover parens/punctuation from around the number.
    """
    remainder = _ENGLISH_ARTICLE_RE.sub(" ", line)
    ar_marker = find_article_marker(remainder)
    if ar_marker is not None:
        _, start, end = ar_marker
        remainder = remainder[:start] + " " + remainder[end:]
    return remainder


def parse_pages_to_articles(pages: Iterable[Tuple[int, str]]) -> List[Article]:
    """Convert (page_number, page_text) pairs into a list of Article records.

    Lines are scanned in page order. A line starting a new "المادة N" marker
    closes the previous article and opens a new one. Hierarchy header lines
    (الكتاب / الباب / الفصل) update the running book/chapter/section context
    that gets attached to subsequently parsed articles.
    """
    hierarchy = _HierarchyState()
    articles: List[Article] = []
    current: Optional[_ArticleBuilder] = None

    for page_number, page_text in pages:
        if not page_text:
            continue
        for raw_line in page_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            hierarchy_match = _match_hierarchy(line)
            if hierarchy_match:
                field_name, value = hierarchy_match
                setattr(hierarchy, field_name, value)
                # A new book/chapter/section resets the narrower levels.
                if field_name == "book":
                    hierarchy.chapter = None
                    hierarchy.section = None
                elif field_name == "chapter":
                    hierarchy.section = None
                continue

            if _RANGE_REPEAL_RE.search(line):
                # Explicit "articles X-Y repealed" notice: not article body
                # text, not a heading. Skip it; the missing numbers already
                # surface via validation's gap report.
                continue

            marker = find_article_marker(line)
            if marker is not None and not _looks_like_heading(line, marker):
                marker = None
            if marker is None:
                marker = _find_english_only_heading(line)
            if marker is not None:
                article_number, _, _ = marker
                if current is not None:
                    articles.append(current.build())
                current = _ArticleBuilder(
                    article_number=article_number,
                    source_page=page_number,
                    hierarchy=_HierarchyState(
                        book=hierarchy.book,
                        chapter=hierarchy.chapter,
                        section=hierarchy.section,
                    ),
                )
                # Strip the marker text(s) themselves (e.g. "Article 105
                # مادة ١٠٥") out of the heading line so they never pollute
                # text_ar/text_en; article_number is already captured
                # separately. Any genuine body-text fragment that shares
                # the heading line (real in this PDF, e.g. "Article 277
                # If the option belongs to the debtor and not مادة ٢٧٧")
                # is preserved and still added as real content.
                remainder = _strip_markers(line)
                if remainder.strip():
                    current.add_line(remainder)
                continue

            if current is not None:
                current.add_line(line)
            # Lines appearing before the first article marker (e.g. cover
            # page / preamble) are intentionally dropped.

    if current is not None:
        articles.append(current.build())

    return articles
