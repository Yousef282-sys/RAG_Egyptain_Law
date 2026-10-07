"""Validation for parsed Article records and the corpus as a whole."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from .schema import Article

MIN_TEXT_LEN = 5
MAX_TEXT_LEN = 20_000


@dataclass
class ValidationReport:
    total_articles: int
    article_issues: dict  # {article_number: [issue, ...]}
    number_gaps: List[int]
    duplicate_numbers: List[int]

    @property
    def is_clean(self) -> bool:
        return not self.article_issues and not self.duplicate_numbers

    def to_dict(self) -> dict:
        return {
            "total_articles": self.total_articles,
            "article_issues": self.article_issues,
            "number_gaps": self.number_gaps,
            "duplicate_numbers": self.duplicate_numbers,
            "is_clean": self.is_clean,
        }


def validate_article(article: Article) -> List[str]:
    """Return a list of human-readable issues for a single article. Empty = ok."""
    issues: List[str] = []

    if not isinstance(article.article_number, int):
        issues.append("article_number is not an integer")

    if not article.text_ar or not article.text_ar.strip():
        issues.append("text_ar is empty")
    else:
        length = len(article.text_ar)
        if length < MIN_TEXT_LEN and not article.is_repealed:
            issues.append(f"text_ar suspiciously short ({length} chars)")
        if length > MAX_TEXT_LEN:
            issues.append(f"text_ar suspiciously long ({length} chars)")

    if article.is_repealed and article.text_ar and len(article.text_ar) > MAX_TEXT_LEN:
        issues.append("repealed article has unexpectedly long text_ar")

    if not isinstance(article.is_repealed, bool):
        issues.append("is_repealed is not a boolean")

    if not article.citation:
        issues.append("citation is empty")

    return issues


def find_number_gaps(article_numbers: Sequence[int]) -> List[int]:
    """Report missing integers in the sorted range of article numbers.

    Gaps are reported, never silently dropped. A gap may be legitimate
    (e.g. an article was repealed and removed from later reprints), but
    it must always show up here rather than being assumed away.
    """
    if not article_numbers:
        return []
    unique_sorted = sorted(set(article_numbers))
    lo, hi = unique_sorted[0], unique_sorted[-1]
    present = set(unique_sorted)
    return [n for n in range(lo, hi + 1) if n not in present]


def find_duplicate_numbers(article_numbers: Sequence[int]) -> List[int]:
    seen = set()
    dupes = set()
    for n in article_numbers:
        if n in seen:
            dupes.add(n)
        seen.add(n)
    return sorted(dupes)


def validate_corpus(articles: Sequence[Article]) -> ValidationReport:
    """Validate every article and check corpus-level continuity/duplicates."""
    article_issues = {}
    for article in articles:
        issues = validate_article(article)
        if issues:
            article_issues[article.article_number] = issues

    numbers = [a.article_number for a in articles if isinstance(a.article_number, int)]
    gaps = find_number_gaps(numbers)
    dupes = find_duplicate_numbers(numbers)

    return ValidationReport(
        total_articles=len(articles),
        article_issues=article_issues,
        number_gaps=gaps,
        duplicate_numbers=dupes,
    )
