"""Data schema for a single Egyptian Civil Code article record."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class Article:
    """One structured record for a single civil code article."""

    article_number: int
    book: Optional[str]
    chapter: Optional[str]
    section: Optional[str]
    topic: Optional[str]
    text_ar: str
    text_en: Optional[str]
    is_repealed: bool
    source_page: int
    citation: str

    def to_dict(self) -> dict:
        return asdict(self)


REQUIRED_STRING_FIELDS = ("text_ar", "citation")
OPTIONAL_STRING_FIELDS = ("book", "chapter", "section", "topic", "text_en")
