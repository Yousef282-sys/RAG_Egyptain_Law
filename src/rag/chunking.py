"""Turn `data/processed/civil_code.json` articles into retrieval chunks.

Default: one chunk per article. Unusually long articles are split by
sentence/paragraph boundary so each chunk stays under MAX_CHUNK_CHARS,
while every resulting chunk still carries the SAME article_number and
metadata as its parent article -- article-number provenance is never
lost, and `topic` is passed through as-is (never invented; it is `None`
in the Day 1 corpus).
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

# Chosen from the real corpus's length distribution: only ~13 of 1095
# articles exceed this, so splitting is exercised but rare, matching the
# "default = one article" requirement.
MAX_CHUNK_CHARS = 1000

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.\u06D4])\s+")


@dataclass
class Chunk:
    chunk_id: str
    article_number: int
    book: Optional[str]
    chapter: Optional[str]
    section: Optional[str]
    topic: Optional[str]
    citation: str
    source_page: int
    is_repealed: bool
    text_ar: str
    text_en: Optional[str]
    chunk_index: int
    chunk_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def load_articles(json_path: str | Path) -> List[dict]:
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _split_long_text(text: str, max_chars: int) -> List[str]:
    """Split text on sentence boundaries into pieces <= max_chars each.
    Never fabricates content -- pure substring splitting.
    """
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces: List[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) > max_chars and current:
            pieces.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current.strip())
    return pieces


def chunk_article(article: dict, max_chars: int = MAX_CHUNK_CHARS) -> List[Chunk]:
    """Split one article dict (Day 1 schema) into 1+ Chunk records, all
    sharing the same article_number/book/chapter/section/citation/
    source_page/is_repealed/topic metadata.
    """
    article_number = article["article_number"]
    text_ar = article.get("text_ar") or ""
    text_en = article.get("text_en") or ""
    common = dict(
        article_number=article_number,
        book=article.get("book"),
        chapter=article.get("chapter"),
        section=article.get("section"),
        topic=article.get("topic"),
        citation=article.get("citation", ""),
        source_page=article.get("source_page"),
        is_repealed=bool(article.get("is_repealed", False)),
    )

    longest = max(len(text_ar), len(text_en))
    if longest <= max_chars:
        return [
            Chunk(
                chunk_id=f"{article_number}-0",
                text_ar=text_ar,
                text_en=text_en or None,
                chunk_index=0,
                chunk_count=1,
                **common,
            )
        ]

    ar_parts = _split_long_text(text_ar, max_chars)
    en_parts = _split_long_text(text_en, max_chars)
    n = max(len(ar_parts), len(en_parts), 1)
    chunks = []
    for i in range(n):
        chunks.append(
            Chunk(
                chunk_id=f"{article_number}-{i}",
                text_ar=ar_parts[i] if i < len(ar_parts) else "",
                text_en=(en_parts[i] if i < len(en_parts) else "") or None,
                chunk_index=i,
                chunk_count=n,
                **common,
            )
        )
    return chunks


def chunk_articles(articles: List[dict], max_chars: int = MAX_CHUNK_CHARS) -> List[Chunk]:
    chunks: List[Chunk] = []
    for article in articles:
        chunks.extend(chunk_article(article, max_chars))
    return chunks
