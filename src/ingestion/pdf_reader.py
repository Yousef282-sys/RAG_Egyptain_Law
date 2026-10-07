"""Extract per-page text from a PDF file.

Kept separate from parsing logic so parser.py can be tested without a
real PDF, and so the extraction backend can be swapped/improved once
we can see how the real Egyptian Civil Code PDF is laid out (columns,
bilingual interleaving, etc.) without touching parsing code.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from .bidi_fix import fix_rtl_text


def extract_pages_text(pdf_path: str | Path) -> List[Tuple[int, str]]:
    """Return a list of (page_number, page_text), 1-indexed pages.

    Tries pdfplumber first (better layout/column handling), falls back
    to pypdf if pdfplumber is unavailable or fails to open the file.
    Raises FileNotFoundError if pdf_path does not exist.

    Neither backend runs the Unicode Bidi algorithm, so Arabic runs come
    back in visual (reversed) order; fix_rtl_text() corrects that before
    the parser ever sees the text.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    try:
        pages = _extract_with_pdfplumber(path)
    except ImportError:
        pages = _extract_with_pypdf(path)

    return [(i, fix_rtl_text(text)) for i, text in pages]


def _extract_with_pdfplumber(path: Path) -> List[Tuple[int, str]]:
    import pdfplumber  # local import: optional dependency

    pages: List[Tuple[int, str]] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            pages.append((i, text))
    return pages


def _extract_with_pypdf(path: Path) -> List[Tuple[int, str]]:
    from pypdf import PdfReader  # local import: optional dependency

    reader = PdfReader(str(path))
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages
