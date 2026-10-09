"""CLI for the Civil Code PDF -> structured JSON ingestion pipeline.

Usage:
    python -m src.ingestion.cli ingest \
        --pdf data/raw/civil_code.pdf \
        --out data/processed/civil_code.json

    python -m src.ingestion.cli validate \
        --json data/processed/civil_code.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .parser import parse_pages_to_articles
from .pdf_reader import extract_pages_text
from .schema import Article
from .validation import validate_corpus


def run_ingest(pdf_path: str, out_path: str) -> int:
    try:
        pages = extract_pages_text(pdf_path)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    articles = parse_pages_to_articles(pages)
    if not articles:
        print(
            "ERROR: no articles were parsed from the PDF. "
            "Check the PDF layout / extraction backend before trusting output.",
            file=sys.stderr,
        )
        return 1

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as f:
        json.dump([a.to_dict() for a in articles], f, ensure_ascii=False, indent=2)

    report = validate_corpus(articles)
    print(f"Parsed {report.total_articles} articles -> {out}")
    if not report.is_clean:
        print(
            f"WARNING: {len(report.article_issues)} articles have validation "
            f"issues, {len(report.duplicate_numbers)} duplicate numbers, "
            f"{len(report.number_gaps)} number gaps. Run 'validate' for details.",
            file=sys.stderr,
        )
    return 0


def run_validate(json_path: str) -> int:
    path = Path(json_path)
    if not path.exists():
        print(f"ERROR: JSON not found: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        raw: List[dict] = json.load(f)
    articles = [Article(**item) for item in raw]

    report = validate_corpus(articles)
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.is_clean else 2


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Egyptian Civil Code ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="PDF -> structured JSON")
    p_ingest.add_argument("--pdf", required=True, help="Path to the source PDF")
    p_ingest.add_argument("--out", default="data/processed/civil_code.json")

    p_validate = sub.add_parser("validate", help="Validate an existing JSON output")
    p_validate.add_argument("--json", required=True)

    args = parser.parse_args(argv)

    if args.command == "ingest":
        return run_ingest(args.pdf, args.out)
    if args.command == "validate":
        return run_validate(args.json)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
