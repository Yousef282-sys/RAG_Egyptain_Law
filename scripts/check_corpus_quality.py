"""Corpus quality gate (CI-safe, deterministic).

`python -m src.ingestion.cli validate` always exits 2 on this corpus,
because `duplicate_numbers` is always non-empty -- articles 1 and 2 are
legitimately duplicated (the Promulgation Law's own front-matter articles
vs. the Civil Code proper restarting numbering; documented since Day 1,
see reports/day1_spot_check.md). Using that raw exit code as a CI gate
would fail on every single correct run.

This script instead encodes the exact known-acceptable baseline and
fails ONLY on a genuine deviation from it -- a real, deterministic
regression check, not a rubber stamp.
"""
from __future__ import annotations

import json
import subprocess
import sys

CORPUS_PATH = "data/processed/civil_code.json"

EXPECTED_TOTAL_ARTICLES = 1095
EXPECTED_DUPLICATE_NUMBERS = [1, 2]
EXPECTED_GAP_COUNT = 56
EXPECTED_ARTICLE_ISSUES = {"1022": ["text_ar is empty"]}


def run_validate() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "src.ingestion.cli", "validate", "--json", CORPUS_PATH],
        capture_output=True,
        text=True,
    )
    # exit code is intentionally NOT used as the pass/fail signal here --
    # see module docstring. The JSON report on stdout is what's checked.
    return json.loads(result.stdout)


def check(report: dict) -> list:
    """Return a list of human-readable problems; empty list = pass."""
    problems = []

    if report["total_articles"] != EXPECTED_TOTAL_ARTICLES:
        problems.append(
            f"total_articles changed: expected {EXPECTED_TOTAL_ARTICLES}, "
            f"got {report['total_articles']}"
        )

    if report["duplicate_numbers"] != EXPECTED_DUPLICATE_NUMBERS:
        problems.append(
            f"duplicate_numbers changed: expected {EXPECTED_DUPLICATE_NUMBERS} "
            f"(the documented Promulgation Law / Civil Code restart), got "
            f"{report['duplicate_numbers']}"
        )

    if len(report["number_gaps"]) != EXPECTED_GAP_COUNT:
        problems.append(
            f"number_gaps count changed: expected {EXPECTED_GAP_COUNT}, "
            f"got {len(report['number_gaps'])}"
        )

    if report["article_issues"] != EXPECTED_ARTICLE_ISSUES:
        problems.append(
            f"article_issues changed: expected {EXPECTED_ARTICLE_ISSUES} "
            f"(the documented article-1022 empty-text_ar gap), got "
            f"{report['article_issues']}"
        )

    return problems


def main() -> int:
    report = run_validate()
    problems = check(report)
    if problems:
        print("Corpus quality gate: FAILED")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(
        f"Corpus quality gate: PASSED "
        f"({report['total_articles']} articles, known-acceptable duplicates/gaps unchanged)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
