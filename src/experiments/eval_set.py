"""Evaluation sets built from the real corpus.

TWO sets live in data/eval/:

* `retrieval_eval.json` -- the CURATED set (>= 50 bilingual Arabic/English
  questions, src/experiments/curated_questions.py) with a corpus-derived
  `reference` answer per question. Used for retrieval metrics AND the four
  RAGAS metrics (context precision/recall need the reference).
  Build/refresh: `python -m src.experiments.eval_set`.
* `selfretrieval_snippets.json` -- the original Day-4 lexical
  self-retrieval check described below (kept for reproducibility of the
  Day-4 results; it has no reference answers, so it cannot drive RAGAS).

Legacy self-retrieval fixture: each question is a short verbatim snippet taken from a real article's own
`text_ar` -- a "given a distinctive piece of this article's own text, can
the system find it again" self-retrieval check. This is NOT an
independently-authored natural-language legal question with a verified
answer (constructing those would need domain-expert legal judgment to
confirm the "correct" article, which we do not have and will not
approximate) -- it is a lexical sanity check with unambiguous, real
ground truth: `expected_article_number` is read directly from
`data/processed/civil_code.json`, never invented.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from src.experiments.curated_questions import CURATED_QUESTIONS

DEFAULT_EVAL_SET_PATH = Path("data/eval/retrieval_eval.json")
SELFRETRIEVAL_PATH = Path("data/eval/selfretrieval_snippets.json")
MIN_EVAL_QUESTIONS = 50  # Handbook: >= 50 evaluation questions
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_OR_DIGIT_RE = re.compile(r"[A-Za-z0-9]")
DEFAULT_SEED = 42
DEFAULT_N = 15
DEFAULT_SNIPPET_WORDS = 8


def build_eval_set_from_corpus(
    corpus_json_path: str,
    n: int = DEFAULT_N,
    snippet_words: int = DEFAULT_SNIPPET_WORDS,
    seed: int = DEFAULT_SEED,
) -> List[Dict]:
    """Deterministically sample n articles (fixed seed) and build one
    self-retrieval question per article from its own opening words.
    Skips articles whose text_ar is empty or too short for a clean
    snippet (e.g. the known Day-1 gap, article 1022) -- never fabricates
    text for them.
    """
    with open(corpus_json_path, encoding="utf-8") as f:
        articles = json.load(f)

    candidates = [
        a
        for a in articles
        if a.get("text_ar") and len(a["text_ar"].split()) >= snippet_words
    ]
    rng = random.Random(seed)
    picked = rng.sample(candidates, min(n, len(candidates)))

    eval_set = []
    for article in picked:
        words = article["text_ar"].split()[:snippet_words]
        eval_set.append(
            {
                "question": " ".join(words),
                "expected_article_number": article["article_number"],
                "citation": article.get("citation", ""),
            }
        )
    return eval_set


def save_eval_set(eval_set: List[Dict], path: str | Path = DEFAULT_EVAL_SET_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_set, f, ensure_ascii=False, indent=2)


def load_eval_set(path: str | Path = DEFAULT_EVAL_SET_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_or_build_eval_set(
    corpus_json_path: str, path: str | Path = DEFAULT_EVAL_SET_PATH
) -> List[Dict]:
    """Legacy helper: load `path`, or build the Day-4 snippet set there.
    New code should use load_curated_eval_set (never silently builds).
    """
    path = Path(path)
    if path.exists():
        return load_eval_set(path)
    eval_set = build_eval_set_from_corpus(corpus_json_path)
    save_eval_set(eval_set, path)
    return eval_set


# --------------------------- curated set ---------------------------------


def english_extract(article: Dict) -> str:
    """The article's English text, taken verbatim from the corpus.

    The source PDF is two-column bilingual and the extraction interleaved
    Arabic and English fragments across `text_ar`/`text_en` (see
    src/rag/chunking.full_text). Keeping only the Latin-script tokens of
    the concatenation recovers the English text; for the short,
    single-sentence articles used in the curated set this is the clean
    English sentence(s).
    """
    joined = " ".join(filter(None, [article.get("text_ar"), article.get("text_en")]))
    return " ".join(
        w for w in joined.split() if _LATIN_OR_DIGIT_RE.search(w) and not _ARABIC_RE.search(w)
    )


def build_curated_eval_set(corpus_json_path: str | Path) -> List[Dict]:
    with open(corpus_json_path, encoding="utf-8") as f:
        articles = json.load(f)
    counts = Counter(a["article_number"] for a in articles)
    by_number = {a["article_number"]: a for a in articles}

    eval_set = []
    for i, (number, language, question) in enumerate(CURATED_QUESTIONS, start=1):
        if counts.get(number) != 1:
            raise ValueError(f"article {number} must occur exactly once in the corpus")
        article = by_number[number]
        eval_set.append(
            {
                "id": f"q{i:03d}",
                "language": language,
                "question": question,
                "expected_article_number": number,
                "expected_article_numbers": [number],
                "citation": article.get("citation", ""),
                "reference": english_extract(article),
                "reference_source": f"corpus: Latin-script text of article {number}",
            }
        )
    return eval_set


def validate_eval_set(eval_set: List[Dict], corpus_json_path: str | Path) -> List[str]:
    """Return a list of problems (empty == valid). Checks size, schema,
    that every expected article exists exactly once in the corpus, and that
    every reference is the corpus-derived extract (never hand-written).
    """
    problems: List[str] = []
    if len(eval_set) < MIN_EVAL_QUESTIONS:
        problems.append(f"only {len(eval_set)} questions; need >= {MIN_EVAL_QUESTIONS}")
    with open(corpus_json_path, encoding="utf-8") as f:
        articles = json.load(f)
    counts = Counter(a["article_number"] for a in articles)
    by_number = {a["article_number"]: a for a in articles}

    ids = Counter(item.get("id") for item in eval_set)
    questions = Counter(item.get("question") for item in eval_set)
    for item in eval_set:
        qid = item.get("id", "?")
        for key in ("id", "language", "question", "expected_article_numbers", "reference"):
            if not item.get(key):
                problems.append(f"{qid}: missing {key}")
        if ids[item.get("id")] > 1:
            problems.append(f"{qid}: duplicate id")
        if questions[item.get("question")] > 1:
            problems.append(f"{qid}: duplicate question")
        if item.get("language") not in ("ar", "en"):
            problems.append(f"{qid}: language must be 'ar' or 'en'")
        has_arabic = bool(_ARABIC_RE.search(item.get("question", "")))
        if (item.get("language") == "ar") != has_arabic:
            problems.append(f"{qid}: language label does not match the question script")
        for number in item.get("expected_article_numbers", []):
            if counts.get(number) != 1:
                problems.append(f"{qid}: article {number} not found exactly once in corpus")
            elif item.get("reference") != english_extract(by_number[number]) and len(
                item["expected_article_numbers"]
            ) == 1:
                problems.append(f"{qid}: reference is not the corpus extract of article {number}")
    return problems


def load_curated_eval_set(
    path: str | Path = DEFAULT_EVAL_SET_PATH, corpus_json_path: str | Path | None = None
) -> List[Dict]:
    """Load the curated set. Never builds or substitutes another set:
    raises if the file is missing, and (when corpus_json_path is given)
    if it fails validate_eval_set -- so a gate can never run on a wrong
    or undersized set.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"curated eval set not found: {path}")
    eval_set = load_eval_set(path)
    if corpus_json_path is not None:
        problems = validate_eval_set(eval_set, corpus_json_path)
        if problems:
            raise ValueError("invalid eval set: " + "; ".join(problems[:5]))
    return eval_set


if __name__ == "__main__":
    curated = build_curated_eval_set("data/processed/civil_code.json")
    save_eval_set(curated, DEFAULT_EVAL_SET_PATH)
    langs = Counter(item["language"] for item in curated)
    print(f"wrote {len(curated)} questions to {DEFAULT_EVAL_SET_PATH} ({dict(langs)})")
