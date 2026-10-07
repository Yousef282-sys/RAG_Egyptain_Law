"""Reproducible retrieval evaluation fixture, built from the real corpus.

Each question is a short verbatim snippet taken from a real article's own
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
from pathlib import Path
from typing import Dict, List

DEFAULT_EVAL_SET_PATH = Path("data/eval/retrieval_eval.json")
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
    path = Path(path)
    if path.exists():
        return load_eval_set(path)
    eval_set = build_eval_set_from_corpus(corpus_json_path)
    save_eval_set(eval_set, path)
    return eval_set
