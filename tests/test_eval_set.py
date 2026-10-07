import tempfile
from pathlib import Path

from src.experiments.eval_set import (
    build_eval_set_from_corpus,
    load_eval_set,
    load_or_build_eval_set,
    save_eval_set,
)

CORPUS_PATH = "data/processed/civil_code.json"


def test_build_eval_set_from_real_corpus_returns_real_article_numbers():
    import json

    with open(CORPUS_PATH, encoding="utf-8") as f:
        articles = json.load(f)
    real_numbers = {a["article_number"] for a in articles}

    eval_set = build_eval_set_from_corpus(CORPUS_PATH, n=10)
    assert len(eval_set) == 10
    for item in eval_set:
        assert item["expected_article_number"] in real_numbers
        assert item["question"]  # non-empty, derived from real text


def test_build_eval_set_is_deterministic_with_fixed_seed():
    set1 = build_eval_set_from_corpus(CORPUS_PATH, n=5, seed=42)
    set2 = build_eval_set_from_corpus(CORPUS_PATH, n=5, seed=42)
    assert set1 == set2


def test_build_eval_set_skips_articles_with_empty_text_ar():
    # article 1022 (Day 1, documented gap) must never be picked, since a
    # snippet cannot be built from empty text without fabricating one
    eval_set = build_eval_set_from_corpus(CORPUS_PATH, n=200)
    assert all(item["expected_article_number"] != 1022 for item in eval_set)


def test_save_and_load_eval_set_round_trip():
    eval_set = build_eval_set_from_corpus(CORPUS_PATH, n=3)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "eval.json"
        save_eval_set(eval_set, path)
        loaded = load_eval_set(path)
    assert loaded == eval_set


def test_load_or_build_eval_set_builds_once_then_reuses_file():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "eval.json"
        first = load_or_build_eval_set(CORPUS_PATH, path)
        assert path.exists()
        second = load_or_build_eval_set(CORPUS_PATH, path)
        assert first == second
