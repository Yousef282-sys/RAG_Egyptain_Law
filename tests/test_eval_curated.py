"""The curated evaluation set (>= 50 bilingual questions) is grounded in the
real corpus: every expected article exists once, every reference is the
corpus-derived extract (never hand-written), nothing duplicated.
"""
import copy
import json
import tempfile
from pathlib import Path

from src.experiments.curated_questions import CURATED_QUESTIONS
from src.experiments.eval_set import (
    MIN_EVAL_QUESTIONS,
    SELFRETRIEVAL_PATH,
    build_curated_eval_set,
    english_extract,
    load_curated_eval_set,
    load_eval_set,
    validate_eval_set,
)

CORPUS = "data/processed/civil_code.json"


def _items():
    return load_curated_eval_set()  # the committed file


def test_committed_eval_set_has_at_least_50_questions():
    assert len(_items()) >= MIN_EVAL_QUESTIONS == 50


def test_committed_eval_set_is_valid_against_the_real_corpus():
    assert validate_eval_set(_items(), CORPUS) == []


def test_committed_file_matches_what_the_builder_produces_from_the_corpus():
    # the JSON file is regenerable (python -m src.experiments.eval_set) and has no hand edits
    assert _items() == build_curated_eval_set(CORPUS)


def test_eval_set_is_bilingual_with_substantial_arabic_and_english():
    langs = [x["language"] for x in _items()]
    assert langs.count("ar") >= 20 and langs.count("en") >= 20


def test_every_item_has_reference_and_expected_articles_for_ragas_and_retrieval():
    for x in _items():
        assert x["reference"].strip() and x["expected_article_numbers"] == [x["expected_article_number"]]
        assert x["citation"].endswith(str(x["expected_article_number"]))


def test_expected_articles_are_unique_across_questions():
    numbers = [x["expected_article_number"] for x in _items()]
    assert len(set(numbers)) == len(numbers)
    assert len(CURATED_QUESTIONS) == len(numbers)


def test_references_are_verbatim_corpus_text_not_hand_written():
    articles = {a["article_number"]: a for a in json.load(open(CORPUS, encoding="utf-8"))}
    for x in _items():
        extract = english_extract(articles[x["expected_article_number"]])
        assert x["reference"] == extract
        # every reference word really occurs in the article's corpus text
        corpus_words = (articles[x["expected_article_number"]]["text_ar"] + " "
                        + (articles[x["expected_article_number"]]["text_en"] or "")).split()
        assert set(x["reference"].split()) <= set(corpus_words)


def test_validate_flags_undersized_duplicate_and_tampered_sets():
    good = _items()
    assert any("only" in p for p in validate_eval_set(good[:10], CORPUS))
    dup = copy.deepcopy(good)
    dup[1]["question"] = dup[0]["question"]
    assert any("duplicate question" in p for p in validate_eval_set(dup, CORPUS))
    bad_ref = copy.deepcopy(good)
    bad_ref[0]["reference"] = "An invented legal rule."
    assert any("not the corpus extract" in p for p in validate_eval_set(bad_ref, CORPUS))
    bad_art = copy.deepcopy(good)
    bad_art[0]["expected_article_numbers"] = [99999]
    assert any("not found" in p for p in validate_eval_set(bad_art, CORPUS))
    wrong_lang = copy.deepcopy(good)
    wrong_lang[0]["language"] = "ar"
    assert any("language label" in p for p in validate_eval_set(wrong_lang, CORPUS))


def test_load_curated_never_builds_or_substitutes_another_set():
    with tempfile.TemporaryDirectory() as d:
        missing = Path(d) / "nope.json"
        try:
            load_curated_eval_set(missing)
        except FileNotFoundError:
            assert not missing.exists()  # nothing was created
        else:
            raise AssertionError("expected FileNotFoundError")
        small = Path(d) / "small.json"
        small.write_text(json.dumps(_items()[:5], ensure_ascii=False), encoding="utf-8")
        try:
            load_curated_eval_set(small, corpus_json_path=CORPUS)
        except ValueError:
            return
    raise AssertionError("expected ValueError for undersized set")


def test_original_selfretrieval_snippet_set_is_preserved_separately():
    old = load_eval_set(SELFRETRIEVAL_PATH)
    assert len(old) == 15 and "reference" not in old[0]
