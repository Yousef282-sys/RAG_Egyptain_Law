from src.api.validation import (
    DEFAULT_MAX_QUESTION_LENGTH,
    validate_question,
    validate_top_k,
)


def test_validate_top_k_accepts_positive_integer_strings():
    assert validate_top_k("5") == 5
    assert validate_top_k("1") == 1
    assert validate_top_k("100") == 100


def test_validate_top_k_rejects_zero():
    raised = False
    try:
        validate_top_k("0")
    except ValueError:
        raised = True
    assert raised


def test_validate_top_k_rejects_negative():
    raised = False
    try:
        validate_top_k("-3")
    except ValueError:
        raised = True
    assert raised


def test_validate_top_k_rejects_non_integer():
    raised = False
    try:
        validate_top_k("not-a-number")
    except ValueError:
        raised = True
    assert raised


def test_validate_question_accepts_normal_question():
    q = "ما هو عقد البيع؟"
    assert validate_question(q) == q


def test_validate_question_rejects_empty_string():
    raised = False
    try:
        validate_question("")
    except ValueError:
        raised = True
    assert raised


def test_validate_question_rejects_whitespace_only():
    raised = False
    try:
        validate_question("   \n\t  ")
    except ValueError:
        raised = True
    assert raised


def test_validate_question_rejects_none():
    raised = False
    try:
        validate_question(None)
    except ValueError:
        raised = True
    assert raised


def test_validate_question_accepts_up_to_max_length():
    q = "x" * DEFAULT_MAX_QUESTION_LENGTH
    assert validate_question(q) == q


def test_validate_question_rejects_over_max_length():
    q = "x" * (DEFAULT_MAX_QUESTION_LENGTH + 1)
    raised = False
    try:
        validate_question(q)
    except ValueError:
        raised = True
    assert raised


def test_validate_question_respects_custom_max_length():
    raised = False
    try:
        validate_question("hello world", max_length=5)
    except ValueError:
        raised = True
    assert raised
    assert validate_question("hi", max_length=5) == "hi"
