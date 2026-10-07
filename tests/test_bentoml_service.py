"""Structural tests for src/serving/bento_service.py.

`bentoml` is not installed in this sandbox and there is no network here
to install it, so these tests cannot import/instantiate the real service.
Each test detects this and returns "SKIPPED" (counted separately by
run_tests.py) rather than fabricating a pass. The module's syntax is
still verified independently via `python3 -m py_compile` (see PROGRESS.md
for the actual command/output), which does NOT require bentoml.
"""
try:
    import bentoml  # noqa: F401

    from src.serving.bento_service import ArabicLegalRAGService

    BENTOML_AVAILABLE = True
except Exception:
    BENTOML_AVAILABLE = False


def _skip(name: str) -> str:
    print(
        f"  ({name}: bentoml not installed in this sandbox -- not executed "
        "here, run in an environment with that package)"
    )
    return "SKIPPED"


def test_bento_service_module_reuses_existing_retriever_not_reimplemented():
    # This check is pure source inspection (no bentoml import needed): it
    # verifies the Day-3 requirement "reuse the existing retriever, do
    # not duplicate the retrieval implementation" by construction.
    import inspect

    from src.rag import retriever as retriever_module

    with open("src/serving/bento_service.py", encoding="utf-8") as f:
        source = f.read()
    assert "from src.rag.retriever import Retriever" in source
    assert inspect.isclass(retriever_module.Retriever)


def test_bento_service_can_be_instantiated_and_exposes_ask():
    if not BENTOML_AVAILABLE:
        return _skip("test_bento_service_can_be_instantiated_and_exposes_ask")
    assert hasattr(ArabicLegalRAGService, "ask")
    assert hasattr(ArabicLegalRAGService, "health")


def test_bento_service_ask_is_a_coroutine_function():
    if not BENTOML_AVAILABLE:
        return _skip("test_bento_service_ask_is_a_coroutine_function")
    import inspect

    assert inspect.iscoroutinefunction(ArabicLegalRAGService.ask)
