"""Semantic embedding configuration (Phase 2). The sentence-transformers
package is replaced by an in-memory stand-in (tests/fakes.py): these tests
verify our configuration/prefix/strictness logic, NOT model quality.
"""
import os
import warnings

from src.rag.embeddings import (
    EmbeddingBackendUnavailable,
    SentenceTransformerEmbedder,
    TfidfFallbackEmbedder,
    default_prefixes,
    embedder_from_manifest,
    get_embedder,
    resolve_backend,
)
from tests.fakes import fake_sentence_transformers, installed


def test_semantic_and_tfidf_are_labelled_distinctly():
    assert TfidfFallbackEmbedder.kind == "tfidf-lexical"
    assert SentenceTransformerEmbedder.kind == "semantic"
    assert TfidfFallbackEmbedder().describe()["embedding_kind"] == "tfidf-lexical"


def test_e5_models_get_query_passage_prefixes_others_do_not():
    assert default_prefixes("intfloat/multilingual-e5-small") == {
        "query_prefix": "query: ", "passage_prefix": "passage: "}
    assert default_prefixes("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") == {
        "query_prefix": "", "passage_prefix": ""}


def test_prefixes_applied_to_documents_and_queries_and_normalize_flag_passed():
    st = fake_sentence_transformers()
    with installed(st):
        e = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        e.embed_documents(["doc one", "doc two"])
        e.embed_query("what?")
        calls = st.instances[0].calls
    assert calls[0]["texts"] == ["passage: doc one", "passage: doc two"]
    assert calls[1]["texts"] == ["query: what?"]
    assert calls[0]["normalize"] is True


def test_describe_records_reproducible_configuration():
    st = fake_sentence_transformers()
    with installed(st):
        e = SentenceTransformerEmbedder("some/model", revision="abc123")
        d = e.describe()
        assert st.instances[0].kwargs == {"revision": "abc123"}  # pinned revision forwarded
    assert d["embedding_kind"] == "semantic" and d["embedder_model"] == "some/model"
    assert d["model_revision"] == "abc123" and d["normalize_embeddings"] is True
    assert set(d) >= {"query_prefix", "passage_prefix", "embedder_class"}


def test_from_manifest_recreates_identical_configuration():
    st = fake_sentence_transformers()
    with installed(st):
        original = SentenceTransformerEmbedder("intfloat/multilingual-e5-small")
        rebuilt = embedder_from_manifest(original.describe())
        assert rebuilt.describe() == original.describe()


def test_model_name_comes_from_env_default():
    # Fresh interpreter (never reload modules in-process: that would
    # create duplicate classes and break isinstance checks elsewhere).
    import subprocess
    import sys

    env = dict(os.environ, EMBEDDING_MODEL="my/custom-model")
    out = subprocess.run(
        [sys.executable, "-c", "import src.rag.embeddings as e; print(e.DEFAULT_MODEL_NAME)"],
        capture_output=True, text=True, env=env, check=True,
    ).stdout.strip()
    assert out == "my/custom-model"


def test_strict_backend_raises_instead_of_falling_back_to_tfidf():
    st = fake_sentence_transformers(fail_with=OSError("offline"))
    with installed(st):
        try:
            get_embedder("any/model", backend="sentence-transformers")
        except EmbeddingBackendUnavailable as exc:
            assert "offline" in str(exc)
            return
    raise AssertionError("strict backend must raise, not return TF-IDF")


def test_auto_backend_falls_back_loudly_never_silently():
    st = fake_sentence_transformers(fail_with=OSError("offline"))
    with installed(st), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        e = get_embedder("any/model", backend="auto")
    assert isinstance(e, TfidfFallbackEmbedder)
    assert any("LEXICAL" in str(w.message) for w in caught)


def test_auto_backend_returns_semantic_when_available():
    with installed(fake_sentence_transformers()):
        assert isinstance(get_embedder("m", backend="auto"), SentenceTransformerEmbedder)


def test_tfidf_backend_explicit_and_invalid_backend_rejected():
    assert isinstance(get_embedder(backend="tfidf"), TfidfFallbackEmbedder)
    try:
        resolve_backend("bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_embedding_backend_env_var_is_honoured():
    saved = os.environ.get("EMBEDDING_BACKEND")
    try:
        os.environ["EMBEDDING_BACKEND"] = "tfidf"
        assert isinstance(get_embedder(), TfidfFallbackEmbedder)
    finally:
        if saved is None:
            os.environ.pop("EMBEDDING_BACKEND", None)
        else:
            os.environ["EMBEDDING_BACKEND"] = saved
