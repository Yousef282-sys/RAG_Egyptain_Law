import numpy as np

from src.rag.embeddings import TfidfFallbackEmbedder, get_embedder

# Synthetic text fixtures only.


def test_tfidf_embed_shape():
    embedder = TfidfFallbackEmbedder(max_features=32)
    texts = ["نص تجريبي اول", "نص تجريبي ثاني مختلف", "جملة عربية اخرى"]
    vectors = embedder.embed(texts)
    assert vectors.shape[0] == 3
    assert vectors.shape[1] <= 32
    assert vectors.dtype == np.dtype("float32")


def test_tfidf_embed_is_deterministic_after_fit():
    embedder = TfidfFallbackEmbedder(max_features=32)
    texts = ["نص تجريبي اول", "نص تجريبي ثاني"]
    embedder.fit(texts)
    v1 = embedder.embed(texts)
    v2 = embedder.embed(texts)
    assert np.allclose(v1, v2)


def test_tfidf_similar_texts_have_higher_similarity_than_unrelated():
    embedder = TfidfFallbackEmbedder(max_features=64)
    corpus = [
        "عقد البيع بين البائع والمشتري",
        "عقد البيع والمشتري والثمن",
        "قواعد كرة القدم والحكم",
    ]
    embedder.fit(corpus)
    vectors = embedder.embed(corpus)

    def cosine(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    sim_related = cosine(vectors[0], vectors[1])
    sim_unrelated = cosine(vectors[0], vectors[2])
    assert sim_related > sim_unrelated


def test_tfidf_save_and_load_round_trip(tmp_path=None):
    import tempfile
    from pathlib import Path

    embedder = TfidfFallbackEmbedder(max_features=32)
    corpus = ["نص تجريبي اول", "نص تجريبي ثاني"]
    embedder.fit(corpus)
    v_before = embedder.embed(["نص تجريبي اول"])

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "vectorizer.pkl"
        embedder.save(path)
        loaded = TfidfFallbackEmbedder.load(path)
        v_after = loaded.embed(["نص تجريبي اول"])

    assert np.allclose(v_before, v_after)


def test_get_embedder_returns_something_with_embed_method():
    # in this sandbox (no sentence-transformers, no network) this must
    # fall back to TfidfFallbackEmbedder without raising.
    embedder = get_embedder()
    assert hasattr(embedder, "embed")
    vectors = embedder.embed(["نص تجريبي"])
    assert vectors.shape[0] == 1
