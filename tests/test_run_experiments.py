from src.experiments.run_experiments import CONFIGURATIONS


def test_at_least_5_configurations_defined():
    assert len(CONFIGURATIONS) >= 5


def test_configurations_are_genuinely_different():
    # every config must differ from every other in at least one field
    # (chunk_size / top_k / tfidf params) -- guards against accidentally
    # "running the same config 5 times" to pad the count.
    seen = set()
    for c in CONFIGURATIONS:
        key = (c["chunk_size"], c["top_k"], c["tfidf_max_features"], c["tfidf_ngram_range"])
        assert key not in seen, f"duplicate configuration: {c['run_name']}"
        seen.add(key)


def test_configurations_have_required_fields():
    required = {"run_name", "chunk_size", "top_k", "tfidf_max_features", "tfidf_ngram_range"}
    for c in CONFIGURATIONS:
        assert required.issubset(c.keys())


def test_configurations_vary_chunk_size():
    sizes = {c["chunk_size"] for c in CONFIGURATIONS}
    assert len(sizes) > 1


def test_configurations_vary_top_k_or_embedding_config():
    top_ks = {c["top_k"] for c in CONFIGURATIONS}
    ngram_ranges = {c["tfidf_ngram_range"] for c in CONFIGURATIONS}
    assert len(top_ks) > 1 or len(ngram_ranges) > 1
