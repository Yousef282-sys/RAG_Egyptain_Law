"""Tests for Dockerfile / .dockerignore.

These are pure file-content checks -- they do NOT build or run the image.
This sandbox has no `docker` binary at all (confirmed: `docker --version`
-> "not found"), so the actual build/run/health/ask verification could
not be performed here. See PROGRESS.md / README for the exact commands
to run for real once Docker is available, and for the explicit statement
that this was never claimed to have been tested.
"""
from pathlib import Path

DOCKERFILE = Path("Dockerfile")
DOCKERIGNORE = Path(".dockerignore")


def test_dockerfile_exists():
    assert DOCKERFILE.exists()


def test_dockerfile_uses_uvicorn_to_start_the_api():
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "uvicorn" in content
    assert "src.api.main:app" in content


def test_dockerfile_exposes_a_port():
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "EXPOSE" in content


def test_dockerfile_has_a_healthcheck_hitting_health_endpoint():
    content = DOCKERFILE.read_text(encoding="utf-8")
    assert "HEALTHCHECK" in content
    assert "/health" in content


def test_dockerfile_does_not_bake_in_generated_vector_store():
    content = DOCKERFILE.read_text(encoding="utf-8")
    # the Dockerfile's comments legitimately mention the path when
    # explaining why it's excluded; what must never appear is a COPY of it
    assert "COPY data/vector_store" not in content


def test_dockerignore_excludes_generated_vector_store_and_raw_pdf():
    content = DOCKERIGNORE.read_text(encoding="utf-8")
    assert "data/vector_store" in content
    assert "data/raw" in content


def test_compose_defines_api_indexer_and_optional_vllm():
    # Structure check only -- compose was NOT run (no docker in sandbox).
    import yaml  # PyYAML is already used by tests/test_ci_workflow.py

    compose = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))
    services = compose["services"]
    assert {"api", "indexer", "vllm"} <= set(services)
    assert "gpu" in services["vllm"]["profiles"]  # GPU service is opt-in
    env = services["api"]["environment"]
    assert "GENERATION_MODEL" in env and "VLLM_BASE_URL" in env
    assert "8000:8000" in services["api"]["ports"]
