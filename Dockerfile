# Arabic Legal RAG -- API image.
#
# NOT BUILT OR RUN in the development sandbox that authored this file: no
# `docker` binary is available there (confirmed: `docker --version` ->
# "not found") and no network access to pull a base image either. This
# Dockerfile is standard and written to be correct, but its build/run has
# NOT been verified here -- see PROGRESS.md / README for exact commands
# to test it for real once Docker is available.

FROM python:3.11-slim

WORKDIR /app

# System deps for pdfplumber/pypdf (Day 1) stay minimal; no extra system
# packages are required by the Day 2/3 RAG or API layers themselves.
COPY pyproject.toml ./
COPY src ./src

# Install the project plus the extras needed to actually run the API
# (fastapi/pydantic/uvicorn) and the real RAG backend
# (sentence-transformers/chromadb). Split into two layers so dependency
# installs cache independently of source changes below.
RUN pip install --no-cache-dir ".[api,rag]"

# The vector index (data/vector_store/) is a generated artifact and is
# intentionally NOT baked into the image -- mount it at runtime (see
# README "Docker usage") so the image stays small and the index can be
# rebuilt/updated without rebuilding the image.
COPY data/processed ./data/processed

EXPOSE 8000

# Fails fast with a clear container-level signal if the process itself
# stops responding; does not require the vector index to be present
# (health is intentionally index-independent, see src/api/main.py).
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
