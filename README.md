# Egyptian Legal RAG
### Arabic Legal Document Question Answering with Retrieval-Augmented Generation

An end-to-end Retrieval-Augmented Generation (RAG) system for exploring the Egyptian Civil Code through source-grounded question answering. The project combines Arabic document processing, article-aware retrieval, vector search, language-model generation, and a FastAPI service in a modular Python architecture.

## Project Overview

Legal question answering requires relevant source passages, clear citations, and a traceable path from a question to the underlying document. This project builds that path by transforming Civil Code source material into structured article records, indexing the content, retrieving relevant provisions, and using a language model to produce a response grounded in the retrieved context.

## Highlights

- **Arabic legal document ingestion:** PDF-to-JSON processing with article-level structure and validation.
- **Bilingual content handling:** Preserves Arabic and English fields when available in the source.
- **Article-aware chunking:** Organizes legal text into searchable chunks with source metadata.
- **Flexible retrieval backends:** Supports multilingual sentence-transformer embeddings and a TF-IDF lexical fallback.
- **Persistent vector storage:** Supports ChromaDB and a NumPy-based vector-store implementation.
- **LLM integration:** Connects retrieved legal context to a Groq-hosted language model.
- **Source-aware responses:** Returns retrieved legal passages and citation metadata with API responses.
- **FastAPI service:** Provides health-check and question-answering endpoints.
- **Request validation:** Handles input validation and structured API error responses.
- **Privacy-conscious observability:** Uses structured request-event logging with an explicit field allowlist.
- **Evaluation tooling:** Includes retrieval metrics, evaluation datasets, experiment-runner interfaces, and RAGAS integration.
- **Automated quality checks:** Includes tests, corpus validation, configuration checks, and a custom test runner.
- **Deployment and workflow configuration:** Includes Docker, BentoML, Locust, DVC, and GitHub Actions components.

## Architecture

```text
Civil Code PDF
     |
     v
PDF Ingestion and Validation
     |
     v
Structured Article JSON
     |
     v
Article-aware Chunking
     |
     v
Embedding Generation
     |
     v
Persistent Vector Store
     |
     v
Question -> Retrieval of Relevant Provisions
     |
     v
Retrieved Legal Context
     |
     v
Groq Language Model
     |
     v
Answer with Supporting Sources

Supporting modules:
FastAPI | Logging | Monitoring | Evaluation | Tests | CI
````

## Technology Stack

| Area                   | Technologies                         |
| ---------------------- | ------------------------------------ |
| Language               | Python 3.10+                         |
| Document processing    | pdfplumber, pypdf                    |
| Data processing        | NumPy, scikit-learn                  |
| Embeddings             | sentence-transformers, TF-IDF        |
| Vector storage         | ChromaDB, NumPy-based store          |
| API                    | FastAPI, Pydantic, Uvicorn           |
| Generation             | Groq API                             |
| Testing                | pytest, custom test runner           |
| Code quality           | Ruff                                 |
| Experiment tracking    | MLflow integration                   |
| Evaluation             | Retrieval metrics, RAGAS integration |
| Observability          | JSONL logging, optional Langfuse     |
| Deployment             | Docker, BentoML                      |
| Data pipelines         | DVC configuration                    |
| Continuous integration | GitHub Actions                       |

## Repository Structure

```text
.
├── src/
│   ├── ingestion/       # PDF ingestion and corpus validation
│   ├── rag/             # Chunking, embeddings, retrieval, generation
│   ├── api/             # FastAPI endpoints and request validation
│   ├── experiments/     # Evaluation and experiment tracking
│   ├── observability/   # Structured logging and monitoring
│   └── serving/         # BentoML integration
├── data/
│   ├── raw/             # Source documents
│   ├── processed/       # Structured legal corpus
│   ├── eval/            # Evaluation datasets
│   └── vector_store/    # Persistent retrieval index
├── tests/               # Automated tests
├── scripts/             # Quality and evaluation scripts
├── reports/             # Project reports
├── .github/workflows/   # CI configuration
├── Dockerfile
├── dvc.yaml
├── pyproject.toml
├── README.md
└── README_OLD_REFERENCE.md
```

## Getting Started

### 1. Requirements

* Python 3.10 or later.
* The project's source PDF and/or processed corpus.
* A Groq API key for hosted language-model generation.
* Internet connectivity for dependency installation and hosted API requests.

### 2. Create a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the API and retrieval extras:

```bash
python -m pip install -e ".[api,rag]"
```

Install the test dependencies:

```bash
python -m pip install -e ".[dev]"
```

### 4. Configure the Groq API key

Set the key in the current terminal session. Do not commit API credentials to source control.

**Windows PowerShell**

```powershell
$env:GROQ_API_KEY = Read-Host "Enter your Groq API key"
```

**Linux / macOS**

```bash
export GROQ_API_KEY="YOUR_GROQ_API_KEY"
```

Use the generation and model environment-variable names expected by the current implementation. Refer to `src/rag/generation.py` and `src/api/main.py` for the exact configuration.

### 5. Prepare the retrieval index

If the processed corpus is available and you intentionally need to create an index, use:

```powershell
python -c "from src.rag.vector_store import build_index_from_json; print(build_index_from_json('data/processed/civil_code.json', 'data/vector_store'))"
```

The vector index is a generated artifact. Preserve an existing working index unless you specifically intend to rebuild it.

### 6. Start the API

```powershell
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001
```

Open the interactive API documentation in a browser:

```text
http://127.0.0.1:8001/docs
```

Check service health:

```powershell
Invoke-RestMethod http://127.0.0.1:8001/health
```

### 7. Ask a legal question

```powershell
$body = @{
    question = "ما هو عقد البيع؟"
} | ConvertTo-Json -Compress

Invoke-RestMethod `
    -Uri "http://127.0.0.1:8001/ask" `
    -Method Post `
    -ContentType "application/json; charset=utf-8" `
    -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

The response contains an answer and the retrieved source provisions, according to the API response schema.

## Testing

Run the standard test suite:

```powershell
python -m pytest -q
```

Run the project's custom test runner:

```powershell
python run_tests.py
```

Check the corpus against its documented quality baseline:

```powershell
python scripts/check_corpus_quality.py
```

Record the command output and test environment when reporting test results.

## Evaluation and Observability

The project includes retrieval evaluation utilities, an evaluation dataset, and experiment-runner interfaces for comparing retrieval configurations. The observability module records structured request events and provides retrieval monitoring summaries.

Optional integrations include MLflow for experiment tracking, RAGAS for generated-answer evaluation, and Langfuse for tracing. Each integration can be configured separately according to its dependencies and service requirements.

## Deployment and Engineering Workflow

The repository includes configuration and implementation components for:

* Docker-based packaging.
* BentoML serving.
* Locust load testing.
* DVC pipeline stages.
* GitHub Actions continuous integration.
* Automated corpus and configuration checks.

See the corresponding project files and reports for component-specific commands and details.

## Documentation and Reports

For additional implementation details, consult:

* `WINDOWS_SETUP.md` — Windows setup guidance.
* `PROJECT_STRUCTURE.txt` — repository inventory.
* `PROGRESS.md` — development history.
* `REMAINING_WORK.md` — project work tracking.
* `reports/PROJECT_HANDOVER.md` — project handover.
* `reports/requirement_matrix.md` — requirement mapping.
* `README_OLD_REFERENCE.md` — preserved previous README and development log.

## Project Goals

This project brings together document engineering, Arabic text processing, retrieval-augmented generation, API development, evaluation tooling, and deployment configuration in one modular legal-document QA pipeline.
