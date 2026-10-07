# Arabic Legal RAG — Complete Project Handover & File Structure Report

Generated as a **reporting-only** task. No project files were modified,
moved, or deleted, and no commit was created while producing this
report. All facts below were gathered by directly inspecting the actual
working tree in this session — nothing is assumed or invented.

---

**Addendum (Day 10 portability audit, added after this report was
originally written — original findings below are preserved
unchanged):** two genuine packaging gaps were found and fixed:
(1) `src/observability/langfuse_adapter.py` imports `langfuse`, but no
`langfuse` extra existed in `pyproject.toml` — added
(`langfuse = ["langfuse>=2.0"]`). (2) `README.md`'s Day 1 section
claimed the source PDF was "not included" — stale; it has been included
since this project's Day 6 recovery. Both are now fixed. See
`AI_CONTEXT.md` and `REMAINING_WORK.md` (new, Day 10) for the
model-agnostic briefing and unfinished-work checklist this addendum
supplements.

---

## 1. Complete Project Directory Tree

(`.git`, `__pycache__`, `.pytest_cache`, `venv`/`.venv` excluded, per
your instructions — this working tree currently has no `venv`/`.venv`
directory at all.)

```text
arabic-legal-rag/
├── .github/
│   └── workflows/
│       └── ci.yml
├── .dockerignore
├── .gitignore
├── Dockerfile
├── PROGRESS.md
├── README.md
├── bentofile.yaml
├── data/
│   ├── eval/
│   │   └── retrieval_eval.json
│   ├── processed/
│   │   ├── .gitkeep
│   │   └── civil_code.json
│   ├── raw/
│   │   ├── .gitkeep
│   │   └── civil_code.pdf
│   └── vector_store/
│       └── manifest.json
├── dvc.yaml
├── pyproject.toml
├── reports/
│   ├── .gitkeep
│   ├── day1_spot_check.md
│   ├── day4_experiment_results.json
│   ├── day9_report.md
│   ├── requirement_matrix.md
│   └── PROJECT_HANDOVER.md        (this file)
├── run_tests.py
├── scripts/
│   ├── check_corpus_quality.py
│   └── ragas_gate.py
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── validation.py
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── eval_set.py
│   │   ├── evaluate.py
│   │   ├── ragas_eval.py
│   │   ├── run_experiments.py
│   │   └── tracking.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── arabic_utils.py
│   │   ├── bidi_fix.py
│   │   ├── cli.py
│   │   ├── parser.py
│   │   ├── pdf_reader.py
│   │   ├── schema.py
│   │   └── validation.py
│   ├── observability/
│   │   ├── __init__.py
│   │   ├── langfuse_adapter.py
│   │   ├── logger.py
│   │   └── monitoring.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   └── serving/
│       ├── __init__.py
│       └── bento_service.py
└── tests/
    ├── __init__.py
    ├── load/
    │   └── locustfile.py
    ├── test_api.py
    ├── test_api_validation.py
    ├── test_arabic_utils.py
    ├── test_bentoml_service.py
    ├── test_bidi_fix.py
    ├── test_chunking.py
    ├── test_ci_workflow.py
    ├── test_docker_config.py
    ├── test_dvc_pipeline.py
    ├── test_embeddings.py
    ├── test_eval_set.py
    ├── test_evaluate.py
    ├── test_integration_pipeline.py
    ├── test_langfuse_adapter.py
    ├── test_load_locustfile.py
    ├── test_mlflow_tracking.py
    ├── test_observability_logger.py
    ├── test_observability_monitoring.py
    ├── test_parser.py
    ├── test_quality_gates.py
    ├── test_ragas_eval.py
    ├── test_retriever.py
    ├── test_run_experiments.py
    ├── test_validation.py
    └── test_vector_store.py
```
**Note:** there is no `requirements.txt` in this project — dependencies
are declared only in `pyproject.toml` (see Section 4). The example tree
in your request showed `requirements.txt`; this project does not have
one, and I have not invented one.

---

## 2. File-by-File Inventory

### Root / config files
| File | Purpose | Essential for | Tested? |
|---|---|---|---|
| `pyproject.toml` | Package metadata, base deps (`pdfplumber`, `pypdf`, `numpy`, `scikit-learn`), 9 optional extras (`dev`, `rag`, `api`, `serving`, `load`, `experiments`, `lint`, `dvc`, `ragas`), `ruff`/`pytest` config | Execution, configuration | Parsed with `tomllib` repeatedly; install itself never run (no network) |
| `run_tests.py` | Bespoke test runner: imports every `tests/test_*.py`, calls every `test_*` function, treats a returned `"SKIPPED"` string as a skip (not a pass/fail), prints a final `N passed, N failed, N skipped` line | Testing | Run dozens of times this project; this is the primary verification tool used throughout |
| `Dockerfile` | `python:3.11-slim` base; copies `pyproject.toml` + `src/` + `data/processed/`; installs `.[api,rag]`; `EXPOSE 8000`; `HEALTHCHECK` hits `/health`; `CMD` runs `uvicorn src.api.main:app` | Docker execution | Structurally tested (`tests/test_docker_config.py`); **never actually built** (no `docker` binary in the dev sandbox) |
| `.dockerignore` | Excludes `__pycache__`, `data/vector_store/`, `data/raw/*.pdf`, `reports/`, `.git/`, `tests/`, docs from the Docker build context | Docker | Same as above — structurally checked, build never run |
| `dvc.yaml` | One pipeline stage, `extract_corpus`: `deps` = source PDF + 7 ingestion modules; `cmd` = the real `src/ingestion/cli.py ingest` command; `outs` = `data/processed/civil_code.json`. Deliberately does NOT hard-gate on `validate`'s exit code (see comments in the file for why) | DVC pipeline | Structurally tested (`tests/test_dvc_pipeline.py`, including a real byte-for-byte reproducibility check of the underlying command); `dvc` itself never installed/run |
| `bentofile.yaml` | Minimal BentoML build config pointing at `src.serving.bento_service:ArabicLegalRAGService` | BentoML packaging | Never built — no `bentoml` package |
| `.github/workflows/ci.yml` | 3 jobs: `test` (lint+pytest+run_tests.py+corpus gate), `ragas-quality-gate` (fails closed by design), `dvc-pipeline` (fails closed — PDF absent from any fresh checkout) | CI/CD | YAML-parsed and structurally tested (`tests/test_ci_workflow.py`); **never executed on GitHub** (no remote configured) |
| `.gitignore` | Excludes `data/vector_store/*` (generated index), `__pycache__/`, `*.pyc`, `.pytest_cache/`, `data/raw/*.pdf` (the source PDF), `mlruns/`, `mlartifacts/` | Git hygiene | N/A |
| `README.md` | Full user-facing documentation, day-by-day architecture notes, every command this project's own development used | Documentation | N/A |
| `PROGRESS.md` | Running log of every day's work, including the Day 6→7 recovery incident, exact test counts and commit hashes at each stage | Documentation | N/A |

### `src/ingestion/` — PDF → structured JSON (Day 1)
| File | Purpose | Key functions/classes |
|---|---|---|
| `pdf_reader.py` | Extracts raw per-page text via `pdfplumber` (primary) or `pypdf` (fallback) | `extract_pages_text()` |
| `bidi_fix.py` | Fixes Arabic text that PDF extractors return in visual (reversed) order; preserves digit-group order | `fix_rtl_text()`, `fix_rtl_line()` |
| `arabic_utils.py` | Arabic-Indic digit normalization, article-number extraction (two heading formats), repeal-marker detection | `extract_article_number()`, `find_article_marker()`, `is_repealed()` |
| `parser.py` | Core state machine turning page text into `Article` records: hierarchy tracking (book/chapter/section), heading-vs-citation disambiguation, range-repeal-notice skipping | `parse_pages_to_articles()`, `_ArticleBuilder` |
| `schema.py` | `Article` dataclass (10 fields matching the JSON corpus schema) | `Article` |
| `validation.py` | Per-article and corpus-level validation: empty-text checks, duplicate/gap detection | `validate_corpus()`, `ValidationReport` |
| `cli.py` | `ingest`/`validate` subcommands tying the above together | `run_ingest()`, `run_validate()`, `main()` |

**Dependencies:** each module only depends on earlier ones in this
list, plus `pdfplumber`/`pypdf` (declared base deps). No circular
dependencies. **Tested:** extensively — `test_arabic_utils.py` (12),
`test_bidi_fix.py` (6), `test_parser.py` (20), `test_validation.py` (9)
— all pass for real, plus the corpus quality gate re-runs the real
pipeline's `validate` output against a known-good baseline.

### `src/rag/` — chunking → embeddings → vector store → retrieval (Day 2)
| File | Purpose | Key functions/classes |
|---|---|---|
| `chunking.py` | Article JSON → `Chunk` records; splits only the ~13/1095 articles over 1000 chars (sentence-boundary split), always preserving `article_number` | `chunk_articles()`, `Chunk` |
| `embeddings.py` | `SentenceTransformerEmbedder` (real path, needs `sentence-transformers`) and `TfidfFallbackEmbedder` (the path actually used everywhere in this project so far — scikit-learn, offline, lexical only) | `get_embedder()` |
| `vector_store.py` | `SimpleVectorStore` (numpy cosine similarity — the path actually used) and `ChromaVectorStore` (real path, needs `chromadb`, never executed) | `build_index_from_json()`, `SimpleVectorStore`, `ChromaVectorStore` |
| `retriever.py` | `Retriever.retrieve(question, top_k)` → list of dicts with `article_number`, `text_ar`, `text_en`, `citation`, `score` | `Retriever` |

**Dependencies:** `chunking.py` ← `schema`-shaped dicts from the
ingestion JSON; `vector_store.py` ← `chunking.Chunk` + `embeddings`;
`retriever.py` ← `vector_store`. **Tested:** `test_chunking.py` (6),
`test_embeddings.py` (5), `test_vector_store.py` (4), `test_retriever.py`
(5) — all real, plus `test_integration_pipeline.py` runs the entire
chain against the actual 1095-article corpus.

### `src/api/` — FastAPI service (Days 2–3, 7)
| File | Purpose |
|---|---|
| `main.py` | `GET /health` (process-liveness only), `POST /ask` (validates via `AskRequest`, calls `Retriever` through a lazy singleton, returns `{answer, sources}`, logs one structured event per request, returns 503 if the index can't load / 500 on a retrieval exception) |
| `validation.py` | `validate_top_k()`, `validate_question()` — **deliberately has no fastapi/pydantic import**, so its logic is unit-testable in this sandbox even though fastapi itself is never installed here |

**Dependencies:** `main.py` imports `fastapi`, `pydantic` (both
uninstalled here), plus `src.api.validation`, `src.observability.logger`,
`src.rag.retriever`. **Tested:** `validation.py`'s logic — yes, for
real, 11/11 tests (`test_api_validation.py`). `main.py`'s actual HTTP
behavior — **no**, `tests/test_api.py` (12 tests) all honestly skip
because `fastapi` is not installed.

### `src/observability/` — logging, monitoring, optional tracing (Day 8)
| File | Purpose |
|---|---|
| `logger.py` | `build_event(**fields)` enforces an explicit allowlist (`ALLOWED_EVENT_FIELDS`) and raises `ValueError` on anything else — the question text itself can never be logged. `log_event()` appends one JSON line, never raises on I/O failure |
| `monitoring.py` | `summarize_retrieval()` computes request count, empty-result rate, latency stats from real recorded events only; `print_summary()` is a CLI entry point |
| `langfuse_adapter.py` | `send_trace()` returns `{"sent": False, "reason": ...}` unless `langfuse` is installed AND credentials are set AND a real call succeeds — never fabricates a trace id |

**Tested:** `test_observability_logger.py` (10), `test_observability_monitoring.py`
(9), `test_langfuse_adapter.py` (5) — all real, all pass. Additionally
verified by direct manual execution (see `PROGRESS.md`'s Day 8 section
for the actual recorded output).

### `src/experiments/` — evaluation, MLflow, RAGAS interface (Day 4, 7)
| File | Purpose |
|---|---|
| `eval_set.py` | Builds a 15-question, deterministic (seed 42) self-retrieval eval set from the real corpus's own article text |
| `evaluate.py` | `evaluate_retriever()` — hit-rate@k, mean reciprocal rank (lexical, NOT semantic/faithfulness) |
| `run_experiments.py` | `CONFIGURATIONS` (5 genuinely different chunk-size/top-k/TF-IDF configs) + `run_all()` — actually builds and evaluates each locally; attempts MLflow logging, catches `MLflowUnavailable` |
| `tracking.py` | `log_experiment_run()`, `register_best_run_artifact()` — both raise `MLflowUnavailable` when `mlflow` isn't installed (true here) rather than faking a run/registration |
| `ragas_eval.py` | `build_ragas_records()` (reuses `Retriever`, needs a caller-supplied `generate_fn`) + `evaluate_faithfulness()` (raises `ImportError` without `ragas`) |

**Tested:** `test_eval_set.py` (5), `test_evaluate.py` (3),
`test_run_experiments.py` (5), `test_mlflow_tracking.py` (5, 1 skips —
needs real mlflow), `test_ragas_eval.py` (4, all real since ragas's
absence IS the condition under test) — all pass.
**Actually executed:** `run_experiments.py` was run for real this
project's Day 4/9 — see `reports/day4_experiment_results.json` for the
genuine, measured local results. **Never executed against real MLflow.**

### `src/serving/` — BentoML (Day 3)
| File | Purpose |
|---|---|
| `bento_service.py` | `ArabicLegalRAGService` class wrapping the same `Retriever` (no duplicated retrieval logic); async `ask()` + `health()` methods |

**Tested:** `test_bentoml_service.py` (3 tests; 1 real structural check
that it reuses `Retriever` rather than reimplementing it; 2 skip — need
`bentoml`). **Never served.**

### `scripts/`
| File | Purpose |
|---|---|
| `check_corpus_quality.py` | Deterministic CI-safe gate: compares `validate`'s JSON output against a hardcoded known-good baseline (1095 articles, duplicates `[1,2]`, 56 gaps, 1 documented empty-text issue) — fails only on genuine deviation |
| `ragas_gate.py` | CI-facing RAGAS gate: fails closed (exit 1) unless BOTH `ragas` is importable AND a `GENERATION_MODEL` env var is set — currently always exit 1 here, by design |

**Tested:** `test_quality_gates.py` (9, real, including a live run
against the actual current corpus). **Actually executed** — both
scripts run directly and produce real exit codes/output every session.

### `tests/load/locustfile.py`
Locust load-test definition targeting `POST /ask` with 5 real Arabic
legal questions, weighted tasks, `--users 50` is the documented target.
**Never executed** — no `locust` package, no running API to target.

---

## 3. Implementation and Execution Status Matrix

| Component | Relevant Files | Implemented? | Actually Executed? | Missing Dependencies | Next Action |
|---|---|---|---|---|---|
| 1. PDF ingestion | `src/ingestion/*.py` | Yes | **Yes** — real PDF, real output, MD5-verified | none | none — done |
| 2. JSON corpus | `data/processed/civil_code.json` | Yes | **Yes** — 1095 articles, re-verified this session | none | none — done |
| 3. Chunking | `src/rag/chunking.py` | Yes | **Yes** — 1107 chunks from real corpus | none | none — done |
| 4. Embeddings | `src/rag/embeddings.py` | Yes (both paths) | TF-IDF path: **Yes**. Semantic path: **No** | `sentence-transformers` + network | install extra `rag`, rebuild index |
| 5. Vector store | `src/rag/vector_store.py` | Yes (both paths) | numpy path: **Yes**. Chroma path: **No** | `chromadb` | install extra `rag` |
| 6. Retrieval | `src/rag/retriever.py` | Yes | **Yes** — real queries, real results | none | none — done |
| 7. FastAPI | `src/api/*.py` | Yes | Validation logic: **Yes**. HTTP server: **No** | `fastapi`, `pydantic`, `uvicorn`, `httpx` | install extra `api`, run `uvicorn` |
| 8. Docker | `Dockerfile`, `.dockerignore` | Yes | **No** | `docker` Desktop/Engine | `docker build`/`run` on Windows |
| 9. BentoML | `src/serving/bento_service.py`, `bentofile.yaml` | Yes | **No** | `bentoml` | install extra `serving` |
| 10. Locust load testing | `tests/load/locustfile.py` | Yes | **No — zero benchmark numbers exist** | `locust`, a running API | install extra `load`, run API first |
| 11. MLflow | `src/experiments/tracking.py`, `run_experiments.py` | Yes | Local eval: **Yes**. MLflow logging: **No** | `mlflow` | install extra `experiments` |
| 12. DVC | `dvc.yaml` | Yes | **No** | `dvc` AND a DVC remote (PDF is git-ignored) | install extra `dvc`; separately, configure a remote |
| 13. GitHub Actions / CI | `.github/workflows/ci.yml` | Yes | **No — never run on GitHub** | a GitHub remote + push | `git remote add origin ...; git push` |
| 14. RAGAS | `scripts/ragas_gate.py`, `src/experiments/ragas_eval.py` | Yes | **No — fails closed by design** | `ragas` AND a real generation backend (neither exists) | build a generation backend first; then install `ragas` |
| 15. Langfuse | `src/observability/langfuse_adapter.py` | Yes | **No** | `langfuse` + server + credentials | `pip install langfuse`; set env vars |
| 16. Monitoring | `src/observability/monitoring.py` | Yes | **Yes** — real local log, real summary computed | none | none — done (local only) |
| 17. Guardrails | `src/api/validation.py`, `src/observability/logger.py`'s allowlist | Yes | **Yes** — tested at logic layer | none (HTTP-layer enforcement untested, see #7) | none — done at logic layer |
| 18. Integration tests | `tests/test_integration_pipeline.py` | Yes | **Yes** — 5/5 real, no mocks | none | none — done |

---

## 4. Dependencies and Installation

**Only dependency file present:** `pyproject.toml` (no
`requirements.txt`, no `Pipfile`, no `poetry.lock`).

**Base dependencies (always installed):**
```
pdfplumber>=0.11.0
pypdf>=4.0.0
numpy>=1.24
scikit-learn>=1.3
```

**Optional extras (install only what you need):**
| Extra | Packages | Needed for |
|---|---|---|
| `dev` | `pytest>=8.0.0` | running tests via `pytest` instead of `run_tests.py` |
| `rag` | `sentence-transformers>=2.2`, `chromadb>=0.4` | real semantic embeddings + Chroma |
| `api` | `fastapi>=0.110`, `pydantic>=2.0`, `uvicorn>=0.27`, `httpx>=0.27` | running the API, `TestClient`-based tests |
| `serving` | `bentoml>=1.2` | BentoML serving |
| `load` | `locust>=2.20` | load testing |
| `experiments` | `mlflow>=2.10` | MLflow tracking/registry |
| `lint` | `ruff>=0.4` | linting |
| `dvc` | `dvc>=3.0` | DVC pipeline execution |
| `ragas` | `ragas>=0.1`, `datasets>=2.14` | RAGAS faithfulness (also needs a generation backend — see #14 above) |

**Known missing-dependency facts (not compatibility issues — simply
never installed in the dev sandbox):** none of the above extras, nor
`docker`, have ever been installed in this project's development
environment. No version-conflict or compatibility problem has been
observed, because nothing beyond the base 4 packages has ever actually
been installed and exercised together.

**Suggested Windows installation commands (not executed — proposed only):**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev,rag,api,serving,load,experiments,lint,dvc,ragas]"
```

---

## 5. Data and Large Files

- **Original PDF:** `data/raw/civil_code.pdf` (2,599,703 bytes). MD5
  `5086ef5f16f15d856278db943e58e948` — confirmed identical to your
  original upload.
- **Processed JSON:** `data/processed/civil_code.json` (1,345,184
  bytes, 1095 articles). MD5 `ac90c9c1001e961d7aa451ca6984c1c8`.
- **Both files exist in the current working tree right now** (verified
  by direct `ls`/`md5sum` this session). Whether they're inside any
  particular ZIP depends on which export you're using — the most recent
  verified one (`arabic-legal-rag-FINAL-day9.zip`, delivered in the
  previous turn) was confirmed to include both, hash-checked from inside
  the archive itself.
- **Git tracking status:** the PDF is **git-ignored**
  (`.gitignore`: `data/raw/*.pdf`) — it is NOT tracked by git. The JSON
  is **not** git-ignored — it IS a normal tracked file in the current
  repository (DVC has never actually run to take over its management;
  see Section 6).
- **What should be DVC-managed (once DVC is actually set up):** the raw
  PDF (currently has no `.dvc` pointer — `dvc add` has never been run)
  and, once `dvc repro` actually runs, the processed JSON as a pipeline
  output (`dvc.lock` would then be the source of truth for its hash).
  Neither exists yet in this repository.
- **What must NOT be committed to GitHub:** the raw PDF as a plain git
  blob (defeats the point of DVC tracking it, and it's already
  git-ignored); `data/vector_store/` (regenerable, already git-ignored);
  `__pycache__/`, `.pytest_cache/` (already git-ignored); `mlruns/`/
  `mlartifacts/` if MLflow is ever run locally (already git-ignored).
- **Hardcoded Linux-specific paths:** none found in any source file.
  The only `/tmp/...` or `/home/...`-shaped strings in the codebase are
  (a) a deliberately-nonexistent path used inside one test as a
  negative test case, and (b) two test assertions that explicitly check
  such strings are ABSENT from the CI workflow. Verified via
  `git grep` across all `.py`/`.yml`/`.toml` files.

---

## 6. Git and GitHub Status

- **Latest commit:** `21ec815` — "Update PROGRESS.md with Day 9 summary"
- **Working tree status:** clean (no uncommitted changes) as of this
  report, apart from this new report file itself, which has
  deliberately **not** been committed (per your "do not create a new
  commit" instruction).
- **Remote configured:** **No** (`git remote -v` returns nothing).
- **Ready to connect to GitHub:** structurally yes (a normal git repo
  with a sensible `.gitignore` and a CI workflow already in place) —
  but see the important caveat below.
- **Before the first push, check:**
  1. This repository's history starts from a recovery commit
     (`9ee3052`) after an earlier sandbox reset destroyed the original
     git history — see `PROGRESS.md`'s "RECOVERY NOTE" section for the
     full story. The old commit hashes referenced anywhere else (e.g.
     `fa30fb5`, prior `9611fa5`) do **not** exist in this repository and
     never will.
  2. Confirm `data/raw/civil_code.pdf` really is excluded before
     pushing (`git status` should never show it as staged) — already
     true, but worth re-checking after any local `git add .`.
  3. Decide whether you want the JSON corpus committed as a plain file
     (current state) or migrated to real DVC tracking first.
- **Secrets/credentials scan:** searched all tracked files for
  patterns like `api_key=`, `secret=`, `password=`, `token=`, AWS-style
  keys, and PEM headers. **None found.** The only matches were the
  *names* of environment variables this project reads (e.g.
  `LANGFUSE_SECRET_KEY` as a variable name, never a value) — no actual
  secret value is stored anywhere in the tree.

---

## 7. Windows Execution Plan

| # | Step | Command / File | Prerequisites | Expected result | Tested? |
|---|---|---|---|---|---|
| 1 | Extract the ZIP | Windows Explorer or `Expand-Archive` | the ZIP file | A folder with the tree in Section 1 | Not tested on Windows |
| 2 | Create a venv | `python -m venv .venv` then `.venv\Scripts\activate` | Python 3.10+ installed | An activated `(.venv)` prompt | Not tested on Windows |
| 3 | Install dependencies | `pip install -e ".[dev,rag,api,serving,load,experiments,lint,dvc,ragas]"` | step 2, internet access | All packages install without error | **Not tested anywhere** — this exact combined extras install has never been run, on any OS, in this project's history |
| 4 | Run the test suite | `pytest -q` or `python run_tests.py` | step 3 | Expect noticeably **fewer skips** than this sandbox's 18, since the real packages would now be present; exact Windows numbers are unknown until run | Only the Linux-sandbox numbers (165/0/18) are confirmed |
| 5 | Prepare the corpus | Already present at `data/processed/civil_code.json`; to rebuild from the PDF: `python -m src.ingestion.cli ingest --pdf data/raw/civil_code.pdf --out data/processed/civil_code.json` | step 3 | Re-running should reproduce the exact same file (MD5 `ac90c9c1001e961d7aa451ca6984c1c8`) — verified reproducible on Linux; not yet cross-checked on Windows | Linux: yes. Windows: no |
| 6 | Start the API | `uvicorn src.api.main:app --reload` | step 3, step 5, an index built (`python -c "from src.rag.vector_store import build_index_from_json; build_index_from_json('data/processed/civil_code.json','data/vector_store')"`) | `http://localhost:8000/health` returns `{"status":"ok"}` | **Never run anywhere, Linux or Windows** |
| 7 | Build and run Docker | `docker build -t arabic-legal-rag .` then `docker run -p 8000:8000 -v "%cd%\data\vector_store:/app/data/vector_store" arabic-legal-rag` | Docker Desktop for Windows (with WSL2 backend) running | Container starts; `/health` reachable | **Never built, anywhere** |
| 8 | Run MLflow | `python -m src.experiments.run_experiments` (logs to MLflow if installed) then `mlflow ui` | step 3 (`experiments` extra) | A real experiment named `arabic-legal-rag` with 5 runs appears in the MLflow UI | Local (non-MLflow) evaluation: yes, this sandbox. Actual MLflow logging: **never done** |
| 9 | Run DVC | `dvc init` (first time only) → `dvc add data/raw/civil_code.pdf` → `dvc repro` → `dvc status` | step 3 (`dvc` extra) | `dvc repro` rebuilds the JSON and reports it as reproducible | **Never run anywhere** — `dvc` itself has never been installed in this project |
| 10 | Run RAGAS | Requires building a real generation backend first (does not exist in this codebase yet), then `pip install -e ".[ragas]"` and extending `scripts/ragas_gate.py`'s real path per its own docstring | a generation backend + `ragas` + an LLM judge | A real faithfulness score vs. `RAGAS_FAITHFULNESS_THRESHOLD` | **Not reachable without new development** — this is not just a missing-package issue |
| 11 | Set up GitHub Actions | `git remote add origin <your-repo-url>`, `git push -u origin master`, then check the Actions tab | a GitHub account/repo, network access | The `test` job should pass; `ragas-quality-gate` and `dvc-pipeline` are **expected to fail** until steps 9–10 are genuinely completed (by design, documented in the workflow's own comments) | **Never run** |

---

## 8. Final Handover Summary

**Most important files to understand first, in order:**
1. `PROGRESS.md` — the full narrative, including the Day 6→7 recovery
   incident; read this before anything else to understand what's real
   vs. rebuilt.
2. `reports/requirement_matrix.md` and `reports/day9_report.md` — the
   most recent, most complete status snapshot.
3. `src/ingestion/cli.py`, `src/rag/retriever.py`, `src/api/main.py` —
   the three files that define the actual data flow.
4. `pyproject.toml` — what's installable and how.

**Correct order to run the project:**
install deps → run tests → build the vector index → start the API →
(optionally) Docker/MLflow/DVC/RAGAS, each independently, in any order,
once their specific prerequisites are met.

**Potential Linux→Windows compatibility issues (anticipated, not
observed, since nothing has run on Windows yet):**
- Path separators: the codebase consistently uses `pathlib.Path`, which
  handles this automatically — no raw string path-joining with `/` was
  found in source files besides `pathlib`-based code.
- `uvicorn`/`fastapi`/`chromadb`/`sentence-transformers` installs can be
  slower or require build tools on Windows (some have compiled
  dependencies) — not something this sandbox can predict without
  actually running it.
- Docker Desktop on Windows requires WSL2 or Hyper-V — a Windows-side
  setup step with no code-level equivalent to verify in advance.
- Shell syntax: Windows PowerShell/cmd differ from the bash commands
  used throughout this project's own development — Section 7 gives
  Windows-flavored equivalents, but they are proposed, not tested.

**What's still needed to demonstrate actual execution of Docker,
MLflow, DVC, and RAGAS:** for all four, the missing piece is simply
*running them for the first time, anywhere* — the code is written and
locally tested to the extent possible without the tool itself. RAGAS
additionally needs a generation backend that doesn't exist in this
codebase yet (a new piece of work, not just an install).

**Requirements that cannot be confirmed without the original course
handbook:** this project has never had an actual handbook document
provided to it at any point — every "the handbook requires X" statement
in this project's history came from your own day-by-day prompts, not
from a document I've read. I cannot confirm this project satisfies an
M1–M5 structure, specific point allocations, or any requirement not
explicitly stated to me in conversation. `reports/requirement_matrix.md`
is built from the cumulative stated requirements only, and says so
explicitly.
