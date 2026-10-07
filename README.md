# Arabic Legal Document Q&A / RAG — Track 2

## Project overview

### Purpose
This project builds a Retrieval-Augmented-Generation (RAG) pipeline for
answering questions about the **Egyptian Civil Code**, a real bilingual
(Arabic/English) legal document. It ingests the source PDF into
structured, per-article JSON, indexes it for retrieval, and exposes it
through a FastAPI `/ask` endpoint — currently **retrieval-only** (no
generation model is wired in yet; see Known Limitations).

### Architecture overview
```
PDF (data/raw/civil_code.pdf)
  → src/ingestion/   → data/processed/civil_code.json (1095 articles)
  → src/rag/chunking.py → chunks
  → src/rag/embeddings.py → vectors (TF-IDF fallback in use; real
    sentence-transformer path exists but untested here)
  → src/rag/vector_store.py → persisted index (numpy fallback in use;
    Chroma path exists but untested here)
  → src/rag/retriever.py → top-k results
  → src/api/main.py → FastAPI GET /health, POST /ask
  → src/observability/ → structured local logging + monitoring + optional Langfuse
  → src/experiments/ → evaluation + MLflow tracking interface + RAGAS interface
  → scripts/, .github/workflows/, Dockerfile, dvc.yaml, bentofile.yaml →
    quality gates, CI, Docker, DVC, BentoML packaging
```

### Features
- Reproducible, byte-for-byte deterministic PDF→JSON ingestion with
  Arabic bidi-text correction, hierarchy tracking, and a validation gate.
- Article-level chunking, embeddings, and local vector search.
- A FastAPI service with input validation, structured logging, and
  honest 503/500 error handling.
- CI workflow, Docker packaging, DVC pipeline definition, MLflow
  experiment-tracking interface, and a fail-closed RAGAS quality-gate
  interface — all implemented; see `reports/PROJECT_HANDOVER.md` and
  `reports/requirement_matrix.md` for exactly what has and hasn't been
  executed.

### Repository structure
See `PROJECT_STRUCTURE.txt` for the full tree, or
`reports/PROJECT_HANDOVER.md` for a file-by-file inventory.

### Prerequisites
- Python 3.10+ (3.11 recommended — this is what CI uses).
- No external services are required to run the core pipeline and test
  suite offline. Docker, MLflow, DVC, RAGAS, and Langfuse each need
  their own real installation/service — see their sections below and
  `REMAINING_WORK.md`.

### Installation
```bash
pip install -e .                  # base: ingestion + core RAG only
pip install -e ".[dev]"           # + pytest
pip install -e ".[dev,rag,api,serving,load,experiments,lint,dvc,ragas,langfuse]"  # everything
```
**Verified in this project's own development environment:** only the
base install (no extras) has ever actually been exercised — the
combined-extras install above is correct per `pyproject.toml` but has
never been run end-to-end anywhere (see `WINDOWS_SETUP.md` for the
tested/proposed distinction on every command).

### Windows setup
See `WINDOWS_SETUP.md` for copy-paste PowerShell commands.

### Linux / macOS setup
The same commands work with `bash`/`zsh` syntax: `python3 -m venv .venv
&& source .venv/bin/activate` instead of the PowerShell activation line,
`$(pwd)` instead of `%cd%` for Docker volume mounts. Everything else
(pip install, pytest, uvicorn, etc.) is identical. This project's own
development and testing has only ever run on Linux.

### Running tests
```bash
pytest -q              # standard
python3 run_tests.py   # bespoke runner used throughout this project's
                        # own development (works even without pytest
                        # installed; treats a "SKIPPED" return as a
                        # skip, not a pass)
```
Current result (this environment, no optional packages installed):
**165 passed, 0 failed, 18 skipped**. See `reports/day9_report.md` for
the exact skip reasons.

### Running the API
```bash
python -c "from src.rag.vector_store import build_index_from_json; build_index_from_json('data/processed/civil_code.json', 'data/vector_store')"
uvicorn src.api.main:app --reload
```
Requires the `api` extra. **Never run in this project's own development
sandbox** (no `fastapi`/`uvicorn` installed there) — see the Day 2/3
sections below for the full design and `WINDOWS_SETUP.md` for the
tested/proposed status of each command.

### Docker / MLflow / DVC / RAGAS / GitHub instructions
See the dedicated Day 3–9 sections below for design detail, and
`WINDOWS_SETUP.md` for exact commands with their tested/proposed status.
**None of these five have ever been executed in this project's
development history** — all are implemented and locally unit-tested
where that's possible without the tool itself, never claimed as working
end-to-end.

### Troubleshooting
- **`ModuleNotFoundError` for any optional package** — install the
  relevant extra (see Installation above); this project's own sandbox
  has never had network access to verify any of them installs cleanly.
- **`/ask` returns 503** — the vector index hasn't been built yet; run
  the index-build command under "Running the API" above.
- **Tests show more skips than expected** — each skip prints the exact
  missing package; install it and re-run.
- **Corpus looks different from what's documented** — run
  `python3 scripts/check_corpus_quality.py`; it fails loudly if the
  corpus deviates from the known-good baseline (1095 articles).

### Known limitations
- Retrieval uses a TF-IDF lexical fallback, not a real semantic
  embedding model (no network to install `sentence-transformers` in
  this project's development sandbox).
- No generation backend exists — `/ask` is retrieval-only, always
  prefixed `[retrieval-only, no generation model configured]`.
- Docker, BentoML, Locust, MLflow, DVC, GitHub Actions, RAGAS, and
  Langfuse are all implemented but have **never been executed** in this
  project's history — see `reports/requirement_matrix.md` and
  `REMAINING_WORK.md` for the complete, itemized list.
- This project's git history starts from a recovery commit after an
  earlier development-sandbox reset destroyed the original history —
  see `PROGRESS.md`'s "RECOVERY NOTE" section.
- No actual course handbook document has ever been provided to this
  project — stated requirements come from cumulative task instructions,
  not a verified handbook.

---

## Day-by-day development log

The sections below are this project's own chronological development
record (Days 1–9), kept for full implementation detail and reasoning.
New readers should start with the overview above and
`reports/PROJECT_HANDOVER.md`; use this log when you need the "why" or
exact history behind a specific design decision.

## Day 1: Civil Code PDF → structured JSON ingestion

### What it does
Converts the Egyptian Civil Code PDF into one JSON record per article, with
fields: `article_number, book, chapter, section, topic, text_ar, text_en,
is_repealed, source_page, citation`. Handles Arabic-Indic article numbers,
tracks book/chapter/section hierarchy, flags explicitly repealed articles,
and best-effort splits bilingual (Arabic/English) lines.

### Input
The real PDF lives at:
```
data/raw/civil_code.pdf
```
It **is included** in this repository/package (it is git-ignored, so it
will not show up via `git status`, but the file itself is present on
disk and in any packaged ZIP of this project — verify with
`Get-Item data\raw\civil_code.pdf` on Windows or `ls -la data/raw/` on
Linux/Mac). Do not substitute another document if you ever need to
regenerate it.

### Output
```
data/processed/civil_code.json
```

### Run ingestion
```bash
pip install -e .
python -m src.ingestion.cli ingest --pdf data/raw/civil_code.pdf --out data/processed/civil_code.json
```

### Run validation
```bash
python -m src.ingestion.cli validate --json data/processed/civil_code.json
```
Reports per-article issues (missing/short text, bad types), duplicate
article numbers, and any gaps in the article-number sequence. Gaps are
reported, not silently ignored — some may be legitimate (repealed/removed
articles), so they need a human look.

### Run tests
```bash
python3 run_tests.py        # no network in this environment, so no pytest available
# once pytest is installed elsewhere: pytest -q
```
All tests use small synthetic Arabic snippets — none are real Civil Code text.

## Day 2: chunking → embeddings → vector store → retrieval → `/ask`

### Architecture
```
data/processed/civil_code.json
        |  src/rag/chunking.py    (article -> 1+ Chunk, metadata preserved)
        v
     Chunk list
        |  src/rag/embeddings.py  (text -> vector)
        v
   vector array
        |  src/rag/vector_store.py (persist + cosine search)
        v
   data/vector_store/  (vectors.npy, metadata.json, manifest.json, ...)
        ^
        |  src/rag/retriever.py   (question -> embed -> search -> top_k chunks)
        |
   src/api/main.py  FastAPI: GET /health, POST /ask
```

### Embedding model
Configured via the `EMBEDDING_MODEL` env var; default is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (small,
CPU-friendly, Arabic-capable) through `sentence-transformers`.

**Sandbox note:** this development sandbox has no network access and does
not have `sentence-transformers` installed, so the real model cannot be
loaded here. `get_embedder()` automatically falls back to a deterministic,
offline scikit-learn TF-IDF vectorizer in that case. This is a real
limitation, not a hidden shortcut — TF-IDF is lexical (word-overlap) only,
not semantic, so retrieval quality is materially weaker than a real
sentence-transformer would give, especially for Arabic. The actual index
in this repo was built in TF-IDF fallback mode; see PROGRESS.md for the
observed retrieval quality. Install `sentence-transformers` (extra `rag`)
in an environment with network access to use the real model — no code
changes needed, `get_embedder()` will pick it up automatically.

### Vector store
`src/rag/vector_store.py` prefers Chroma (`chromadb`, persistent, local)
if installed; otherwise falls back to a simple numpy cosine-similarity
store. Both use the same on-disk location, **`data/vector_store/`**
(gitignored — rebuild it, don't commit it). Same sandbox caveat as above:
`chromadb` isn't installed here, so the numpy fallback is what actually
built the real index in this repo.

### Build / rebuild the index
```bash
python3 -c "from src.rag.vector_store import build_index_from_json; \
  print(build_index_from_json('data/processed/civil_code.json', 'data/vector_store'))"
```

### Run the API
Requires the `api` extra (`fastapi`, `pydantic>=2`, `uvicorn`) — **not
installed in this sandbox**; install and run in an environment with
network access:
```bash
pip install -e ".[api]"
uvicorn src.api.main:app --reload
```

### `/ask` example
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هو عقد البيع؟"}'
```
```json
{
  "answer": "[retrieval-only, no generation model configured] The most relevant provision found is BOOK II ... المادة 465. See the sources list for the retrieved article text.",
  "sources": [
    {"article_number": 465, "citation": "BOOK II ... المادة 465", "text": "..."}
  ]
}
```
No generation model is configured, so `/ask` returns a clearly-labeled
retrieval-only answer (never presented as LLM output) plus the real
retrieved sources. An empty or whitespace-only `question` returns HTTP 422
(Pydantic validation).

### Day 2 known limitations
- Retrieval currently runs on the TF-IDF fallback embedder (see above) —
  lexical, not semantic. Some smoke-test queries returned topically
  adjacent but not the most precise article (see PROGRESS.md for the 5
  real observations). Expect a real sentence-transformer to do better,
  particularly on paraphrased/semantic queries.
- The FastAPI app (`src/api/main.py`) and its tests
  (`tests/test_api.py`) could not be executed in this sandbox — no
  `fastapi`/`pydantic` v2/`httpx` available, no network to install them.
  The code is standard FastAPI/Pydantic v2 and is expected to run as-is;
  `test_api.py` explicitly SKIPS (not fabricates a pass) when these
  imports are unavailable. Run it for real wherever those packages are
  installed.
- `ChromaVectorStore` (in `src/rag/vector_store.py`) is written but
  untested in this sandbox (`chromadb` unavailable). `SimpleVectorStore`
  (numpy fallback) is what was actually built and tested here.
- Article 1022's `text_ar` is still empty (Day 1, documented, unresolved
  source gap) — it is embedded/retrieved on `text_en` only.

## Day 3: real service, Docker, BentoML, load testing

### Environment status (this sandbox)
No network access and none of the following are installed here (all
confirmed by direct attempts, not assumed):
`fastapi`, `pydantic` v2, `uvicorn`, `httpx`, `sentence-transformers`,
`chromadb`, `bentoml`, `locust`, `docker` (no binary at all). Day 3 code
is written to be correct and standard, but **could not be executed** for
the API/Docker/BentoML/Locust layers in this sandbox — see PROGRESS.md
for exactly what was and wasn't verified, and the commands to run for
real once these are available.

### Architecture (updated)
```
data/processed/civil_code.json
        |  src/rag/chunking.py
        v
     Chunk list -> src/rag/embeddings.py -> src/rag/vector_store.py
        v
   data/vector_store/  (persisted index, loaded once, reused)
        ^
        |  src/rag/retriever.py
        |
   src/api/main.py        FastAPI:  GET /health, POST /ask
   src/serving/bento_service.py   BentoML: ask(), health() -- same retriever, no reimplementation
        ^
        |  tests/load/locustfile.py  Locust load test against POST /ask
```

### API usage (env vars)
- `VECTOR_STORE_DIR` — path to the built index (default `data/vector_store`)
- `RAG_TOP_K` — chunks retrieved per question (default `5`)
- `EMBEDDING_MODEL` — sentence-transformer model name (Day 2; default
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
- `MAX_QUESTION_LENGTH` — max accepted `/ask` question length in
  characters (Day 7; default `2000`)
- `RAG_LOG_DIR` — directory for the local structured request log (Day 8;
  default `logs/`, file is `logs/requests.jsonl`)
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` — optional, only used if
  `langfuse` is installed (Day 8; unset in this sandbox)

The index is loaded once, lazily, on first request and reused for every
request after that (a module-level singleton) — never rebuilt per
request. If loading fails (index missing/corrupt), `/ask` returns a
clean **HTTP 503** with a message pointing at how to build the index,
instead of crashing the process; `/health` stays independent of index
state (reports the process is up, not the index).

```bash
pip install -e ".[api,rag]"
uvicorn src.api.main:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "ما هو عقد البيع؟"}'
```
**Not run in this sandbox** (no fastapi/uvicorn here) — commands above
are correct-as-written, not verified end-to-end locally.

### Docker usage
```bash
docker build -t arabic-legal-rag .
docker run -p 8000:8000 -v "$(pwd)/data/vector_store:/app/data/vector_store" arabic-legal-rag
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "ما هو عقد البيع؟"}'
```
The vector index is intentionally **not** baked into the image (kept
small, rebuildable independently) — mount `data/vector_store/` as shown,
or run the index-build step inside the running container first.
**Not built or run in this sandbox** — there is no `docker` binary here
at all. The Dockerfile/`.dockerignore` were reviewed by hand and
structurally tested (`tests/test_docker_config.py`); the actual
`docker build`/`docker run`/`/health`/`/ask` steps above have not been
executed anywhere yet.

### BentoML usage
```bash
pip install -e ".[rag]" "bentoml>=1.2"
bentoml serve src.serving.bento_service:ArabicLegalRAGService
```
`src/serving/bento_service.py` wraps the *same* `Retriever` used by the
FastAPI app — no retrieval logic is duplicated. **Not executed in this
sandbox** (no `bentoml` package, no network to install it).

### Locust usage
```bash
pip install -e ".[load]"
locust -f tests/load/locustfile.py --host http://localhost:8000
# or headless, 50 concurrent users (handbook target):
locust -f tests/load/locustfile.py --host http://localhost:8000 \
  --users 50 --spawn-rate 10 --run-time 2m --headless --csv reports/load_test
```
**Not executed in this sandbox** — no `locust` package and no running
API instance to point it at. `tests/load/locustfile.py`'s syntax is
verified (`python3 -m py_compile`), and `tests/test_load_locustfile.py`
checks its structure whenever `locust` is available.

### Day 3 known limitations
See PROGRESS.md for the full, itemized list of what could and could not
be verified in this sandbox (API, Docker, BentoML, Locust all written
but blocked purely by missing network/packages, not by design issues).

## Day 4: MLflow experiment tracking and registry

### Environment status (this sandbox)
`mlflow` is not installed and there is no network here to install it
(confirmed: `pip install mlflow` fails, same as fastapi/chromadb/etc. in
Day 2-3). Per the Day 4 instructions, no MLflow runs were faked. The
integration code (`src/experiments/`) is written to use real MLflow when
available, and everything that does **not** require MLflow itself
(building real configs from the real corpus, evaluating retrieval,
saving results) was **actually executed** in this sandbox — see
PROGRESS.md for the real numbers.

### Architecture
```
src/experiments/eval_set.py     -- reproducible eval fixture, built from
                                    the real corpus (data/eval/retrieval_eval.json)
src/experiments/evaluate.py     -- hit_rate@k, mean reciprocal rank
src/experiments/tracking.py     -- thin MLflow wrapper: log_experiment_run(),
                                    register_best_run_artifact(); raises
                                    MLflowUnavailable (never fakes a run)
                                    if mlflow isn't installed
src/experiments/run_experiments.py -- runs 5 real configurations end to
                                    end (chunk -> embed -> index -> eval),
                                    logs to MLflow if available, always
                                    saves reports/day4_experiment_results.json
```
Experiment name: **`arabic-legal-rag`**.

### Run the experiment runner
```bash
python3 -m src.experiments.run_experiments
```
Builds/loads `data/eval/retrieval_eval.json` (15 questions, fixed seed
42, deterministic), runs the 5 configurations in
`CONFIGURATIONS` (`src/experiments/run_experiments.py`), evaluates each
with `hit_rate_at_k` + `mean_reciprocal_rank`, attempts MLflow logging
(skips cleanly with a printed message if `mlflow` unavailable), then
attempts registry (same). Always writes
`reports/day4_experiment_results.json` regardless of MLflow availability
— that file is the durable, inspectable record of what was measured.
**Actually run in this sandbox** — see PROGRESS.md for the real output.

### What's tracked per run
`chunk_size`, `top_k`, `tfidf_max_features`, `tfidf_ngram_range`,
`embedder_class`, `embedder_model` (always explicitly named — never
labels TF-IDF as a semantic model), `backend`; metrics:
`hit_rate_at_<k>`, `mean_reciprocal_rank`, `chunk_count`,
`build_seconds`, `eval_seconds`.

### Evaluation set
15 questions in `data/eval/retrieval_eval.json`, each built from a real
article's own opening words (`src/experiments/eval_set.py`,
`build_eval_set_from_corpus`) — a **self-retrieval lexical sanity check**
("given a snippet of this article's own text, can the system find it
again"), not an independently-authored natural-language legal question
with expert-verified ground truth (which this project has no way to
construct or validate honestly). `expected_article_number` for every
question is read directly from `data/processed/civil_code.json`, never
invented. Deterministic (fixed seed 42) and reproducible.

### Where MLflow artifacts/store would live
Not created in this sandbox (no mlflow). With mlflow installed, the
default local tracking URI (`./mlruns/`) applies unless
`MLFLOW_TRACKING_URI` is set; inspect runs with `mlflow ui` in the
project root, or `mlflow.search_runs(experiment_names=["arabic-legal-rag"])`.

### Registry
`register_best_run_artifact()` registers the one genuine trained-model
artifact in this pipeline — the fitted TF-IDF vectorizer (`sklearn`) used
by the offline fallback embedder — under
`arabic-legal-rag-tfidf-embedder`, tied to the best-scoring run. The RAG
system as a whole is **not** treated as a conventional ML model. If the
best run instead used a real sentence-transformer (pretrained, not
locally trained), the function returns `{"registered": False, "reason":
...}` rather than fabricating a registration — there is no locally-fit
artifact to version in that case. **Not executed in this sandbox** (no
mlflow) — confirmed by a real, passing test that it raises
`MLflowUnavailable` rather than faking a result.

### Day 4 known limitations
- No MLflow run was ever created in this sandbox — 0 real MLflow runs,
  by design, not a placeholder. The 5 configurations WERE actually run
  and evaluated locally (real numbers in PROGRESS.md); they were simply
  not logged to MLflow.
- All 5 configurations necessarily used the TF-IDF fallback embedder
  (documented since Day 2) — no real semantic-model comparison exists
  yet.
- The eval set measures self-retrieval lexical recall only, not RAG
  answer quality or semantic retrieval accuracy — named and scoped
  honestly as such.

## Day 5: DVC pipeline for the corpus (data/raw/civil_code.pdf → data/processed/civil_code.json)

### Environment status (this sandbox)
`dvc` is not installed and there is no network here to install it
(confirmed: `pip install dvc` -> "No matching distribution found", same
signature as every prior day's blocked package). `dvc init`, `dvc add`,
`dvc repro`, and `dvc status` were **never executed**. `dvc.yaml` (pure
YAML, hand-authored, no `dvc` package needed to write or parse it) IS
real and structurally verified. `git` IS available and was used for
real — see below.

### Pipeline
```
data/raw/civil_code.pdf
        |  extract_corpus stage (dvc.yaml)
        |  cmd: python -m src.ingestion.cli ingest --pdf data/raw/civil_code.pdf --out data/processed/civil_code.json
        v
data/processed/civil_code.json
```
The stage command is exactly the existing Day 1 CLI (`src/ingestion/cli.py`)
— no second parser was written for DVC. `deps:` lists the PDF plus every
Day 1 ingestion module the command actually imports; `outs:` is the JSON.

### Run it
```bash
dvc repro          # rebuild civil_code.json from the tracked PDF
dvc status          # what's out of date vs. the pipeline
```
**Not run in this sandbox** — no `dvc` binary. `tests/test_dvc_pipeline.py`
statically verifies `dvc.yaml`'s structure (real deps/outs, reuses the
existing CLI, no duplicate parser) using PyYAML instead, and separately
verifies actual pipeline **reproducibility** by literally running the CLI
twice against the real PDF and comparing output — see PROGRESS.md for
the real, observed result.

### DVC metadata location
`dvc.yaml` lives at the project root and is committed to git (verified —
see PROGRESS.md for the actual `git status`/`git log` output). Per-file
DVC tracking (`data/raw/civil_code.pdf.dvc`, `dvc.lock`) was **not**
created — those specifically represent "dvc has run and computed a real
hash," and hand-writing them without running `dvc add`/`dvc repro` would
misrepresent that as having happened. They do not exist in this repo yet;
generating them for real is the first Day 6 step.

### Known limitation
`data/raw/civil_code.pdf` is already git-ignored (since Day 2/3, for
size reasons); `data/processed/civil_code.json` is currently a normal
git-tracked file (not yet DVC-managed) since no `dvc add`/`dvc repro` has
run. Once DVC is available and `dvc repro` is run for real, DVC will want
to manage the JSON output itself (typically removing it from direct git
tracking in favor of a `.dvc`/`dvc.lock` pointer) — not done yet, since
faking that transition without a real dvc run would be dishonest.

## Day 6: GitHub Actions CI + quality gates

### Environment status (this sandbox)
This sandbox has no network access and no `git remote` configured
(confirmed: empty `git remote -v`; a direct HTTP request to github.com
is denied by this sandbox's network policy). **The workflow below has
never executed on GitHub** — it is written for GitHub's own (networked)
runners, and its structure is verified locally with PyYAML
(`tests/test_ci_workflow.py`, 10 tests, all pass for real). `ragas` and
`dvc` are also not installed here (same no-network signature as every
prior day) — see "RAGAS gate activation" and "DVC status" below for
exactly what that means.

### What runs on push/PR
`.github/workflows/ci.yml`, three jobs:
1. **`test`** (required): checkout, Python 3.11 + pip cache, install via
   `pip install -e ".[dev,rag,api,serving,load,experiments]"` (the
   project's own `pyproject.toml` — no separate dependency list), `ruff
   check .`, `pytest -q`, `python3 run_tests.py` (the bespoke runner used
   throughout this sandbox, run as a parity check), and
   `scripts/check_corpus_quality.py` — a deterministic corpus regression
   gate (see below).
2. **`ragas-quality-gate`** (required, `needs: test`): runs
   `scripts/ragas_gate.py`. **Currently always fails closed** — see
   below, this is intentional.
3. **`dvc-pipeline`** (required, `needs: test`): attempts `dvc repro` /
   `dvc status`. **Currently always fails closed** at a PDF-presence
   check — see "DVC status" below.

### Corpus quality gate (not a rubber stamp)
`python -m src.ingestion.cli validate` always exits 2 on this corpus
because articles 1 and 2 are legitimately duplicated (documented since
Day 1: the Promulgation Law's own front matter vs. the Civil Code proper
restarting numbering). Using that raw exit code as a CI gate would fail
on every correct run. `scripts/check_corpus_quality.py` instead encodes
the exact known-acceptable baseline (1095 articles, duplicates `[1, 2]`,
56 gaps, one documented empty-`text_ar` case at article 1022) and fails
**only** on a genuine deviation — a real regression check. Run it
yourself: `python3 scripts/check_corpus_quality.py` (tested — passes
against the real corpus right now).

### RAGAS gate activation
`scripts/ragas_gate.py` fails closed (never silently passes, never
substitutes lexical hit-rate/MRR as if it were faithfulness) whenever a
**real** faithfulness score can't be computed. Two independent
requirements must BOTH be true first:
1. `ragas` (+ `datasets`, + a configured LLM judge) importable/usable.
2. A **real generation backend** configured via the `GENERATION_MODEL`
   env var — RAGAS faithfulness measures whether a *generated answer* is
   supported by its context, and this pipeline is retrieval-only through
   Day 5 (`/ask` literally returns
   `"[retrieval-only, no generation model configured]..."`) — there is
   no generated answer yet for RAGAS to score.

Neither is true today, so this gate **always** exits 1 in this project,
regardless of whether `ragas` happens to install. This is intentional,
fail-closed behavior for a required release check, not a bug — see the
script's own docstring for the full reasoning.

### Threshold configuration
`RAGAS_FAITHFULNESS_THRESHOLD` env var (default `0.7`) — read but not
yet actionable, since the gate never reaches the real-scoring code path
in this project as of Day 6.

### DVC status and limitations
Even once `dvc` is installed (real, on a GitHub runner — the sandbox
that authored this still can't), `dvc repro` cannot run in a **fresh** CI
checkout: `data/raw/civil_code.pdf` is git-ignored (Day 2/3, for size)
and no DVC remote is configured to fetch it (Day 5 deliberately avoided
adding one — "Do NOT add unnecessary remote storage"). The `dvc-pipeline`
job detects this explicitly and fails with a clear message rather than
silently skipping or passing. Configuring a real DVC remote (or another
way to make the PDF available to CI) is Day 7+ scope.

### How to inspect GitHub Actions results
Once this repo is pushed to GitHub: the **Actions** tab on the repo page
lists runs per commit/PR; each job's steps show their real output
(including the two gates' exact printed reasoning above). Nothing here
substitutes for actually looking at a real run — this sandbox cannot
produce one.

## ⚠️ Recovery note (before Day 7)
Between Day 6 and Day 7, the development sandbox's working directory was
lost to an environment reset (not caused by any project command). The
git history through Day 6 (`cb480dc` … `9611fa5`) is **permanently
unrecoverable** — confirmed by an exhaustive filesystem search (no
`.git` objects, no other archives anywhere). The project was restored
from a user-provided backup of the Day 6 state (verified: corpus JSON
MD5 `ac90c9c1001e961d7aa451ca6984c1c8`, source PDF MD5 matching the
original upload, 119/0/13 baseline tests passing) and re-initialized as
a **new** git repository. Day 7 and Day 8 below are **rebuilt from
scratch** against that restored baseline — not restored from the lost
originals. They are new, independently-verified work, not a claim of
recovering the previous Day 7/8 commits.

## Day 7 (rebuilt): API audit, CI consistency, RAGAS integration interface

### What's new
- `src/api/validation.py` — dependency-free (no fastapi/pydantic)
  validation logic: `validate_top_k()` (fails fast on `RAG_TOP_K<=0` or
  non-integer) and `validate_question()` (rejects empty/whitespace and,
  new, over-length questions via `MAX_QUESTION_LENGTH`, default 2000
  chars). Kept separate from `src/api/main.py` specifically so it's
  unit-testable in this sandbox without fastapi/pydantic installed.
- `.github/workflows/ci.yml` / `pyproject.toml`: `ruff`, `dvc`, `ragas`
  were each installed via a separate, hand-written `pip install <pkg>`
  line instead of through this project's own extras mechanism —
  inconsistent with every other dependency in the workflow, and a
  version-drift risk. Fixed: added `lint`, `dvc`, `ragas` extras to
  `pyproject.toml`; all three workflow steps now install via
  `pip install -e ".[<extra>]"`. 2 regression tests guard against this
  reappearing.
- `src/experiments/ragas_eval.py` — `build_ragas_records()` (reuses the
  existing `Retriever` + Day 4 eval set to build real
  `(question, contexts, answer)` triples from a caller-supplied
  `generate_fn`) and `evaluate_faithfulness()` (thin `ragas.evaluate`
  wrapper, raises `ImportError` if `ragas` absent). `scripts/ragas_gate.py`
  still fails closed for the same two reasons as before (no `ragas`, no
  generation backend) — this makes the "once both exist" path concrete
  and tested, not runnable yet.

## Day 8 (rebuilt): observability and guardrails

### What's new
`src/observability/`:
- **`logger.py`** — `build_event()` enforces an explicit allowlist
  (`ALLOWED_EVENT_FIELDS`: request id, timestamp, route, latency,
  status, result count, question length, error type) — anything else,
  including raw question text or any secret, is **refused with a
  `ValueError`**. `log_event()` appends one JSON line per request to
  `logs/requests.jsonl` (configurable via `RAG_LOG_DIR`) and never
  raises.
- **`monitoring.py`** — `summarize_retrieval()` computes request count,
  empty-result rate, average result count, and latency stats from
  **actually-recorded events only**. Explicitly not an answer-quality or
  faithfulness metric (see Day 4/`src/experiments/` for those).
- **`langfuse_adapter.py`** — optional. `send_trace()` returns
  `{"sent": False, "reason": ...}` unless `langfuse` is installed AND
  `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are set AND a real client
  call succeeds.

`/ask` now logs one event per request (success, 503, and 500 paths) via
this system automatically.

### Inspect logs locally
```bash
cat logs/requests.jsonl
python3 -m src.observability.monitoring
```

### Langfuse activation prerequisites (not met in this sandbox)
1. `pip install langfuse`. 2. A real, reachable Langfuse server.
3. `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` set to real credentials.

## What was genuinely executed this session (Day 7 + Day 8 rebuild)
- `python3 run_tests.py` — multiple times, final: **160 passed, 0
  failed, 18 skipped**.
- Manual end-to-end logging/monitoring verification (real Python
  execution, real file I/O, output inspected directly).
- Corpus integrity re-verified: 1095 articles, MD5
  `ac90c9c1001e961d7aa451ca6984c1c8` (unchanged).
- `python3 -m py_compile` on every new/modified file.
- YAML/TOML structural parsing of `ci.yml`/`pyproject.toml`.

## What was NOT executed (unchanged limitations)
`ruff`, `dvc`, `langfuse`, `ragas` are not installed (no network — not
retried). No `docker` binary. No GitHub remote; GitHub Actions has never
run. No real generation backend exists, so RAGAS faithfulness remains
unmeasurable even if the package were installed.
