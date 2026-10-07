# AI Context Briefing — Arabic Legal RAG Project

**Read this entire file before making any changes.** It is written to
be self-contained: you should not need the user to re-explain this
project's history. If you are a different AI assistant (ChatGPT, Gemini,
Copilot, or a different Claude session with no memory of this one),
this document is your primary source of truth. Where this document and
your own inference disagree, trust this document and the actual files
on disk — verify by reading them, not by assuming.

## 1. Project objective and scope
Build a Retrieval-Augmented-Generation (RAG) system that answers
questions about the **Egyptian Civil Code**, a real bilingual
(Arabic/English) legal document, as a 10-day structured MLOps project
(ingestion → RAG pipeline → API → serving/Docker → experiments/MLflow →
DVC → CI/CD+RAGAS gate → observability → integration audit → final
packaging). As of this document, the project is through "Day 9" of that
structure, now being packaged for handover.

## 2. The use case
Input: the real Egyptian Civil Code PDF (`data/raw/civil_code.pdf`,
2,599,703 bytes). Output: a structured JSON corpus
(`data/processed/civil_code.json`) of **1095 articles**, each with
Arabic and English text, hierarchy (book/chapter/section), repeal
status, and citation. The system retrieves relevant articles for a
natural-language legal question and returns them with their real
article numbers and citations — it does **not** currently generate a
natural-language answer (see "What has NOT been executed" below).

## 3. Current architecture
```
PDF → src/ingestion/ → processed JSON (1095 articles)
    → src/rag/chunking.py → chunks (article-level, splits only long articles)
    → src/rag/embeddings.py → vectors (TF-IDF fallback IN USE; real
      sentence-transformer path exists in code, never executed)
    → src/rag/vector_store.py → index (numpy-cosine fallback IN USE;
      Chroma path exists in code, never executed)
    → src/rag/retriever.py → Retriever.retrieve(question, top_k)
    → src/api/main.py → FastAPI GET /health, POST /ask (never run — no fastapi installed)
    → src/observability/ → structured JSON logging + monitoring + optional Langfuse (logging/monitoring ARE verified; Langfuse is not)
    → src/experiments/ → eval set, hit-rate/MRR evaluation, MLflow
      tracking interface, RAGAS interface (local eval IS verified;
      MLflow/RAGAS execution is not)
```
Supporting infra (all implemented, none executed — see section 9):
`Dockerfile`/`.dockerignore`, `dvc.yaml`, `bentofile.yaml` +
`src/serving/bento_service.py`, `.github/workflows/ci.yml`,
`tests/load/locustfile.py`.

## 4. Complete repository structure
See `PROJECT_STRUCTURE.txt` (plain text, always current as of
packaging) or `reports/PROJECT_HANDOVER.md` (narrated, file-by-file).
Top level:
```
src/{ingestion,rag,api,observability,experiments,serving}/
tests/ (+ tests/load/)
scripts/
data/{raw,processed,eval,vector_store}/
reports/
.github/workflows/
Dockerfile, .dockerignore, dvc.yaml, bentofile.yaml, pyproject.toml
README.md, PROGRESS.md, AI_CONTEXT.md, WINDOWS_SETUP.md,
REMAINING_WORK.md, PROJECT_STRUCTURE.txt
```

## 5. Data schema
Each article record in `data/processed/civil_code.json`:
```json
{
  "article_number": 1,
  "book": null,
  "chapter": null,
  "section": null,
  "topic": null,
  "text_ar": "...",
  "text_en": null,
  "is_repealed": false,
  "source_page": 1,
  "citation": "المادة 1"
}
```
`topic` is always `null` by design (never inferred/invented). One known
article (1022) has empty `text_ar` — a genuine, documented source-PDF
extraction gap, not a bug to silently "fix" by inventing text.

## 6. Corpus size and verified integrity
- **1095 articles**, `article_number` range 1–1149 (gaps are real,
  documented repealed-article ranges, not bugs).
- `data/processed/civil_code.json` MD5: `ac90c9c1001e961d7aa451ca6984c1c8`
- `data/raw/civil_code.pdf` MD5: `5086ef5f16f15d856278db943e58e948`
- **Before trusting either file, re-verify these hashes yourself** —
  this document can go stale; the hash is the ground truth.
- `scripts/check_corpus_quality.py` is a deterministic regression gate
  against a known-good baseline (1095 articles, duplicate article
  numbers `[1, 2]` — a real, intentional document-structure fact, not a
  bug — 56 gaps, 1 documented empty-text issue at article 1022). Run it
  before and after any change that touches ingestion or the corpus.

## 7. What has actually been implemented
Everything listed in Section 3's diagram has real, reviewable source
code: PDF ingestion with Arabic bidi-text correction, chunking,
embeddings (two backends), vector store (two backends), retrieval,
FastAPI app with validation, structured logging with a field allowlist,
retrieval monitoring, an MLflow tracking interface, a DVC pipeline
definition, a GitHub Actions workflow, a RAGAS quality-gate interface, a
BentoML service wrapper, and a Locust load-test script.

## 8. What has actually been tested (and how)
`python3 run_tests.py` (or `pytest -q`) → **165 passed, 0 failed, 18
skipped** as of this document (re-run it yourself — don't trust a
number that might be stale). Tests that run for real include: all of
ingestion (parser, bidi-fix, arabic-utils, validation — 47+ tests),
chunking/embeddings/vector-store/retriever (TF-IDF/numpy paths), API
*validation logic* (not the HTTP layer), observability logging and
monitoring (including live manual verification, not just unit tests),
quality gates, CI-workflow YAML structure, DVC-yaml structure, and a
full local end-to-end integration test
(`tests/test_integration_pipeline.py`) against the real corpus.

## 9. What has NOT been executed (be precise about this distinction)
- The FastAPI HTTP layer itself (`fastapi`/`pydantic`/`uvicorn` never
  installed in this project's development sandbox).
- `docker build`/`docker run` (no `docker` binary available there).
- `bentoml serve` (no `bentoml` installed).
- Locust load test (no `locust`, no running server) — **zero benchmark
  numbers of any kind exist in this project**.
- Real MLflow run logging/registry (no `mlflow` installed) — 5
  configurations WERE run and evaluated **locally** (see
  `reports/day4_experiment_results.json`), just never logged to MLflow.
- `dvc init`/`dvc add`/`dvc repro`/`dvc status` (no `dvc` installed;
  also, even once installed, the source PDF is git-ignored with no DVC
  remote configured yet, so a fresh checkout can't fetch it).
- GitHub Actions (no git remote has ever been configured for this repo).
- A real RAGAS faithfulness score (no `ragas` package AND no generation
  backend exists anywhere in this codebase — this is two independent
  blockers, not one).
- A real Langfuse trace (no package, no server, no credentials).
- A real sentence-transformer embedding model / Chroma vector store
  (both have a real code path; neither has ever been loaded/installed).

**Do not write, in a report or to the user, that any of the above
"works" or "passed" unless you have just run it yourself and can show
the real output.**

## 10. Missing dependencies and environment blockers
The development sandbox that built this project had **no network
access** at any point in its history, and no `docker` binary. This is
the single root cause behind every item in Section 9 — not a design
flaw, just an environment constraint. See `pyproject.toml`'s
`[project.optional-dependencies]` for the exact extras
(`dev, rag, api, serving, load, experiments, lint, dvc, ragas,
langfuse`) needed to unblock each one.

## 11. Test results and skip reasons (as of this document)
165 passed, 0 failed, 18 skipped. Skip breakdown by file (each skip
prints its own exact reason at runtime — trust that output over this
summary if they ever disagree): `test_api.py` (~12, fastapi/pydantic/
httpx absent), `test_bentoml_service.py` (2, bentoml absent),
`test_dvc_pipeline.py` (1, dvc binary absent), `test_load_locustfile.py`
(2, locust absent), `test_mlflow_tracking.py` (1, mlflow absent).

## 12. Current Git commit and repository state
Run `git log --oneline -1` and `git status` yourself — do not trust a
hash written in this document without checking, since it will go stale
the moment a new commit is made. **Important history note:** this
repository's git history starts from a recovery commit (not the
project's true beginning) after an earlier development-sandbox reset
destroyed the original history. See `PROGRESS.md`'s "RECOVERY NOTE"
section for the full story — commit hashes referenced in older
conversation transcripts or exported reports may not exist in this
repository.

## 13. Known bugs or limitations
- Embeddings/retrieval are lexical (TF-IDF), not semantic — expect
  retrieval quality to differ materially from a real sentence-transformer
  once one is wired in.
- `/ask`'s answer is always prefixed
  `[retrieval-only, no generation model configured]` — this is
  deliberate, not a placeholder bug; do not remove this prefix without
  actually wiring in a real generation backend.
- No answer-quality/faithfulness metric exists — only lexical
  hit-rate/MRR (`src/experiments/evaluate.py`), explicitly scoped as
  such, never to be confused with or relabeled as semantic quality.
- Article 1022 has empty `text_ar` (documented source gap, not a code
  bug — do not try to fabricate text for it).

## 14. Course requirements currently known
**No actual course handbook document has ever been provided to this
project at any point.** Every requirement referenced anywhere in this
project's documentation (BentoML, MLflow, DVC, a RAGAS faithfulness
gate with a specific threshold, "50 concurrent users" for load testing,
etc.) came from a human's own task instructions in conversation, not
from reading a handbook. `reports/requirement_matrix.md` says this
explicitly and should not be cited as if it reflects verified handbook
compliance.

## 15. The handbook situation (repeated for emphasis)
If the user gives you an actual handbook document in a future session,
re-derive requirements from it directly and flag any place where this
project's prior assumptions (inferred from conversation only) turn out
to be wrong — do not assume prior assumptions were correct.

## 16. Remaining implementation and verification tasks
See `REMAINING_WORK.md` for the full, itemized checklist. Short version:
everything in Section 9 needs to actually be run, for the first time,
in an environment that has the relevant tool/network/credential. A real
generation backend needs to be designed and built before RAGAS can ever
produce a real score — this is new development, not just an install.

## 17. Instructions to you, the next AI assistant
1. **Inspect before editing.** Read the actual current file content —
   this document, `PROGRESS.md`, and the code itself — before changing
   anything. Do not assume a file's content matches what's described
   here if you haven't just read it.
2. **Preserve the corpus.** Never regenerate or hand-edit
   `data/processed/civil_code.json` or touch `data/raw/civil_code.pdf`
   without a clearly demonstrated bug and the user's awareness. Run
   `scripts/check_corpus_quality.py` before and after any change that
   could plausibly affect it.
3. **Never fabricate results.** If you can't actually run something
   (missing package, no credentials, no network), say so plainly and
   explain exactly what's missing — do not describe untested code as
   "working" or invent metrics/benchmarks/traces.
4. **Make small, verifiable changes.** Run the test suite
   (`pytest -q` or `python3 run_tests.py`) after every change. Report
   exact pass/fail/skip counts, not vague summaries.
5. **The user may not know this project's technical history in detail**
   — this document and `PROGRESS.md` are more reliable than asking them
   to re-explain it, but it's fine to ask them for new credentials,
   decisions, or a real course handbook if one becomes available.
6. **You cannot access local files automatically.** If you're a
   different AI product than the one that wrote this document, you
   likely need the user to upload this ZIP (or the relevant files)
   directly — don't assume you already have filesystem access just
   because this document describes file paths.
