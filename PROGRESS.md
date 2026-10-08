# PROGRESS

## Day 1 status: COMPLETE (final cleanup pass done)

Real Egyptian Civil Code PDF (`llm_final_pro.pdf`, uploaded by user) at
`data/raw/civil_code.pdf`, ingested end-to-end, validated, spot-checked,
and cleaned up over three passes. See `reports/day1_spot_check.md`
(original) and `reports/day1_spot_check.md` (v2, this session, appended
below the original content's findings) for full detail.

### This session's fixes (final cleanup)
1. **7 individual gaps investigated** (277, 452, 714, 746, 898, 1022,
   1090): all confirmed present in the PDF. 6 were mis-rejected by the
   heading heuristic (trailing English sentence between the English and
   Arabic markers, or a missing leading "A" in "Article") -- fixed and
   recovered. 1 (1022) has a genuine, singular extraction gap -- its
   Arabic marker/text is entirely absent from the source even though the
   English text is present (verified by comparing to neighboring articles
   1021/1023, which both have normal bilingual pairing). Not fabricated;
   left as an explicit, flagged validation issue
   (`text_ar is empty`) rather than guessed at.
2. **False duplicate at article 1148 fixed**: a short, English-free
   wrapped citation line (`.المادة ١١٤٨`, only a trailing period as
   "extra" content) was slipping past the Arabic-only heading check.
   Tightened to require the match be the entire line, whitespace-trimmed
   only (no punctuation stripped). Verified: exactly one occurrence of
   1148 now, with intact real content.
3. **Second-order regression caught and fixed**: loosening the heading
   rule for fix #1 initially introduced 5 NEW false duplicates (298, 864,
   982, 1033, 1050) -- real cross-reference sentences citing another
   article's number in both languages on one line. Fixed by requiring the
   English marker to open the line (nothing before it); true headings
   always start the line, citation sentences never do.
4. **Heading text no longer leaks into `text_ar`/`text_en`**: the marker
   text itself ("Article 105 مادة ١٠٥") is now stripped from the heading
   line before any remainder is added to the article body.
   `article_number` already carries the number separately. Any genuine
   body-text fragment sharing the heading line (real in this PDF) is
   preserved, not discarded.
5. Added regression tests for all of the above (6 new tests, synthetic
   data, explicitly labeled non-real). Full suite: **47 tests, 0
   failures**.

### Current state after this session
- **1095 articles** parsed to `data/processed/civil_code.json` (up from
  1089 last session; +6 recovered gap articles, -1148's false duplicate
  net effect already included).
- Validation: **1 article-level issue** (1022's empty `text_ar`,
  documented, not silently dropped), **2 duplicate numbers** (`[1, 2]`
  -- both the legitimate Promulgation Law / Civil Code numbering
  restart, documented in the original spot check), **56 number gaps**
  (all accounted for: 27 in range 54-80 + 29 in range 389-417, both
  explicit in-PDF repeal notices; no unexplained individual gaps
  remain).
- 29 articles spot-checked this session (the original 22 + the 7 gap
  cases), all correct except the documented 1022 case.

### Files changed this session
```
src/ingestion/parser.py          (heading heuristic tightened/extended:
                                   English marker must open the line;
                                   Arabic-only marker must be the whole
                                   line, unstripped; new
                                   _find_english_only_heading fallback;
                                   new _strip_markers to keep heading
                                   text out of text_ar/text_en)
tests/test_parser.py             (7 new regression tests)
data/processed/civil_code.json   (regenerated)
reports/day1_spot_check.md       (rewritten with v2 findings)
PROGRESS.md                      (this file)
```

### Tests run/results
- `python3 run_tests.py` -> **47 passed, 0 failed**.
- `python3 -m src.ingestion.cli ingest --pdf data/raw/civil_code.pdf --out data/processed/civil_code.json`
  -> 1095 articles parsed.
- `python3 -m src.ingestion.cli validate --json data/processed/civil_code.json`
  -> 1 article-level issue (1022, documented); 2 duplicates (legitimate);
     56 gaps (fully explained).

### Known limitations (unresolved, documented, not guessed at)
- Article 1022's `text_ar` is empty -- genuine, unresolved source
  extraction gap (no Arabic marker/text present anywhere near it in the
  extracted PDF text). Flagged by validation.
- Bilingual `text_ar`/`text_en` split is still a per-line Latin-ratio
  heuristic, not a true column-based split -- an occasional line could
  still be misclassified.
- `topic` field is still always `None` -- no reliable automatic signal
  designed for it.
- The two duplicate `article_number=1` and `=2` records are a genuine,
  intentional document-structure fact (Promulgation Law's own front-matter
  articles vs. the Civil Code proper restarting numbering), not a bug --
  documented, not "fixed away".

### Exact next step (Day 2+, not started, out of scope for Day 1)
Anything beyond ingestion (embeddings, vector DB, API, MLflow, DVC, CI/CD,
etc.) is explicitly out of scope for Day 1 and intentionally not started.

---

# DAY 2

## Day 2 status: COMPLETE, with one major environment-forced limitation (documented, not hidden)

This sandbox has **no network access** and does not have `fastapi`,
`pydantic` (v2), `uvicorn`, `httpx`, `sentence-transformers`, or
`chromadb` installed, and none could be installed here (`pip install`
fails with no matching distribution — confirmed before starting). All
Day 2 code is written to use the real intended stack (a real
sentence-transformer model, Chroma, FastAPI) and is structured so that
stack is picked up automatically once available — but in THIS sandbox,
the embedding and vector-store layers ran in documented offline fallback
modes, and the FastAPI layer could be written but not executed/tested.
Nothing below claims a result that wasn't actually observed.

### What was built
- `src/rag/chunking.py` — JSON article -> 1+ `Chunk` (default: one
  chunk per article; splits by sentence boundary only for the ~13/1095
  articles over 1000 chars, always preserving article_number + all
  metadata; `topic` passed through as `None`, never invented).
- `src/rag/embeddings.py` — `SentenceTransformerEmbedder` (real, primary;
  requires `sentence-transformers` + network) and `TfidfFallbackEmbedder`
  (offline, deterministic, scikit-learn); `get_embedder()` picks
  whichever loads. **In this sandbox, only the TF-IDF fallback actually
  ran** (confirmed: no `sentence-transformers` package, no network).
- `src/rag/vector_store.py` — `ChromaVectorStore` (real, primary; written
  but untested here — no `chromadb` installed) and `SimpleVectorStore`
  (numpy cosine-similarity, offline); `get_vector_store()` picks whichever
  loads. **In this sandbox, only `SimpleVectorStore` actually ran.**
  Persists to `data/vector_store/` (vectors.npy, metadata.json,
  manifest.json, tfidf_vectorizer.pkl). Added to `.gitignore`.
- `src/rag/retriever.py` — `Retriever.retrieve(question, top_k)` ->
  list of dicts with `article_number`, `text_ar`, `text_en`, `citation`,
  `score`. Independently testable without any LLM.
- `src/api/main.py` — FastAPI `GET /health`, `POST /ask` per the required
  request/response shape; Pydantic validation rejects empty/whitespace
  questions (422); sources cite `article_number` + `citation`, never
  chunk IDs or page numbers; `/ask` returns a clearly-labeled
  `[retrieval-only, no generation model configured]` answer — never
  pretends an LLM generated it. **Written but NOT executed in this
  sandbox** (no fastapi/pydantic v2/httpx here).
- `pyproject.toml` updated: numpy/scikit-learn added as base deps; new
  `rag` extra (sentence-transformers, chromadb) and `api` extra (fastapi,
  pydantic, uvicorn, httpx) added — none installable here, needed for the
  parts of Day 2 this sandbox couldn't run.
- `.gitignore` added (vector store artifacts, `__pycache__`, raw PDF).
- README updated with Day 2 architecture, setup, index build/rebuild,
  API run instructions, `/ask` example, embedding model choice, and
  known limitations.

### Tests run and actual result
`python3 run_tests.py` -> **66 passed, 0 failed, 5 skipped**.
- Day 1 tests: 47 (unchanged, still passing).
- New Day 2 tests, ACTUALLY EXECUTED (19): chunking (6), embeddings (5),
  vector store (4), retriever (4) — all against tiny synthetic fixtures
  or the offline TF-IDF fallback, no network/model download required.
- `tests/test_api.py` (5 tests): **honestly SKIPPED**, not fabricated as
  passing — the module detects at import time that
  `fastapi`/`pydantic`/`httpx` are unavailable and each test prints a
  skip notice and returns `"SKIPPED"`, which `run_tests.py` now counts
  separately from real passes. These must be run for real in an
  environment with those packages installed before the `/ask` endpoint
  is considered verified.

### Index build result (real corpus, actually run)
```
python3 -c "from src.rag.vector_store import build_index_from_json; \
  print(build_index_from_json('data/processed/civil_code.json', 'data/vector_store'))"
```
- Input: 1095 articles from `data/processed/civil_code.json`.
- Output: **1107 chunks** indexed (1095 articles; ~12 articles over
  1000 chars split into 2+ chunks each).
- Backend actually used: `simple-numpy` / `TfidfFallbackEmbedder`
  (confirmed via `manifest.json`) — NOT a real sentence-transformer,
  because none is installed/reachable here.
- Build time: ~0.9s (TF-IDF fit + transform over 1107 chunks).
- Store size on disk: ~19 MB (mostly `vectors.npy`, a dense 1107 x
  ~4096 float32 matrix — gitignored, rebuildable from the JSON).

### 5 real retrieval smoke-test observations (actual output, not invented)
1. **"ما هو عقد البيع؟"** (What is a sale contract?) -> top hits: arts.
   465, 475, 466 — all correctly in the Sale chapter (Book II, Ch I,
   Sec I). Topically correct chapter; not necessarily the code's own
   opening "definition" article for sale, but genuinely on-topic.
2. **"متى ينتهي عقد الايجار؟"** (When does a lease end?) -> top hit:
   art. 598, whose text is literally "A lease ends at the expiration of
   the agreed term" — a strong, precise match. Also arts. 543 (loan
   term) and 612 (agricultural lease), both topically related.
3. **"ما هي احكام حق الانتفاع؟"** (Rules of usufruct?) -> arts. 511, 84,
   85 — only tangentially related (a partnership-share article
   mentioning usufruct, plus unrelated "consumable/fungible things"
   articles). The Civil Code's actual dedicated Usufruct section was
   **not** surfaced. This is the clearest case of the TF-IDF fallback's
   documented lexical-only weakness.
4. **"كيف يثبت الرهن الحيازي؟"** (How is a possessory pledge
   established?) -> arts. 1114, 239, 1047 — adjacent pledge/mortgage
   articles (Book IV), but not precisely the possessory-pledge
   subsection; partially relevant only.
5. **"ما هي مسؤولية الوكيل عن اعمال الوكالة؟"** (Agent's liability for
   acts of mandate?) -> arts. 715, 710, 708 — all correctly from the
   Mandate chapter (Book II, Ch III). Strong, on-topic match.

**Honest summary: 3/5 queries returned clearly on-topic, useful results
(2 strong, 1 good); 2/5 returned only topically-adjacent results, missing
the most precise article.** This is consistent with, and was predicted
by, the documented TF-IDF-fallback limitation (lexical overlap only, no
semantic understanding) — no accuracy score is claimed beyond this
qualitative, per-query observation.

### Known blockers / limitations going into Day 3
- **Biggest blocker for real quality**: this sandbox cannot install or
  run a real sentence-transformer or chromadb (no network). The code
  already supports both via `get_embedder()`/`get_vector_store()` with
  zero required code changes — Day 3 (or before it) should rebuild the
  index in an environment with network access and `pip install -e
  ".[rag]"` to get real semantic embeddings, then re-run the same 5
  smoke-test queries to compare.
- **FastAPI app unexecuted**: `src/api/main.py` and `tests/test_api.py`
  need `pip install -e ".[api]"` in an environment with network access,
  then `python3 run_tests.py` (should show 0 skipped) and a manual
  `uvicorn src.api.main:app` smoke test.
- Article 1022's empty `text_ar` (Day 1) is unchanged; it is currently
  embedded/retrieved using `text_en` only.
- `topic` is still always `None` (Day 1 known limitation, intentionally
  not invented here either).

### Exact starting point for Day 3
1. In an environment with network access: `pip install -e ".[rag,api]"`.
2. Rebuild the index: `python3 -c "from src.rag.vector_store import
   build_index_from_json; build_index_from_json('data/processed/civil_code.json',
   'data/vector_store')"` — confirm `manifest.json` now shows a real
   sentence-transformer model name, not `tfidf-fallback`.
3. Re-run the exact same 5 smoke-test questions above and compare
   results/quality against this session's TF-IDF baseline.
4. Run `python3 run_tests.py` — `tests/test_api.py`'s 5 tests should now
   execute (0 skipped) instead of skipping.
5. Only once both of the above are verified for real should Day 3
   (generation/LLM integration, evaluation, etc.) begin.

---

# DAY 3

## Day 3 status: COMPLETE for everything verifiable in this sandbox; API/Docker/BentoML/Locust execution BLOCKED by environment (documented, not hidden)

### Phase 1 — environment check (actual, confirmed results)
Checked directly (import attempts / `pip install` / `docker --version`),
not assumed:
| Dependency | Available? |
|---|---|
| fastapi | No |
| pydantic v2 | No |
| uvicorn | No |
| httpx | No |
| sentence-transformers | No |
| chromadb | No |
| bentoml | No |
| locust | No |
| docker (binary) | No |
| numpy, scikit-learn | Yes (already present, used since Day 2) |

`pip install fastapi` (and a `pip download numpy` probe) both failed with
"No matching distribution found" — confirming no network access, exactly
as before Day 2. Per instructions, did not keep retrying; proceeded with
whatever is honestly verifiable offline.

### Phase 2 — `/ask` API: fixed and improved (code-reviewed, NOT executed)
`src/api/main.py` changes:
- Config now via env vars: `VECTOR_STORE_DIR` (default
  `data/vector_store`), `RAG_TOP_K` (default `5`).
- Index load stays a lazy, module-level singleton (already true in Day
  2) — confirmed never rebuilt per request.
- Added explicit error handling: a failed index load now caches the
  error and returns a clean **HTTP 503** with a helpful message on every
  `/ask` call, instead of raising an uncaught exception each time or
  crashing the process. A retrieval-time exception returns **HTTP 500**
  with detail, not a bare crash.
- `/health` deliberately stays independent of index state (reports the
  process is up).
- Existing requirements unchanged and preserved: empty/whitespace
  question -> 422 (Pydantic validator), sources cite `article_number` +
  `citation` (never chunk IDs/page numbers), retrieval-only answer
  clearly labeled `[retrieval-only, no generation model configured]` —
  never presented as LLM output.
- **Not executed**: no fastapi/pydantic/uvicorn in this sandbox. Syntax
  verified with `python3 -m py_compile src/api/main.py` (passed).

### Phase 3 — Docker (written, NOT built/run: no `docker` binary here)
- `Dockerfile`: `python:3.11-slim` base, installs `.[api,rag]`, copies
  `src/` + `data/processed/` only (NOT `data/vector_store/` — mounted at
  runtime instead, per instructions), `EXPOSE 8000`, `HEALTHCHECK` against
  `/health`, starts with `uvicorn src.api.main:app`.
- `.dockerignore`: excludes `data/vector_store/`, raw PDF, tests,
  reports, git metadata.
- `tests/test_docker_config.py` (6 tests, all pass for real — pure
  file-content checks, no `docker` needed): Dockerfile exists, starts via
  uvicorn on the right module, exposes a port, has a `/health`
  healthcheck, does not `COPY` the generated vector store,
  `.dockerignore` excludes it and the raw PDF.
- **`docker build`/`docker run`/`/health`/`/ask` against the container
  were NOT run** — `docker --version` returns "not found" in this
  sandbox. Not claimed to work; commands documented in README for
  someone with Docker to verify.

### Phase 4 — BentoML (written, NOT executed: no `bentoml` package here)
- `src/serving/bento_service.py`: class-based `@bentoml.service`
  (BentoML >= 1.2 API) wrapping the *same* `Retriever` from
  `src/rag/retriever.py` — no retrieval logic duplicated (verified by a
  real, executed test: `test_bento_service_module_reuses_existing_retriever_not_reimplemented`,
  which checks the import statement and the class via plain source
  inspection, no bentoml needed).
- `ask()` is `async def`; `health()` mirrors the FastAPI `/health`.
- Streaming: explicitly NOT implemented. No generation model is
  configured (retrieval-only), so there is no token stream to relay —
  faking one over an already-complete string would misrepresent it as
  real token-by-token generation, which the instructions disallow. Noted
  in the module docstring that `ask` is already `async`, so adding a real
  streaming generator later needs no other architecture change.
- `bentofile.yaml` added (minimal build config).
- `tests/test_bentoml_service.py`: 1 test runs for real (reuse check,
  above); 2 tests SKIP (instantiate/coroutine checks — need `bentoml`).
- **Not executed**: `bentoml serve ...` was not run. Syntax verified with
  `python3 -m py_compile src/serving/bento_service.py` (passed).

### Phase 5 — Locust (written, NOT executed: no `locust`, no running server)
- `tests/load/locustfile.py`: `AskUser(HttpUser)`, `wait_time=between(1,3)`,
  weighted tasks (`/ask` x3, `/health` x1), 5 real Arabic legal questions
  (same set as the Day 2 retrieval smoke test) as request bodies,
  configurable `--host`.
- `tests/test_load_locustfile.py`: syntax check via `py_compile` runs for
  real and passes (does not need `locust` installed); 2 structural checks
  (HttpUser subclass, task/body shape) SKIP — need `locust`.
- **No load test was run — no numbers of any kind are reported.**
  Reasons: `locust` is not installed (no network), and there is no
  running instance of the API to target anyway (fastapi/uvicorn also
  unavailable). Nothing under `reports/` claims a benchmark result.

### Phase 6 — full test suite (actual result)
`python3 run_tests.py` -> **75 passed, 0 failed, 11 skipped**.
- Day 1 (47) + Day 2 (19) tests: unchanged, all still passing.
- New Day 3 tests actually executed (9): Docker config (6),
  BentoML reuse-check (1), Locust syntax (1), retriever
  missing-index error (1).
- Skipped (11, up from 5 in Day 2 — honestly, not silently): 7 in
  `test_api.py` (added 2 new: env-var defaults, 503-on-missing-index),
  2 in `test_bentoml_service.py`, 2 in `test_load_locustfile.py`. Every
  skip prints the exact missing package and returns the literal string
  `"SKIPPED"`, which `run_tests.py` counts separately from real passes —
  none of these are silently or dishonestly counted as passing.

### Files changed/added this session
```
src/api/main.py                  (env-var config, 503 error handling)
src/serving/__init__.py          (new)
src/serving/bento_service.py     (new)
bentofile.yaml                   (new)
tests/load/locustfile.py         (new)
tests/test_api.py                (2 new tests: config defaults, 503 path)
tests/test_bentoml_service.py    (new)
tests/test_load_locustfile.py    (new)
tests/test_docker_config.py      (new)
tests/test_retriever.py          (1 new test: missing-index error)
Dockerfile                       (new)
.dockerignore                    (new)
pyproject.toml                   (serving, load extras added)
README.md                        (Day 3 section appended)
PROGRESS.md                      (this section)
```

### Docker result
Written and structurally tested (6/6 config tests pass); **build/run
never attempted** — no `docker` binary in this sandbox. Not claimed to
work end-to-end.

### BentoML result
Written; reuse-of-existing-retriever verified for real (1 test);
instantiate/serve behavior **not executed** — no `bentoml` package.

### Locust/load-test result
Script written and syntax-verified; **no load test was run, no
latency/throughput/error numbers exist** — no `locust`, no reachable
running API. This is the honest, complete result: zero numbers, by
design, not a placeholder.

### Blockers
Single root cause for everything not executed this session: **no network
access in this sandbox**, confirmed directly (failed `pip install`,
absent `docker` binary). Every blocked phase (API runtime test, Docker
build/run, BentoML serve, Locust run) traces back to this one blocker,
not to a code defect found so far.

### Exact starting point for Day 4
1. In an environment with network access: `pip install -e
   ".[rag,api,serving,load]"`.
2. Run `python3 run_tests.py` — expect 0 skipped (all 86 tests, current
   75 + the 11 now-unblocked, should execute and pass; fix anything that
   doesn't before moving on).
3. Start the API for real: `uvicorn src.api.main:app` — manually hit
   `/health` and `/ask` with a real question; confirm the 503 path by
   temporarily pointing `VECTOR_STORE_DIR` at a bad path.
4. `docker build -t arabic-legal-rag .` then `docker run` per the README
   — confirm `/health` and `/ask` inside the container for real.
5. `bentoml serve src.serving.bento_service:ArabicLegalRAGService` —
   confirm `ask`/`health` work for real.
6. Run the Locust load test for real, including the handbook's
   50-concurrent-user target, and save actual output under `reports/`
   (e.g. `reports/load_test_stats.csv`) — only real, observed numbers.
7. Only once all of the above are verified for real should Day 4 (MLflow
   experiments, per the stop condition) begin.

---

# DAY 4

## Day 4 status: COMPLETE for everything verifiable in this sandbox; MLflow execution BLOCKED by environment (0 real MLflow runs, honestly)

### Phase 1 — environment check (actual, confirmed result)
`import mlflow` -> `ModuleNotFoundError`. `pip install mlflow` -> "No
matching distribution found" (same no-network signature as every prior
day). Did not retry repeatedly. Proceeded per instructions: implemented
the integration code, ran everything that does NOT itself require
mlflow, and documented the blocker clearly rather than faking runs.

### Phase 2 — MLflow integration (written; execution blocked)
`src/experiments/tracking.py`:
- `get_mlflow()` — returns the module if importable, else `None`, never
  raises. Tested for real (always executable): confirms it returns
  `None` cleanly here.
- `log_experiment_run(config, metrics, artifact_paths, run_name)` — logs
  params/metrics/artifacts to a real MLflow run under experiment
  `arabic-legal-rag` if mlflow is available; **raises
  `MLflowUnavailable` otherwise — never fakes a run record.** Tested for
  real: confirms it raises in this environment.
- `register_best_run_artifact(results, metric_name, model_name)` — Phase
  5 registry function (see below). Also raises `MLflowUnavailable` when
  mlflow is absent; tested for real.

### Phase 3 — 5 real configurations: ACTUALLY RUN (not MLflow runs — see below)
`python3 -m src.experiments.run_experiments` was executed for real in
this sandbox. **0 MLflow runs were created** (mlflow unavailable, by
design — never faked). All 5 configurations WERE genuinely built and
evaluated locally (real corpus, real chunking, real TF-IDF fit, real
retrieval, real metrics) and the results were saved to
`reports/day4_experiment_results.json`.

Every configuration necessarily used **`TfidfFallbackEmbedder`**
(`embedder_model: "tfidf-fallback (offline, lexical only -- not
semantic)"`, `backend: "simple-numpy"`) — no `sentence-transformers`/
network in this sandbox, same as Day 2-3. Never labeled as a semantic
model. No fallback-vs-real-model comparison exists (nothing to compare
against).

| run_name | chunk_size | top_k | tfidf_max_features | ngram_range | hit_rate_at_k | mean_reciprocal_rank | chunk_count | build_s | eval_s |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 1000 | 5 | 4096 | (1,1) | 1.000 | 0.906 | 1107 | 0.14 | 0.13 |
| smaller_chunks | 500 | 5 | 4096 | (1,1) | 1.000 | 0.922 | 1343 | 0.15 | 0.17 |
| larger_chunks | 2000 | 5 | 4096 | (1,1) | 1.000 | 0.906 | 1095 | 0.13 | 0.14 |
| smaller_top_k | 1000 | 3 | 4096 | (1,1) | 0.933 | 0.889 | 1107 | 0.13 | 0.14 |
| bigram_tfidf | 1000 | 5 | 4096 | (1,2) | 1.000 | 0.806 | 1107 | 0.27 | 0.14 |

(`hit_rate_at_k` is at the run's own `top_k`, e.g. `hit_rate_at_3` for
`smaller_top_k` — see raw JSON for exact metric names.)

### Phase 4 — evaluation (real, honestly-scoped)
`data/eval/retrieval_eval.json`: 15 questions, deterministic (seed 42),
built by `src/experiments/eval_set.py` from 15 real articles' own opening
8 words each (never fabricated); `expected_article_number` read directly
from `data/processed/civil_code.json`. Article 1022 (Day 1's empty-
`text_ar` gap) is structurally excluded (can't build a snippet from empty
text) — confirmed by a passing test.

Metrics computed: `hit_rate_at_k` (does the expected article appear in
the top-k results) and `mean_reciprocal_rank`. **Named and scoped
honestly as a self-retrieval lexical sanity check**, not a semantic RAG
quality score — the near-perfect hit rates above are expected and
explained by this (the query IS literally a substring of the target
article's own text), not evidence of strong semantic retrieval. No
RAGAS/semantic metric was attempted (dependency unavailable) or faked.

### Phase 5 — registry (written; execution blocked)
`register_best_run_artifact()`: finds the best run by a given metric,
and — only if that run's embedder is the locally-trained
`TfidfFallbackEmbedder` (real scikit-learn model, not a pretrained one)
— logs it via `mlflow.sklearn.log_model` and registers it under
`arabic-legal-rag-tfidf-embedder` via `mlflow.register_model`. The RAG
system as a whole is never treated as a conventional ML model. Called
from `run_experiments.py`'s `__main__`; in this sandbox it printed
`[mlflow] registry step skipped: mlflow is not installed...` — **0
artifacts registered**, honestly, not faked.

### Phase 6 — tests (actual result)
`python3 run_tests.py` -> **92 passed, 0 failed, 12 skipped** (up from
75/0/11 at end of Day 3; all Day 1-3 tests still passing, unchanged).
New Day 4 tests: `test_eval_set.py` (5, all run for real),
`test_evaluate.py` (3, all run for real), `test_mlflow_tracking.py` (5:
4 run for real — including confirming the real `MLflowUnavailable`
behavior for both `log_experiment_run` and `register_best_run_artifact`
— 1 skipped: the "creates a real run" test, which genuinely needs
mlflow), `test_run_experiments.py` (5, all run for real — configuration
count/uniqueness/required-fields checks, pure Python, no mlflow needed).
No skipped test was turned into a fake pass.

### Files changed/added this session
```
src/rag/embeddings.py               (additive: TfidfFallbackEmbedder now
                                      takes ngram_range, backward compatible)
src/experiments/__init__.py         (new)
src/experiments/eval_set.py         (new)
src/experiments/evaluate.py         (new)
src/experiments/tracking.py         (new)
src/experiments/run_experiments.py  (new)
data/eval/retrieval_eval.json       (new, generated -- real corpus data)
reports/day4_experiment_results.json (new, generated -- real measured results)
tests/test_eval_set.py              (new)
tests/test_evaluate.py              (new)
tests/test_mlflow_tracking.py       (new)
tests/test_run_experiments.py       (new)
pyproject.toml                      (experiments extra: mlflow>=2.10)
.gitignore                          (mlruns/, mlartifacts/)
README.md                           (Day 4 section appended)
PROGRESS.md                         (this section)
```

### Known limitations
- **0 real MLflow runs** exist anywhere — mlflow was never installed in
  this sandbox (no network). Nothing under `mlruns/` or any MLflow store
  was created or claimed.
- The 5 configurations' real, measured results exist only in
  `reports/day4_experiment_results.json` (local JSON), not as MLflow
  runs, until re-run with mlflow installed.
- All 5 configurations used the TF-IDF fallback (documented since Day
  2) — results say nothing about a real sentence-transformer's
  performance.
- Evaluation is a self-retrieval lexical sanity check (15 questions),
  not a semantic-quality or RAGAS-style metric.
- Registry: 0 artifacts registered (same root blocker).

### Exact starting point for Day 5
1. In an environment with network access: `pip install -e
   ".[rag,api,serving,load,experiments]"`.
2. Re-run `python3 -m src.experiments.run_experiments` for real — confirm
   5 real MLflow runs appear under experiment `arabic-legal-rag`
   (`mlflow ui` or `mlflow.search_runs`), with the exact same configs/
   metrics already recorded in `reports/day4_experiment_results.json` as
   a sanity cross-check.
3. Confirm the registry step actually registers
   `arabic-legal-rag-tfidf-embedder` version 1 (or whichever run wins).
4. Run `python3 run_tests.py` — expect the 1 currently-skipped mlflow
   test (`test_log_experiment_run_creates_a_real_run_when_mlflow_available`)
   to now execute and pass.
5. Optionally, once `sentence-transformers` is also available, add a 6th
   configuration using the real embedder and compare its metrics against
   the TF-IDF baselines above for the first time — do not claim this
   comparison before it is actually run.
6. Only once the above are verified for real should Day 5 (DVC, per the
   stop condition) begin.

---

# DAY 5

## Day 5 status: COMPLETE for everything verifiable in this sandbox; `dvc` execution BLOCKED (no network, confirmed)

### Phase 1 — inspection (confirmed facts, not assumed)
- Source PDF: `data/raw/civil_code.pdf` (already present from Day 1;
  already git-ignored via `.gitignore`'s `data/raw/*.pdf`).
- Exact ingestion command (unchanged since Day 1, confirmed by reading
  `src/ingestion/cli.py` and `README.md`):
  `python -m src.ingestion.cli ingest --pdf data/raw/civil_code.pdf --out data/processed/civil_code.json`
- Exact validation command: `python -m src.ingestion.cli validate --json data/processed/civil_code.json`
  — returns exit code 2 whenever `duplicate_numbers` is non-empty, which
  it always is here (articles 1 and 2 are legitimately duplicated —
  Promulgation Law vs. Civil Code renumbering, documented since Day 1).
  This means `validate`'s exit code cannot be used as a blind
  pass/fail gate for the DVC stage without the pipeline "failing" on
  every correct run — see Phase 4 below for how this was handled
  honestly instead of ignored.
- **Determinism: verified for real.** Ran the ingestion command twice
  against the real PDF, to two separate output files, and compared
  MD5 hashes: **identical** (`ac90c9c1001e961d7aa451ca6984c1c8` both
  times, matching the already-committed `data/processed/civil_code.json`
  exactly). Byte-for-byte reproducible — not assumed, checked, and now
  covered by a real, passing test
  (`test_pipeline_output_is_byte_for_byte_reproducible`) that reruns
  this exact check.

### Phase 2 — DVC environment (actual, confirmed result)
`import dvc` -> `ModuleNotFoundError`. `pip install dvc` -> "No matching
distribution found" (identical no-network signature to mlflow/fastapi/
chromadb/etc. every prior day). Did not retry repeatedly. `dvc init` was
never run — no `.dvc/` directory exists. Per instructions, created the
pipeline definition (`dvc.yaml`, pure hand-authored YAML — writing it
does not require the `dvc` package) and tested its structure statically
with PyYAML (which IS available). `dvc repro`/`dvc status` themselves
were not run — no `dvc` binary at all (`shutil.which("dvc")` -> `None`,
checked directly in the test suite).

### Phase 3 — DVC tracking (partially done; per-file tracking explicitly NOT faked)
- `data/raw/civil_code.pdf`: already excluded from git via
  `.gitignore`'s `data/raw/*.pdf` (set up in Day 2/3) — the "large PDF
  not committed as an ordinary git blob" requirement is already
  satisfied, confirmed by a real `git status` showing the PDF absent
  from the staged file list.
- `data/raw/civil_code.pdf.dvc` and `dvc.lock`: **deliberately not
  created.** These files specifically represent "dvc has run and
  computed a real content hash" — hand-writing them (even with a
  correctly-computed MD5) without actually running `dvc add`/`dvc repro`
  would misrepresent that as having happened. Documented as the first
  concrete Day 6 step instead of faked here.
- `data/processed/civil_code.json`: currently a normal, git-tracked file
  (not yet DVC-managed) since no `dvc repro`/`dvc add` has run to bring
  it under DVC's management. This is an accurate, unfaked description of
  the current state, not a target state.

### Phase 4 — `dvc.yaml` (written, real, structurally verified)
`extract_corpus` stage:
- `cmd`: the exact, unmodified Day 1 ingestion CLI command (confirmed
  identical to the one documented in README's "Run ingestion" section).
- `deps`: the source PDF + every `src/ingestion/*.py` module the command
  actually imports (`cli.py`, `pdf_reader.py`, `bidi_fix.py`,
  `arabic_utils.py`, `parser.py`, `schema.py`, `validation.py`) — all 8
  paths confirmed to exist on disk by a real, passing test.
- `outs`: `data/processed/civil_code.json`.
- Validation is **not** wired as a hard pass/fail gate on the stage —
  explained in an inline comment in `dvc.yaml` itself (see Phase 1 above
  for why: the "not clean" exit code is a false positive caused by a
  real, intentional, documented corpus characteristic, not a defect).
  `run_ingest` already prints the validation WARNING to stderr on every
  run, so the information is still surfaced, just not used to fail the
  stage.
- No second/duplicate parser was created anywhere — confirmed by a real,
  passing test restricting every `.py` dependency to
  `src/ingestion/`.

### Phase 5 — reproducibility (verified for real, see Phase 1)
Byte-for-byte identical output across two runs of the real command
against the real PDF. Not claimed beyond what was checked: this
confirms the *ingestion command itself* is deterministic; it does not
confirm `dvc repro`'s own caching/hash-comparison behavior, since `dvc
repro` was never run.

### Phase 6 — git status / dvc status (git run for real; dvc blocked)
`git init` was run for real in this session (no prior git repo existed).
`git add -A` + `git status --short` showed 58 files staged, with
`data/raw/civil_code.pdf` correctly **absent** (gitignored) and
`data/processed/civil_code.json` correctly **present** (still plain
git-tracked, per Phase 3 above). Committed
(`cb480dc "Day 5: initial commit with DVC pipeline definition
(dvc.yaml)"`). `dvc status` was not run (no `dvc` binary).

### Phase 7 — tests (actual result)
`python3 run_tests.py` -> **100 passed, 0 failed, 13 skipped** (up from
92/0/12 at end of Day 4; all Day 1-4 tests still passing, unchanged).
New `tests/test_dvc_pipeline.py` (9 tests): 8 run for real (YAML
structure, correct deps/outs, reuses existing CLI, no duplicate parser,
all referenced paths exist on disk, byte-for-byte reproducibility); 1
skips (`dvc status` via the actual binary — genuinely needs `dvc`
installed).

### Files changed/added this session
```
dvc.yaml                       (new -- pipeline definition)
tests/test_dvc_pipeline.py     (new -- 9 tests, 8 run for real)
README.md                      (Day 5 section appended)
PROGRESS.md                    (this section)
.git/                          (new -- git repo initialized this session)
```

### `dvc repro` result
**Not executed** — no `dvc` binary in this sandbox.

### `dvc status` result
**Not executed** — no `dvc` binary in this sandbox. (`git status` WAS
run for real — see Phase 6.)

### Reproducibility verification result
**Verified for real**: two runs of the exact `dvc.yaml` stage command
against the real tracked PDF produce byte-for-byte identical JSON
(MD5 `ac90c9c1001e961d7aa451ca6984c1c8` both times). Article count (1095),
article numbers, repeal flags, hierarchy metadata, and text fields are
therefore also identical between runs (implied by the byte-for-byte
match, not spot-checked separately since the stronger byte-level claim
already subsumes it).

### Known limitations
- 0 real `dvc repro`/`dvc status`/`dvc add` executions — `dvc` was never
  installed in this sandbox (no network). Nothing under `.dvc/` or
  `dvc.lock` was created or claimed.
- `data/processed/civil_code.json` is not yet DVC-managed (still a plain
  git-tracked file) — bringing it under DVC's management is a Day 6 task,
  not done here to avoid faking it.
- `data/raw/civil_code.pdf.dvc` does not exist — same reason.
- The "no hard-fail on validate" design decision (Phase 4) means the DVC
  stage cannot currently catch a *genuinely new* validation regression
  automatically; it only surfaces it via the printed WARNING. Revisiting
  this (e.g. a stricter check that distinguishes the known-acceptable
  duplicates from unexpected new ones) is a reasonable future
  improvement, not attempted here since it would require modifying Day 1
  validation logic, out of scope for "do not redo Days 1-4."

### Exact starting point for Day 6
1. In an environment with network access: `pip install dvc`.
2. `dvc init` (inside this git repo).
3. `dvc add data/raw/civil_code.pdf` — confirm the real, dvc-generated
   `data/raw/civil_code.pdf.dvc` and compare its md5 against
   `hashlib.md5(open('data/raw/civil_code.pdf','rb').read()).hexdigest()`
   computed independently, as a sanity cross-check.
4. `dvc repro` — confirm it runs `extract_corpus`, produces
   `dvc.lock`, and that the resulting `data/processed/civil_code.json`
   has the same MD5 (`ac90c9c1001e961d7aa451ca6984c1c8`) already
   recorded in this file, confirming true end-to-end reproducibility
   through `dvc repro` itself (not just the raw CLI command).
5. `dvc status` — confirm it reports up to date after step 4.
6. Update `.gitignore`/git tracking so `data/processed/civil_code.json`
   moves from a plain git blob to DVC-managed (matching what `dvc add`/
   `dvc repro` naturally produces), commit the resulting `.dvc`/
   `dvc.lock` files.
7. Run `python3 run_tests.py` — expect the 1 currently-skipped dvc test
   (`test_dvc_binary_status_check`) to now execute and pass.
8. Only once the above are verified for real should Day 6 begin.

---

# DAY 6

## Day 6 status: COMPLETE for everything verifiable locally; GitHub execution itself never occurred (no network/remote — documented, not faked)

### Phase 1 — inspection (confirmed)
- `PROGRESS.md`, `README.md`, `pyproject.toml`, `run_tests.py`, `tests/`,
  `src/`, `dvc.yaml` all reviewed before any change.
- `git status`: clean at session start (Day 5's two commits were the
  most recent). Branch: `master`. `git remote -v`: empty — no GitHub
  remote configured, confirmed before writing any workflow.
- `ragas`: `ModuleNotFoundError`. `pip install ragas` -> "No matching
  distribution found" (same no-network signature as every prior day).
- Lint tools (`ruff`/`flake8`/`black`): none installed here either.
- `act` (local GitHub Actions runner): not found. Direct request to
  `github.com`: denied by this sandbox's network policy
  (`x-deny-reason: host_not_allowed`). **This workflow has never run
  anywhere** — not locally (no `act`), not on GitHub (no network/remote).

### Phase 2 — GitHub Actions workflow (written; never executed anywhere)
`.github/workflows/ci.yml`:
- Triggers on `push` and `pull_request` (the `"on"` key is explicitly
  quoted in the YAML to avoid the classic PyYAML boolean-coercion gotcha
  where an unquoted `on:` parses as the boolean `True` — verified this
  matters and fixed it, confirmed by re-parsing).
- `test` job: checkout, Python 3.11 with pip caching, installs via
  `pip install -e ".[dev,rag,api,serving,load,experiments]"` (the
  project's own `pyproject.toml` config, no separate dependency list),
  `ruff check .` (new minimal lint config added to `pyproject.toml`,
  `select = ["E9", "F"]` — syntax errors + pyflakes only, not a full
  style rewrite of 5 days of code), `pytest -q`, `python3 run_tests.py`
  (parity check against the bespoke runner used throughout this
  sandbox), and `scripts/check_corpus_quality.py`.
- Does not reference any local sandbox path — checked by a real, passing
  test (`test_ci_does_not_reference_any_local_sandbox_path`).

### Phase 3 — RAGAS quality gate (written; fails closed, honestly, for two independent reasons)
`scripts/ragas_gate.py`: fails closed (exit 1) unless BOTH (1) `ragas` is
importable AND (2) a real generation backend is configured
(`GENERATION_MODEL` env var). **Never** substitutes the Day 4 lexical
hit-rate/MRR numbers as if they were faithfulness — confirmed by a real,
passing test that scans the script's own output for a forbidden
"faithfulness score" line. Run for real in this sandbox:
```
RAGAS faithfulness gate: UNAVAILABLE -- failing closed.
  - `ragas` package is not installed/importable.
  - no generation backend is configured (GENERATION_MODEL env var unset)...
Activation requirements to make this gate real: ...
Configured threshold (unused until then): 0.7
```
Exit code: **1**. This is the honest, current, correct state — this
pipeline has been retrieval-only by design since Day 2, so even a
hypothetical `ragas`-available environment could not compute a real
faithfulness score yet (there is no generated answer to evaluate). The
`ragas-quality-gate` CI job wraps this script and is expected to fail on
every run until a real generation backend exists — documented as
intentional, not silenced.

### Phase 4 — DVC integration in CI (written; honestly reports a structural blocker, not just "dvc not installed")
The `dvc-pipeline` job installs `dvc` for real (this WILL work on a
GitHub runner, unlike this sandbox) but explicitly checks whether
`data/raw/civil_code.pdf` is present in the checkout **before** attempting
`dvc repro`. It is not: the PDF is git-ignored (Day 2/3) and no DVC
remote exists to fetch it (Day 5 intentionally did not add one). The job
step detects this and exits 1 with a clear explanatory message — a
required-but-currently-blocked check, reported honestly, never silently
skipped or passed. No `dvc.lock`/`.dvc` file was hand-created anywhere in
this session (same policy as Day 5).

### Phase 5 — tests (actual result)
`python3 run_tests.py` -> **119 passed, 0 failed, 13 skipped** (up from
100/0/13 at end of Day 5 — 19 new tests, all executable and run for real,
skip count unchanged). All Day 1-5 tests still passing.
- `tests/test_ci_workflow.py` (10 tests, all real): workflow YAML valid,
  triggers correct, test job has checkout+setup-python, install command
  uses `pyproject.toml` extras (not a hand-written duplicate list), runs
  both `pytest` and `run_tests.py`, runs the corpus gate, RAGAS job
  exists and depends on `test`, RAGAS job never hardcodes a fake passing
  score, DVC job exists and checks for the PDF, no local sandbox path
  anywhere in the workflow.
- `tests/test_quality_gates.py` (9 tests, all real): corpus gate passes
  on the known-good baseline and against the REAL current corpus
  end-to-end; fails on synthetic new-duplicate/count-change/new-issue
  scenarios (regression protection actually exercised, not just
  asserted); RAGAS gate correctly reports unavailable in this sandbox,
  fails closed (nonzero exit), and never prints a fabricated numeric
  score.

### Files changed/added this session
```
.github/workflows/ci.yml        (new)
scripts/check_corpus_quality.py (new)
scripts/ragas_gate.py           (new)
tests/test_ci_workflow.py       (new)
tests/test_quality_gates.py     (new)
pyproject.toml                  (lint extra + [tool.ruff] config added)
README.md                       (Day 6 section appended)
PROGRESS.md                     (this section)
```

### Actual GitHub Actions execution status
**Never executed, anywhere.** No `act` binary in this sandbox (checked
directly), no network to GitHub (checked directly, request denied), no
`git remote` configured (checked directly, empty). Everything above that
says "real"/"actual" refers to local execution in this sandbox
(`scripts/*.py` run directly, `pytest`/`run_tests.py` run directly, YAML
structurally parsed) — never to an actual GitHub Actions run, which does
not exist yet for this project.

### Known limitations
- The workflow has 0 real GitHub executions to point to.
- The `ragas-quality-gate` job is expected to **always fail** until a
  real generation backend exists (Day 7+ scope) — by design, not a bug.
- The `dvc-pipeline` job is expected to **always fail** at the
  PDF-presence check until a DVC remote (or equivalent) is configured —
  by design, not a bug.
- `ruff`'s actual lint output on this codebase is unknown — never run
  here (no network/package). The rule set is deliberately minimal
  (`E9`, `F` only) to reduce the chance of a large, unreviewed
  first-run diff, but even that hasn't been checked for real yet.

### Exact starting point for Day 7
1. Push this repository to a real GitHub remote; confirm the `test` job
   passes for real (all extras install, lint, pytest, run_tests.py,
   corpus gate) — this is the first real GitHub Actions execution this
   project will ever have.
2. Fix whatever `ruff check .` flags for real (unknown until run).
3. Decide and implement a real generation backend (wire up
   `GENERATION_MODEL`), then extend `scripts/ragas_gate.py`'s real path
   to actually call `ragas.evaluate(...)` against real
   (question, retrieved_contexts, generated_answer) triples — only then
   should the `ragas-quality-gate` job be expected to pass.
4. Configure a real DVC remote (or another way to make
   `data/raw/civil_code.pdf` available to CI) so `dvc-pipeline` can get
   past its PDF-presence check and actually attempt `dvc repro`/`dvc status`.
5. Re-run `python3 run_tests.py` after all of the above — expect the
   currently-skipped mlflow/dvc/bentoml/locust/api tests to also start
   executing once their packages are genuinely installed in a networked
   environment.
6. Only once real GitHub Actions results exist for all three jobs should
   Day 7 (serving/optimization) begin.

---

# RECOVERY NOTE (between Day 6 and Day 7)

The development sandbox's working directory (`/home/claude/project`),
including the full git history through Day 6 (`cb480dc` … `9611fa5`,
and the Day 7/Day 8 commits that followed it, `fa30fb5` and `9611fa5`
respectively per prior session reports), was lost to an environment
reset between sessions — not caused by any project command; confirmed
by direct inspection (the directory simply did not exist). An
exhaustive filesystem search (`.git` directories, loose/packed git
objects, any other zip/tar/backup, all other home directories, `/tmp`)
found **nothing recoverable** beyond a user-provided backup zip of the
Day 6 state.

**Recovery performed:**
1. Verified the user's backup zip (`arabic-legal-rag-day6.zip`, inside
   `files__3_.zip`) with `unzip -t` — no errors.
2. Extracted it to a fresh `/home/claude/project`.
3. Verified integrity against documented baselines: corpus JSON MD5
   `ac90c9c1001e961d7aa451ca6984c1c8` (exact match), source PDF MD5
   `5086ef5f16f15d856278db943e58e948` (exact match against the original
   upload), 1095 articles, `python3 run_tests.py` → 119 passed, 0
   failed, 13 skipped (exact match to the documented Day 6 result).
4. Created a separate backup
   (`/mnt/user-data/outputs/day6-recovery-baseline-backup.zip`, via
   `git archive` on the baseline commit) before any further
   modification.
5. Re-initialized git (`git init`) and committed the verified Day 6
   state as commit `9ee3052` ("Restore: Day 6 baseline recovered from
   user-provided backup zip...").

**The old commit hashes are not recovered and never will be** — this is
a new git history starting from `9ee3052`. Day 7 and Day 8 below are
**rebuilt from scratch** against this restored baseline, as new,
independently-written and independently-tested work — not a restoration
of the lost originals, even though the design intentionally follows the
same approach described in the lost work's own prior session reports
(which I have no way to verify matched the actual lost files byte-for-
byte).

---

# DAY 7 (rebuilt)

## Status: COMPLETE for everything verifiable locally; same environment blockers as before (ruff/dvc/ragas/GitHub Actions)

### What was rebuilt
- `src/api/validation.py` (new): `validate_top_k`, `validate_question`
  (with `MAX_QUESTION_LENGTH`), dependency-free.
- `src/api/main.py`: wired to use the above; `RAG_TOP_K` validated at
  import time.
- `.github/workflows/ci.yml` + `pyproject.toml`: fixed `ruff`/`dvc`/
  `ragas` installs to go through the project's own extras instead of
  separate hand-written `pip install` lines. Added `lint`, `dvc`,
  `ragas` extras.
- `src/experiments/ragas_eval.py` (new): `build_ragas_records()` +
  `evaluate_faithfulness()`, reusing the existing `Retriever`.
- `scripts/ragas_gate.py`: comment updated to reference the new
  interface; fail-closed behavior unchanged and re-verified.

### Tests (actual result)
`tests/test_api_validation.py` (11, new), `tests/test_ragas_eval.py` (4,
new), `tests/test_ci_workflow.py` (+2), `tests/test_api.py` (+4, all
skip — fastapi unavailable). Running after Day 7's changes alone:
**136 passed, 0 failed, 18 skipped**.

### Committed
`0e27354` — "Day 7 (rebuilt): API validation module, CI extras
consistency fix, RAGAS integration interface"

---

# DAY 8 (rebuilt)

## Status: COMPLETE for everything verifiable locally; Langfuse never executed (no package/credentials/network)

### What was rebuilt
- `src/observability/logger.py` (new): `build_event()` (allowlist-
  enforced), `log_event()` (never raises, call-time-resolved default
  path — verified this matters with a dedicated regression test),
  `now_iso()`.
- `src/observability/monitoring.py` (new): `summarize_retrieval()`
  (real data only, explicitly scoped away from answer-quality/
  faithfulness), `print_summary()`.
- `src/observability/langfuse_adapter.py` (new): `send_trace()` fails
  honestly (`sent: False` + reason).
- `src/api/main.py`: `/ask` now logs one event per request (success,
  503, 500 paths) with zero sensitive content.

### Manual end-to-end verification (real, this session)
```
events: 3
{'n_requests': 3, 'error_count': 1, 'error_rate': 0.333...,
 'empty_result_count': 2, 'empty_result_rate': 0.666...,
 'avg_result_count': 1.0, 'avg_latency_ms': 8.533...,
 'min_latency_ms': 5.0, 'max_latency_ms': 12.5}
```
Raw log file inspected directly: 3 JSON lines, only allowlisted fields,
no question text, no secrets.

### Tests (actual result)
`tests/test_observability_logger.py` (10, new), 
`tests/test_observability_monitoring.py` (9, new),
`tests/test_langfuse_adapter.py` (5, new), `tests/test_api.py` (+1
logging-specific case, skip — fastapi unavailable).

**Final combined Day 7 + Day 8 test result:**
```
$ python3 run_tests.py
160 passed, 0 failed, 18 skipped
```

### Corpus integrity (re-verified after all Day 7/8 changes)
- MD5: `ac90c9c1001e961d7aa451ca6984c1c8` — **unchanged, exact match**.
- Article count: **1095** — unchanged.
- `scripts/check_corpus_quality.py` → PASSED, exit 0.

### Lint/DVC/Langfuse/Docker status (checked, not retried)
`ruff`, `dvc`, `langfuse`: `ModuleNotFoundError` — not installed, no
network, not retried. `docker`: no binary. None of these were executed;
none of their results are claimed.

### Files added/modified this session (Day 7 + Day 8 combined)
```
NEW:
  src/api/validation.py
  src/experiments/ragas_eval.py
  src/observability/__init__.py
  src/observability/logger.py
  src/observability/monitoring.py
  src/observability/langfuse_adapter.py
  tests/test_api_validation.py
  tests/test_ragas_eval.py
  tests/test_observability_logger.py
  tests/test_observability_monitoring.py
  tests/test_langfuse_adapter.py
MODIFIED:
  src/api/main.py
  scripts/ragas_gate.py
  pyproject.toml
  .github/workflows/ci.yml
  tests/test_api.py (+5 tests)
  tests/test_ci_workflow.py (+2 tests)
  README.md
  PROGRESS.md (this file)
```

### Git commits this session
```
9ee3052  Restore: Day 6 baseline recovered from user-provided backup zip
0e27354  Day 7 (rebuilt): API validation module, CI extras consistency
         fix, RAGAS integration interface
<Day 8 commit — see below>
```

### What remains blocked (unchanged root cause: no network)
- `ruff check .`: never run anywhere.
- Real GitHub Actions execution: never occurred (no remote, no network).
- Real RAGAS faithfulness score: blocked by missing `ragas` AND no
  generation backend.
- Real `dvc repro`/`dvc status`: blocked by missing `dvc` AND the PDF
  being absent from any fresh checkout (git-ignored, no DVC remote).
- Real Langfuse trace: blocked by missing package, credentials, server.
- Docker build/run: no `docker` binary.

### Exact Day 9 starting point
Same as documented before the environment reset: once network access
exists, install every extra, push to a real GitHub remote, run `ruff`
for real and fix findings, configure a DVC remote, wire a real
generation backend before RAGAS can produce a real score, and configure
real Langfuse credentials before any trace can be genuinely sent. Until
then, Day 9 should continue the same pattern as Days 7-8: honest,
testable groundwork rather than claimed executions of blocked
integrations.

---

# DAY 9

## Status: COMPLETE — requirement matrix, integration audit, release readiness, submission prep

Full detail in `reports/day9_report.md` and `reports/requirement_matrix.md`.
Summary:
- Built a requirement traceability matrix from this project's cumulative
  stated requirements (explicitly **not** verified against an actual
  course handbook document — none has ever been provided to this
  session).
- Ran a real end-to-end integration check (corpus → chunking → vector
  store → retriever → logger → monitoring) against the restored corpus
  — all real execution, output recorded in the Day 9 report.
- Audited Dockerfile/bentofile.yaml/dvc.yaml path consistency after the
  Day 7/8 rebuild — found them already correct; found and fixed one
  real gap (README's env-var list was missing `MAX_QUESTION_LENGTH` and
  `RAG_LOG_DIR`, both genuinely used in code).
- Added `tests/test_integration_pipeline.py` (5 new tests, all real, no
  mocks) formalizing the integration audit as a permanent regression
  test.
- Reviewed Day 4's experiment results and re-stated their scope/limits
  explicitly (lexical self-retrieval sanity check, not semantic/answer
  quality — unchanged from Day 4, nothing new fabricated).
- Wrote a deliverables checklist and a beginner Windows execution
  checklist (explicitly labeled as unexecuted guidance, not a claim
  that Windows execution occurred).

### Test results
`python3 run_tests.py` → **165 passed, 0 failed, 18 skipped** (up from
160/0/18 at the end of Day 8).

### Corpus integrity (re-verified)
MD5 `ac90c9c1001e961d7aa451ca6984c1c8`, 1095 articles — unchanged.
`scripts/check_corpus_quality.py` → PASSED, exit 0.

### Committed
`21840e2` — "Day 9: requirement traceability matrix, integration audit
(1 real gap fixed), integration tests, release readiness, submission
checklists"


---

# PHASE 2 (semantic RAG + evaluation + MLflow foundation)

## Status: code COMPLETE; everything needing sentence-transformers / chromadb / ragas / mlflow / vLLM is UNVERIFIED (not installable here, no network)

**Implemented:** strict/auto/tfidf embedding backends with labelled `embedding_kind`; manifest-driven vector-store backend (fixes Phase 1 build/load mismatch), Chroma fixes (cosine, None-metadata, clean rebuild), chunk `overlap`; full-article text (`text_ar + text_en`) for embedding/sources/contexts; 57-question curated bilingual eval set with corpus-extracted references; four-metric RAGAS integration; faithfulness gate at 0.75 (exit 1/2/0); 7-config MLflow runner, documented best-run selection, registry via pyfunc model; compose index volume made writable + HF cache.

**VERIFIED (real execution):** 253 tests passed / 0 failed / 21 skipped; curated set validates against corpus; 2 TF-IDF control runs on the 57 questions (config tfidf_control_c1000: hit@5 0.860, MRR@10 0.747; tfidf_control_bigram_c500_o100: hit@5 0.789, MRR@10 0.689) in `reports/phase2_experiment_results.json`; gate fails closed (exit 1) in this sandbox.

**UNVERIFIED / BLOCKED:** any semantic-embedding result; real Chroma; RAGAS scores; MLflow runs and registry entries (none exist); vLLM generation. Tests for those use stand-ins (`tests/fakes.py`) that check our wiring only.

**Caveats:** the eval questions were written from the article text, so they favour lexical overlap and are easier than real user questions; Articles 1 and 2 are duplicated in the corpus and excluded. Package versions are pinned to the APIs written against (`ragas>=0.2.6,<0.3`, `mlflow>=2.13,<3`) but never installed here.
