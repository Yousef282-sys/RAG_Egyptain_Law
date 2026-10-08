# Requirement Traceability Matrix

**Important caveat before reading this matrix:** no course handbook
document has ever been uploaded to this conversation. Every requirement
below is drawn from the cumulative, explicit instructions given across
Days 1–9 of this project (PDF→JSON ingestion, RAG pipeline, API,
Docker/BentoML/Locust, MLflow, DVC, GitHub Actions + RAGAS gate,
observability). This is **not** a verified mapping against an actual
"M1–M5" handbook structure — I have never seen that document. Where the
prompts used handbook-sounding language (e.g. "the handbook requires
BentoML", "50 concurrent users, as required by the handbook"), I've
carried that requirement forward as stated, without independent
verification that it's accurately transcribed.

Status legend:
- **Implemented & locally verified** — code exists, ran successfully in
  this sandbox, with real output to show for it.
- **Implemented, externally unverified** — code exists and is
  structurally/unit-tested, but its real target (GitHub, a real model,
  a real server) was never reached from this sandbox.
- **Blocked** — cannot be done here at all (no network/binary/credential
  of a kind no amount of local coding fixes).

| # | Requirement | Implementation | Test/Evidence | Status | Blocker |
|---|---|---|---|---|---|
| 1 | PDF → structured JSON corpus | `src/ingestion/` (parser, bidi_fix, arabic_utils, validation) | `tests/test_parser.py` etc., `reports/day1_spot_check.md`, corpus MD5-verified | **Implemented & locally verified** | none |
| 2 | Article chunking | `src/rag/chunking.py` | `tests/test_chunking.py` | **Implemented & locally verified** | none |
| 3 | Embeddings (semantic) | `src/rag/embeddings.py` — strict `SentenceTransformerEmbedder` (configurable model/prefixes/revision via `EMBEDDING_MODEL`, `EMBEDDING_BACKEND`) + explicit `TfidfFallbackEmbedder` (`embedding_kind: tfidf-lexical`, never called semantic) | `tests/test_embeddings_semantic.py` (against a sentence-transformers stand-in) | **VERIFIED: config/labels/strictness logic. UNVERIFIED: real model load + retrieval quality** | `sentence-transformers` not installed, no network |
| 4 | Vector store | `src/rag/vector_store.py` — Chroma (cosine, metadata round-trip, clean rebuild) + numpy fallback; backend always reopened from the index manifest; all of article_number/citation/book/chapter/section/topic/source_page/is_repealed preserved | `tests/test_vector_store_backends.py` (numpy store real; Chroma vs an in-memory stand-in) | **VERIFIED: numpy path + metadata. UNVERIFIED: real chromadb** | `chromadb` not installed |
| 5 | Retrieval | `src/rag/retriever.py` | `tests/test_retriever.py`, real queries run against real corpus | **Implemented & locally verified** | none |
| 6 | FastAPI `/ask`, `/health` | `src/api/main.py` | `tests/test_api.py` (structurally correct; execution gated) | **Implemented, externally unverified** | `fastapi`/`pydantic`/`uvicorn` not installed |
| 7 | API input validation (422 on empty/long/malformed) | `src/api/validation.py` | `tests/test_api_validation.py` (11 tests, run for real — dependency-free by design) | **Implemented & locally verified** (validation logic); **externally unverified** (actual HTTP layer) | fastapi unavailable for the HTTP-layer tests |
| 8 | Docker image | `Dockerfile`, `.dockerignore` | `tests/test_docker_config.py` (structural only) | **Implemented, externally unverified** | no `docker` binary in this sandbox |
| 9 | BentoML serving | `src/serving/bento_service.py`, `bentofile.yaml` | `tests/test_bentoml_service.py` (1 real structural test, rest skip) | **Implemented, externally unverified** | `bentoml` not installed |
| 10 | Load testing, 50 concurrent users | `tests/load/locustfile.py` | `tests/test_load_locustfile.py` (syntax verified; execution skips) | **Implemented, never executed — zero benchmark numbers exist** | `locust` not installed, no running server |
| 11 | MLflow experiment tracking, ≥5 runs | `src/experiments/run_experiments.py` (7 configs: 5 semantic + 2 TF-IDF controls; logs chunk_size, overlap, embedding_model, faithfulness + 3 more RAGAS metrics, retrieval metrics, artifacts) | `tests/test_run_experiments.py`, `tests/test_mlflow_registry.py` (stand-in mlflow); `reports/phase2_experiment_results.json` = 2 real TF-IDF control runs, 5 semantic runs honestly SKIPPED | **Implemented. UNVERIFIED: no real MLflow run exists** | `mlflow`, `sentence-transformers`, vLLM/judge not available |
| 12 | MLflow best config / registry | `tracking.select_best_run` (documented rule: faithfulness if every run has it, else MRR@10) + `register_best_configuration` (pyfunc model = config + index, alias `champion`) | selection logic tested for real; registration vs stand-in mlflow | **Selection VERIFIED. Registration UNVERIFIED — nothing is registered** | `mlflow` not installed |
| 13 | DVC pipeline (`dvc.yaml`) | `dvc.yaml` | `tests/test_dvc_pipeline.py` (structural + real reproducibility check of the underlying command, not of `dvc` itself) | **Implemented, `dvc` itself never executed** | `dvc` not installed; even if it were, no DVC remote exists to fetch the git-ignored PDF in a fresh checkout |
| 14 | GitHub Actions CI (`ci.yml`) | `.github/workflows/ci.yml` | `tests/test_ci_workflow.py` (structural, incl. 2 new consistency-regression tests) | **Implemented, never executed on GitHub** | no git remote, no network to github.com |
| 15 | RAGAS 4 metrics + faithfulness gate ≥ 0.75 | `scripts/ragas_gate.py` (exit 1 unavailable / 2 below 0.75 / 0 pass), `src/experiments/ragas_eval.py` (faithfulness, answer_relevancy, context_precision, context_recall), CI job at 0.75 | `tests/test_quality_gates.py`, `tests/test_ragas_integration.py`: decision logic + every fail-closed branch with injected inputs | **Gate logic VERIFIED; RAGAS evaluation UNVERIFIED — no RAGAS score exists** | `ragas`, a vLLM generation endpoint and a judge LLM are unavailable here |
| 16 | Structured observability logging | `src/observability/logger.py` | `tests/test_observability_logger.py` (10 real tests) + manual end-to-end run (see PROGRESS.md) | **Implemented & locally verified** | none |
| 17 | Retrieval monitoring/summary | `src/observability/monitoring.py` | `tests/test_observability_monitoring.py` (9 real tests) + manual run | **Implemented & locally verified** | none |
| 18 | Optional Langfuse tracing | `src/observability/langfuse_adapter.py` | `tests/test_langfuse_adapter.py` (5 real tests) | **Implemented, correctly reports non-sent; never sends a real trace** | `langfuse` not installed, no server, no credentials |
| 19 | Reproducible corpus (byte-for-byte) | `src/ingestion/cli.py` + `dvc.yaml` | re-run twice, MD5-compared, identical | **Implemented & locally verified** | none |
| 20 | Known-good corpus regression gate | `scripts/check_corpus_quality.py` | run repeatedly, PASSED every time this session | **Implemented & locally verified** | none |
| 21 | Final submission packaging (this task) | `reports/day9_report.md` (being written), deliverables checklist, Windows checklist | this session | **In progress** | n/a |
| 18 | ≥ 50 evaluation questions (bilingual) | `data/eval/retrieval_eval.json` (57: 28 EN + 29 AR) built by `python -m src.experiments.eval_set` from `curated_questions.py`; references are extracted verbatim from the corpus | `tests/test_eval_curated.py` (validated against the corpus: article exists once, reference == corpus extract) | **VERIFIED** (questions are paraphrases of article text; not independently reviewed by a legal expert) | none |

## What "complete all deliverables" cannot mean here, honestly
Items 6, 8–14, 18 above have a **real implementation that compiles,
passes every test that can run without the missing dependency, and is
ready to execute** — but their actual execution requires something this
sandbox does not have access to (network, a `docker` binary, a GitHub
remote, real credentials, a real ML model). No amount of additional
local work changes that. I will not mark these "complete" in the sense
of "executed successfully" — only "implemented and ready," which is a
different, narrower claim.
