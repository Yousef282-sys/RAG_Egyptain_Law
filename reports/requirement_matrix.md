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
| 3 | Embeddings (semantic) | `src/rag/embeddings.py` — `SentenceTransformerEmbedder` (real path) + `TfidfFallbackEmbedder` (actual path used) | `tests/test_embeddings.py` | **Implemented, TF-IDF path verified; semantic path externally unverified** | no network to install/download a real sentence-transformer model |
| 4 | Vector store | `src/rag/vector_store.py` — `ChromaVectorStore` (real path) + `SimpleVectorStore` (actual path used) | `tests/test_vector_store.py`, real 1107-chunk index built | **Implemented, numpy fallback verified; Chroma path externally unverified** | `chromadb` not installed, no network |
| 5 | Retrieval | `src/rag/retriever.py` | `tests/test_retriever.py`, real queries run against real corpus | **Implemented & locally verified** | none |
| 6 | FastAPI `/ask`, `/health` | `src/api/main.py` | `tests/test_api.py` (structurally correct; execution gated) | **Implemented, externally unverified** | `fastapi`/`pydantic`/`uvicorn` not installed |
| 7 | API input validation (422 on empty/long/malformed) | `src/api/validation.py` | `tests/test_api_validation.py` (11 tests, run for real — dependency-free by design) | **Implemented & locally verified** (validation logic); **externally unverified** (actual HTTP layer) | fastapi unavailable for the HTTP-layer tests |
| 8 | Docker image | `Dockerfile`, `.dockerignore` | `tests/test_docker_config.py` (structural only) | **Implemented, externally unverified** | no `docker` binary in this sandbox |
| 9 | BentoML serving | `src/serving/bento_service.py`, `bentofile.yaml` | `tests/test_bentoml_service.py` (1 real structural test, rest skip) | **Implemented, externally unverified** | `bentoml` not installed |
| 10 | Load testing, 50 concurrent users | `tests/load/locustfile.py` | `tests/test_load_locustfile.py` (syntax verified; execution skips) | **Implemented, never executed — zero benchmark numbers exist** | `locust` not installed, no running server |
| 11 | MLflow experiment tracking, ≥5 runs | `src/experiments/tracking.py`, `run_experiments.py` | `reports/day4_experiment_results.json` — 5 real configs run and evaluated **locally**, never logged as real MLflow runs | **Implemented, local evaluation verified; MLflow logging never executed** | `mlflow` not installed |
| 12 | MLflow model/artifact registry | `src/experiments/tracking.py::register_best_run_artifact` | fails closed with real `MLflowUnavailable`, tested | **Implemented, never executed** | `mlflow` not installed |
| 13 | DVC pipeline (`dvc.yaml`) | `dvc.yaml` | `tests/test_dvc_pipeline.py` (structural + real reproducibility check of the underlying command, not of `dvc` itself) | **Implemented, `dvc` itself never executed** | `dvc` not installed; even if it were, no DVC remote exists to fetch the git-ignored PDF in a fresh checkout |
| 14 | GitHub Actions CI (`ci.yml`) | `.github/workflows/ci.yml` | `tests/test_ci_workflow.py` (structural, incl. 2 new consistency-regression tests) | **Implemented, never executed on GitHub** | no git remote, no network to github.com |
| 15 | RAGAS faithfulness quality gate | `scripts/ragas_gate.py`, `src/experiments/ragas_eval.py` | fails closed, tested for real (package absent + no generation backend — both independently confirmed) | **Implemented, correctly fails closed; never produces a real score** | `ragas` not installed AND no generation backend exists anywhere in this project |
| 16 | Structured observability logging | `src/observability/logger.py` | `tests/test_observability_logger.py` (10 real tests) + manual end-to-end run (see PROGRESS.md) | **Implemented & locally verified** | none |
| 17 | Retrieval monitoring/summary | `src/observability/monitoring.py` | `tests/test_observability_monitoring.py` (9 real tests) + manual run | **Implemented & locally verified** | none |
| 18 | Optional Langfuse tracing | `src/observability/langfuse_adapter.py` | `tests/test_langfuse_adapter.py` (5 real tests) | **Implemented, correctly reports non-sent; never sends a real trace** | `langfuse` not installed, no server, no credentials |
| 19 | Reproducible corpus (byte-for-byte) | `src/ingestion/cli.py` + `dvc.yaml` | re-run twice, MD5-compared, identical | **Implemented & locally verified** | none |
| 20 | Known-good corpus regression gate | `scripts/check_corpus_quality.py` | run repeatedly, PASSED every time this session | **Implemented & locally verified** | none |
| 21 | Final submission packaging (this task) | `reports/day9_report.md` (being written), deliverables checklist, Windows checklist | this session | **In progress** | n/a |

## What "complete all deliverables" cannot mean here, honestly
Items 6, 8–14, 18 above have a **real implementation that compiles,
passes every test that can run without the missing dependency, and is
ready to execute** — but their actual execution requires something this
sandbox does not have access to (network, a `docker` binary, a GitHub
remote, real credentials, a real ML model). No amount of additional
local work changes that. I will not mark these "complete" in the sense
of "executed successfully" — only "implemented and ready," which is a
different, narrower claim.
