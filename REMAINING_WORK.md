# Remaining Work Checklist

Everything below is **implemented in code** but has **never been
executed** in this project's development history, unless otherwise
noted. Check items off only after you have personally run the command
and seen real output — not after reading the code and judging it looks
correct.

## Docker
- [ ] `docker build -t arabic-legal-rag .` succeeds without error
- [ ] `docker run` starts the container and `/health` responds
- [ ] `/ask` responds correctly from inside the container (requires the
      vector index to be built/mounted first)
- [ ] Decide whether to bake a pre-built index into the image or always
      mount/build it at runtime (current `Dockerfile` assumes the
      latter — see its comments)

## Real MLflow tracking
- [ ] `pip install mlflow` succeeds
- [ ] `python -m src.experiments.run_experiments` produces 5 **real**
      MLflow runs under the experiment name `arabic-legal-rag` (not
      just the local JSON output already in
      `reports/day4_experiment_results.json`)
- [ ] `mlflow ui` shows those runs with correct params/metrics
- [ ] `register_best_run_artifact()` (`src/experiments/tracking.py`)
      successfully registers a model version in the MLflow Model
      Registry

## DVC initialization and pipeline execution
- [ ] `dvc init` succeeds in this repository
- [ ] `dvc add data/raw/civil_code.pdf` produces a real `.dvc` file with
      a genuinely computed hash
- [ ] `dvc repro` runs the `extract_corpus` stage and produces
      `dvc.lock`
- [ ] Confirm the MD5 of the `dvc repro`-produced JSON matches
      `ac90c9c1001e961d7aa451ca6984c1c8` (true reproducibility, not just
      the raw CLI command's reproducibility, which IS already verified)
- [ ] `dvc status` reports clean
- [ ] Decide on and configure a DVC remote (S3, GDrive, or similar) so
      a fresh clone can actually `dvc pull` the PDF — currently there is
      none, which is also what blocks the CI `dvc-pipeline` job

## RAGAS execution with a real generation backend
- [ ] **Design and build a generation backend first** — this project is
      retrieval-only; there is nothing that generates an answer yet.
      This is new development, not a missing-package problem.
- [ ] Wire the new backend into `src/experiments/ragas_eval.py`'s
      `GenerateFn` interface (already defined, never implemented)
- [ ] `pip install ragas datasets` succeeds
- [ ] `scripts/ragas_gate.py` produces a **real** faithfulness score
      (not the current permanent "UNAVAILABLE, failing closed" state)
- [ ] Decide on and justify `RAGAS_FAITHFULNESS_THRESHOLD`

## GitHub remote and CI verification
- [ ] Create a GitHub repository and add it as `origin`
- [ ] Push this project's history (note: this is a recovery-era history
      starting from a reconstruction commit — see `PROGRESS.md`'s
      recovery note — not the project's original, now-lost commits)
- [ ] Confirm the `.github/workflows/ci.yml` `test` job passes for real
      on GitHub's runners
- [ ] Decide what to do about the `ragas-quality-gate` and
      `dvc-pipeline` jobs, which will fail until the items above are
      done — options: leave them failing (honest, visible), mark them
      non-blocking/`continue-on-error` until ready, or complete the
      underlying work first

## Load testing
- [ ] `pip install locust`
- [ ] Have a running API instance (local or deployed) to target
- [ ] Run `locust -f tests/load/locustfile.py --host <url> --users 50
      --spawn-rate 10 --run-time 2m --headless --csv reports/load_test`
- [ ] Save and report the **real** output — this project currently has
      **zero** load-test numbers of any kind

## Final course rubric validation
- [ ] **Obtain the actual course handbook document** — none has ever
      been provided to this project. Everything in
      `reports/requirement_matrix.md` is built from cumulative task
      instructions, not a verified handbook, and should be re-checked
      against the real document once available.
- [ ] Re-map every requirement once the handbook is available; flag any
      place this project's assumed requirements turn out to be wrong

## Submission documentation
- [x] `README.md` — beginner-friendly overview + day-by-day log
- [x] `reports/PROJECT_HANDOVER.md` — file inventory + status matrix
- [x] `AI_CONTEXT.md` — model-agnostic briefing for a future assistant
- [x] `WINDOWS_SETUP.md` — tested/proposed command reference
- [x] `REMAINING_WORK.md` — this file
- [x] `PROJECT_STRUCTURE.txt` — plain-text directory tree
- [ ] A final submission write-up incorporating **real** results from
      the items above, once they've actually been executed — do not
      submit this project claiming Docker/MLflow/DVC/RAGAS/CI "work"
      until you've personally run each one and can show real output
