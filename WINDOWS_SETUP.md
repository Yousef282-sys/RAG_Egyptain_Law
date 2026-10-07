# Windows Setup Guide (PowerShell)

**Status key used throughout this document:**
- ✅ **Previously verified** — this exact command (or its Linux
  equivalent) has actually been run in this project's development
  history, with real output recorded.
- ⚠️ **Proposed but not yet executed** — correct as far as the code and
  standard tool documentation indicate, but never actually run, on
  Windows or anywhere else, in this project's history. Don't assume it
  will work without errors.

## 1. Extract the ZIP ⚠️
```powershell
Expand-Archive -Path arabic-legal-rag-COMPLETE-PORTABLE.zip -DestinationPath .\arabic-legal-rag
cd .\arabic-legal-rag
```

## 2. Create a virtual environment ⚠️
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
If PowerShell blocks the activation script, run once as Administrator:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## 3. Install dependencies ⚠️
```powershell
# minimum to run ingestion/tests offline:
pip install -e .
# everything (needed for API/Docker-adjacent/MLflow/DVC/RAGAS/etc. work):
pip install -e ".[dev,rag,api,serving,load,experiments,lint,dvc,ragas,langfuse]"
```
This combined-extras install has never been run anywhere in this
project's history — some packages (e.g. `sentence-transformers`,
`chromadb`) can take a while and may need Microsoft C++ Build Tools on
Windows for certain versions. If one extra fails, try installing the
others individually and troubleshoot the failing one on its own.

## 4. Run the test suite ✅ (verified on Linux; Windows re-verification proposed)
```powershell
pytest -q
# or, the bespoke runner used throughout this project's own development:
python run_tests.py
```
**Verified result (Linux, no optional packages installed): 165 passed,
0 failed, 18 skipped.** On Windows, with more extras installed, expect
fewer skips — the exact Windows numbers are unknown until you run it.

## 5. Prepare / validate the corpus ✅ (validation command verified; already-present corpus confirmed)
The processed corpus already exists at `data\processed\civil_code.json`
(1095 articles) and the source PDF at `data\raw\civil_code.pdf` — both
are included in the package (see `reports/PROJECT_HANDOVER.md` Section
5 for hash verification). To validate or rebuild:
```powershell
python -m src.ingestion.cli validate --json data\processed\civil_code.json
python scripts\check_corpus_quality.py
# only if you need to rebuild from the PDF (not required normally):
python -m src.ingestion.cli ingest --pdf data\raw\civil_code.pdf --out data\processed\civil_code.json
```

## 6. Start the API ⚠️
```powershell
python -c "from src.rag.vector_store import build_index_from_json; build_index_from_json('data/processed/civil_code.json', 'data/vector_store')"
uvicorn src.api.main:app --reload
```
Requires the `api` extra (step 3). Never run anywhere in this project's
history — `fastapi`/`uvicorn` were never installed in its development
sandbox.

## 7. Test endpoints ⚠️
```powershell
# in a second PowerShell window, with the server from step 6 running:
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod -Method Post -Uri http://localhost:8000/ask `
  -ContentType "application/json" `
  -Body '{"question": "ما هو عقد البيع؟"}'
```

## 8. Run Docker ⚠️
Requires Docker Desktop for Windows (WSL2 backend recommended) running.
```powershell
docker build -t arabic-legal-rag .
docker run -p 8000:8000 -v "${PWD}\data\vector_store:/app/data/vector_store" arabic-legal-rag
Invoke-RestMethod http://localhost:8000/health
```
Never built anywhere in this project's history (no `docker` binary in
its development sandbox). The `Dockerfile` has only been reviewed and
structurally tested, never actually built.

## 9. Start MLflow ⚠️
```powershell
pip install -e ".[experiments]"
python -m src.experiments.run_experiments
mlflow ui
```
Then open `http://localhost:5000` in a browser. The 5 experiment
configurations have been run and evaluated **locally** before (see
`reports/day4_experiment_results.json`), but never logged to a real
MLflow server — `mlflow` itself has never been installed anywhere in
this project's history.

## 10. Run DVC ⚠️
```powershell
pip install -e ".[dvc]"
dvc init
dvc add data\raw\civil_code.pdf
dvc repro
dvc status
```
Never run anywhere in this project's history. Note: `dvc repro` reuses
the existing ingestion CLI (`dvc.yaml`'s `extract_corpus` stage) — it
does not reimplement anything.

## 11. Run RAGAS ⚠️ — needs more than an install
```powershell
pip install -e ".[ragas]"
python scripts\ragas_gate.py
```
This will currently **always exit 1** ("UNAVAILABLE -- failing closed"),
even with `ragas` installed, because this project has **no generation
backend** — RAGAS faithfulness scores how well a *generated* answer is
supported by context, and nothing generates an answer here yet (see
`src/experiments/ragas_eval.py` for the exact interface a future
generation backend would plug into). This is not a packaging problem to
fix with `pip install` — it needs new development.

## 12. Connect GitHub ⚠️
```powershell
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin master
```
No remote has ever been configured for this repository. Before pushing:
check `git status` is clean, check `data\raw\civil_code.pdf` is NOT
staged (it's git-ignored by design), and read
`reports/PROJECT_HANDOVER.md` Section 6 for the full pre-push checklist.
After pushing, the `.github/workflows/ci.yml` workflow's `test` job
should run automatically; its `ragas-quality-gate` and `dvc-pipeline`
jobs are **expected to fail** until steps 10–11 above are genuinely
completed — this is by design, not a bug to silence.

---
**If anything in this document conflicts with what you actually
observe when running these commands, trust your own command output —
this document was accurate at the time it was written but cannot
account for Windows-specific issues that were never tested.**
