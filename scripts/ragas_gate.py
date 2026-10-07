"""RAGAS faithfulness quality gate for CI.

Fails closed (nonzero exit) whenever a REAL faithfulness score cannot be
computed here -- this script NEVER silently passes and NEVER substitutes
a different metric (the Day 4 lexical hit-rate/MRR numbers are NOT
faithfulness and are never reported as such by this script).

TWO independent requirements must BOTH be met before a real score can
even be attempted:

1. The `ragas` package (plus its own dependencies -- `datasets`, and an
   LLM judge, typically configured via an API key) must be
   importable/usable.
2. This RAG pipeline needs a REAL generation backend. RAGAS faithfulness
   measures whether a *generated answer* is supported by its retrieved
   context. Through Day 5, this pipeline is retrieval-only by design
   (see src/api/main.py's literal `"[retrieval-only, no generation model
   configured]"` answer string) -- there is no generated answer for
   RAGAS to score yet. Running faithfulness against that placeholder
   string would not measure anything real, so this script does not
   attempt it.

As of Day 6, requirement (2) is never true, so this gate ALWAYS fails
closed in this project regardless of whether `ragas` itself happens to
be installed -- this is an honest, documented reflection of the current
architecture, not a bug to work around.
"""
from __future__ import annotations

import os
import sys

DEFAULT_THRESHOLD = float(os.environ.get("RAGAS_FAITHFULNESS_THRESHOLD", "0.7"))


def ragas_importable() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def generation_backend_configured() -> bool:
    # No real generation backend exists anywhere in this pipeline as of
    # Day 6 (Day 2-5: retrieval-only, by explicit design). A future day
    # that adds one should set GENERATION_MODEL; until then this is
    # always False, and the gate always fails closed as a direct, correct
    # consequence -- not a bug.
    return bool(os.environ.get("GENERATION_MODEL"))


def run_gate(threshold: float = DEFAULT_THRESHOLD) -> int:
    ragas_ok = ragas_importable()
    generation_ok = generation_backend_configured()

    if not (ragas_ok and generation_ok):
        print("RAGAS faithfulness gate: UNAVAILABLE -- failing closed.")
        if not ragas_ok:
            print("  - `ragas` package is not installed/importable.")
        if not generation_ok:
            print(
                "  - no generation backend is configured (GENERATION_MODEL "
                "env var unset). This pipeline is retrieval-only through "
                "Day 5 -- there is no generated answer for RAGAS to score."
            )
        print(
            "Activation requirements to make this gate real: "
            '(1) `pip install ragas datasets` and an LLM judge configured, '
            "AND (2) a real generation backend wired up and exported as "
            "GENERATION_MODEL. Both are currently absent in this project."
        )
        print(f"Configured threshold (unused until then): {threshold}")
        return 1

    # Real path -- only reachable once BOTH requirements above are
    # genuinely met. Not implemented/exercised in this sandbox because it
    # is never reachable here. The orchestration this would call already
    # exists and is unit-tested (src/experiments/ragas_eval.py,
    # build_ragas_records + evaluate_faithfulness) -- it reuses the
    # existing Retriever and Day 4 eval set, takes a real generate_fn,
    # and would compare the measured faithfulness score against
    # `threshold` here.
    raise NotImplementedError(
        "Reachable only once ragas AND a real generation backend are both "
        "available -- not exercised in this project as of Day 7. Once a "
        "real generate_fn exists: eval_set = load_or_build_eval_set(...); "
        "records = build_ragas_records(eval_set, Retriever(), generate_fn); "
        "scores = evaluate_faithfulness(records); compare against threshold."
    )


if __name__ == "__main__":
    sys.exit(run_gate())
