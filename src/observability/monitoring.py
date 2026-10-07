"""Retrieval monitoring: summarizes ONLY actually-recorded local signals
(request count, empty-result rate, result count, latency) from the JSON
log written by src/observability/logger.py.

SCOPE NOTE: these are operational signals (did retrieval return
anything, how fast, how many results) -- NOT answer-quality or
faithfulness metrics. Those are measured separately by
src/experiments/evaluate.py (lexical hit-rate/MRR) and, once a real
generation backend + `ragas` exist, src/experiments/ragas_eval.py. This
module never computes or reports either of those.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.observability.logger import DEFAULT_LOG_FILE


def load_events(log_file: str | Path = DEFAULT_LOG_FILE) -> List[Dict[str, Any]]:
    """Load every JSON-line event from the log file. Returns an empty
    list if the file doesn't exist yet -- never fabricates data.
    """
    path = Path(log_file)
    if not path.exists():
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize_retrieval(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute local retrieval signals from real recorded events only.
    Returns n_requests=0 and no other keys when there is no data.
    """
    ask_events = [e for e in events if e.get("route") == "/ask"]
    n = len(ask_events)
    if n == 0:
        return {"n_requests": 0}

    latencies = [e["latency_ms"] for e in ask_events if "latency_ms" in e]
    result_counts = [e["result_count"] for e in ask_events if "result_count" in e]
    empty_count = sum(1 for e in ask_events if e.get("result_count") == 0)
    error_count = sum(1 for e in ask_events if str(e.get("status", "")).startswith("error"))

    summary: Dict[str, Any] = {
        "n_requests": n,
        "error_count": error_count,
        "error_rate": error_count / n,
    }
    if result_counts:
        summary["empty_result_count"] = empty_count
        summary["empty_result_rate"] = empty_count / n
        summary["avg_result_count"] = sum(result_counts) / len(result_counts)
    if latencies:
        summary["avg_latency_ms"] = sum(latencies) / len(latencies)
        summary["min_latency_ms"] = min(latencies)
        summary["max_latency_ms"] = max(latencies)
    return summary


def print_summary(log_file: str | Path = DEFAULT_LOG_FILE) -> Dict[str, Any]:
    """CLI entry point: `python3 -m src.observability.monitoring`."""
    events = load_events(log_file)
    summary = summarize_retrieval(events)
    if summary["n_requests"] == 0:
        print(f"No /ask requests recorded yet in {log_file}.")
    else:
        print(f"Retrieval monitoring summary ({log_file}):")
        for key, value in summary.items():
            print(f"  {key}: {value}")
    return summary


if __name__ == "__main__":
    print_summary()
