"""Locust load test for POST /ask.

NOT EXECUTED in this sandbox: `locust` is not installed here (no network
access to install it), and there is no running instance of the API to
point it at anyway (fastapi/uvicorn also unavailable here -- see
PROGRESS.md). This file's syntax was verified with
`python3 -m py_compile tests/load/locustfile.py`, which does not require
`locust` itself. No load-test numbers are reported from this sandbox;
none were fabricated.

Run for real once the API is up somewhere reachable:
    locust -f tests/load/locustfile.py --host http://localhost:8000
Then open http://localhost:8089 and set users=50 (the handbook's target
concurrency), spawn rate as desired, and start the run. Or headless:
    locust -f tests/load/locustfile.py --host http://localhost:8000 \
        --users 50 --spawn-rate 10 --run-time 2m --headless \
        --csv reports/load_test
"""
from __future__ import annotations

import random

from locust import HttpUser, between, task

# A handful of real Arabic legal questions (same flavor as the Day 2
# retrieval smoke test) so the load test exercises realistic request
# bodies, not a single cached/degenerate query.
SAMPLE_QUESTIONS = [
    "ما هو عقد البيع؟",
    "متى ينتهي عقد الايجار؟",
    "ما هي احكام حق الانتفاع؟",
    "كيف يثبت الرهن الحيازي؟",
    "ما هي مسؤولية الوكيل عن اعمال الوكالة؟",
]


class AskUser(HttpUser):
    # Reasonable think-time between requests for a Q&A-style client,
    # not a tight hammering loop.
    wait_time = between(1, 3)

    @task(3)
    def ask_question(self) -> None:
        question = random.choice(SAMPLE_QUESTIONS)
        self.client.post("/ask", json={"question": question}, name="/ask")

    @task(1)
    def health_check(self) -> None:
        self.client.get("/health", name="/health")
