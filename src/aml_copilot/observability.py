import sqlite3
import time
from contextlib import closing
from contextvars import ContextVar

import numpy as np

from . import config

# set once per request so every llm call can be tied back to its run
run_id = ContextVar("run_id", default="-")

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS llm_calls (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, run_id TEXT, step TEXT,
        model TEXT, latency_ms REAL, prompt_tokens INTEGER,
        completion_tokens INTEGER, cost_usd REAL, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS runs (
        run_id TEXT PRIMARY KEY, ts REAL, account_id TEXT, latency_ms REAL,
        n_claims INTEGER, n_unsupported INTEGER, attempts INTEGER, status TEXT)""",
]


def _connect():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.METRICS_DB)
    for stmt in SCHEMA:
        conn.execute(stmt)
    return conn


def estimate_cost(model, prompt_tokens, completion_tokens):
    price_in, price_out = config.PRICES.get(model, (0.0, 0.0))
    return (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000


def log_call(step, model, latency_ms, prompt_tokens, completion_tokens, error=None):
    cost = estimate_cost(model, prompt_tokens, completion_tokens)
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT INTO llm_calls (ts, run_id, step, model, latency_ms, prompt_tokens,"
            " completion_tokens, cost_usd, error) VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), run_id.get(), step, model, latency_ms, prompt_tokens,
             completion_tokens, cost, error),
        )
        conn.commit()


def log_run(rid, account_id, latency_ms, verification, attempts, status):
    verification = verification or {}
    with closing(_connect()) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?)",
            (rid, time.time(), account_id, latency_ms,
             verification.get("n_claims", 0), verification.get("n_unsupported", 0),
             attempts, status),
        )
        conn.commit()


def summary():
    with closing(_connect()) as conn:
        calls = conn.execute(
            "SELECT step, latency_ms, prompt_tokens, completion_tokens, cost_usd, error FROM llm_calls"
        ).fetchall()
        runs = conn.execute(
            "SELECT latency_ms, n_claims, n_unsupported, status FROM runs"
        ).fetchall()

    by_step = {}
    for step, lat, pt, ct, cost, err in calls:
        s = by_step.setdefault(step, {"calls": 0, "latency": [], "tokens": 0, "cost": 0.0, "errors": 0})
        s["calls"] += 1
        s["latency"].append(lat)
        s["tokens"] += (pt or 0) + (ct or 0)
        s["cost"] += cost or 0
        s["errors"] += 1 if err else 0

    steps = [
        {"step": name, "calls": s["calls"], "avg_latency_ms": round(float(np.mean(s["latency"])), 1),
         "tokens": s["tokens"], "cost_usd": round(s["cost"], 5), "errors": s["errors"]}
        for name, s in by_step.items()
    ]

    run_lat = [r[0] for r in runs]
    ok_runs = [r for r in runs if r[3] == "ok"]
    claims = sum(r[1] for r in ok_runs)
    unsupported = sum(r[2] for r in ok_runs)
    return {
        "runs": len(runs),
        "failed_runs": len(runs) - len(ok_runs),
        "p50_run_latency_ms": round(float(np.percentile(run_lat, 50)), 1) if run_lat else None,
        "p95_run_latency_ms": round(float(np.percentile(run_lat, 95)), 1) if run_lat else None,
        "total_cost_usd": round(sum(s["cost_usd"] for s in steps), 5),
        "avg_cost_per_run_usd": round(sum(s["cost_usd"] for s in steps) / len(runs), 5) if runs else None,
        "unsupported_claim_rate": round(unsupported / claims, 4) if claims else None,
        "by_step": steps,
    }
