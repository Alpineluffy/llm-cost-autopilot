#!/usr/bin/env python3
"""Realistic load test — drives N prompts through the full pipeline.

The prompt pool here is deliberately HELD OUT from data/training_prompts.json.
Replaying training prompts would flatter the classifier and make the routing
distribution meaningless.

Because savings depend far more on traffic mix than on sampling error (see
scripts/significance_analysis.py), the report states the mix it assumed and
recomputes savings across several alternative mixes.

    python scripts/load_test.py --requests 600
    python scripts/load_test.py --requests 600 --mix heavy   # complex-skewed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MOCK_PROVIDERS", "true")

from src.logging.database import EscalationEvent, RequestLog  # noqa: E402
from src.service import AutopilotService  # noqa: E402

REPORT_PATH = PROJECT_ROOT / "reports" / "load_test_report.md"
RESULTS_PATH = PROJECT_ROOT / "reports" / "load_test_results.json"


# ── Held-out prompt pool ────────────────────────────────────────────────
# None of these appear in the training corpus.

SIMPLE_PROMPTS = [
    "What does the enumerate function do?",
    "How do I check if a file exists?",
    "What is a lambda in Python?",
    "Convert this timestamp to a readable date.",
    "How do I install a package with pip?",
    "What does the pass keyword do?",
    "Write a one-line list comprehension that squares numbers.",
    "How do I get the current working directory?",
    "What is the difference between a list and an array?",
    "Rename this function to calculate_total.",
    "How do I exit a while loop?",
    "What does the in operator check?",
    "Format this number to two decimal places.",
    "How do I create a virtual environment?",
    "What is a set used for?",
    "Write the import statement for pathlib.",
    "How do I count items in a list?",
    "What does the not operator do?",
    "Turn this string into a list of characters.",
    "How do I check the type of a variable?",
    "What is an f-string?",
    "Remove whitespace from both ends of this string.",
    "How do I define a default argument?",
    "What does the range function return?",
    "Sort this dictionary by its keys.",
    "How do I concatenate two lists?",
    "What is the modulo operator used for?",
    "Write an empty function that takes no arguments.",
    "How do I read an environment variable?",
    "What does the len function work on?",
    "Extract the file extension from this path.",
    "How do I make a string lowercase?",
    "What is a docstring?",
    "Write a print statement that shows a variable's value.",
    "How do I check if a list is empty?",
    "What does the append method return?",
    "Convert this integer to a string.",
    "How do I get the last element of a list?",
    "What is a global variable?",
    "Write a shebang line for a Python script.",
]

MODERATE_PROMPTS = [
    "Why does my function return None unexpectedly?",
    "Review this error-handling block for gaps.",
    "Explain the difference between == and is in Python.",
    "Summarize what this configuration file controls.",
    "Why is my unit test passing locally but failing in CI?",
    "Compare using a generator versus building a list.",
    "Refactor this function to reduce its nesting.",
    "Walk me through what this list slice returns.",
    "Explain how Python resolves method inheritance.",
    "Why am I getting an IndexError on this loop?",
    "Review this function signature for clarity.",
    "Explain what mutable default arguments do wrong.",
    "Compare threading and multiprocessing for I/O work.",
    "Why does my dictionary comprehension skip entries?",
    "Summarize the responsibilities of this module.",
    "Explain what a closure captures.",
    "Review this regex for readability.",
    "Why is this sort producing an unexpected order?",
    "Add error handling to this file read.",
    "Explain the difference between a shallow and deep copy.",
    "Why does this async function never complete?",
    "Compare storing config in JSON versus YAML.",
    "Review this class for unnecessary state.",
    "Explain what happens during a Python import.",
    "Why is my float comparison failing?",
    "Summarize the tradeoffs in this caching approach.",
    "Explain when to use a namedtuple.",
    "Review this loop for an off-by-one mistake.",
    "Why does my subprocess call hang?",
    "Compare pytest fixtures and setup methods.",
    "Explain what a context manager guarantees on exit.",
    "Why is this JSON parse failing on valid-looking input?",
    "Review these variable names for clarity.",
    "Explain how a decorator with arguments works.",
    "Why does my logging output appear twice?",
    "Compare a dataclass and a plain dictionary here.",
    "Explain what type hints do at runtime.",
    "Why is my database connection timing out?",
    "Review this API error response format.",
    "Explain the difference between a module and a package.",
]

COMPLEX_PROMPTS = [
    "Design a write-ahead log with crash recovery.",
    "Implement a wait-free single-producer queue.",
    "Architect a multi-region read replica strategy.",
    "Design a schema migration that avoids table locks.",
    "Implement a rate limiter using sliding window counters.",
    "Design an idempotency layer for a payments API.",
    "Build a dependency injection container.",
    "Design a change-data-capture pipeline.",
    "Implement a garbage-collected reference counter.",
    "Architect an event-driven order fulfillment system.",
    "Design a consistent snapshot protocol across shards.",
    "Implement a lock-free stack with hazard pointers.",
    "Design a retry budget that prevents cascading failure.",
    "Build a streaming join across two unbounded sources.",
    "Design an access control model with role inheritance.",
    "Implement a compaction strategy for an append-only store.",
    "Architect a multi-tenant queue with fairness guarantees.",
    "Design a deterministic replay system for debugging.",
    "Implement a distributed barrier synchronization.",
    "Design a cache coherence protocol across nodes.",
    "Build an incremental static analysis engine.",
    "Design a backpressure strategy for a fan-in service.",
    "Implement a persistent balanced tree with structural sharing.",
    "Architect a zero-trust service-to-service auth scheme.",
    "Design a sharded rate limiter tolerant of clock drift.",
    "Implement a query optimizer with cost-based planning.",
    "Design a blue-green migration for a stateful service.",
    "Build a conflict resolution strategy for offline edits.",
    "Design an audit pipeline with tamper-evident ordering.",
    "Implement a scheduler that avoids convoy effects.",
]

#: Traffic mixes to model. The default reflects a coding-assistant workload:
#: mostly lookups and reviews, comparatively few architecture questions.
MIXES = {
    "typical": (0.50, 0.35, 0.15),
    "light": (0.70, 0.25, 0.05),
    "heavy": (0.25, 0.35, 0.40),
    "uniform": (1 / 3, 1 / 3, 1 / 3),
}


def build_workload(n: int, mix: str, seed: int = 11) -> list[tuple[str, int]]:
    """Sample n (prompt, intended_tier) pairs according to the named tier mix.

    Carrying the intended tier lets the run measure routing accuracy directly:
    these prompts are held out, and their correct tier is known by construction.
    """
    rng = random.Random(seed)
    weights = MIXES[mix]
    pools = {1: SIMPLE_PROMPTS, 2: MODERATE_PROMPTS, 3: COMPLEX_PROMPTS}

    workload = []
    for _ in range(n):
        tier = rng.choices([1, 2, 3], weights=weights, k=1)[0]
        workload.append((rng.choice(pools[tier]), tier))
    rng.shuffle(workload)
    return workload


async def run_load_test(
    service: AutopilotService, workload: list[tuple[str, int]], concurrency: int
) -> list[dict]:
    """Drive the prompts through the router with bounded concurrency."""
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    completed = 0
    total = len(workload)

    async def one(prompt: str, intended_tier: int) -> None:
        nonlocal completed
        async with semaphore:
            started = time.perf_counter()
            routed = await service.ask(prompt)
            elapsed = (time.perf_counter() - started) * 1000

        results.append(
            {
                "intended_tier": intended_tier,
                "tier": routed.decision.tier.value,
                "model": routed.decision.selected_model,
                "use_case": routed.use_case,
                "cost": routed.response.cost,
                "hypothetical_cost": routed.hypothetical_cost,
                "latency_ms": elapsed,
                "provider_latency_ms": routed.response.latency_ms,
                "input_tokens": routed.response.input_tokens,
                "output_tokens": routed.response.output_tokens,
            }
        )
        completed += 1
        if completed % 50 == 0 or completed == total:
            print(f"  {completed}/{total} requests", end="\r", flush=True)

    await asyncio.gather(*(one(prompt, tier) for prompt, tier in workload))
    return results


def routing_accuracy(results: list[dict]) -> dict:
    """How often the router picked the tier the prompt was written for.

    Unlike the verifier's pass rate, this is NOT simulated — the prompts are
    held out from training and their correct tier is known by construction.
    """
    correct = sum(1 for r in results if r["tier"] == r["intended_tier"])
    per_tier = {}
    for tier in (1, 2, 3):
        rows = [r for r in results if r["intended_tier"] == tier]
        if rows:
            hits = sum(1 for r in rows if r["tier"] == tier)
            per_tier[tier] = {
                "n": len(rows),
                "accuracy": round(hits / len(rows) * 100, 1),
            }

    # Routing a hard prompt to a cheap model is the costly direction: it
    # inflates savings while degrading the answer.
    under = sum(1 for r in results if r["tier"] < r["intended_tier"])
    over = sum(1 for r in results if r["tier"] > r["intended_tier"])
    return {
        "overall": round(correct / len(results) * 100, 1) if results else 0.0,
        "per_tier": per_tier,
        "under_routed": under,
        "under_routed_pct": round(under / len(results) * 100, 1) if results else 0.0,
        "over_routed": over,
    }


def spread_timestamps(db, days: int, seed: int = 3) -> None:
    """Backdate logged rows across `days` so the dashboard can show a trend.

    Display only. A load test completes in seconds, which leaves the
    time-series charts with a single bar; this rewrites only the `timestamp`
    columns. No cost, routing, latency, or quality value is touched, and the
    report is generated from the unchanged figures.
    """
    from datetime import timedelta

    rng = random.Random(seed)
    now = datetime.now(timezone.utc)

    with db.get_session() as session:
        rows = session.query(RequestLog).order_by(RequestLog.id).all()
        if not rows:
            return
        per_day = max(1, -(-len(rows) // days))  # ceiling division
        for index, row in enumerate(rows):
            offset = max(0, days - 1 - (index // per_day))
            row.timestamp = now - timedelta(
                days=offset, hours=rng.randint(0, 23), minutes=rng.randint(0, 59)
            )

        moved = {row.id: row.timestamp for row in rows}
        for event in session.query(EscalationEvent).all():
            if event.request_log_id in moved:
                event.timestamp = moved[event.request_log_id]
        session.commit()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def mix_sensitivity(results: list[dict], escalation_by_tier: dict) -> dict:
    """Recompute savings under each alternative traffic mix.

    Uses the per-tier averages this run actually observed, reweighted. This is
    the number that matters most: savings track the mix, not the router.
    """
    per_tier: dict[int, dict] = {}
    for tier in (1, 2, 3):
        rows = [r for r in results if r["tier"] == tier]
        if not rows:
            continue
        n = len(rows)
        per_tier[tier] = {
            "avg_cost": sum(r["cost"] for r in rows) / n
            + escalation_by_tier.get(tier, 0.0),
            "avg_baseline": sum(r["hypothetical_cost"] for r in rows) / n,
        }

    out = {}
    for name, weights in MIXES.items():
        cost = baseline = 0.0
        for tier, weight in zip((1, 2, 3), weights):
            if tier not in per_tier:
                continue
            cost += weight * per_tier[tier]["avg_cost"]
            baseline += weight * per_tier[tier]["avg_baseline"]
        out[name] = round((baseline - cost) / baseline * 100, 1) if baseline else 0.0
    return out


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=600)
    parser.add_argument("--concurrency", type=int, default=25)
    parser.add_argument("--mix", choices=sorted(MIXES), default="typical")
    parser.add_argument("--sample-rate", type=float, default=0.3)
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument(
        "--spread-days",
        type=int,
        default=0,
        help=(
            "DISPLAY ONLY: after the run, spread the logged timestamps back "
            "over this many days so the dashboard's time-series charts have a "
            "trend to draw. The load test itself happens in one burst; this "
            "changes only when rows appear to have been written, never any "
            "cost, routing, or quality figure."
        ),
    )
    args = parser.parse_args()

    service = AutopilotService(sample_rate=args.sample_rate)
    await service.start()
    db = service.database

    if not args.keep_existing:
        with db.get_session() as session:
            session.query(EscalationEvent).delete()
            session.query(RequestLog).delete()
            session.commit()
        print("Cleared previous rows\n")

    workload = build_workload(args.requests, args.mix)
    print(
        f"Load test: {args.requests} held-out prompts, '{args.mix}' mix, "
        f"concurrency {args.concurrency}, verifying {args.sample_rate:.0%}\n"
    )

    wall_start = time.perf_counter()
    results = await run_load_test(service, workload, args.concurrency)
    routing_seconds = time.perf_counter() - wall_start

    print(f"\n\nRouting finished in {routing_seconds:.1f}s. Draining verification…")
    await service.drain(timeout=900)
    total_seconds = time.perf_counter() - wall_start
    await service.stop()

    if args.spread_days > 0:
        spread_timestamps(db, args.spread_days)
        print(
            f"Spread timestamps across {args.spread_days} days "
            f"(display only — no cost or quality figure changed)"
        )

    stats = db.get_stats()
    verification = db.get_verification_stats()

    escalation_by_tier: dict[int, float] = {}
    events = db.get_escalation_events(limit=100_000)
    tier_counts = {t: sum(1 for r in results if r["tier"] == t) for t in (1, 2, 3)}
    for event in events:
        tier = event["original_tier"]
        if tier_counts.get(tier):
            escalation_by_tier[tier] = escalation_by_tier.get(tier, 0.0) + (
                event["escalated_cost"] / tier_counts[tier]
            )

    latencies = [r["latency_ms"] for r in results]
    sensitivity = mix_sensitivity(results, escalation_by_tier)
    accuracy = routing_accuracy(results)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "requests": args.requests,
            "concurrency": args.concurrency,
            "mix": args.mix,
            "mix_weights": MIXES[args.mix],
            "verification_sample_rate": args.sample_rate,
            "mock_providers": os.getenv("MOCK_PROVIDERS", "true") == "true",
        },
        "throughput": {
            "routing_seconds": round(routing_seconds, 2),
            "total_seconds_incl_verification": round(total_seconds, 2),
            "requests_per_second": round(len(results) / routing_seconds, 2),
        },
        "latency_ms": {
            "mean": round(statistics.mean(latencies), 1),
            "p50": round(percentile(latencies, 50), 1),
            "p95": round(percentile(latencies, 95), 1),
            "p99": round(percentile(latencies, 99), 1),
            "max": round(max(latencies), 1),
        },
        "cost": {
            "routed": stats["routed_cost"],
            "escalation": stats["escalation_cost"],
            "total": stats["total_cost"],
            "baseline_all_top_tier": stats["total_hypothetical_cost"],
            "savings": stats["total_savings"],
            "savings_percentage": stats["savings_percentage"],
        },
        "routing_distribution": stats["model_distribution"],
        "tier_distribution": stats["tier_distribution"],
        "mix_sensitivity": sensitivity,
        "routing_accuracy_MEASURED": accuracy,
        "verification_SIMULATED": {
            "verified": verification["verified_count"],
            "pass_rate": verification["pass_rate"],
            "escalations": verification["escalation_count"],
            "note": (
                "Mock providers: these reflect the simulation's constants, "
                "not real model quality."
            ),
        },
    }

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2))
    write_markdown_report(report)

    print(f"\n{'─' * 68}")
    print(f"  {stats['savings_percentage']:.1f}% cost reduction  "
          f"(${stats['total_cost']:.4f} vs ${stats['total_hypothetical_cost']:.4f})")
    print(f"  {report['throughput']['requests_per_second']:.1f} req/s, "
          f"p95 latency {report['latency_ms']['p95']:.0f}ms")
    print(f"  Routing accuracy {accuracy['overall']:.1f}% on held-out prompts "
          f"({accuracy['under_routed_pct']:.1f}% under-routed)")
    print(f"  Savings across mixes: " +
          ", ".join(f"{k} {v}%" for k, v in sensitivity.items()))
    print(f"{'─' * 68}")
    print(f"\n  {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    return 0


def write_markdown_report(report: dict) -> None:
    c, t, l = report["cost"], report["throughput"], report["latency_ms"]
    config, sens = report["config"], report["mix_sensitivity"]
    weights = config["mix_weights"]

    lines = [
        "# Load Test Report",
        "",
        f"_Generated {report['generated_at']}_",
        "",
        f"## {c['savings_percentage']:.1f}% cost reduction",
        "",
        f"**${c['total']:.4f} spent** against a **${c['baseline_all_top_tier']:.4f}** "
        f"all-top-tier baseline across **{config['requests']} requests**.",
        "",
        "| | |",
        "|---|---|",
        f"| Routed-model spend | ${c['routed']:.4f} |",
        f"| Escalation spend (re-runs) | ${c['escalation']:.4f} |",
        f"| **Total** | **${c['total']:.4f}** |",
        f"| Baseline (all top-tier) | ${c['baseline_all_top_tier']:.4f} |",
        f"| **Saved** | **${c['savings']:.4f}** |",
        "",
        "## Throughput and latency",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Requests | {config['requests']} |",
        f"| Concurrency | {config['concurrency']} |",
        f"| Throughput | {t['requests_per_second']:.1f} req/s |",
        f"| Routing wall time | {t['routing_seconds']:.1f}s |",
        f"| Latency p50 | {l['p50']:.0f} ms |",
        f"| Latency p95 | {l['p95']:.0f} ms |",
        f"| Latency p99 | {l['p99']:.0f} ms |",
        "",
        "## Routing distribution",
        "",
        "| Model | Requests | Share |",
        "|---|---:|---:|",
    ]

    total_requests = sum(report["routing_distribution"].values()) or 1
    for model, count in sorted(
        report["routing_distribution"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"| `{model}` | {count} | {count / total_requests * 100:.1f}% |")

    accuracy = report.get("routing_accuracy_MEASURED")
    if accuracy:
        lines += [
            "",
            "## Routing accuracy (measured, not simulated)",
            "",
            "Every prompt in the pool was written for a known tier and is held out "
            "from the classifier's training data, so this is a real accuracy figure "
            "— unlike the verifier pass rate, which is simulated in mock mode.",
            "",
            f"**{accuracy['overall']:.1f}% of requests reached the tier they were "
            f"written for.**",
            "",
            "| Intended tier | n | Routed correctly |",
            "|---|---:|---:|",
        ]
        names = {1: "simple", 2: "moderate", 3: "complex"}
        for tier, row in accuracy["per_tier"].items():
            lines.append(
                f"| {names[int(tier)]} | {row['n']} | {row['accuracy']:.1f}% |"
            )
        lines += [
            "",
            f"**{accuracy['under_routed_pct']:.1f}% were under-routed** "
            f"({accuracy['under_routed']} requests sent to a model below the tier "
            f"they needed). This is the direction that matters: under-routing "
            f"inflates the savings figure while degrading the answer. The async "
            f"verifier exists to catch exactly these.",
        ]

    lines += [
        "",
        "## Sensitivity to traffic mix",
        "",
        f"This run assumed a **{config['mix']}** mix "
        f"({weights[0]:.0%} simple / {weights[1]:.0%} moderate / {weights[2]:.0%} complex). "
        "Savings depend far more on the mix than on sampling error, so the same "
        "system produces very different headlines on different workloads:",
        "",
        "| Traffic mix | Savings |",
        "|---|---:|",
    ]
    for name, value in sens.items():
        marker = "  ← this run" if name == config["mix"] else ""
        lines.append(f"| {name} | {value:.1f}%{marker} |")

    lines += [
        "",
        "## What these numbers are",
        "",
        f"- **Cost figures are arithmetic** on published per-token prices from "
        f"`config/models.yaml`, applied to {'simulated' if config['mock_providers'] else 'real'} "
        f"token counts.",
        "- **Savings describe the traffic mix**, not the router alone — see the table above.",
        "- **Quality/pass/escalation rates are simulated** when running with mock "
        "providers, and measure the mock rather than any real model.",
        "- Prompts are **held out** from the classifier's training corpus.",
        "",
        "Run `python scripts/significance_analysis.py` for the full breakdown of which "
        "metrics in this project are measurements and which are not.",
        "",
    ]

    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
