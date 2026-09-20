#!/usr/bin/env python3
"""Seed the database with backdated traffic so the dashboard has a history.

Routes real prompts through the real pipeline (in mock provider mode), then
backdates the resulting rows across the requested window so the time-series
charts have something to show.

    python scripts/seed_demo_data.py --days 21 --per-day 12
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MOCK_PROVIDERS", "true")

from src.logging.database import EscalationEvent, RequestLog  # noqa: E402
from src.service import AutopilotService  # noqa: E402

PROMPTS = [
    "What is a variable in Python?",
    "Print hello world in Go.",
    "How do I import a module?",
    "What does len() do?",
    "Convert this list to a set.",
    "Extract the name, email, and phone number from this text as JSON",
    "Classify the sentiment of this review as positive or negative",
    "Summarize this authentication module in two sentences",
    "Explain what a Python decorator is",
    "Why is my loop running one extra time?",
    "Review this function for readability.",
    "Write a Python function that merges two sorted linked lists",
    "Compare REST and GraphQL for a small app.",
    "Design a distributed rate limiter that scales across regions and "
    "handles clock skew, with trade-offs between accuracy and latency",
    "Implement Raft consensus.",
    "Design a CRDT for collaborative text editing.",
    "Architect a multi-tenant SaaS database schema.",
    "Build a thread-safe LRU cache.",
]


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--per-day", type=int, default=12)
    parser.add_argument(
        "--reset", action="store_true", help="Delete existing rows first"
    )
    args = parser.parse_args()

    service = AutopilotService(sample_rate=1.0)
    await service.start()

    db = service.database
    if args.reset:
        with db.get_session() as session:
            session.query(EscalationEvent).delete()
            session.query(RequestLog).delete()
            session.commit()
        print("Cleared existing rows")

    rng = random.Random(7)
    total = args.days * args.per_day
    print(f"Generating {total} requests across {args.days} days…")

    for day in range(args.days):
        for _ in range(args.per_day):
            await service.ask(rng.choice(PROMPTS))
        print(f"  day {day + 1}/{args.days}", end="\r", flush=True)

    print("\nWaiting for verification to finish…")
    await service.drain(timeout=180)
    await service.stop()

    # Backdate the rows so the charts span the window rather than one instant.
    now = datetime.now(timezone.utc)
    with db.get_session() as session:
        rows = session.query(RequestLog).order_by(RequestLog.id).all()

        # Spread whatever is in the table evenly across the window. Deriving
        # the offset from --per-day instead would push rows into the FUTURE
        # when the table already holds more rows than this run created.
        per_day = max(1, -(-len(rows) // args.days))  # ceiling division
        for index, row in enumerate(rows):
            day_offset = max(0, args.days - 1 - (index // per_day))
            row.timestamp = now - timedelta(
                days=day_offset,
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )

        # Keep each escalation aligned with the request that triggered it.
        by_id = {row.id: row.timestamp for row in rows}
        for event in session.query(EscalationEvent).all():
            if event.request_log_id in by_id:
                event.timestamp = by_id[event.request_log_id]
        session.commit()

    stats = db.get_stats()
    verification = db.get_verification_stats()
    print(
        f"\nSeeded {stats['total_requests']} requests\n"
        f"  spend      {stats['total_cost']:.6f} vs baseline "
        f"{stats['total_hypothetical_cost']:.6f}\n"
        f"  saved      {stats['savings_percentage']:.1f}%\n"
        f"  verified   {verification['verified_count']} "
        f"(pass rate {verification['pass_rate']:.0f}%)\n"
        f"  escalated  {verification['escalation_count']}\n"
        f"\nNow run:  streamlit run dashboard/app.py"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
