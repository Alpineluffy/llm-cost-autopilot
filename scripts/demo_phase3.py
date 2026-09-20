#!/usr/bin/env python3
"""End-to-end demonstration of the Phase 3 verification loop.

Routes a batch of prompts, verifies every one against the highest-tier model,
escalates the failures, and shows the feedback the classifier accumulated.

    python scripts/demo_phase3.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("MOCK_PROVIDERS", "true")

from src.service import AutopilotService  # noqa: E402

DEMO_PROMPTS = [
    # extraction → field coverage, bar 4.5
    "Extract the name, email, and phone number from this contact block as JSON",
    # classification → label match, bar 5.0
    "Classify the sentiment of this review as positive, negative, or neutral",
    # summarization → rubric, bar 4.0
    "Summarize this authentication module in two sentences",
    # code generation → rubric, bar 3.8
    "Write a Python function that merges two sorted linked lists",
    # explanation → rubric, bar 3.5
    "Explain what a Python decorator is",
    # complex → already top tier, nothing to escalate to
    "Design a distributed rate limiter that scales across regions and "
    "handles clock skew, with trade-offs between accuracy and latency",
]

RULE = "─" * 78


def header(text: str) -> None:
    print(f"\n{RULE}\n{text}\n{RULE}")


async def main() -> int:
    # Isolated scratch state so the demo never pollutes real data.
    tmp = tempfile.mkdtemp(prefix="autopilot-demo-")
    service = AutopilotService(
        db_url=f"sqlite:///{tmp}/demo.db",
        sample_rate=1.0,  # verify everything, so the demo is deterministic
        feedback_path=f"{tmp}/feedback_data.json",
        state_path=f"{tmp}/feedback_state.json",
        model_output_path=f"{tmp}/classifier_model.pkl",
    )
    await service.start()

    header("1. ROUTING — each prompt goes to the cheapest capable model")
    log_ids = []
    for prompt in DEMO_PROMPTS:
        result = await service.ask(prompt)
        d = result.decision
        saved = result.hypothetical_cost - result.response.cost
        print(
            f"\n  {prompt[:64]}{'…' if len(prompt) > 64 else ''}\n"
            f"    tier={d.tier.label:<9} model={d.selected_model:<14} "
            f"cost=${result.response.cost:.6f}  saved=${saved:.6f}"
        )

    header("2. VERIFICATION — comparing each against the reference model")
    await service.drain()

    for row in reversed(service.database.get_recent_logs(limit=len(DEMO_PROMPTS))):
        verdict = "PASS" if (row["quality_gap"] or 0) <= 0 else "FAIL"
        log_ids.append(row["id"])
        print(
            f"\n  [{verdict}] {row['prompt_preview'][:58]}\n"
            f"    use_case={row['use_case']:<15} strategy={row['verification_strategy']}\n"
            f"    score={row['quality_score']:.2f}/5  bar={row['quality_threshold']}  "
            f"gap={row['quality_gap']:.2f}  divergence={row['divergence']:.0%}"
        )

    header("3. ESCALATION — failures re-run one tier up")
    events = service.database.get_escalation_events()
    if not events:
        print("\n  No escalations triggered.")
    for e in events:
        served = "served to user" if e["returned_to_user"] else "logged only (over budget)"
        print(
            f"\n  {e['original_model']} → {e['escalated_model']}  [{e['use_case']}]\n"
            f"    cost delta  ${e['cost_delta']:+.6f}  "
            f"(${e['original_cost']:.6f} → ${e['escalated_cost']:.6f})\n"
            f"    quality gap {e['quality_gap']:.2f} below the bar\n"
            f"    latency     {e['escalation_latency_ms']:.0f}ms — {served}\n"
            f"    trigger     {(e['trigger_reason'] or '')[:64]}"
        )

    header("4. FEEDBACK — every failure becomes a training example")
    summary = service.feedback.summary()
    print(
        f"\n  Accumulated examples : {summary['feedback_examples']}\n"
        f"  By use case          : {summary['by_use_case'] or '—'}\n"
        f"  Retrain due          : {summary['retrain_due']}\n"
        f"  Reason               : {summary['retrain_reason']}"
    )

    print("\n  Forcing a retrain to show the flywheel closing…")
    metrics = service.force_retrain()
    print(
        f"    retrained on {metrics['train_size'] + metrics['test_size']} examples "
        f"({metrics['feedback_examples_used']} from failures) "
        f"→ accuracy {metrics['accuracy']:.1%}"
    )

    header("SUMMARY")
    stats = service.stats()
    r, v, w = stats["requests"], stats["verification"], stats["worker"]
    print(
        f"\n  Requests routed     : {r['total_requests']}\n"
        f"  Spend vs all-GPT-4o : ${r['total_cost']:.6f} vs "
        f"${r['total_hypothetical_cost']:.6f}  "
        f"({r['savings_percentage']:.1f}% saved)\n"
        f"  Verified            : {w['verified']}  "
        f"(pass {w['passed']}, fail {w['failed']})\n"
        f"  Escalated           : {w['escalated']}  "
        f"(served {w['escalations_served']})\n"
        f"  Escalation spend    : ${v['escalation_cost_delta']:+.6f}\n"
        f"  Feedback recorded   : {w['feedback_recorded']}"
    )

    await service.stop()
    print(f"\n  Demo state written to {tmp}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
