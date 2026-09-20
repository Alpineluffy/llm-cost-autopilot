"""Phase 4 — the savings headline must not overstate itself.

The money-shot metric is the number this project is judged on, so escalation
spend has to be counted against it. Re-running a failed request on GPT-4o is a
real cost of operating this router; excluding it would inflate the headline.
"""

from __future__ import annotations

import pytest

from src.classifier.tiers import ComplexityTier
from src.models.response import LLMResponse
from src.verifier.use_cases import PolicyBook
from src.verifier.worker import VerificationJob, VerificationWorker


def _response(text: str, model_id: str = "gpt-4o-mini", cost: float = 0.001):
    return LLMResponse(
        text=text, model_id=model_id, provider="openai",
        input_tokens=100, output_tokens=50, cost=cost, latency_ms=400.0,
    )


def _log(database, cost: float = 0.001, hypothetical: float = 0.01) -> int:
    return database.log_request(
        {
            "prompt_hash": "h", "prompt_preview": "Classify the sentiment",
            "complexity_tier": 2, "routed_model": "gpt-4o-mini",
            "provider": "openai", "input_tokens": 100, "output_tokens": 50,
            "cost": cost, "latency_ms": 400.0,
            "hypothetical_cost": hypothetical,
            "was_escalated": False, "was_verified": False,
            "use_case": "classification",
        }
    )


def test_savings_exclude_nothing_when_there_are_no_escalations(database):
    _log(database, cost=0.001, hypothetical=0.01)
    stats = database.get_stats()

    assert stats["escalation_cost"] == 0
    assert stats["total_cost"] == pytest.approx(0.001)
    assert stats["savings_percentage"] == pytest.approx(90.0, abs=0.1)


@pytest.mark.asyncio
async def test_escalation_spend_is_counted_against_the_headline(
    registry, providers, database, request_logger, routing_config
):
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "Label: negative"

    worker = VerificationWorker(
        registry=registry, providers=providers, database=database,
        request_logger=request_logger, routing_config=routing_config,
        policy_book=PolicyBook(routing_config), sample_rate=1.0,
    )

    # Realistic: the cheap model cost a fraction of the top-tier baseline.
    log_id = _log(database, cost=0.0001, hypothetical=0.01)

    await worker.start()
    await worker.enqueue_now(
        VerificationJob(
            log_id=log_id,
            prompt="Classify the sentiment of this review",
            response=_response("Label: positive", cost=0.0001),  # diverges → escalates
            tier=ComplexityTier.MODERATE,
            model_name="gpt-4o-mini",
        )
    )
    await worker.drain()
    await worker.stop()

    stats = database.get_stats()

    assert stats["escalation_cost"] > 0, "escalation produced no recorded cost"
    assert stats["routed_cost"] == pytest.approx(0.0001)
    assert stats["total_cost"] == pytest.approx(
        stats["routed_cost"] + stats["escalation_cost"]
    )

    # The headline must reflect the true cost, not just the routed cost.
    naive = (0.01 - stats["routed_cost"]) / 0.01 * 100
    honest = (0.01 - stats["total_cost"]) / 0.01 * 100
    assert stats["savings_percentage"] == pytest.approx(honest, abs=0.1)
    assert stats["savings_percentage"] < naive, (
        "savings headline ignores escalation spend and overstates itself"
    )


def test_savings_are_zero_when_nothing_was_routed(database):
    stats = database.get_stats()
    assert stats["total_requests"] == 0
    assert stats["savings_percentage"] == 0


@pytest.mark.asyncio
async def test_time_series_agrees_with_the_headline(
    registry, providers, database, request_logger, routing_config
):
    """The chart must not tell a different story than the headline metric."""
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "Label: negative"

    worker = VerificationWorker(
        registry=registry, providers=providers, database=database,
        request_logger=request_logger, routing_config=routing_config,
        policy_book=PolicyBook(routing_config), sample_rate=1.0,
    )
    await worker.start()
    for _ in range(3):
        log_id = _log(database, cost=0.0001, hypothetical=0.01)
        await worker.enqueue_now(
            VerificationJob(
                log_id=log_id,
                prompt="Classify the sentiment of this review",
                response=_response("Label: positive", cost=0.0001),
                tier=ComplexityTier.MODERATE,
                model_name="gpt-4o-mini",
            )
        )
    await worker.drain()
    await worker.stop()

    headline = database.get_stats()["total_savings"]
    daily = sum(
        row["hypothetical_cost"] - row["actual_cost"]
        for row in database.get_daily_costs(days=90)
    )
    weekly = sum(row["savings"] for row in database.get_weekly_costs(weeks=52))

    # Tolerance covers per-period rounding at 6 decimal places.
    assert daily == pytest.approx(headline, abs=1e-5)
    assert weekly == pytest.approx(headline, abs=1e-5)
