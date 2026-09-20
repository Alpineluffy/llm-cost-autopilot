"""Integration — the Phase 3 loop end to end.

These are the tests that would have caught the original defect: every
component existed, but `enqueue` had no callers, so nothing ever ran.
"""

from __future__ import annotations

import pytest

from src.classifier.model import ComplexityClassifier
from src.classifier.tiers import ComplexityTier
from src.models.response import LLMResponse
from src.router.engine import RoutingEngine
from src.verifier.feedback import FeedbackLoop
from src.verifier.use_cases import PolicyBook
from src.verifier.worker import VerificationJob, VerificationWorker


def _response(text: str, model_id: str = "gpt-4o-mini", cost: float = 0.0001):
    return LLMResponse(
        text=text,
        model_id=model_id,
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        cost=cost,
        latency_ms=400.0,
    )


@pytest.fixture
def feedback(tmp_path) -> FeedbackLoop:
    return FeedbackLoop(
        classifier=ComplexityClassifier(),
        feedback_path=str(tmp_path / "feedback.json"),
        state_path=str(tmp_path / "state.json"),
        model_path=str(tmp_path / "model.pkl"),
        min_examples_to_retrain=2,
    )


@pytest.fixture
def worker(registry, providers, database, request_logger, routing_config, feedback):
    return VerificationWorker(
        registry=registry,
        providers=providers,
        database=database,
        request_logger=request_logger,
        routing_config=routing_config,
        policy_book=PolicyBook(routing_config),
        feedback_loop=feedback,
        sample_rate=1.0,
    )


def _log_a_request(database, tier: int = 2, prompt: str = "test") -> int:
    return database.log_request(
        {
            "prompt_hash": "abc123",
            "prompt_preview": prompt[:200],
            "complexity_tier": tier,
            "routed_model": "gpt-4o-mini",
            "provider": "openai",
            "input_tokens": 100,
            "output_tokens": 50,
            "cost": 0.0001,
            "latency_ms": 400.0,
            "hypothetical_cost": 0.001,
            "was_escalated": False,
            "was_verified": False,
            "use_case": "classification",
        }
    )


# ── The wiring that was missing ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_routing_engine_notifies_its_completion_hook(registry):
    """The original bug: route_request never handed off for verification."""
    seen = []

    async def hook(prompt, routed):
        seen.append((prompt, routed))

    engine = RoutingEngine(registry=registry, on_complete=hook)
    await engine.route_request("What is a variable?")

    assert len(seen) == 1, "completed request never reached the verification hook"
    assert seen[0][0] == "What is a variable?"


@pytest.mark.asyncio
async def test_hook_failure_does_not_break_the_user_response(registry):
    """Verification is best-effort; it must never surface to the caller."""
    async def exploding_hook(prompt, routed):
        raise RuntimeError("verification subsystem is down")

    engine = RoutingEngine(registry=registry, on_complete=exploding_hook)
    result = await engine.route_request("What is a variable?")

    assert result.response.text  # the user still got their answer


# ── Verify → escalate → feed back ───────────────────────────────────────

@pytest.mark.asyncio
async def test_failure_escalates_and_records_feedback(
    worker, database, feedback, providers
):
    """One failing request should produce all three Phase 3 side effects."""
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "negative"
    log_id = _log_a_request(database)

    await worker.start()
    await worker.enqueue_now(
        VerificationJob(
            log_id=log_id,
            prompt="Classify the sentiment of this review",
            response=_response("positive"),  # diverges from the reference
            tier=ComplexityTier.MODERATE,
            model_name="gpt-4o-mini",
        )
    )
    await worker.drain()
    await worker.stop()

    # 3.2 — the verification outcome was persisted
    log = database.get_recent_logs(1)[0]
    assert log["was_verified"] is True
    assert log["use_case"] == "classification"
    assert log["quality_gap"] > 0

    # 3.3 — the escalation was recorded with cost delta and quality gap
    events = database.get_escalation_events()
    assert len(events) == 1
    assert events[0]["original_model"] == "gpt-4o-mini"
    assert events[0]["cost_delta"] > 0
    assert events[0]["quality_gap"] == 4.0

    # 3.4 — the failure became a training example
    assert feedback.feedback_count == 1


@pytest.mark.asyncio
async def test_passing_request_causes_no_escalation_or_feedback(
    worker, database, feedback, providers
):
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "positive"
    log_id = _log_a_request(database)

    await worker.start()
    await worker.enqueue_now(
        VerificationJob(
            log_id=log_id,
            prompt="Classify the sentiment of this review",
            response=_response("positive"),  # matches the reference
            tier=ComplexityTier.MODERATE,
            model_name="gpt-4o-mini",
        )
    )
    await worker.drain()
    await worker.stop()

    assert database.get_recent_logs(1)[0]["quality_gap"] == 0
    assert database.get_escalation_events() == []
    assert feedback.feedback_count == 0
    assert worker.stats["passed"] == 1


@pytest.mark.asyncio
async def test_top_tier_failure_records_feedback_without_escalating(
    worker, database, feedback, providers
):
    """Nothing to escalate to, but the classifier should still not learn a
    tier correction it cannot act on."""
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "negative"
    log_id = _log_a_request(database, tier=3)

    await worker.start()
    await worker.enqueue_now(
        VerificationJob(
            log_id=log_id,
            prompt="Classify the sentiment of this review",
            response=_response("positive", model_id="gpt-4o"),
            tier=ComplexityTier.COMPLEX,
            model_name="gpt-4o",
        )
    )
    await worker.drain()
    await worker.stop()

    assert database.get_escalation_events() == []
    assert feedback.feedback_count == 0  # already at the top tier


@pytest.mark.asyncio
async def test_worker_survives_a_failing_job(worker, database, providers):
    """A bad job must not kill the loop for every job behind it."""
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "negative"

    await worker.start()
    await worker.enqueue_now(
        VerificationJob(
            log_id=999_999,  # no such row
            prompt="Classify this",
            response=_response("positive"),
            tier=ComplexityTier.MODERATE,
            model_name="gpt-4o-mini",
        )
    )
    good_id = _log_a_request(database)
    await worker.enqueue_now(
        VerificationJob(
            log_id=good_id,
            prompt="Classify the sentiment",
            response=_response("negative"),
            tier=ComplexityTier.MODERATE,
            model_name="gpt-4o-mini",
        )
    )
    await worker.drain()
    await worker.stop()

    assert worker.stats["verified"] == 2


# ── Sampling ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_zero_sample_rate_queues_nothing(
    registry, providers, database, request_logger, routing_config
):
    worker = VerificationWorker(
        registry=registry,
        providers=providers,
        database=database,
        request_logger=request_logger,
        routing_config=routing_config,
        sample_rate=0.0,
    )
    job = VerificationJob(
        log_id=1, prompt="p", response=_response("x"),
        tier=ComplexityTier.MODERATE, model_name="gpt-4o-mini",
    )
    assert await worker.enqueue(job) is False


@pytest.mark.asyncio
async def test_full_sample_rate_queues_everything(worker):
    job = VerificationJob(
        log_id=1, prompt="p", response=_response("x"),
        tier=ComplexityTier.MODERATE, model_name="gpt-4o-mini",
    )
    assert await worker.enqueue(job) is True


# ── Aggregates ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verification_stats_aggregate_correctly(
    worker, database, providers
):
    providers["anthropic"]._responses["claude-sonnet-4-6"] = "negative"
    await worker.start()
    for i in range(3):
        log_id = _log_a_request(database, prompt=f"prompt {i}")
        await worker.enqueue_now(
            VerificationJob(
                log_id=log_id,
                prompt=f"Classify the sentiment of review {i}",
                response=_response("positive"),
                tier=ComplexityTier.MODERATE,
                model_name="gpt-4o-mini",
            )
        )
    await worker.drain()
    await worker.stop()

    stats = database.get_verification_stats()
    assert stats["verified_count"] == 3
    assert stats["failed_count"] == 3
    assert stats["escalation_count"] == 3
    assert stats["escalation_cost_delta"] > 0
    assert stats["by_use_case"]["classification"]["count"] == 3
