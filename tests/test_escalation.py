"""Phase 3.3 — auto-escalation, latency budget, and the audit trail."""

from __future__ import annotations

import pytest

from src.classifier.tiers import ComplexityTier
from src.models.response import LLMResponse
from src.router.escalation import EscalationManager


def _response(model_id: str = "llama3.2", cost: float = 0.0) -> LLMResponse:
    return LLMResponse(
        text="cheap answer",
        model_id=model_id,
        provider="ollama",
        input_tokens=100,
        output_tokens=50,
        cost=cost,
        latency_ms=1200.0,
    )


@pytest.fixture
def manager(registry, providers, routing_config) -> EscalationManager:
    return EscalationManager(registry, providers, routing_config)


async def _escalate(manager, tier=ComplexityTier.SIMPLE, **kwargs):
    defaults = dict(
        prompt="test prompt",
        original_response=_response(),
        original_tier=tier,
        quality_score=2.0,
        quality_threshold=3.5,
    )
    defaults.update(kwargs)
    return await manager.escalate(**defaults)


# ── Tier movement ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_simple_escalates_to_moderate(manager):
    result = await _escalate(manager, ComplexityTier.SIMPLE)
    assert result is not None
    assert result.escalated_tier is ComplexityTier.MODERATE
    assert result.escalated_model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_moderate_escalates_to_complex(manager):
    result = await _escalate(manager, ComplexityTier.MODERATE)
    assert result.escalated_tier is ComplexityTier.COMPLEX
    assert result.escalated_model == "gpt-4o"


@pytest.mark.asyncio
async def test_complex_cannot_escalate(manager):
    """Nothing above the top tier — must return None, not loop or crash."""
    assert await _escalate(manager, ComplexityTier.COMPLEX) is None


@pytest.mark.asyncio
async def test_disabled_escalation_returns_none(registry, providers, routing_config):
    routing_config["escalation"]["enabled"] = False
    manager = EscalationManager(registry, providers, routing_config)
    assert await _escalate(manager) is None


# ── Cost and quality deltas ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_records_cost_delta_and_quality_gap(manager):
    result = await _escalate(
        manager, quality_score=2.0, quality_threshold=3.5,
        original_response=_response(cost=0.0),
    )
    assert result.quality_gap == 1.5
    assert result.cost_delta == pytest.approx(result.escalated_cost)
    assert result.cost_delta > 0


@pytest.mark.asyncio
async def test_audit_dict_carries_every_required_field(manager):
    """The spec requires all four on every escalation event."""
    event = (await _escalate(manager)).to_dict()
    for key in ("original_model", "escalated_model", "cost_delta", "quality_gap"):
        assert key in event, f"escalation audit is missing {key}"


# ── Latency budget ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_result_served_when_within_budget(manager):
    result = await _escalate(manager, latency_budget_ms=10_000)
    assert result.returned_to_user
    assert result.better_result is not None


@pytest.mark.asyncio
async def test_result_withheld_when_over_budget(manager):
    """Over budget the escalation is still logged, just not served."""
    result = await _escalate(manager, latency_budget_ms=0.001)
    assert not result.returned_to_user
    assert result.better_result is None
    assert result.escalated_model  # still recorded for the audit trail


@pytest.mark.asyncio
async def test_no_budget_always_serves(manager):
    result = await _escalate(manager, latency_budget_ms=None)
    assert result.returned_to_user


# ── Retries ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_falls_back_to_secondary_model_on_failure(
    registry, providers, routing_config
):
    providers["openai"].fail_models.add("gpt-4o-mini")
    manager = EscalationManager(registry, providers, routing_config)

    result = await _escalate(manager, ComplexityTier.SIMPLE)
    assert result is not None
    assert result.escalated_model == registry.get("claude-haiku-4-5").model_id
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_max_retries_zero_does_not_retry(
    registry, providers, routing_config
):
    routing_config["escalation"]["max_retries"] = 0
    providers["openai"].fail_models.add("gpt-4o-mini")
    manager = EscalationManager(registry, providers, routing_config)

    assert await _escalate(manager, ComplexityTier.SIMPLE) is None


@pytest.mark.asyncio
async def test_returns_none_when_all_attempts_fail(
    registry, providers, routing_config
):
    providers["openai"].fail_models.add("gpt-4o-mini")
    providers["anthropic"].fail_models.add("claude-haiku-4-5")
    manager = EscalationManager(registry, providers, routing_config)

    assert await _escalate(manager, ComplexityTier.SIMPLE) is None
