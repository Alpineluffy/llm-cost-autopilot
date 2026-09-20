"""Auto-escalation — re-routes to a higher-tier model when verification fails.

Phase 3.3 — when the verifier catches a routing failure, re-run the request
on the next tier up and return the better result *if latency permits*. Every
escalation is recorded with the original model, the escalated model, the cost
delta, and the quality gap that triggered it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

from src.classifier.tiers import ComplexityTier
from src.models.providers.base import BaseProvider
from src.models.registry import ModelRegistry
from src.models.response import LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class EscalationResult:
    """Record of an escalation event — the full audit row."""

    original_model: str
    escalated_model: str
    original_cost: float
    escalated_cost: float
    cost_delta: float
    quality_gap: float
    original_tier: ComplexityTier
    escalated_tier: ComplexityTier
    new_response: LLMResponse

    # Phase 3.3 additions
    use_case: str = "general"
    trigger_reason: str = ""
    returned_to_user: bool = False   # Did latency permit serving this result?
    latency_budget_ms: Optional[float] = None
    escalation_latency_ms: float = 0.0
    attempts: int = 1

    @property
    def better_result(self) -> Optional[LLMResponse]:
        """The escalated response, when it is fit to serve to the user."""
        return self.new_response if self.returned_to_user else None

    def to_dict(self) -> dict:
        return {
            "original_model": self.original_model,
            "escalated_model": self.escalated_model,
            "original_cost": round(self.original_cost, 8),
            "escalated_cost": round(self.escalated_cost, 8),
            "cost_delta": round(self.cost_delta, 8),
            "quality_gap": round(self.quality_gap, 3),
            "original_tier": self.original_tier.value,
            "escalated_tier": self.escalated_tier.value,
            "use_case": self.use_case,
            "trigger_reason": self.trigger_reason,
            "returned_to_user": self.returned_to_user,
            "latency_budget_ms": self.latency_budget_ms,
            "escalation_latency_ms": round(self.escalation_latency_ms, 2),
            "attempts": self.attempts,
        }


class EscalationManager:
    """Re-runs failed requests on a higher tier and reports the cost/quality delta."""

    TIER_ESCALATION = {
        ComplexityTier.SIMPLE: ComplexityTier.MODERATE,
        ComplexityTier.MODERATE: ComplexityTier.COMPLEX,
        ComplexityTier.COMPLEX: ComplexityTier.COMPLEX,  # Already at max
    }

    def __init__(
        self,
        registry: ModelRegistry,
        providers: dict[str, BaseProvider],
        routing_config: dict,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._routing_config = routing_config

    # ── Config helpers ──────────────────────────────────────────────────

    @property
    def _escalation_config(self) -> dict:
        return self._routing_config.get("escalation", {}) or {}

    @property
    def enabled(self) -> bool:
        return bool(self._escalation_config.get("enabled", True))

    @property
    def max_retries(self) -> int:
        return int(self._escalation_config.get("max_retries", 1))

    @property
    def default_latency_budget_ms(self) -> Optional[float]:
        budget = self._escalation_config.get("latency_budget_ms")
        return float(budget) if budget is not None else None

    def _candidates_for(self, tier: ComplexityTier) -> list[str]:
        """Primary then fallback model names for a tier, in attempt order."""
        tier_config = (
            self._routing_config.get("routing", {}).get(tier.routing_key, {}) or {}
        )
        names = [tier_config.get("primary"), tier_config.get("fallback")]
        seen, ordered = set(), []
        for name in names:
            if name and name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered or ["gpt-4o"]

    # ── Main entry point ────────────────────────────────────────────────

    async def escalate(
        self,
        prompt: str,
        original_response: LLMResponse,
        original_tier: ComplexityTier,
        quality_score: float,
        quality_threshold: float,
        use_case: str = "general",
        trigger_reason: str = "",
        latency_budget_ms: Optional[float] = None,
    ) -> Optional[EscalationResult]:
        """Re-run the request one tier up.

        Returns None if escalation is disabled, the request is already at the
        highest tier, or every attempt failed.
        """
        if not self.enabled:
            logger.info("Escalation disabled by config")
            return None

        escalated_tier = self.TIER_ESCALATION[original_tier]
        if escalated_tier == original_tier:
            logger.info("Already at highest tier — cannot escalate further")
            return None

        if latency_budget_ms is None:
            latency_budget_ms = self.default_latency_budget_ms

        candidates = self._candidates_for(escalated_tier)
        attempts = 0
        start = time.perf_counter()

        for model_name in candidates[: self.max_retries + 1]:
            attempts += 1
            try:
                model_config = self._registry.get(model_name)
            except KeyError:
                logger.error(f"Escalation model '{model_name}' not in registry")
                continue

            provider = self._providers.get(model_config.provider)
            if provider is None:
                logger.error(f"No provider for '{model_config.provider}'")
                continue

            try:
                new_response = await provider.send_request(
                    prompt=prompt, model_config=model_config
                )
            except Exception as e:
                logger.error(f"Escalation attempt {attempts} failed: {e}")
                continue

            elapsed_ms = (time.perf_counter() - start) * 1000
            cost_delta = new_response.cost - original_response.cost
            quality_gap = round(quality_threshold - quality_score, 3)

            # Latency gate: serve the better result only if the budget allows.
            returned_to_user = (
                latency_budget_ms is None or elapsed_ms <= latency_budget_ms
            )

            logger.info(
                f"Escalated [{use_case}]: {original_response.model_id} → "
                f"{new_response.model_id} (cost delta ${cost_delta:+.6f}, "
                f"quality gap {quality_gap:.2f}, {elapsed_ms:.0f}ms, "
                f"{'served' if returned_to_user else 'logged only — over budget'})"
            )

            return EscalationResult(
                original_model=original_response.model_id,
                escalated_model=new_response.model_id,
                original_cost=original_response.cost,
                escalated_cost=new_response.cost,
                cost_delta=cost_delta,
                quality_gap=quality_gap,
                original_tier=original_tier,
                escalated_tier=escalated_tier,
                new_response=new_response,
                use_case=use_case,
                trigger_reason=trigger_reason,
                returned_to_user=returned_to_user,
                latency_budget_ms=latency_budget_ms,
                escalation_latency_ms=elapsed_ms,
                attempts=attempts,
            )

        logger.error(f"All {attempts} escalation attempts failed")
        return None
