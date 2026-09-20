"""Core routing engine — ties classifier → provider selection → response.

This is the heart of the system: it classifies the prompt, picks the
cheapest capable model, sends the request, logs everything, and queues
for async verification.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional

import yaml

from src.classifier.model import ComplexityClassifier
from src.classifier.tiers import ComplexityTier
from src.models.providers.base import BaseProvider
from src.models.providers.openai_provider import OpenAIProvider
from src.models.providers.anthropic_provider import AnthropicProvider
from src.models.providers.ollama_provider import OllamaProvider
from src.models.registry import ModelConfig, ModelRegistry
from src.models.response import LLMResponse

logger = logging.getLogger(__name__)

#: Called after every completed request. The verification layer subscribes to
#: this so the engine never has to import the verifier (which would be a cycle:
#: verifier → logger → engine).
CompletionHook = Callable[[str, "RoutedResponse"], Awaitable[None]]


@dataclass
class RoutingDecision:
    """Captures the full routing decision for audit trail."""
    tier: ComplexityTier
    confidence: dict[str, float]
    selected_model: str
    is_fallback: bool
    reasoning: str


@dataclass
class RoutedResponse:
    """Response bundled with routing metadata."""
    response: LLMResponse
    decision: RoutingDecision
    hypothetical_cost: float  # What the most expensive model would have cost

    #: Set by the completion hook once the request is logged. Carried per
    #: instance rather than on the service, which would race across requests.
    log_id: Optional[int] = None
    use_case: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "log_id": self.log_id,
            "use_case": self.use_case,
            "response": self.response.to_dict(),
            "routing": {
                "tier": self.decision.tier.label,
                "confidence": self.decision.confidence,
                "selected_model": self.decision.selected_model,
                "is_fallback": self.decision.is_fallback,
                "reasoning": self.decision.reasoning,
            },
            "hypothetical_cost": round(self.hypothetical_cost, 8),
            "cost_savings": round(self.hypothetical_cost - self.response.cost, 8),
        }


class RoutingEngine:
    """Orchestrates the full request pipeline.

    1. Classify prompt complexity
    2. Look up routing config → model
    3. Send via appropriate provider
    4. Compute hypothetical cost (what GPT-4o would have cost)
    5. Return response with full routing metadata
    """

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        classifier: Optional[ComplexityClassifier] = None,
        routing_config_path: Optional[str] = None,
        on_complete: Optional[CompletionHook] = None,
    ) -> None:
        self._registry = registry or ModelRegistry()
        self._classifier = classifier or ComplexityClassifier()
        self._routing_config = self._load_routing_config(routing_config_path)
        self._on_complete = on_complete

        # Initialize providers
        self._providers: dict[str, BaseProvider] = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "ollama": OllamaProvider(),
        }

    def set_completion_hook(self, hook: CompletionHook) -> None:
        """Register the post-response hook (used to queue async verification)."""
        self._on_complete = hook

    def _load_routing_config(self, path: Optional[str] = None) -> dict:
        if path is None:
            path = os.path.join(
                Path(__file__).resolve().parent.parent.parent, "config", "routing.yaml"
            )
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ── Main entry point ────────────────────────────────────────────────

    async def route_request(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> RoutedResponse:
        """Route a prompt to the optimal model and return the response."""

        # Step 1: Classify complexity
        if self._classifier.is_trained:
            tier, confidence = self._classifier.predict_with_confidence(prompt)
        else:
            # Fallback: default to moderate if classifier isn't trained
            tier = ComplexityTier.MODERATE
            confidence = {"simple": 0.0, "moderate": 1.0, "complex": 0.0}

        # Step 2: Look up model from routing config
        model_name, is_fallback = self._select_model(tier)
        model_config = self._registry.get(model_name)

        # Step 3: Get the provider and send request
        provider = self._providers[model_config.provider]

        try:
            response = await provider.send_request(
                prompt=prompt,
                model_config=model_config,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            # Try fallback model
            model_name, _ = self._select_model(tier, force_fallback=True)
            model_config = self._registry.get(model_name)
            provider = self._providers[model_config.provider]
            is_fallback = True

            response = await provider.send_request(
                prompt=prompt,
                model_config=model_config,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        # Step 4: Compute hypothetical cost (most expensive model)
        most_expensive = self._registry.get_highest_quality()
        hypothetical_cost = most_expensive.compute_cost(
            response.input_tokens, response.output_tokens
        )

        # Step 5: Build routing decision
        decision = RoutingDecision(
            tier=tier,
            confidence=confidence,
            selected_model=model_name,
            is_fallback=is_fallback,
            reasoning=self._build_reasoning(tier, model_name, confidence),
        )

        routed = RoutedResponse(
            response=response,
            decision=decision,
            hypothetical_cost=hypothetical_cost,
        )

        # Step 6: Hand off for async verification. The user's response is
        # already built, so a failure here must never surface to them.
        if self._on_complete is not None:
            try:
                await self._on_complete(prompt, routed)
            except Exception as e:
                logger.error(f"Completion hook failed: {e}")

        return routed

    # ── Model selection ─────────────────────────────────────────────────

    def _select_model(
        self, tier: ComplexityTier, force_fallback: bool = False
    ) -> tuple[str, bool]:
        """Look up the model for a tier from routing config."""
        routing = self._routing_config.get("routing", {})
        tier_config = routing.get(tier.routing_key, {})

        if force_fallback or not tier_config.get("primary"):
            fallback = tier_config.get("fallback", "gpt-4o-mini")
            return fallback, True

        primary = tier_config["primary"]

        # Verify model exists in registry
        if primary in self._registry:
            return primary, False

        fallback = tier_config.get("fallback", "gpt-4o-mini")
        return fallback, True

    def _build_reasoning(
        self, tier: ComplexityTier, model: str, confidence: dict
    ) -> str:
        top_conf = max(confidence.values()) if confidence else 0
        return (
            f"Classified as {tier.label} (confidence: {top_conf:.0%}). "
            f"Routed to {model} per tier_{tier.value} config."
        )

    # ── Config hot-reload ───────────────────────────────────────────────

    def update_routing_config(self, new_config: dict) -> None:
        """Update tier-to-model mappings without restart."""
        if "routing" in new_config:
            self._routing_config["routing"] = new_config["routing"]

    def get_routing_config(self) -> dict:
        return self._routing_config

    @property
    def registry(self) -> ModelRegistry:
        return self._registry
