"""Standardized LLM response object returned by all providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class LLMResponse:
    """Unified response from any LLM provider.

    Every provider adapter MUST return this exact structure so the router,
    logger, and verifier can operate provider-agnostically.
    """

    text: str
    model_id: str
    provider: str
    input_tokens: int
    output_tokens: int
    cost: float
    latency_ms: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary."""
        return {
            "text": self.text,
            "model_id": self.model_id,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": round(self.cost, 8),
            "latency_ms": round(self.latency_ms, 2),
            "timestamp": self.timestamp.isoformat(),
        }
