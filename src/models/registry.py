"""Model configuration and registry — loads from config/models.yaml."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    """Specification for a single LLM model."""

    name: str                         # Registry key (e.g. "gpt-4o")
    provider: str                     # "openai" | "anthropic" | "ollama"
    model_id: str                     # API model identifier
    display_name: str
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    max_tokens: int
    context_window: int
    quality_tier: str                 # "high" | "medium" | "low"
    avg_latency_ms: int
    supports_streaming: bool = True

    # ── Cost helpers ────────────────────────────────────────────────────

    @property
    def input_cost_per_token(self) -> float:
        return self.cost_per_1m_input_tokens / 1_000_000

    @property
    def output_cost_per_token(self) -> float:
        return self.cost_per_1m_output_tokens / 1_000_000

    def compute_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate the dollar cost for a given token count."""
        return (
            input_tokens * self.input_cost_per_token
            + output_tokens * self.output_cost_per_token
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "display_name": self.display_name,
            "cost_per_1m_input_tokens": self.cost_per_1m_input_tokens,
            "cost_per_1m_output_tokens": self.cost_per_1m_output_tokens,
            "max_tokens": self.max_tokens,
            "context_window": self.context_window,
            "quality_tier": self.quality_tier,
            "avg_latency_ms": self.avg_latency_ms,
            "supports_streaming": self.supports_streaming,
        }


class ModelRegistry:
    """Central registry of all available LLM models.

    Loads from a YAML config and provides fast lookups by name, provider,
    or quality tier.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        if config_path is None:
            config_path = os.path.join(
                Path(__file__).resolve().parent.parent.parent, "config", "models.yaml"
            )
        self._models: dict[str, ModelConfig] = {}
        self._load(config_path)

    # ── Loading ─────────────────────────────────────────────────────────

    def _load(self, path: str) -> None:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        for name, cfg in raw.get("models", {}).items():
            self._models[name] = ModelConfig(name=name, **cfg)

    # ── Lookups ─────────────────────────────────────────────────────────

    def get(self, name: str) -> ModelConfig:
        """Get a model by its registry key. Raises KeyError if not found."""
        if name not in self._models:
            raise KeyError(
                f"Model '{name}' not in registry. "
                f"Available: {list(self._models.keys())}"
            )
        return self._models[name]

    def get_by_provider(self, provider: str) -> list[ModelConfig]:
        """Return all models for a given provider."""
        return [m for m in self._models.values() if m.provider == provider]

    def get_by_tier(self, tier: str) -> list[ModelConfig]:
        """Return all models in a quality tier ('high', 'medium', 'low')."""
        return [m for m in self._models.values() if m.quality_tier == tier]

    def get_cheapest(self, tier: Optional[str] = None) -> ModelConfig:
        """Return the cheapest model, optionally filtered by quality tier."""
        candidates = (
            self.get_by_tier(tier) if tier else list(self._models.values())
        )
        if not candidates:
            raise ValueError(f"No models found for tier '{tier}'")
        return min(
            candidates,
            key=lambda m: m.cost_per_1m_input_tokens + m.cost_per_1m_output_tokens,
        )

    def get_highest_quality(self) -> ModelConfig:
        """Return the highest-quality (most expensive) model."""
        return max(
            self._models.values(),
            key=lambda m: m.cost_per_1m_input_tokens + m.cost_per_1m_output_tokens,
        )

    def list_all(self) -> list[ModelConfig]:
        return list(self._models.values())

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models
