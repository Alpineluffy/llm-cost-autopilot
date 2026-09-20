"""Shared fixtures — scripted providers so tests never touch a real API."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["MOCK_PROVIDERS"] = "true"

from src.logging.database import Database  # noqa: E402
from src.logging.logger import RequestLogger  # noqa: E402
from src.models.providers.base import BaseProvider  # noqa: E402
from src.models.registry import ModelRegistry  # noqa: E402
from src.models.response import LLMResponse  # noqa: E402


class ScriptedProvider(BaseProvider):
    """Returns canned text keyed by model name, so outputs are deterministic."""

    def __init__(self, name: str, responses: dict[str, str]) -> None:
        self._name = name
        self._responses = responses
        self.calls: list[tuple[str, str]] = []
        self.fail_models: set[str] = set()

    @property
    def provider_name(self) -> str:
        return self._name

    async def send_request(
        self,
        prompt: str,
        model_config,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        self.calls.append((model_config.name, prompt))

        if model_config.name in self.fail_models:
            raise RuntimeError(f"Simulated failure for {model_config.name}")

        text = self._responses.get(model_config.name, f"[{model_config.name}] answer")
        input_tokens, output_tokens = 100, 50
        return LLMResponse(
            text=text,
            model_id=model_config.model_id,
            provider=self._name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=model_config.compute_cost(input_tokens, output_tokens),
            latency_ms=float(model_config.avg_latency_ms),
        )


@pytest.fixture
def registry() -> ModelRegistry:
    return ModelRegistry()


@pytest.fixture
def routing_config() -> dict:
    """Mirrors config/routing.yaml, with escalation fully enabled."""
    return {
        "routing": {
            "tier_1": {"primary": "llama3.2", "fallback": "claude-haiku-4-5"},
            "tier_2": {"primary": "gpt-4o-mini", "fallback": "claude-haiku-4-5"},
            "tier_3": {"primary": "gpt-4o", "fallback": "claude-sonnet-4-6"},
        },
        "verification": {"judge_model": "gpt-4o", "sample_rate": 1.0},
        "escalation": {
            "enabled": True,
            "max_retries": 1,
            "latency_budget_ms": 10_000,
        },
        "feedback": {"retrain_interval_days": 7, "min_examples_to_retrain": 2},
    }


@pytest.fixture
def responses() -> dict[str, str]:
    """Default canned outputs — overridden per test as needed."""
    return {}


@pytest.fixture
def providers(responses) -> dict[str, BaseProvider]:
    return {
        "openai": ScriptedProvider("openai", responses),
        "anthropic": ScriptedProvider("anthropic", responses),
        "ollama": ScriptedProvider("ollama", responses),
    }


@pytest.fixture
def database(tmp_path) -> Database:
    return Database(f"sqlite:///{tmp_path}/test.db")


@pytest.fixture
def request_logger(database) -> RequestLogger:
    return RequestLogger(database)
