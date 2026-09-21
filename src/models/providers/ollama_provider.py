"""Ollama provider adapter — connects to locally running Ollama instance.

Unlike cloud providers, Ollama is always "real" (no mock needed) since it
runs locally at zero cost. Falls back to mock if Ollama is unreachable.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from typing import Optional

import httpx

from src.models.providers.base import BaseProvider
from src.models.registry import ModelConfig
from src.models.providers.mock_content import generate_mock_response
from src.models.providers.mode import ProviderConfigurationError, resolve_mock_mode
from src.models.providers.tokens import count_tokens
from src.models.response import LLMResponse


class OllamaProvider(BaseProvider):
    """Ollama local LLM adapter via HTTP API."""

    def __init__(self) -> None:
        self._base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        # No credential to check, but the same rule applies: mock only when asked.
        self._mock = resolve_mock_mode("OllamaProvider", requires_credential=False)

    @property
    def provider_name(self) -> str:
        return "ollama"

    async def send_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if self._mock:
            return await self._mock_request(prompt, model_config, max_tokens, temperature)

        try:
            return await self._real_request(prompt, model_config, max_tokens, temperature)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            # Previously this fell back to mock, which meant an Ollama outage
            # silently turned real traffic into fabricated answers. Surface it
            # instead: the router's own fallback will pick another model.
            raise ProviderConfigurationError(
                f"OllamaProvider: cannot reach Ollama at {self._base_url} "
                f"({type(exc).__name__}). Start Ollama, point OLLAMA_BASE_URL "
                f"at a running instance, or set MOCK_PROVIDERS=true."
            ) from exc

    # ── Real API call ───────────────────────────────────────────────────

    async def _real_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int],
        temperature: float,
    ) -> LLMResponse:
        start = time.perf_counter()

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": model_config.model_id,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "num_predict": max_tokens or model_config.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        latency_ms = (time.perf_counter() - start) * 1000

        # Ollama returns token counts in eval_count / prompt_eval_count
        # Ollama usually reports counts; when it omits them, tokenize rather
        # than guess — these feed the savings baseline directly.
        response_text = data.get("response", "")
        input_tokens = data.get("prompt_eval_count") or count_tokens(
            prompt, model_config.model_id
        )
        output_tokens = data.get("eval_count") or count_tokens(
            response_text, model_config.model_id
        )

        return LLMResponse(
            text=response_text,
            model_id=model_config.model_id,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=model_config.compute_cost(input_tokens, output_tokens),  # $0 for local
            latency_ms=latency_ms,
        )

    # ── Mock for when Ollama is unavailable ─────────────────────────────

    async def _mock_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int],
        temperature: float,
    ) -> LLMResponse:
        base_latency = model_config.avg_latency_ms
        jitter = random.uniform(-0.2, 0.2) * base_latency
        await asyncio.sleep((base_latency + jitter) / 1000)

        mock_text = self._generate_mock_response(prompt, model_config)
        input_tokens = count_tokens(prompt, model_config.model_id)
        output_tokens = count_tokens(mock_text, model_config.model_id)

        return LLMResponse(
            text=mock_text,
            model_id=model_config.model_id,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=0.0,  # Local model — always free
            latency_ms=base_latency + jitter,
        )

    def _generate_mock_response(self, prompt: str, model_config: ModelConfig) -> str:
        """Delegate to the shared generator so all providers stay comparable."""
        return generate_mock_response(
            prompt, model_config.display_name, model_config.quality_tier
        )
