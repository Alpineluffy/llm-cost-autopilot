"""OpenAI provider adapter — supports GPT-4o and GPT-4o-mini.

Runs mocked only when MOCK_PROVIDERS is explicitly true; a missing key in
live mode raises rather than silently fabricating responses.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import random
import time
from typing import Optional

from src.models.providers.base import BaseProvider
from src.models.registry import ModelConfig
from src.models.providers.mock_content import generate_mock_response
from src.models.providers.mode import require_sdk, resolve_mock_mode
from src.models.providers.tokens import count_tokens
from src.models.response import LLMResponse


class OpenAIProvider(BaseProvider):
    """OpenAI API adapter with automatic mock fallback."""

    def __init__(self) -> None:
        self._api_key = os.getenv("OPENAI_API_KEY", "")
        self._mock = resolve_mock_mode(
            "OpenAIProvider", self._api_key, "OPENAI_API_KEY"
        )
        self._client = None

        if not self._mock:
            sdk = require_sdk("OpenAIProvider", "openai", "openai")
            self._client = sdk.AsyncOpenAI(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "openai"

    async def send_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        if self._mock:
            return await self._mock_request(prompt, model_config, max_tokens, temperature)
        return await self._real_request(prompt, model_config, max_tokens, temperature)

    # ── Real API call ───────────────────────────────────────────────────

    async def _real_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int],
        temperature: float,
    ) -> LLMResponse:
        start = time.perf_counter()

        response = await self._client.chat.completions.create(
            model=model_config.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens or model_config.max_tokens,
            temperature=temperature,
        )

        latency_ms = (time.perf_counter() - start) * 1000
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            text=choice.message.content or "",
            model_id=model_config.model_id,
            provider=self.provider_name,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost=model_config.compute_cost(usage.prompt_tokens, usage.completion_tokens),
            latency_ms=latency_ms,
        )

    # ── Mock for development ────────────────────────────────────────────

    async def _mock_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int],
        temperature: float,
    ) -> LLMResponse:
        # Simulate realistic latency
        base_latency = model_config.avg_latency_ms
        jitter = random.uniform(-0.15, 0.15) * base_latency
        await asyncio.sleep((base_latency + jitter) / 1000)

        # Generate deterministic mock response based on prompt
        mock_text = self._generate_mock_response(prompt, model_config)
        input_tokens = count_tokens(prompt, model_config.model_id)
        output_tokens = count_tokens(mock_text, model_config.model_id)

        return LLMResponse(
            text=mock_text,
            model_id=model_config.model_id,
            provider=self.provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=model_config.compute_cost(input_tokens, output_tokens),
            latency_ms=base_latency + jitter,
        )

    def _generate_mock_response(self, prompt: str, model_config: ModelConfig) -> str:
        """Delegate to the shared generator so all providers stay comparable."""
        return generate_mock_response(
            prompt, model_config.display_name, model_config.quality_tier
        )
