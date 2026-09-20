"""Abstract base class for all LLM provider adapters."""

from __future__ import annotations

import abc
from typing import Optional

from src.models.registry import ModelConfig
from src.models.response import LLMResponse


class BaseProvider(abc.ABC):
    """Every provider must implement send_request().

    The contract: accept a prompt + model config, return a standardized
    LLMResponse. All provider-specific logic lives inside the subclass.
    """

    @abc.abstractmethod
    async def send_request(
        self,
        prompt: str,
        model_config: ModelConfig,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> LLMResponse:
        """Send a prompt to the model and return a standardized response.

        Args:
            prompt: The user prompt text.
            model_config: Configuration for the target model.
            max_tokens: Override for max output tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with text, token counts, cost, and latency.
        """
        ...

    @property
    @abc.abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'openai', 'anthropic')."""
        ...
