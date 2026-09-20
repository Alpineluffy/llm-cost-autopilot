"""Complexity tier definitions for prompt classification."""

from __future__ import annotations

from enum import IntEnum


class ComplexityTier(IntEnum):
    """Three-tier complexity system for routing decisions.

    Each tier maps to a cost/quality band in the routing config:
    - SIMPLE  → cheapest model (Ollama / Llama local)
    - MODERATE → mid-tier (GPT-4o-mini / Haiku)
    - COMPLEX  → highest quality (GPT-4o / Sonnet)
    """

    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3

    @property
    def label(self) -> str:
        return {
            1: "simple",
            2: "moderate",
            3: "complex",
        }[self.value]

    @property
    def description(self) -> str:
        return {
            1: (
                "Simple tasks: reformatting, extraction, basic Q&A, "
                "simple syntax questions, boilerplate code generation"
            ),
            2: (
                "Moderate tasks: code summarization, debugging assistance, "
                "code review, structured analysis, concept explanations"
            ),
            3: (
                "Complex tasks: multi-step reasoning, architecture design, "
                "complex algorithm implementation, nuanced code generation, "
                "performance optimization with trade-offs"
            ),
        }[self.value]

    @property
    def routing_key(self) -> str:
        """Key used to look up the model in routing.yaml."""
        return f"tier_{self.value}"

    @classmethod
    def from_string(cls, s: str) -> ComplexityTier:
        mapping = {"simple": 1, "moderate": 2, "complex": 3}
        return cls(mapping.get(s.lower(), 2))
