"""Token counting for paths where the provider doesn't report usage.

Provider-reported counts are always authoritative and are used when present.
This module covers the gaps: Ollama responses that omit eval counts, and mock
responses.

Accuracy matters more here than it first appears. The savings headline is
computed as `most_expensive_model.compute_cost(input_tokens, output_tokens)`,
so the counterfactual scales linearly with these numbers — and a local model
handling half the traffic means a sloppy estimate moves the headline directly.
The previous `len(text.split()) * 2` heuristic overcounts ordinary prose and
undercounts dense code.
"""

from __future__ import annotations

import functools
import logging
import re

logger = logging.getLogger(__name__)

#: tiktoken has no encoding for non-OpenAI models. cl100k_base is a reasonable
#: stand-in: Llama and Claude tokenizers land within roughly 10-20% of it on
#: English and code, far closer than a word-count heuristic.
_DEFAULT_ENCODING = "cl100k_base"


@functools.lru_cache(maxsize=8)
def _encoder(encoding_name: str):
    """Load and cache a tiktoken encoder. Returns None if unavailable."""
    try:
        import tiktoken

        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # ImportError, or a download failure offline
        logger.warning(
            f"tiktoken unavailable ({exc}); falling back to a heuristic "
            f"token estimate. Cost figures will be approximate."
        )
        return None


@functools.lru_cache(maxsize=4)
def _encoder_for_model(model_id: str):
    """Prefer the model's own encoding when tiktoken knows it."""
    try:
        import tiktoken

        return tiktoken.encoding_for_model(model_id)
    except Exception:
        return _encoder(_DEFAULT_ENCODING)


def heuristic_token_count(text: str) -> int:
    """Estimate tokens without tiktoken.

    Splits on word boundaries and punctuation, which tracks real tokenizers
    better than a flat word multiplier because it charges code and symbols
    closer to their true cost.
    """
    if not text:
        return 0
    pieces = re.findall(r"\w+|[^\w\s]", text)
    # Long words split into multiple tokens; ~4 characters per token is the
    # widely used rule of thumb for English.
    extra = sum(max(0, (len(p) - 1) // 4) for p in pieces if p.isalnum())
    return max(1, len(pieces) + extra)


def count_tokens(text: str, model_id: str = "") -> int:
    """Count tokens in `text`, as accurately as the environment allows."""
    if not text:
        return 0

    encoder = _encoder_for_model(model_id) if model_id else _encoder(_DEFAULT_ENCODING)
    if encoder is None:
        return heuristic_token_count(text)

    try:
        return len(encoder.encode(text))
    except Exception:
        return heuristic_token_count(text)


def count_message_tokens(prompt: str, response: str, model_id: str = "") -> tuple[int, int]:
    """Convenience: count a prompt/response pair in one call."""
    return count_tokens(prompt, model_id), count_tokens(response, model_id)
