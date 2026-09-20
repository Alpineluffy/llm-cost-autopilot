"""Shared mock response generation for development without API keys.

Realism matters here: the verifier grades a cheap model's output by comparing
it against the top-tier model's output, so mocks that differ mainly by model
name make every request look like a routing failure and render the whole
demo meaningless.

Real models mostly agree on the substance of an answer and differ in how
complete it is. These mocks reproduce that: the body is derived from the
prompt (so all models overlap heavily), and lower-quality tiers drop detail
and occasionally the answer outright.
"""

from __future__ import annotations

import hashlib
import random
import re

#: Fraction of the full answer each quality tier produces.
TIER_COMPLETENESS = {"high": 1.0, "medium": 0.75, "low": 0.5}

#: Chance a tier gets a classification label wrong.
TIER_LABEL_ERROR_RATE = {"high": 0.0, "medium": 0.12, "low": 0.3}

SENTIMENT_LABELS = ["positive", "negative", "neutral"]


def _seed(prompt: str) -> int:
    return int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)


def _is_classification(prompt: str) -> bool:
    return bool(
        re.search(r"\b(classify|categori[sz]e|sentiment|label|yes or no)\b",
                  prompt, re.I)
    )


def _is_extraction(prompt: str) -> bool:
    return bool(
        re.search(r"\b(extract|pull out|parse|as json|into json)\b", prompt, re.I)
    )


def _classification_response(prompt: str, quality_tier: str) -> str:
    """Return an actual label, occasionally wrong for weaker models."""
    rng = random.Random(_seed(prompt))
    truth = rng.choice(SENTIMENT_LABELS)

    error_rate = TIER_LABEL_ERROR_RATE.get(quality_tier, 0.2)
    # A separate stream so the error draw does not shift the truth draw.
    if random.Random(_seed(prompt) + 1).random() < error_rate:
        wrong = [label for label in SENTIMENT_LABELS if label != truth]
        return f"Label: {random.Random(_seed(prompt) + 2).choice(wrong)}"

    return f"Label: {truth}"


def _extraction_response(prompt: str, quality_tier: str) -> str:
    """Return JSON fields, with weaker models dropping some of them."""
    fields = {
        "name": "Ada Lovelace",
        "email": "ada@example.com",
        "phone": "+1-555-0101",
        "company": "Analytical Engines Ltd",
        "role": "Lead Engineer",
    }
    completeness = TIER_COMPLETENESS.get(quality_tier, 0.7)
    keep = max(1, round(len(fields) * completeness))
    kept = list(fields.items())[:keep]

    body = ",\n".join(f'  "{k}": "{v}"' for k, v in kept)
    return "{\n" + body + "\n}"


def _prose_response(prompt: str, display_name: str, quality_tier: str) -> str:
    """Build a prompt-derived answer, truncated by quality tier.

    Every model draws from the same prompt-seeded sentence pool, so outputs
    overlap the way real model outputs do. Cheaper tiers return fewer points.
    """
    rng = random.Random(_seed(prompt))
    topic_words = [w for w in re.findall(r"[a-z]{4,}", prompt.lower())][:6]
    topic = " ".join(topic_words[:3]) or "the request"

    pool = [
        f"The core idea behind {topic} is to keep the interface small and the "
        f"behaviour predictable under load.",
        f"In practice you start by identifying the invariants that must hold, "
        f"then design around the failure cases rather than the happy path.",
        f"A common mistake is treating {topic} as a single step when it is "
        f"really a pipeline with distinct validation and transformation stages.",
        f"Consider the edge cases explicitly: empty input, duplicate keys, and "
        f"concurrent modification all change the correct answer here.",
        f"Performance follows from reducing the work done per item, not from "
        f"micro-optimising the inner loop.",
        f"For maintainability, prefer explicit names and a narrow public "
        f"surface so future readers can follow the control flow.",
        f"Finally, add a test that covers the boundary condition, since that "
        f"is where regressions in {topic} usually appear.",
    ]
    rng.shuffle(pool)

    completeness = TIER_COMPLETENESS.get(quality_tier, 0.7)
    keep = max(1, round(len(pool) * completeness))
    body = " ".join(pool[:keep])

    if re.search(r"\b(write|implement|function|code|build)\b", prompt, re.I):
        code = (
            "\n\n```python\n"
            "def solution(items):\n"
            "    result = []\n"
            "    for item in items:\n"
            "        if item is not None:\n"
            "            result.append(transform(item))\n"
            "    return result\n"
            "```"
        )
        if completeness < 0.75:
            # Weaker models return a sketch rather than working code.
            code = "\n\n```python\ndef solution(items):\n    pass\n```"
        body += code

    return f"{body}\n\n— {display_name}"


def generate_mock_response(
    prompt: str, display_name: str, quality_tier: str
) -> str:
    """Produce a realistic mock answer for a model of the given quality tier."""
    if _is_classification(prompt):
        return _classification_response(prompt, quality_tier)
    if _is_extraction(prompt):
        return _extraction_response(prompt, quality_tier)
    return _prose_response(prompt, display_name, quality_tier)
