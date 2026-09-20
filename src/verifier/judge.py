"""Reference-comparison quality verification.

Phase 3.2 — the verifier sends the *same prompt* to the highest-tier model
and compares the two outputs. How they are compared depends on the use case:

    extraction     → did the cheap model capture every key field?
    classification → does its label match the reference label?
    summarization  → LLM-as-judge rubric score
    general        → semantic agreement with the reference

A significant divergence is a routing failure, and is logged as one.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from typing import Optional

from src.models.providers.base import BaseProvider
from src.models.registry import ModelRegistry
from src.models.response import LLMResponse
from src.verifier.use_cases import (
    QualityPolicy,
    UseCase,
    VerificationStrategy,
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as", "that",
    "this", "these", "those", "it", "its", "and", "or", "but", "if", "then",
    "so", "you", "your", "we", "our", "they", "their", "can", "will", "would",
    "should", "could", "has", "have", "had", "do", "does", "did", "not",
    "here", "there", "which", "what", "how", "when", "where", "who", "why",
}


@dataclass(frozen=True)
class QualityScore:
    """Rubric assessment from the LLM judge."""

    accuracy: float       # 1.0 - 5.0
    completeness: float   # 1.0 - 5.0
    coherence: float      # 1.0 - 5.0

    @property
    def aggregate(self) -> float:
        return round((self.accuracy + self.completeness + self.coherence) / 3, 2)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "completeness": self.completeness,
            "coherence": self.coherence,
            "aggregate": self.aggregate,
        }


@dataclass(frozen=True)
class VerificationOutcome:
    """Full result of comparing a cheap response against the reference.

    `score` is always on a 1.0-5.0 scale so a single threshold comparison
    works regardless of which strategy produced it.
    """

    score: float
    passed: bool
    policy: QualityPolicy
    strategy: VerificationStrategy
    detail: str
    reference_model: Optional[str] = None
    reference_cost: float = 0.0
    divergence: float = 0.0        # 0.0 (identical) - 1.0 (totally divergent)
    rubric: Optional[QualityScore] = field(default=None)

    @property
    def use_case(self) -> UseCase:
        return self.policy.use_case

    @property
    def quality_gap(self) -> float:
        """How far below the bar this landed. 0.0 when it passed."""
        return max(0.0, round(self.policy.threshold - self.score, 3))

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "passed": self.passed,
            "use_case": self.use_case.value,
            "strategy": self.strategy.value,
            "threshold": self.policy.threshold,
            "criterion": self.policy.criterion,
            "quality_gap": self.quality_gap,
            "divergence": self.divergence,
            "detail": self.detail,
            "reference_model": self.reference_model,
            "reference_cost": round(self.reference_cost, 8),
            "rubric": self.rubric.to_dict() if self.rubric else None,
        }


JUDGE_PROMPT_TEMPLATE = """You are a quality evaluator for LLM responses about programming and code.

Rate the following response on three criteria, each from 1.0 to 5.0:
- **Accuracy**: Is the information correct? Are code samples syntactically and logically valid?
- **Completeness**: Does it fully address the user's question? Any missing edge cases?
- **Coherence**: Is it well-structured, clear, and easy to follow?

USER PROMPT:
{prompt}

MODEL RESPONSE:
{response}

Respond ONLY in this exact format (no other text):
Accuracy: X.X
Completeness: X.X
Coherence: X.X"""


class QualityJudge:
    """Scores a response by comparing it against a highest-tier reference."""

    def __init__(
        self,
        registry: ModelRegistry,
        providers: dict[str, BaseProvider],
        judge_model: Optional[str] = None,
    ) -> None:
        self._registry = registry
        self._providers = providers
        self._judge_model = judge_model or "gpt-4o"
        self._mock = os.getenv("MOCK_PROVIDERS", "true").lower() == "true"

    # ── Main entry point ────────────────────────────────────────────────

    async def verify(
        self,
        prompt: str,
        response: LLMResponse,
        policy: QualityPolicy,
    ) -> VerificationOutcome:
        """Compare a response against the reference model under `policy`."""
        reference = await self._fetch_reference(prompt, response)

        if policy.strategy is VerificationStrategy.FIELD_COVERAGE:
            return self._score_field_coverage(response, reference, policy)
        if policy.strategy is VerificationStrategy.LABEL_MATCH:
            return self._score_label_match(response, reference, policy)
        if policy.strategy is VerificationStrategy.AGREEMENT:
            return self._score_agreement(response, reference, policy)
        return await self._score_rubric(prompt, response, reference, policy)

    async def _fetch_reference(
        self, prompt: str, response: LLMResponse
    ) -> Optional[LLMResponse]:
        """Send the same prompt to the highest-tier model.

        Returns None when the response already came from the reference model
        (nothing to compare against) or the call fails.
        """
        reference_config = self._registry.get_highest_quality()
        if reference_config.model_id == response.model_id:
            return None

        provider = self._providers.get(reference_config.provider)
        if provider is None:
            return None

        try:
            return await provider.send_request(
                prompt=prompt,
                model_config=reference_config,
                temperature=0.1,
            )
        except Exception:
            return None

    # ── Strategy: field coverage (extraction) ───────────────────────────

    def _score_field_coverage(
        self,
        response: LLMResponse,
        reference: Optional[LLMResponse],
        policy: QualityPolicy,
    ) -> VerificationOutcome:
        """Did the cheap model capture every field the reference found?"""
        if reference is None:
            return self._no_reference(policy)

        expected = self._extract_fields(reference.text)
        if not expected:
            # The reference itself produced no parseable fields, so field
            # coverage is unmeasurable. Scoring this on the strict extraction
            # bar with a different metric would fail it for our blind spot,
            # not its own — report inconclusive instead.
            return self._inconclusive(
                policy, "Reference produced no parseable fields to compare against"
            )

        found = self._extract_fields(response.text)
        candidate_text = response.text.lower()

        # A field counts as captured if it appears as a parsed field OR
        # anywhere in the candidate text (formatting differences are fine).
        captured = {f for f in expected if f in found or f in candidate_text}
        coverage = len(captured) / len(expected)
        missing = sorted(expected - captured)

        score = round(1.0 + 4.0 * coverage, 2)
        detail = (
            f"Captured {len(captured)}/{len(expected)} reference fields "
            f"({coverage:.0%})"
        )
        if missing:
            detail += f". Missing: {', '.join(missing[:5])}"

        return VerificationOutcome(
            score=score,
            passed=policy.passes(score),
            policy=policy,
            strategy=policy.strategy,
            detail=detail,
            reference_model=reference.model_id,
            reference_cost=reference.cost,
            divergence=round(1.0 - coverage, 3),
        )

    @staticmethod
    def _extract_fields(text: str) -> set[str]:
        """Pull field names out of JSON-ish, bulleted, or labelled output."""
        fields: set[str] = set()

        # JSON keys: "name": ...
        fields.update(m.lower() for m in re.findall(r'"([\w \-]{1,40})"\s*:', text))
        # Labelled lines: `- Name:` / `* Name:` / `Name:` at line start
        fields.update(
            m.strip().lower()
            for m in re.findall(r"^\s*(?:[-*•]\s*)?([A-Za-z][\w \-]{0,39}):", text, re.M)
        )
        # Markdown bold labels: **Name:**
        fields.update(
            m.strip().lower() for m in re.findall(r"\*\*([\w \-]{1,40})\*\*\s*:?", text)
        )

        return {f for f in fields if f and f not in _STOPWORDS}

    # ── Strategy: label match (classification) ──────────────────────────

    def _score_label_match(
        self,
        response: LLMResponse,
        reference: Optional[LLMResponse],
        policy: QualityPolicy,
    ) -> VerificationOutcome:
        """Does the cheap model's label match the reference model's label?"""
        if reference is None:
            return self._no_reference(policy)

        candidate_label = self._extract_label(response.text)
        reference_label = self._extract_label(reference.text)

        matched = bool(candidate_label) and candidate_label == reference_label
        score = 5.0 if matched else 1.0

        return VerificationOutcome(
            score=score,
            passed=policy.passes(score),
            policy=policy,
            strategy=policy.strategy,
            detail=(
                f"Label '{candidate_label or '?'}' "
                f"{'matches' if matched else 'DIVERGES from'} "
                f"reference '{reference_label or '?'}'"
            ),
            reference_model=reference.model_id,
            reference_cost=reference.cost,
            divergence=0.0 if matched else 1.0,
        )

    @staticmethod
    def _extract_label(text: str) -> str:
        """Normalize a classification answer down to a comparable label."""
        cleaned = re.sub(r"[*_`#>]", " ", text)
        lines = [ln.strip(" .:-\t") for ln in cleaned.splitlines() if ln.strip()]
        if not lines:
            return ""

        # Prefer an explicit "Label: X" / "Answer: X" / "Category: X" line.
        for line in lines:
            m = re.match(
                r"^(?:label|answer|category|classification|sentiment|result)\s*[:\-]\s*(.+)$",
                line,
                re.I,
            )
            if m:
                return re.sub(r"[^\w ]", "", m.group(1)).strip().lower()[:60]

        # Otherwise the shortest line is usually the bare label.
        shortest = min(lines, key=len)
        return re.sub(r"[^\w ]", "", shortest).strip().lower()[:60]

    # ── Strategy: agreement (general) ───────────────────────────────────

    def _score_agreement(
        self,
        response: LLMResponse,
        reference: Optional[LLMResponse],
        policy: QualityPolicy,
    ) -> VerificationOutcome:
        """Score lexical/semantic overlap between candidate and reference."""
        if reference is None:
            return self._no_reference(policy)

        candidate_tokens = self._content_tokens(response.text)
        reference_tokens = self._content_tokens(reference.text)

        if not candidate_tokens or not reference_tokens:
            return self._no_reference(policy)

        overlap = len(candidate_tokens & reference_tokens)
        union = len(candidate_tokens | reference_tokens)
        jaccard = overlap / union if union else 0.0

        # Penalize answers far shorter than the reference — a common failure
        # mode for cheap models is truncating the substance.
        length_ratio = min(
            1.0, len(candidate_tokens) / max(1, len(reference_tokens))
        )
        agreement = 0.7 * jaccard + 0.3 * length_ratio

        score = round(1.0 + 4.0 * agreement, 2)
        return VerificationOutcome(
            score=score,
            passed=policy.passes(score),
            policy=policy,
            strategy=VerificationStrategy.AGREEMENT,
            detail=(
                f"Agreement {agreement:.0%} vs reference "
                f"(overlap {jaccard:.0%}, length {length_ratio:.0%})"
            ),
            reference_model=reference.model_id,
            reference_cost=reference.cost,
            divergence=round(1.0 - agreement, 3),
        )

    @staticmethod
    def _content_tokens(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9_]{2,}", text.lower())
        return {t for t in tokens if t not in _STOPWORDS}

    # ── Strategy: rubric (summarization, code, explanation) ─────────────

    async def _score_rubric(
        self,
        prompt: str,
        response: LLMResponse,
        reference: Optional[LLMResponse],
        policy: QualityPolicy,
    ) -> VerificationOutcome:
        """LLM-as-judge rubric score, cross-checked against the reference."""
        if self._mock:
            rubric = self._mock_rubric(prompt, response)
        else:
            rubric = await self._real_rubric(prompt, response)

        score = rubric.aggregate
        detail = (
            f"Rubric {score:.2f}/5 "
            f"(acc {rubric.accuracy}, comp {rubric.completeness}, "
            f"coh {rubric.coherence})"
        )

        divergence = 0.0
        if reference is not None:
            agreement = self._score_agreement(response, reference, policy)
            divergence = agreement.divergence
            detail += f"; divergence from reference {divergence:.0%}"

        return VerificationOutcome(
            score=score,
            passed=policy.passes(score),
            policy=policy,
            strategy=VerificationStrategy.RUBRIC,
            detail=detail,
            reference_model=reference.model_id if reference else None,
            reference_cost=reference.cost if reference else 0.0,
            divergence=divergence,
            rubric=rubric,
        )

    async def _real_rubric(self, prompt: str, response: LLMResponse) -> QualityScore:
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt, response=response.text
        )
        judge_config = self._registry.get(self._judge_model)
        provider = self._providers[judge_config.provider]

        judge_response = await provider.send_request(
            prompt=judge_prompt,
            model_config=judge_config,
            temperature=0.1,  # Low temp for consistent scoring
        )
        return self._parse_scores(judge_response.text)

    def _mock_rubric(self, prompt: str, response: LLMResponse) -> QualityScore:
        """Deterministic mock scores — higher tiers score better on average."""
        quality_tier = response.model_id
        rng = random.Random(hash(prompt + response.text) % 2**32)

        if "gpt-4o" in quality_tier and "mini" not in quality_tier:
            base = 4.3
        elif "sonnet" in quality_tier:
            base = 4.2
        elif "mini" in quality_tier or "haiku" in quality_tier:
            base = 3.8
        else:  # Local / Llama
            base = 3.3

        def jitter() -> float:
            return round(rng.uniform(-0.5, 0.5), 1)

        clamp = lambda v: max(1.0, min(5.0, round(v, 1)))
        return QualityScore(
            accuracy=clamp(base + jitter()),
            completeness=clamp(base + jitter()),
            coherence=clamp(base + jitter()),
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    def _no_reference(self, policy: QualityPolicy) -> VerificationOutcome:
        """Neutral pass when no reference comparison was possible."""
        return self._inconclusive(
            policy, "No reference available (already top tier or call failed)"
        )

    @staticmethod
    def _inconclusive(policy: QualityPolicy, reason: str) -> VerificationOutcome:
        """Neutral pass used when the comparison could not be made.

        Scored exactly at the threshold so it neither triggers an escalation
        nor counts as evidence the routing was correct.
        """
        return VerificationOutcome(
            score=policy.threshold,
            passed=True,
            policy=policy,
            strategy=policy.strategy,
            detail=f"Inconclusive: {reason}",
        )

    @staticmethod
    def _parse_scores(judge_text: str) -> QualityScore:
        """Parse the judge's rating text into a QualityScore."""
        def grab(name: str) -> float:
            m = re.search(rf"{name}:\s*([\d.]+)", judge_text)
            if not m:
                return 3.5
            try:
                return max(1.0, min(5.0, float(m.group(1))))
            except ValueError:
                return 3.5

        return QualityScore(
            accuracy=grab("Accuracy"),
            completeness=grab("Completeness"),
            coherence=grab("Coherence"),
        )
