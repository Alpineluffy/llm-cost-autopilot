"""Per-use-case quality policies.

Phase 3.1 — "good enough" is not one number. An extraction task is graded on
whether it captured every key field; a summarization on an LLM-judge score;
a classification on whether the label matches what the high-tier model said.

This module answers two questions for any prompt:
    1. What kind of request is this?      → UseCaseDetector.detect()
    2. What does "good enough" mean here? → QualityPolicy
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class UseCase(str, Enum):
    """Request types that are graded differently."""

    EXTRACTION = "extraction"
    SUMMARIZATION = "summarization"
    CLASSIFICATION = "classification"
    CODE_GENERATION = "code_generation"
    EXPLANATION = "explanation"
    GENERAL = "general"

    @property
    def label(self) -> str:
        return self.value.replace("_", " ")


class VerificationStrategy(str, Enum):
    """How a response is compared against the high-tier reference."""

    FIELD_COVERAGE = "field_coverage"   # Did it capture every key field?
    LABEL_MATCH = "label_match"         # Does the label match the reference?
    RUBRIC = "rubric"                   # LLM-as-judge 1-5 rubric score
    AGREEMENT = "agreement"             # Semantic overlap with the reference


@dataclass(frozen=True)
class QualityPolicy:
    """What "good enough" means for one use case.

    Attributes:
        use_case: The request type this policy governs.
        threshold: Minimum passing score, on a 1.0-5.0 scale.
        strategy: How the score is computed.
        criterion: Human-readable statement of the bar, for logs and audit.
    """

    use_case: UseCase
    threshold: float
    strategy: VerificationStrategy
    criterion: str

    def passes(self, score: float) -> bool:
        return score >= self.threshold

    def to_dict(self) -> dict:
        return {
            "use_case": self.use_case.value,
            "threshold": self.threshold,
            "strategy": self.strategy.value,
            "criterion": self.criterion,
        }


# ── Default policies (overridable from config/routing.yaml) ─────────────

DEFAULT_POLICIES: dict[UseCase, QualityPolicy] = {
    UseCase.EXTRACTION: QualityPolicy(
        use_case=UseCase.EXTRACTION,
        threshold=4.5,
        strategy=VerificationStrategy.FIELD_COVERAGE,
        criterion="Must capture >=90% of the key fields the reference model found",
    ),
    UseCase.SUMMARIZATION: QualityPolicy(
        use_case=UseCase.SUMMARIZATION,
        threshold=4.0,
        strategy=VerificationStrategy.RUBRIC,
        criterion="LLM-as-judge aggregate score above 4.0/5",
    ),
    UseCase.CLASSIFICATION: QualityPolicy(
        use_case=UseCase.CLASSIFICATION,
        threshold=5.0,
        strategy=VerificationStrategy.LABEL_MATCH,
        criterion="Label must match the reference model's label exactly",
    ),
    UseCase.CODE_GENERATION: QualityPolicy(
        use_case=UseCase.CODE_GENERATION,
        threshold=3.8,
        strategy=VerificationStrategy.RUBRIC,
        criterion="Judge rubric above 3.8/5 on accuracy, completeness, coherence",
    ),
    UseCase.EXPLANATION: QualityPolicy(
        use_case=UseCase.EXPLANATION,
        threshold=3.5,
        strategy=VerificationStrategy.RUBRIC,
        criterion="Judge rubric above 3.5/5",
    ),
    UseCase.GENERAL: QualityPolicy(
        use_case=UseCase.GENERAL,
        threshold=3.5,
        strategy=VerificationStrategy.AGREEMENT,
        criterion="Must not diverge significantly from the reference response",
    ),
}


# ── Detection keywords ──────────────────────────────────────────────────

_EXTRACTION_PATTERNS = [
    r"\bextract\b", r"\bpull out\b", r"\bparse\b", r"\bfind all\b",
    r"\blist all\b", r"\bidentify all\b", r"\bget the (?:fields|values|names)\b",
    r"\binto json\b", r"\bas json\b", r"\bkey-value\b",
]

_SUMMARIZATION_PATTERNS = [
    r"\bsummari[sz]e\b", r"\bsummary\b", r"\btl;?dr\b", r"\bcondense\b",
    r"\bin (?:a|one) (?:few |couple )?(?:sentence|paragraph|line)",
    r"\bkey (?:points|takeaways)\b", r"\bbrief(?:ly)? (?:describe|explain)\b",
]

_CLASSIFICATION_PATTERNS = [
    r"\bclassify\b", r"\bcategori[sz]e\b", r"\blabel\b", r"\btag\b",
    r"\bwhich category\b", r"\bis this (?:a|an)\b", r"\bsentiment\b",
    r"\byes or no\b", r"\btrue or false\b", r"\bchoose one\b",
]

_CODE_PATTERNS = [
    # Allow modifiers between the verb and the noun: "write a Python function"
    r"\b(?:write|create|build|generate|add)\s+(?:\w+\s+){0,3}?"
    r"(?:function|class|script|program|method|module|endpoint|query|test)\b",
    r"\bimplement\b", r"\brefactor\b", r"\bdebug\b",
    r"\bfix (?:this|the|my)?\s*(?:bug|error|code|function)\b",
    r"\boptimi[sz]e (?:this|the|my)\b",
    r"\bwrite (?:some |the )?code\b", r"\bcode that\b",
]

_EXPLANATION_PATTERNS = [
    r"\bexplain\b", r"\bwhat is\b", r"\bhow does\b", r"\bwhy (?:is|does|do)\b",
    r"\bdescribe\b", r"\bwalk me through\b", r"\bdifference between\b",
    # The three most common question forms in real traffic, all of which were
    # falling through to GENERAL and skipping the explanation policy.
    r"\bwhat does\b", r"\bwhat are\b",
    r"\bhow do i\b", r"\bhow can i\b", r"\bhow to\b",
    r"\bcompare\b", r"\bversus\b", r"\bpros and cons\b",
    r"\bwhen (?:should i |to )use\b",
]


class UseCaseDetector:
    """Infers the use case of a prompt with ordered keyword heuristics.

    Order matters: a prompt asking to "extract and summarize" is graded as
    extraction, because the stricter field-coverage bar is the safer default.
    """

    _ORDERED_RULES: list[tuple[UseCase, list[str]]] = [
        (UseCase.EXTRACTION, _EXTRACTION_PATTERNS),
        (UseCase.CLASSIFICATION, _CLASSIFICATION_PATTERNS),
        (UseCase.SUMMARIZATION, _SUMMARIZATION_PATTERNS),
        (UseCase.CODE_GENERATION, _CODE_PATTERNS),
        (UseCase.EXPLANATION, _EXPLANATION_PATTERNS),
    ]

    def detect(self, prompt: str) -> UseCase:
        """Classify a prompt into a use case. Falls back to GENERAL."""
        text = prompt.lower()
        for use_case, patterns in self._ORDERED_RULES:
            if any(re.search(p, text) for p in patterns):
                return use_case
        return UseCase.GENERAL


class PolicyBook:
    """Holds the active quality policy for every use case.

    Built from DEFAULT_POLICIES, then overlaid with any `quality_thresholds`
    block in routing.yaml so the bar can be tuned without code changes.
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._policies = dict(DEFAULT_POLICIES)
        self._detector = UseCaseDetector()
        if config:
            self._apply_overrides(config)

    def _apply_overrides(self, config: dict) -> None:
        overrides = config.get("quality_thresholds", {}) or {}
        for key, values in overrides.items():
            try:
                use_case = UseCase(key)
            except ValueError:
                continue

            base = self._policies[use_case]
            strategy = base.strategy
            if "strategy" in values:
                try:
                    strategy = VerificationStrategy(values["strategy"])
                except ValueError:
                    strategy = base.strategy

            self._policies[use_case] = QualityPolicy(
                use_case=use_case,
                threshold=float(values.get("threshold", base.threshold)),
                strategy=strategy,
                criterion=values.get("criterion", base.criterion),
            )

    def for_prompt(self, prompt: str) -> QualityPolicy:
        """Detect the use case of a prompt and return its policy."""
        return self._policies[self._detector.detect(prompt)]

    def for_use_case(self, use_case: UseCase) -> QualityPolicy:
        return self._policies[use_case]

    def all_policies(self) -> dict[str, dict]:
        return {uc.value: p.to_dict() for uc, p in self._policies.items()}
