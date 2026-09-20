"""Phase 3.1 — per-use-case quality policies."""

from __future__ import annotations

import pytest

from src.verifier.use_cases import (
    PolicyBook,
    UseCase,
    UseCaseDetector,
    VerificationStrategy,
)


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("Extract the name and email from this text as JSON", UseCase.EXTRACTION),
        ("Pull out all the function names", UseCase.EXTRACTION),
        ("Classify the sentiment of this review", UseCase.CLASSIFICATION),
        ("Is this a bug report? Yes or no", UseCase.CLASSIFICATION),
        ("Summarize this module in two sentences", UseCase.SUMMARIZATION),
        ("Give me a TLDR of this changelog", UseCase.SUMMARIZATION),
        ("Write a Python function that merges two sorted lists", UseCase.CODE_GENERATION),
        ("Implement a binary search tree", UseCase.CODE_GENERATION),
        ("Refactor this loop", UseCase.CODE_GENERATION),
        ("Explain what a decorator is", UseCase.EXPLANATION),
        ("How does garbage collection work?", UseCase.EXPLANATION),
        # Common question forms that used to fall through to GENERAL
        ("What does the range function return?", UseCase.EXPLANATION),
        ("How do I concatenate two lists?", UseCase.EXPLANATION),
        ("How can I read a file?", UseCase.EXPLANATION),
        ("Compare threading and multiprocessing for I/O work.", UseCase.EXPLANATION),
        ("What are the pros and cons of an ORM?", UseCase.EXPLANATION),
        ("Give me three ideas for a team offsite", UseCase.GENERAL),
    ],
)
def test_detects_use_case(prompt, expected):
    assert UseCaseDetector().detect(prompt) == expected


def test_common_question_forms_are_not_left_as_general():
    """A detector that sends most traffic to GENERAL makes the per-use-case
    policies decorative. These forms dominate real prompts."""
    detector = UseCaseDetector()
    forms = [
        "What does append() do?",
        "How do I install a package?",
        "How to reverse a string",
        "Compare REST and GraphQL",
        "When should I use a set?",
    ]
    assert all(detector.detect(p) is not UseCase.GENERAL for p in forms)


def test_extraction_wins_over_summarization():
    """The stricter bar is the safer default when a prompt asks for both."""
    detected = UseCaseDetector().detect("Extract the key fields and summarize them")
    assert detected is UseCase.EXTRACTION


def test_each_use_case_has_a_distinct_bar():
    book = PolicyBook()
    assert book.for_use_case(UseCase.CLASSIFICATION).threshold == 5.0
    assert book.for_use_case(UseCase.EXTRACTION).threshold == 4.5
    assert book.for_use_case(UseCase.SUMMARIZATION).threshold == 4.0
    assert book.for_use_case(UseCase.EXPLANATION).threshold == 3.5


def test_strategy_matches_use_case():
    book = PolicyBook()
    assert (
        book.for_use_case(UseCase.EXTRACTION).strategy
        is VerificationStrategy.FIELD_COVERAGE
    )
    assert (
        book.for_use_case(UseCase.CLASSIFICATION).strategy
        is VerificationStrategy.LABEL_MATCH
    )
    assert (
        book.for_use_case(UseCase.SUMMARIZATION).strategy
        is VerificationStrategy.RUBRIC
    )


def test_config_overrides_threshold():
    book = PolicyBook({"quality_thresholds": {"summarization": {"threshold": 4.8}}})
    assert book.for_use_case(UseCase.SUMMARIZATION).threshold == 4.8
    # Untouched policies keep their defaults.
    assert book.for_use_case(UseCase.EXTRACTION).threshold == 4.5


def test_config_override_ignores_unknown_use_case():
    book = PolicyBook({"quality_thresholds": {"nonsense": {"threshold": 1.0}}})
    assert book.for_use_case(UseCase.GENERAL).threshold == 3.5


def test_for_prompt_routes_to_the_right_policy():
    policy = PolicyBook().for_prompt("Classify this ticket by priority")
    assert policy.use_case is UseCase.CLASSIFICATION
    assert policy.passes(5.0)
    assert not policy.passes(4.9)
