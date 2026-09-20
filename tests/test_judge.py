"""Phase 3.2 — reference comparison and per-strategy scoring."""

from __future__ import annotations

import pytest

from src.models.response import LLMResponse
from src.verifier.judge import QualityJudge
from src.verifier.use_cases import PolicyBook, UseCase

from tests.conftest import ScriptedProvider


def _response(text: str, model_id: str = "gpt-4o-mini") -> LLMResponse:
    return LLMResponse(
        text=text,
        model_id=model_id,
        provider="openai",
        input_tokens=100,
        output_tokens=50,
        cost=0.0001,
        latency_ms=400.0,
    )


@pytest.fixture
def book() -> PolicyBook:
    return PolicyBook()


def _judge(registry, reference_text: str) -> QualityJudge:
    """A judge whose reference model (claude-sonnet-4-6) returns known text."""
    providers = {
        "openai": ScriptedProvider("openai", {}),
        "anthropic": ScriptedProvider("anthropic", {"claude-sonnet-4-6": reference_text}),
        "ollama": ScriptedProvider("ollama", {}),
    }
    return QualityJudge(registry, providers)


# ── Reference fetching ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sends_prompt_to_highest_tier_model(registry, book):
    judge = _judge(registry, "reference answer")
    outcome = await judge.verify(
        "Explain decorators", _response("some answer"), book.for_use_case(UseCase.GENERAL)
    )
    assert outcome.reference_model == registry.get_highest_quality().model_id


@pytest.mark.asyncio
async def test_top_tier_response_has_no_reference(registry, book):
    """A response already from the reference model is not compared to itself."""
    judge = _judge(registry, "reference answer")
    top = registry.get_highest_quality()
    outcome = await judge.verify(
        "anything", _response("x", model_id=top.model_id),
        book.for_use_case(UseCase.GENERAL),
    )
    assert outcome.reference_model is None
    assert outcome.passed
    assert "Inconclusive" in outcome.detail


# ── Field coverage (extraction) ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_field_coverage_passes(registry, book):
    reference = '{"name": "Ada", "email": "a@b.c", "phone": "555"}'
    judge = _judge(registry, reference)
    outcome = await judge.verify(
        "Extract name, email, phone as JSON",
        _response('{"name": "Ada", "email": "a@b.c", "phone": "555"}'),
        book.for_use_case(UseCase.EXTRACTION),
    )
    assert outcome.score == 5.0
    assert outcome.passed
    assert outcome.divergence == 0.0


@pytest.mark.asyncio
async def test_missing_field_fails_extraction(registry, book):
    reference = '{"name": "Ada", "email": "a@b.c", "phone": "555"}'
    judge = _judge(registry, reference)
    outcome = await judge.verify(
        "Extract name, email, phone as JSON",
        _response('{"name": "Ada"}'),  # dropped two of three fields
        book.for_use_case(UseCase.EXTRACTION),
    )
    assert not outcome.passed
    assert outcome.score < 4.5
    assert "phone" in outcome.detail or "email" in outcome.detail


@pytest.mark.asyncio
async def test_field_coverage_tolerates_format_differences(registry, book):
    """Same fields in prose rather than JSON still count as captured."""
    judge = _judge(registry, '{"name": "Ada", "email": "a@b.c"}')
    outcome = await judge.verify(
        "Extract the fields",
        _response("name: Ada\nemail: a@b.c"),
        book.for_use_case(UseCase.EXTRACTION),
    )
    assert outcome.passed


@pytest.mark.asyncio
async def test_unparseable_reference_is_inconclusive_not_a_failure(registry, book):
    """We must not fail a response for our own inability to measure it."""
    judge = _judge(registry, "just some prose with no fields at all")
    outcome = await judge.verify(
        "Extract the fields",
        _response("also prose"),
        book.for_use_case(UseCase.EXTRACTION),
    )
    assert outcome.passed
    assert "Inconclusive" in outcome.detail


# ── Label match (classification) ────────────────────────────────────────

@pytest.mark.asyncio
async def test_matching_label_passes(registry, book):
    judge = _judge(registry, "positive")
    outcome = await judge.verify(
        "Classify the sentiment", _response("positive"),
        book.for_use_case(UseCase.CLASSIFICATION),
    )
    assert outcome.score == 5.0
    assert outcome.passed


@pytest.mark.asyncio
async def test_diverging_label_fails(registry, book):
    judge = _judge(registry, "negative")
    outcome = await judge.verify(
        "Classify the sentiment", _response("positive"),
        book.for_use_case(UseCase.CLASSIFICATION),
    )
    assert outcome.score == 1.0
    assert not outcome.passed
    assert outcome.divergence == 1.0
    assert outcome.quality_gap == 4.0


@pytest.mark.asyncio
async def test_label_extracted_from_labelled_line(registry, book):
    judge = _judge(registry, "Label: positive")
    outcome = await judge.verify(
        "Classify it",
        _response("After careful analysis:\nAnswer: positive"),
        book.for_use_case(UseCase.CLASSIFICATION),
    )
    assert outcome.passed


# ── Agreement (general) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_identical_text_agrees(registry, book):
    text = "The quick brown fox jumps over the lazy dog repeatedly"
    judge = _judge(registry, text)
    outcome = await judge.verify(
        "anything", _response(text), book.for_use_case(UseCase.GENERAL)
    )
    assert outcome.score == 5.0
    assert outcome.divergence == 0.0


@pytest.mark.asyncio
async def test_divergent_text_fails(registry, book):
    judge = _judge(registry, "Kubernetes orchestrates containerized workloads at scale")
    outcome = await judge.verify(
        "anything",
        _response("Bananas are yellow"),
        book.for_use_case(UseCase.GENERAL),
    )
    assert not outcome.passed
    assert outcome.divergence > 0.5


@pytest.mark.asyncio
async def test_truncated_answer_is_penalized(registry, book):
    """A common cheap-model failure: right topic, missing substance."""
    reference = " ".join(f"detail{i} matters here significantly" for i in range(20))
    judge = _judge(registry, reference)
    outcome = await judge.verify(
        "anything", _response("detail1 matters"), book.for_use_case(UseCase.GENERAL)
    )
    assert not outcome.passed


# ── Rubric parsing ──────────────────────────────────────────────────────

def test_parses_judge_scores():
    score = QualityJudge._parse_scores(
        "Accuracy: 4.5\nCompleteness: 3.0\nCoherence: 4.0"
    )
    assert (score.accuracy, score.completeness, score.coherence) == (4.5, 3.0, 4.0)
    assert score.aggregate == 3.83


def test_unparseable_judge_output_is_neutral():
    score = QualityJudge._parse_scores("I refuse to answer")
    assert score.aggregate == 3.5


def test_out_of_range_scores_are_clamped():
    score = QualityJudge._parse_scores(
        "Accuracy: 9.9\nCompleteness: 0.1\nCoherence: 4.0"
    )
    assert score.accuracy == 5.0
    assert score.completeness == 1.0
