"""Phase 3.4 — failures become training data, retrained on a weekly cadence."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from src.classifier.model import ComplexityClassifier
from src.classifier.tiers import ComplexityTier
from src.verifier.feedback import FeedbackLoop


@pytest.fixture
def loop(tmp_path) -> FeedbackLoop:
    """All state under tmp_path — a retrain must never touch the real model."""
    return FeedbackLoop(
        classifier=ComplexityClassifier(),
        feedback_path=str(tmp_path / "feedback.json"),
        state_path=str(tmp_path / "state.json"),
        model_path=str(tmp_path / "model.pkl"),
        min_examples_to_retrain=2,
    )


# ── Tier inference ──────────────────────────────────────────────────────

def test_small_gap_moves_up_one_tier():
    assert FeedbackLoop.infer_correct_tier(1, quality_gap=0.4) == 2


def test_severe_gap_jumps_two_tiers():
    assert FeedbackLoop.infer_correct_tier(1, quality_gap=2.5) == 3


def test_inference_is_capped_at_the_top_tier():
    assert FeedbackLoop.infer_correct_tier(2, quality_gap=4.0) == int(
        ComplexityTier.COMPLEX
    )
    assert FeedbackLoop.infer_correct_tier(3, quality_gap=4.0) == 3


# ── Recording ───────────────────────────────────────────────────────────

def test_records_failure_with_inferred_tier(loop):
    entry = loop.record_failure(
        prompt="Summarize this", original_tier=1,
        quality_score=2.0, quality_gap=0.5, use_case="summarization",
    )
    assert entry is not None
    assert entry["tier"] == 2
    assert entry["original_tier"] == 1
    assert entry["use_case"] == "summarization"
    assert loop.feedback_count == 1


def test_caller_can_override_inferred_tier(loop):
    entry = loop.record_failure(
        prompt="p", original_tier=1, quality_score=2.0,
        quality_gap=0.1, correct_tier=3,
    )
    assert entry["tier"] == 3


def test_duplicate_prompts_are_not_recorded_twice(loop):
    for _ in range(3):
        loop.record_failure("same prompt", 1, 2.0, 0.5)
    assert loop.feedback_count == 1


def test_top_tier_failure_yields_no_training_example(loop):
    """Nothing to learn when there is no higher tier to route to."""
    assert loop.record_failure("p", original_tier=3, quality_score=2.0,
                               quality_gap=1.0) is None
    assert loop.feedback_count == 0


def test_failures_persist_across_instances(loop, tmp_path):
    loop.record_failure("persisted prompt", 1, 2.0, 0.5)

    reloaded = FeedbackLoop(
        classifier=ComplexityClassifier(),
        feedback_path=str(tmp_path / "feedback.json"),
        state_path=str(tmp_path / "state.json"),
    )
    assert reloaded.feedback_count == 1
    # Dedup survives the reload too.
    assert reloaded.record_failure("persisted prompt", 1, 2.0, 0.5) is None


def test_corrupt_feedback_file_does_not_crash(tmp_path):
    path = tmp_path / "feedback.json"
    path.write_text("{ not valid json")

    loop = FeedbackLoop(
        classifier=ComplexityClassifier(),
        feedback_path=str(path),
        state_path=str(tmp_path / "state.json"),
    )
    assert loop.feedback_count == 0


# ── Weekly cadence ──────────────────────────────────────────────────────

def test_not_due_below_the_example_threshold(loop):
    loop.record_failure("one", 1, 2.0, 0.5)
    due, reason = loop.should_retrain()
    assert not due
    assert "need 2" in reason


def test_first_retrain_is_due_once_enough_examples(loop):
    loop.record_failure("one", 1, 2.0, 0.5)
    loop.record_failure("two", 1, 2.0, 0.5)
    due, _ = loop.should_retrain()
    assert due


def test_not_due_again_within_the_week(loop):
    loop.record_failure("one", 1, 2.0, 0.5)
    loop.record_failure("two", 1, 2.0, 0.5)
    loop.retrain_classifier()

    loop.record_failure("three", 1, 2.0, 0.5)
    loop.record_failure("four", 1, 2.0, 0.5)
    due, reason = loop.should_retrain()
    assert not due
    assert "Next retrain due" in reason


def test_due_again_after_a_week(loop):
    loop.record_failure("one", 1, 2.0, 0.5)
    loop.record_failure("two", 1, 2.0, 0.5)
    loop.retrain_classifier()

    loop.record_failure("three", 1, 2.0, 0.5)
    loop.record_failure("four", 1, 2.0, 0.5)

    next_week = datetime.now(timezone.utc) + timedelta(days=8)
    due, _ = loop.should_retrain(now=next_week)
    assert due


# ── Retraining ──────────────────────────────────────────────────────────

def test_retrain_merges_feedback_into_the_training_set(loop):
    import json
    from pathlib import Path

    corpus = json.loads(
        (Path(__file__).resolve().parent.parent / "data" / "training_prompts.json")
        .read_text()
    )

    loop.record_failure("a novel failing prompt", 1, 2.0, 0.5)
    loop.record_failure("another failing prompt", 1, 2.0, 0.5)

    metrics = loop.retrain_classifier()
    assert metrics["feedback_examples_used"] == 2
    assert metrics["train_size"] + metrics["test_size"] == len(corpus) + 2
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_retrain_produces_a_usable_classifier(loop, tmp_path):
    loop.record_failure("a", 1, 2.0, 0.5)
    loop.record_failure("b", 1, 2.0, 0.5)
    loop.retrain_classifier()

    classifier = ComplexityClassifier()
    classifier.load(str(tmp_path / "model.pkl"))
    assert classifier.predict("What is a variable?") in list(ComplexityTier)


def test_maybe_retrain_respects_the_cadence(loop):
    loop.record_failure("one", 1, 2.0, 0.5)
    assert loop.maybe_retrain() is None  # below threshold

    loop.record_failure("two", 1, 2.0, 0.5)
    assert loop.maybe_retrain() is not None  # now due


def test_state_survives_restart(loop, tmp_path):
    loop.record_failure("one", 1, 2.0, 0.5)
    loop.record_failure("two", 1, 2.0, 0.5)
    loop.retrain_classifier()

    reloaded = FeedbackLoop(
        classifier=ComplexityClassifier(),
        feedback_path=str(tmp_path / "feedback.json"),
        state_path=str(tmp_path / "state.json"),
        model_path=str(tmp_path / "model.pkl"),
        min_examples_to_retrain=2,
    )
    assert reloaded.last_retrained_at is not None
    due, reason = reloaded.should_retrain()
    assert not due


def test_summary_reports_the_flywheel_state(loop):
    loop.record_failure("one", 1, 2.0, 0.5, use_case="extraction")
    summary = loop.summary()
    assert summary["feedback_examples"] == 1
    assert summary["by_use_case"] == {"extraction": 1}
    assert "retrain_due" in summary
