"""Guards against the training corpus re-encoding tier in prompt length.

The original 180-prompt set had tier 1 capped at 63 characters and tier 3
starting at 191, so a single length threshold scored 93% and the classifier
never learned anything else — while reporting 100% accuracy against a test
split drawn from the same skew. These tests fail if that returns.
"""

from __future__ import annotations

import json
import statistics as stats
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src.classifier.features import FeatureExtractor
from src.classifier.model import ComplexityClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "training_prompts.json"
EVAL_PATH = PROJECT_ROOT / "data" / "eval_length_adversarial.json"

CHANCE = 1 / 3


@pytest.fixture(scope="module")
def corpus() -> list[dict]:
    return json.loads(TRAIN_PATH.read_text())


@pytest.fixture(scope="module")
def adversarial() -> list[dict]:
    return json.loads(EVAL_PATH.read_text())


@pytest.fixture(scope="module")
def trained(corpus) -> ComplexityClassifier:
    classifier = ComplexityClassifier()
    classifier.train(
        [d["prompt"] for d in corpus], [d["tier"] for d in corpus]
    )
    return classifier


# ── Corpus shape ────────────────────────────────────────────────────────

def test_every_tier_is_represented(corpus):
    tiers = {d["tier"] for d in corpus}
    assert tiers == {1, 2, 3}


def test_corpus_is_reasonably_balanced(corpus):
    counts = [sum(1 for d in corpus if d["tier"] == t) for t in (1, 2, 3)]
    assert min(counts) / max(counts) > 0.7, f"tier counts too skewed: {counts}"


def test_no_duplicate_prompts(corpus):
    prompts = [d["prompt"] for d in corpus]
    assert len(prompts) == len(set(prompts))


# ── The leak itself ─────────────────────────────────────────────────────

def test_tier_length_ranges_overlap(corpus):
    """Tier 1 and tier 3 must share a length range, or length gives it away."""
    spans = {}
    for tier in (1, 2, 3):
        lengths = [len(d["prompt"]) for d in corpus if d["tier"] == tier]
        spans[tier] = (min(lengths), max(lengths))

    for a, b in ((1, 2), (2, 3), (1, 3)):
        lo = max(spans[a][0], spans[b][0])
        hi = min(spans[a][1], spans[b][1])
        assert lo < hi, f"tier {a} and {b} lengths do not overlap: {spans}"


def test_median_lengths_are_close(corpus):
    """No tier should be dramatically longer than another on average."""
    medians = [
        stats.median([len(d["prompt"]) for d in corpus if d["tier"] == t])
        for t in (1, 2, 3)
    ]
    assert max(medians) / min(medians) < 2.0, f"median lengths diverge: {medians}"


def test_length_alone_cannot_predict_tier(corpus):
    """The regression test for the original defect.

    On the old corpus this scored 93%. Anything above ~50% means complexity
    is encoded in prompt length again.
    """
    X = np.array([[len(d["prompt"])] for d in corpus])
    y = np.array([d["tier"] for d in corpus])
    score = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=0), X, y, cv=5
    ).mean()

    assert score < 0.50, (
        f"length alone predicts tier at {score:.1%} — the corpus has "
        f"re-acquired a length leak"
    )


def test_word_count_alone_cannot_predict_tier(corpus):
    X = np.array([[len(d["prompt"].split())] for d in corpus])
    y = np.array([d["tier"] for d in corpus])
    score = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=0), X, y, cv=5
    ).mean()
    assert score < 0.50, f"word count alone predicts tier at {score:.1%}"


# ── Model health ────────────────────────────────────────────────────────

def test_model_beats_chance_by_a_clear_margin(corpus):
    X = FeatureExtractor().extract_batch([d["prompt"] for d in corpus])
    y = np.array([d["tier"] for d in corpus])
    score = cross_val_score(
        RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        X, y, cv=5,
    ).mean()
    assert score > 0.60, f"CV accuracy {score:.1%} is too close to chance"


def test_length_features_do_not_dominate(corpus):
    """Length may inform the decision; it must not make it."""
    classifier = ComplexityClassifier()
    metrics = classifier.train(
        [d["prompt"] for d in corpus], [d["tier"] for d in corpus]
    )
    importances = metrics["feature_importances"]
    length_weight = sum(
        importances[name]
        for name in ("char_count", "token_count", "sentence_count")
    )
    assert length_weight < 0.50, (
        f"length proxies are {length_weight:.1%} of the decision"
    )


# ── Adversarial behaviour ───────────────────────────────────────────────

def test_short_complex_prompts_are_not_called_simple(trained, adversarial):
    """'Implement Paxos.' is 16 characters and genuinely tier 3."""
    short_complex = [d for d in adversarial if d["tier"] == 3]
    correct = sum(
        1 for d in short_complex if int(trained.predict(d["prompt"])) == 3
    )
    assert correct / len(short_complex) >= 0.7, (
        f"only {correct}/{len(short_complex)} short complex prompts "
        f"classified correctly"
    )


def test_long_simple_prompts_are_not_called_complex(trained, adversarial):
    """A rambling preamble around 'What is a variable?' is still tier 1."""
    long_simple = [d for d in adversarial if d["tier"] == 1]
    not_complex = sum(
        1 for d in long_simple if int(trained.predict(d["prompt"])) < 3
    )
    assert not_complex == len(long_simple), (
        f"{len(long_simple) - not_complex} long simple prompts were "
        f"classified complex"
    )


def test_overall_adversarial_accuracy(trained, adversarial):
    correct = sum(
        1 for d in adversarial if int(trained.predict(d["prompt"])) == d["tier"]
    )
    assert correct / len(adversarial) >= 0.70
