#!/usr/bin/env python3
"""Which of this project's numbers are real measurements, and which are not.

Every metric here falls into one of three buckets:

  MEASURED    a statistic with sampling error we can quantify
  ARITHMETIC  a deterministic calculation — no sampling error, but entirely
              dependent on inputs that were chosen rather than observed
  SIMULATED   a number produced by mock machinery; it measures the mock

Conflating these is how a project ends up claiming "100% accuracy" or a
headline saving that evaporates on real traffic. Run this before quoting any
figure from this repo.

    python scripts/significance_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
    permutation_test_score,
)
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.features import FeatureExtractor  # noqa: E402
from src.classifier.model import ComplexityClassifier  # noqa: E402
from src.logging.database import Database  # noqa: E402

RULE = "─" * 74
BOOTSTRAP_ROUNDS = 5000


def section(title: str, bucket: str) -> None:
    print(f"\n{RULE}\n{title}   [{bucket}]\n{RULE}")


def _forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )


# ── 1. The savings headline ─────────────────────────────────────────────

def analyse_savings(db: Database) -> None:
    section("1. COST SAVINGS HEADLINE", "ARITHMETIC")

    logs = db.get_all_logs()
    if not logs:
        print("\n  No logged requests. Run scripts/seed_demo_data.py first.")
        return

    escalations = {
        e["request_log_id"]: e["escalated_cost"]
        for e in db.get_escalation_events(limit=100_000)
    }
    costs = np.array(
        [
            (log["cost"] + escalations.get(log["id"], 0.0), log["hypothetical_cost"])
            for log in logs
        ]
    )
    actual, baseline = costs[:, 0], costs[:, 1]
    point = (baseline.sum() - actual.sum()) / baseline.sum() * 100

    rng = np.random.default_rng(0)
    boot = np.empty(BOOTSTRAP_ROUNDS)
    for i in range(BOOTSTRAP_ROUNDS):
        idx = rng.integers(0, len(costs), len(costs))
        sample = costs[idx]
        boot[i] = (
            (sample[:, 1].sum() - sample[:, 0].sum()) / sample[:, 1].sum() * 100
        )

    print(f"\n  Point estimate       {point:.1f}%  (n={len(logs)} requests)")
    print(
        f"  Bootstrap 95% CI     "
        f"[{np.percentile(boot, 2.5):.1f}%, {np.percentile(boot, 97.5):.1f}%]"
    )

    # The CI above only captures resampling within this prompt mix. The mix
    # itself is the dominant term, and it was chosen, not observed.
    by_tier: dict[int, list] = {1: [], 2: [], 3: []}
    for log in logs:
        by_tier[log["complexity_tier"]].append(
            (log["cost"] + escalations.get(log["id"], 0.0), log["hypothetical_cost"])
        )

    print("\n  Savings if ALL traffic were one tier:")
    spread = []
    for tier in (1, 2, 3):
        rows = np.array(by_tier[tier]) if by_tier[tier] else None
        if rows is None or not len(rows):
            continue
        value = (rows[:, 1].sum() - rows[:, 0].sum()) / rows[:, 1].sum() * 100
        spread.append(value)
        share = len(rows) / len(logs) * 100
        print(
            f"    tier {tier}  {value:6.1f}%   "
            f"(this dataset is {share:.0f}% tier {tier})"
        )

    if spread:
        print(
            f"\n  VERDICT: the headline moves {min(spread):.0f}%-{max(spread):.0f}% "
            f"depending on traffic mix alone —\n"
            f"  a far larger swing than the +/-{(np.percentile(boot, 97.5) - point):.0f}pp "
            f"sampling interval. This number\n"
            f"  describes the PROMPT MIX, not the router. It is not a finding."
        )


# ── 2. Classifier accuracy ──────────────────────────────────────────────

def analyse_classifier(corpus: list[dict]) -> ComplexityClassifier:
    section("2. CLASSIFIER ACCURACY", "MEASURED")

    X = FeatureExtractor().extract_batch([d["prompt"] for d in corpus])
    y = np.array([d["tier"] for d in corpus])

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    scores = cross_val_score(_forest(), X, y, cv=cv)

    print(f"\n  5-fold x 10 repeats, n={len(y)}")
    print(f"    mean accuracy      {scores.mean():.1%}")
    print(f"    std across folds   {scores.std():.1%}")
    print(
        f"    95% fold range     "
        f"[{np.percentile(scores, 2.5):.1%}, {np.percentile(scores, 97.5):.1%}]"
    )

    _, permuted, p_value = permutation_test_score(
        _forest(), X, y, cv=5, n_permutations=200, random_state=0, n_jobs=-1
    )
    print(
        f"\n  Permutation test (labels shuffled 200x):\n"
        f"    shuffled baseline  {permuted.mean():.1%}\n"
        f"    p-value            {p_value:.4f}"
    )
    verdict = "real signal" if p_value < 0.05 else "NOT distinguishable from chance"
    print(f"\n  VERDICT: {verdict}, but the fold range is wide — treat this as")
    print("  'clearly better than guessing', not as a precise figure. It is also")
    print("  measured on a hand-authored corpus, so it does not predict live traffic.")

    classifier = ComplexityClassifier()
    classifier.train([d["prompt"] for d in corpus], [d["tier"] for d in corpus])
    return classifier


# ── 3. The length leak ──────────────────────────────────────────────────

def analyse_length_leak(corpus: list[dict]) -> None:
    section("3. LENGTH LEAK (the regression guard)", "MEASURED")

    X = np.array([[len(d["prompt"])] for d in corpus])
    y = np.array([d["tier"] for d in corpus])

    score, permuted, p_value = permutation_test_score(
        DecisionTreeClassifier(max_depth=3, random_state=0),
        X, y, cv=5, n_permutations=500, random_state=0, n_jobs=-1,
    )
    print(f"\n  Accuracy from prompt LENGTH alone   {score:.1%}")
    print(f"  Same model on shuffled labels       {permuted.mean():.1%}")
    print(f"  p-value                             {p_value:.4f}")

    if p_value < 0.05:
        print("\n  VERDICT: length still predicts tier. The leak is NOT fully gone.")
    else:
        print("\n  VERDICT: length is statistically indistinguishable from chance.")
        print("  The leak is genuinely closed — this is the strongest result here.")


# ── 4. Adversarial set ──────────────────────────────────────────────────

def analyse_adversarial(classifier: ComplexityClassifier) -> None:
    section("4. LENGTH-ADVERSARIAL SET", "MEASURED (underpowered)")

    path = PROJECT_ROOT / "data" / "eval_length_adversarial.json"
    if not path.exists():
        print("\n  Missing — run scripts/build_training_data.py first.")
        return

    adversarial = json.loads(path.read_text())
    hits = sum(
        1 for a in adversarial if int(classifier.predict(a["prompt"])) == a["tier"]
    )
    n = len(adversarial)
    result = binomtest(hits, n, p=1 / 3)
    ci = result.proportion_ci(confidence_level=0.95)

    print(f"\n  Accuracy    {hits}/{n} = {hits / n:.0%}")
    print(f"  95% CI      [{ci.low:.0%}, {ci.high:.0%}]")
    print(f"  vs chance   p = {result.pvalue:.6f}")
    print(
        f"\n  VERDICT: beats chance, but n={n} makes the interval "
        f"{(ci.high - ci.low) * 100:.0f} points wide.\n"
        f"  Enough to show the model is not length-driven; too small to quote\n"
        f"  '85%' as a real accuracy. Grow this set before trusting the number."
    )


# ── 5. Quality scores ───────────────────────────────────────────────────

def analyse_quality(db: Database) -> None:
    section("5. QUALITY SCORES / PASS RATE / ESCALATION RATE", "SIMULATED")

    scores = db.get_quality_distribution()
    verification = db.get_verification_stats()

    if not scores:
        print("\n  No verified requests yet.")
        return

    print(f"\n  Verified requests   {verification['verified_count']}")
    print(f"  Pass rate           {verification['pass_rate']:.0f}%")
    print(f"  Distinct scores     {len(set(scores))} across {len(scores)} requests")
    print(
        f"\n  In mock mode these come from src/models/providers/mock_content.py,\n"
        f"  where each quality tier emits a fixed fraction of the reference\n"
        f"  answer and a fixed label-error rate. The pass rate is therefore a\n"
        f"  restatement of those constants and the thresholds in routing.yaml."
    )
    print(
        "\n  VERDICT: measures the simulation, not the models. Carries no\n"
        "  information about real model quality. Ignore until MOCK_PROVIDERS=false."
    )


# ── How to actually verify ──────────────────────────────────────────────

def how_to_verify() -> None:
    section("HOW TO GET NUMBERS THAT MEAN SOMETHING", "")
    print(
        """
  1. REAL TOKENS AND PRICES
     Mock token counts are len(text.split()) * 2. Set MOCK_PROVIDERS=false
     with real keys; providers then report true usage. Until then every cost
     figure is a guess about tokenization.

  2. REAL TRAFFIC MIX
     The headline swings 33%-95% on mix alone, so it is only meaningful
     against YOUR prompt distribution. Log a week of production prompts and
     replay them. Report savings with the mix stated alongside it.

  3. HUMAN-LABELLED TEST SET
     The 220-prompt corpus is hand-authored, so accuracy on it measures
     agreement with its author. Have 2+ people independently tier ~300 real
     prompts, keep the ones they agree on, and never train on that set.

  4. A LARGER ADVERSARIAL SET
     n=20 gives a 35-point interval. ~200 prompts would narrow it to roughly
     +/-7 points.

  5. AN A/B TEST — THE ONLY REAL PROOF
     Route half of live traffic through AutoPilot and half to the top-tier
     model. Compare cost AND a quality measure on the same prompts. That is
     the only design that supports a causal claim about savings.

  Until then, state figures as: "on a 220-prompt synthetic corpus with mock
  providers" — accurate, and it costs nothing to say.
"""
    )


def main() -> int:
    corpus = json.loads((PROJECT_ROOT / "data" / "training_prompts.json").read_text())
    db = Database()

    print("\nWHICH NUMBERS IN THIS PROJECT ARE REAL?")
    analyse_savings(db)
    classifier = analyse_classifier(corpus)
    analyse_length_leak(corpus)
    analyse_adversarial(classifier)
    analyse_quality(db)
    how_to_verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
