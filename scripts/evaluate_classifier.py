#!/usr/bin/env python3
"""Honest evaluation of the complexity classifier.

A plain train/test split told us the old classifier was 100% accurate while it
was really just measuring prompt length. This script reports the three numbers
that would have caught that:

  1. Held-out accuracy            — the usual number
  2. Length-only baseline         — what a model sees from length ALONE
  3. Length-adversarial accuracy  — short-complex and long-simple prompts

A healthy classifier scores well on (1) and (3) while (2) stays near chance.
If (1) and (2) are both high, the model is reading length, not meaning.

    python scripts/evaluate_classifier.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.classifier.model import ComplexityClassifier  # noqa: E402
from src.classifier.tiers import ComplexityTier  # noqa: E402

TRAIN_PATH = PROJECT_ROOT / "data" / "training_prompts.json"
EVAL_PATH = PROJECT_ROOT / "data" / "eval_length_adversarial.json"

RULE = "─" * 72


def section(title: str) -> None:
    print(f"\n{RULE}\n{title}\n{RULE}")


def main() -> int:
    data = json.loads(TRAIN_PATH.read_text())
    prompts = [d["prompt"] for d in data]
    labels = [d["tier"] for d in data]

    section("1. HELD-OUT ACCURACY")
    classifier = ComplexityClassifier()
    metrics = classifier.train(prompts, labels)

    # A single split over ~44 test prompts swings several points run to run,
    # so cross-validation is the number to trust here.
    from src.classifier.features import FeatureExtractor
    from sklearn.ensemble import RandomForestClassifier

    X = FeatureExtractor().extract_batch(prompts)
    cv_scores = cross_val_score(
        RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
        X, np.array(labels), cv=5,
    )
    cv_mean = cv_scores.mean()

    print(
        f"\n  5-fold CV accuracy: {cv_mean:.1%} "
        f"(±{cv_scores.std():.1%} across folds)"
    )
    print(
        f"  Single split:       {metrics['accuracy']:.1%} "
        f"({metrics['train_size']} train / {metrics['test_size']} test)"
    )
    for tier_name in ("Simple", "Moderate", "Complex"):
        row = metrics["classification_report"][tier_name]
        print(
            f"    {tier_name:<9} precision {row['precision']:.2f}  "
            f"recall {row['recall']:.2f}  f1 {row['f1-score']:.2f}"
        )

    section("2. LENGTH-ONLY BASELINE  (the leak detector)")
    X_len = np.array([[len(p)] for p in prompts])
    y = np.array(labels)
    baseline = cross_val_score(
        DecisionTreeClassifier(max_depth=3, random_state=0), X_len, y, cv=5
    ).mean()
    print(f"\n  A model given ONLY prompt length scores: {baseline:.1%}")
    print(f"  Chance is 33.3%.")
    if baseline > 0.50:
        print("\n  LEAK: length alone predicts the tier. The corpus encodes")
        print("  complexity in prompt length, so held-out accuracy is inflated.")
    else:
        print("\n  OK: length carries little tier signal, so accuracy in (1)")
        print("  reflects the model reading the prompt, not measuring it.")

    section("3. LENGTH-ADVERSARIAL SET  (short-complex, long-simple)")
    if not EVAL_PATH.exists():
        print("\n  Missing — run scripts/build_training_data.py first.")
        return 1

    adversarial = json.loads(EVAL_PATH.read_text())
    correct = 0
    misses = []
    for item in adversarial:
        predicted = classifier.predict(item["prompt"])
        if int(predicted) == item["tier"]:
            correct += 1
        else:
            misses.append((item, predicted))

    total = len(adversarial)
    print(f"\n  Accuracy on adversarial prompts: {correct}/{total} "
          f"({correct / total:.0%})")

    short_complex = [i for i in adversarial if i["tier"] == 3]
    long_simple = [i for i in adversarial if i["tier"] == 1]
    for name, group in (("short-but-complex", short_complex),
                        ("long-but-simple", long_simple)):
        hits = sum(
            1 for i in group if int(classifier.predict(i["prompt"])) == i["tier"]
        )
        print(f"    {name:<18} {hits}/{len(group)}")

    if misses:
        print("\n  Misclassified:")
        for item, predicted in misses[:8]:
            preview = item["prompt"][:46].replace("\n", " ")
            print(
                f"    {len(item['prompt']):>4} chars  "
                f"expected {ComplexityTier(item['tier']).label:<9} "
                f"got {predicted.label:<9} {preview}…"
            )

    section("4. WHAT THE MODEL IS ACTUALLY USING")
    print()
    length_weight = 0.0
    for name, importance in list(metrics["feature_importances"].items())[:8]:
        marker = ""
        if name in ("char_count", "token_count", "sentence_count"):
            length_weight += importance
            marker = "  ← length proxy"
        print(f"    {name:<24} {importance:.3f}{marker}")

    print(f"\n  Length-proxy features total: {length_weight:.1%} of the decision")
    if length_weight > 0.50:
        print("  WARNING: the model is majority-driven by how long the prompt is.")
    else:
        print("  OK: semantic features outweigh length.")

    section("SUMMARY")
    healthy = (
        baseline <= 0.50
        and correct / total >= 0.70
        and length_weight <= 0.50
        and cv_mean >= 0.60
    )
    print(
        f"\n  CV accuracy            {cv_mean:.1%}   (want ≥60%, chance 33%)\n"
        f"  Length-only baseline   {baseline:.1%}   (want ≈33%)\n"
        f"  Adversarial accuracy   {correct / total:.0%}   (want ≥70%)\n"
        f"  Length-proxy weight    {length_weight:.1%}   (want ≤50%)\n"
    )
    print(f"  Verdict: {'HEALTHY' if healthy else 'NEEDS WORK'}\n")
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
