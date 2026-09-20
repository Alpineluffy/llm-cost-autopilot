#!/usr/bin/env python3
"""Train the complexity classifier from data/training_prompts.json.

    python scripts/train_classifier.py
    python scripts/train_classifier.py --with-feedback   # include failures
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.model import ComplexityClassifier  # noqa: E402
from src.classifier.tiers import ComplexityTier  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = PROJECT_ROOT / "data" / "training_prompts.json"
FEEDBACK_DATA = PROJECT_ROOT / "data" / "feedback_data.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the complexity classifier")
    parser.add_argument(
        "--with-feedback",
        action="store_true",
        help="Include accumulated routing failures from feedback_data.json",
    )
    args = parser.parse_args()

    with open(TRAINING_DATA) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} labelled prompts from training_prompts.json")

    if args.with_feedback and FEEDBACK_DATA.exists():
        with open(FEEDBACK_DATA) as f:
            feedback = json.load(f)
        data += feedback
        print(f"Added {len(feedback)} feedback corrections")

    prompts = [d["prompt"] for d in data]
    labels = [d["tier"] for d in data]

    classifier = ComplexityClassifier()
    metrics = classifier.train(prompts, labels)
    path = classifier.save()

    print(f"\nAccuracy: {metrics['accuracy']:.1%} "
          f"({metrics['train_size']} train / {metrics['test_size']} test)")

    print("\nPer-tier performance:")
    for tier_name in ("Simple", "Moderate", "Complex"):
        row = metrics["classification_report"][tier_name]
        print(
            f"  {tier_name:<9} precision {row['precision']:.2f}  "
            f"recall {row['recall']:.2f}  f1 {row['f1-score']:.2f}"
        )

    print("\nTop features:")
    for name, importance in list(metrics["feature_importances"].items())[:6]:
        print(f"  {name:<22} {importance:.3f}")

    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(f"  {'':<10}" + "".join(f"{t.label:>10}" for t in ComplexityTier))
    for tier, row in zip(ComplexityTier, metrics["confusion_matrix"]):
        print(f"  {tier.label:<10}" + "".join(f"{v:>10}" for v in row))

    print(f"\nSaved model → {os.path.relpath(path, PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
