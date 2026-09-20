"""Complexity classifier — Random Forest on extracted prompt features."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.classifier.features import FeatureExtractor
from src.classifier.tiers import ComplexityTier


class ComplexityClassifier:
    """Classifies prompts into complexity tiers for cost-optimal routing.

    Uses a Random Forest model trained on extracted prompt features.
    Supports training, evaluation, prediction, saving, and loading.
    """

    DEFAULT_MODEL_PATH = os.path.join(
        Path(__file__).resolve().parent.parent.parent, "data", "classifier_model.pkl"
    )

    def __init__(self) -> None:
        self._extractor = FeatureExtractor()
        self._model: Optional[RandomForestClassifier] = None
        self._is_trained = False

    # ── Training ────────────────────────────────────────────────────────

    def train(
        self,
        prompts: list[str],
        labels: list[int],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict:
        """Train the classifier on labeled prompts.

        Args:
            prompts: List of prompt strings.
            labels: List of tier labels (1, 2, 3).
            test_size: Fraction of data to hold out for evaluation.
            random_state: Random seed for reproducibility.

        Returns:
            Dictionary with training metrics.
        """
        X = self._extractor.extract_batch(prompts)
        y = np.array(labels)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        self._model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self._model.fit(X_train, y_train)
        self._is_trained = True

        # Evaluate
        y_pred = self._model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred, target_names=["Simple", "Moderate", "Complex"], output_dict=True
        )
        cm = confusion_matrix(y_test, y_pred).tolist()

        # Feature importances
        importances = dict(
            zip(FeatureExtractor.FEATURE_NAMES, self._model.feature_importances_)
        )

        return {
            "accuracy": round(accuracy, 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "feature_importances": {
                k: round(v, 4) for k, v in sorted(importances.items(), key=lambda x: -x[1])
            },
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    # ── Prediction ──────────────────────────────────────────────────────

    def predict(self, prompt: str) -> ComplexityTier:
        """Classify a single prompt into a complexity tier."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        features = self._extractor.extract(prompt).reshape(1, -1)
        prediction = self._model.predict(features)[0]
        return ComplexityTier(int(prediction))

    def predict_with_confidence(self, prompt: str) -> tuple[ComplexityTier, dict[str, float]]:
        """Classify a prompt and return confidence scores per tier."""
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() or load() first.")

        features = self._extractor.extract(prompt).reshape(1, -1)
        prediction = self._model.predict(features)[0]
        probabilities = self._model.predict_proba(features)[0]

        tier = ComplexityTier(int(prediction))
        confidence = {}
        for i, cls in enumerate(self._model.classes_):
            tier_name = ComplexityTier(int(cls)).label
            confidence[tier_name] = round(float(probabilities[i]), 4)

        return tier, confidence

    # ── Persistence ─────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """Save the trained model to disk."""
        if not self._is_trained:
            raise RuntimeError("No trained model to save.")

        path = path or self.DEFAULT_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self._model, path)
        return path

    def load(self, path: Optional[str] = None) -> None:
        """Load a pre-trained model from disk."""
        path = path or self.DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(f"No model file at {path}")

        self._model = joblib.load(path)
        self._is_trained = True

    @property
    def is_trained(self) -> bool:
        return self._is_trained
