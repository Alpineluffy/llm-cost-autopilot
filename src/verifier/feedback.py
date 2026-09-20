"""Feedback loop — routing failures become new training data for the classifier.

Phase 3.4 — every routing failure the verifier catches is a labelled example
the classifier got wrong: this prompt was routed to tier N and the output did
not hold up, so it belonged at a higher tier. Accumulate those, retrain weekly,
and the router gets smarter at the exact prompts it used to misjudge.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.classifier.model import ComplexityClassifier
from src.classifier.tiers import ComplexityTier
from src.logging.database import Database

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class FeedbackLoop:
    """Accumulates routing failures and retrains the classifier on a cadence."""

    DEFAULT_FEEDBACK_PATH = os.path.join(_PROJECT_ROOT, "data", "feedback_data.json")
    DEFAULT_STATE_PATH = os.path.join(_PROJECT_ROOT, "data", "feedback_state.json")

    #: A failure this far below the bar means the prompt was badly misjudged,
    #: so it jumps two tiers instead of one.
    SEVERE_GAP = 1.0

    def __init__(
        self,
        classifier: ComplexityClassifier,
        database: Optional[Database] = None,
        feedback_path: Optional[str] = None,
        state_path: Optional[str] = None,
        model_path: Optional[str] = None,
        retrain_interval_days: int = 7,
        min_examples_to_retrain: int = 10,
    ) -> None:
        self._classifier = classifier
        self._db = database
        self._feedback_path = feedback_path or self.DEFAULT_FEEDBACK_PATH
        self._state_path = state_path or self.DEFAULT_STATE_PATH
        #: Where a retrain writes the model. None means the classifier's
        #: default path; tests override it so they never clobber real state.
        self._model_path = model_path
        self._retrain_interval = timedelta(days=retrain_interval_days)
        self._min_examples = min_examples_to_retrain

        self._feedback_buffer: list[dict] = []
        self._seen_hashes: set[str] = set()
        self._state: dict = {}

        self._load_feedback()
        self._load_state()

    # ── Persistence ─────────────────────────────────────────────────────

    def _load_feedback(self) -> None:
        if not os.path.exists(self._feedback_path):
            return
        try:
            with open(self._feedback_path, "r") as f:
                self._feedback_buffer = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not read feedback file, starting empty: {e}")
            self._feedback_buffer = []
            return

        self._seen_hashes = {
            e["prompt_hash"] for e in self._feedback_buffer if "prompt_hash" in e
        }
        logger.info(f"Loaded {len(self._feedback_buffer)} feedback examples")

    def _save_feedback(self) -> None:
        os.makedirs(os.path.dirname(self._feedback_path), exist_ok=True)
        with open(self._feedback_path, "w") as f:
            json.dump(self._feedback_buffer, f, indent=2)

    def _load_state(self) -> None:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path, "r") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._state = {}

    def _save_state(self) -> None:
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Tier inference ──────────────────────────────────────────────────

    @staticmethod
    def infer_correct_tier(original_tier: int, quality_gap: float) -> int:
        """Infer the tier this prompt should have been routed to.

        The verifier tells us the prompt was under-served at `original_tier`.
        A small miss means one tier up; a severe miss jumps two. Capped at
        COMPLEX, which is the highest tier available.
        """
        jump = 2 if quality_gap >= FeedbackLoop.SEVERE_GAP else 1
        return min(int(original_tier) + jump, int(ComplexityTier.COMPLEX))

    # ── Recording failures ──────────────────────────────────────────────

    def record_failure(
        self,
        prompt: str,
        original_tier: int,
        quality_score: float,
        quality_gap: float,
        use_case: str = "general",
        correct_tier: Optional[int] = None,
    ) -> Optional[dict]:
        """Record a routing failure as a new training example.

        `correct_tier` is inferred from the size of the quality gap unless the
        caller supplies one. Returns the stored entry, or None if this prompt
        is already in the buffer or was already at the top tier.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        if prompt_hash in self._seen_hashes:
            logger.debug("Feedback example already recorded, skipping")
            return None

        if correct_tier is None:
            correct_tier = self.infer_correct_tier(original_tier, quality_gap)

        if correct_tier == original_tier:
            # Already at the top tier — nothing for the classifier to learn.
            logger.debug("Failure at top tier — no tier correction available")
            return None

        entry = {
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "tier": int(correct_tier),
            "original_tier": int(original_tier),
            "quality_score": round(quality_score, 3),
            "quality_gap": round(quality_gap, 3),
            "use_case": use_case,
            "category": "feedback_correction",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }

        self._feedback_buffer.append(entry)
        self._seen_hashes.add(prompt_hash)
        self._save_feedback()

        logger.info(
            f"Recorded feedback [{use_case}]: tier {original_tier} → "
            f"{correct_tier} (score={quality_score:.2f}, gap={quality_gap:.2f})"
        )
        return entry

    # ── Retraining ──────────────────────────────────────────────────────

    @property
    def last_retrained_at(self) -> Optional[datetime]:
        raw = self._state.get("last_retrained_at")
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return None

    def _new_examples_since_retrain(self) -> int:
        return len(self._feedback_buffer) - self._state.get("examples_at_last_retrain", 0)

    def should_retrain(self, now: Optional[datetime] = None) -> tuple[bool, str]:
        """Is a weekly retrain due? Returns (decision, human-readable reason)."""
        now = now or datetime.now(timezone.utc)
        new_examples = self._new_examples_since_retrain()

        if new_examples < self._min_examples:
            return False, (
                f"Only {new_examples} new failures since last retrain "
                f"(need {self._min_examples})"
            )

        last = self.last_retrained_at
        if last is None:
            return True, f"First retrain with {new_examples} accumulated failures"

        due_at = last + self._retrain_interval
        if now >= due_at:
            return True, (
                f"{new_examples} new failures and last retrain was "
                f"{(now - last).days}d ago"
            )
        return False, f"Next retrain due {due_at.date().isoformat()}"

    def maybe_retrain(self, now: Optional[datetime] = None) -> Optional[dict]:
        """Retrain if the weekly cadence and example threshold are both met."""
        due, reason = self.should_retrain(now)
        if not due:
            logger.info(f"Retrain skipped: {reason}")
            return None

        logger.info(f"Retrain triggered: {reason}")
        return self.retrain_classifier(now=now)

    def retrain_classifier(
        self,
        original_data_path: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> dict:
        """Retrain the classifier on original + accumulated feedback data."""
        now = now or datetime.now(timezone.utc)
        if original_data_path is None:
            original_data_path = os.path.join(
                _PROJECT_ROOT, "data", "training_prompts.json"
            )

        with open(original_data_path, "r") as f:
            original_data = json.load(f)

        all_data = original_data + self._feedback_buffer
        prompts = [d["prompt"] for d in all_data]
        labels = [d["tier"] for d in all_data]

        logger.info(
            f"Retraining with {len(original_data)} original + "
            f"{len(self._feedback_buffer)} feedback = {len(all_data)} examples"
        )

        metrics = self._classifier.train(prompts, labels)
        self._classifier.save(self._model_path)

        self._state.update(
            {
                "last_retrained_at": now.isoformat(),
                "examples_at_last_retrain": len(self._feedback_buffer),
                "last_accuracy": metrics["accuracy"],
                "training_set_size": len(all_data),
            }
        )
        self._save_state()

        logger.info(f"Retrained classifier: accuracy={metrics['accuracy']:.2%}")
        metrics["feedback_examples_used"] = len(self._feedback_buffer)
        return metrics

    # ── Introspection ───────────────────────────────────────────────────

    @property
    def feedback_count(self) -> int:
        return len(self._feedback_buffer)

    def summary(self) -> dict:
        due, reason = self.should_retrain()
        by_use_case: dict[str, int] = {}
        for entry in self._feedback_buffer:
            key = entry.get("use_case", "general")
            by_use_case[key] = by_use_case.get(key, 0) + 1

        return {
            "feedback_examples": len(self._feedback_buffer),
            "new_since_retrain": self._new_examples_since_retrain(),
            "last_retrained_at": self._state.get("last_retrained_at"),
            "last_accuracy": self._state.get("last_accuracy"),
            "retrain_due": due,
            "retrain_reason": reason,
            "by_use_case": by_use_case,
        }
