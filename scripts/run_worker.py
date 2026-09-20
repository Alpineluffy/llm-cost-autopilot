#!/usr/bin/env python3
"""Background maintenance worker — the retrain half of the flywheel.

Scope note: quality verification runs inside the API process, because the
verification queue is an in-memory asyncio.Queue and cannot be consumed from
another container. What *is* separable is the periodic retrain, which reads
accumulated failures from disk and rewrites the classifier — CPU-heavy work
that is better kept off the request path.

    python scripts/run_worker.py
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.classifier.model import ComplexityClassifier  # noqa: E402
from src.logging.database import Database  # noqa: E402
from src.verifier.feedback import FeedbackLoop  # noqa: E402

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("worker")

CHECK_INTERVAL_HOURS = float(os.getenv("RETRAIN_CHECK_INTERVAL_HOURS", "6"))


async def main() -> int:
    classifier = ComplexityClassifier()
    try:
        classifier.load()
        logger.info("Loaded existing classifier")
    except FileNotFoundError:
        logger.warning("No trained classifier yet — a retrain will create one")

    feedback = FeedbackLoop(classifier=classifier, database=Database())
    logger.info(
        f"Maintenance worker started "
        f"(checking every {CHECK_INTERVAL_HOURS}h, "
        f"{feedback.feedback_count} failures accumulated)"
    )

    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stopping.set)
        except NotImplementedError:  # not available on some platforms
            pass

    interval = CHECK_INTERVAL_HOURS * 3600
    while not stopping.is_set():
        try:
            await asyncio.wait_for(stopping.wait(), timeout=interval)
            break  # stop requested
        except asyncio.TimeoutError:
            pass

        try:
            # Re-read from disk: the API process appends failures there.
            feedback._load_feedback()
            metrics = await asyncio.to_thread(feedback.maybe_retrain)
            if metrics:
                logger.info(
                    f"Retrained: accuracy={metrics['accuracy']:.2%} on "
                    f"{metrics['train_size'] + metrics['test_size']} examples "
                    f"({metrics['feedback_examples_used']} from failures)"
                )
        except Exception as e:
            logger.error(f"Retrain check failed: {e}")

    logger.info("Maintenance worker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
