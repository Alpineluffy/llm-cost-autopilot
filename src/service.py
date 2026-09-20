"""Composition root — assembles the router, logger, verifier, and feedback loop.

Nothing else in the codebase wires these together: the router deliberately does
not import the verifier (that would cycle through the logger), so this is the
one place that connects the completion hook to the verification queue.

    service = AutopilotService()
    await service.start()
    result = await service.ask("Summarize this function")
    await service.drain()          # wait for async verification
    await service.stop()
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from src.classifier.model import ComplexityClassifier
from src.logging.database import Database
from src.logging.logger import RequestLogger
from src.models.registry import ModelRegistry
from src.router.engine import RoutedResponse, RoutingEngine
from src.router.escalation import EscalationResult
from src.verifier.feedback import FeedbackLoop
from src.verifier.use_cases import PolicyBook, UseCase, UseCaseDetector
from src.verifier.worker import VerificationJob, VerificationWorker

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class AutopilotService:
    """The full Phase 3 pipeline, assembled and running."""

    def __init__(
        self,
        db_url: Optional[str] = None,
        sample_rate: Optional[float] = None,
        classifier_path: Optional[str] = None,
        feedback_path: Optional[str] = None,
        state_path: Optional[str] = None,
        model_output_path: Optional[str] = None,
    ) -> None:
        self._registry = ModelRegistry()
        self._classifier = ComplexityClassifier()
        self._load_classifier(classifier_path)

        self._engine = RoutingEngine(
            registry=self._registry, classifier=self._classifier
        )
        self._config = self._engine.get_routing_config()

        self._database = Database(db_url)
        self._request_logger = RequestLogger(self._database)
        self._policies = PolicyBook(self._config)
        self._detector = UseCaseDetector()

        feedback_cfg = self._config.get("feedback", {}) or {}
        self._feedback = FeedbackLoop(
            classifier=self._classifier,
            database=self._database,
            feedback_path=feedback_path,
            state_path=state_path,
            model_path=model_output_path,
            retrain_interval_days=int(feedback_cfg.get("retrain_interval_days", 7)),
            min_examples_to_retrain=int(
                feedback_cfg.get("min_examples_to_retrain", 10)
            ),
        )

        verification_cfg = self._config.get("verification", {}) or {}
        if sample_rate is None:
            sample_rate = float(
                os.getenv(
                    "VERIFICATION_SAMPLE_RATE",
                    verification_cfg.get("sample_rate", 0.3),
                )
            )

        self._worker = VerificationWorker(
            registry=self._registry,
            providers=self._engine._providers,
            database=self._database,
            request_logger=self._request_logger,
            routing_config=self._config,
            policy_book=self._policies,
            feedback_loop=self._feedback,
            sample_rate=sample_rate,
            on_escalated=self._handle_escalated,
        )

        # The wiring that was missing: every completed request now reaches
        # the verification queue.
        self._engine.set_completion_hook(self._on_request_complete)

        # Bounded: a long-running process would otherwise retain every
        # escalated response for the life of the process.
        self._escalated_results: OrderedDict[int, EscalationResult] = OrderedDict()
        self._max_retained_escalations = 500

        self._verification_enabled = (
            os.getenv("VERIFICATION_ENABLED", "true").lower() == "true"
        )
        self._retrain_task: Optional[asyncio.Task] = None
        self._retrain_check_hours = float(
            os.getenv("RETRAIN_CHECK_INTERVAL_HOURS", "6")
        )

    def _load_classifier(self, path: Optional[str]) -> None:
        try:
            self._classifier.load(path)
            logger.info("Loaded trained classifier")
        except FileNotFoundError:
            logger.warning(
                "No trained classifier found — routing falls back to MODERATE. "
                "Run scripts/train_classifier.py first."
            )

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self, with_scheduler: bool = False) -> None:
        """Start the verification worker, and optionally the retrain scheduler.

        The scheduler is opt-in so short-lived processes (scripts, tests) do
        not spawn a periodic task they will never use.
        """
        await self._worker.start()
        if with_scheduler and self._retrain_task is None:
            self._retrain_task = asyncio.create_task(self._retrain_loop())
            logger.info(
                f"Retrain scheduler started "
                f"(checks every {self._retrain_check_hours}h)"
            )

    async def stop(self) -> None:
        if self._retrain_task is not None:
            self._retrain_task.cancel()
            try:
                await self._retrain_task
            except asyncio.CancelledError:
                pass
            self._retrain_task = None
        await self._worker.stop()

    async def _retrain_loop(self) -> None:
        """Periodically ask the feedback loop whether a retrain is due.

        The cadence check lives in FeedbackLoop; this only wakes it up. Training
        is CPU-bound sklearn work, so it runs in a thread rather than blocking
        the event loop and stalling in-flight requests.
        """
        interval = self._retrain_check_hours * 3600
        while True:
            try:
                await asyncio.sleep(interval)
                metrics = await asyncio.to_thread(self._feedback.maybe_retrain)
                if metrics:
                    logger.info(
                        f"Scheduled retrain complete: "
                        f"accuracy={metrics['accuracy']:.2%} "
                        f"({metrics['feedback_examples_used']} from failures)"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                # A failed retrain must never kill the scheduler.
                logger.error(f"Scheduled retrain failed: {e}")

    async def drain(self, timeout: float = 30.0) -> None:
        """Block until every queued verification has finished."""
        await self._worker.drain(timeout)

    # ── Request path ────────────────────────────────────────────────────

    async def ask(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> RoutedResponse:
        """Route a prompt, returning as soon as the response is ready.

        Verification happens afterwards, off the request path.
        """
        return await self._engine.route_request(
            prompt=prompt, max_tokens=max_tokens, temperature=temperature
        )

    def detect_use_case(self, prompt: str) -> UseCase:
        """Which grading policy this prompt falls under."""
        return self._detector.detect(prompt)

    async def _on_request_complete(
        self, prompt: str, routed: RoutedResponse
    ) -> None:
        """Log the request, then queue it for async verification."""
        use_case = self._detector.detect(prompt)
        log_id = self._request_logger.log_request(
            prompt, routed, use_case=use_case.value
        )
        # Stamp the response so the caller can poll for a later escalation.
        routed.log_id = log_id
        routed.use_case = use_case.value

        if not self._verification_enabled:
            return

        escalation_cfg = self._config.get("escalation", {}) or {}
        await self._worker.enqueue(
            VerificationJob(
                log_id=log_id,
                prompt=prompt,
                response=routed.response,
                tier=routed.decision.tier,
                model_name=routed.decision.selected_model,
                latency_budget_ms=escalation_cfg.get("latency_budget_ms"),
            )
        )

    async def _handle_escalated(
        self, log_id: int, result: EscalationResult
    ) -> None:
        """Hold onto an escalated result that arrived within the latency budget.

        Verification runs after the response was already returned, so the
        improved answer is retained here for the caller to collect via
        GET /v1/completions/{log_id}/escalation rather than pushed.
        """
        self._escalated_results[log_id] = result
        while len(self._escalated_results) > self._max_retained_escalations:
            self._escalated_results.popitem(last=False)  # evict oldest

    def get_escalated_result(self, log_id: int) -> Optional[EscalationResult]:
        """The improved response for a request, if escalation produced one."""
        return self._escalated_results.get(log_id)

    # ── Feedback flywheel ───────────────────────────────────────────────

    def maybe_retrain(self) -> Optional[dict]:
        """Run the weekly retrain if it is due."""
        return self._feedback.maybe_retrain()

    def force_retrain(self) -> dict:
        """Retrain now, regardless of cadence."""
        return self._feedback.retrain_classifier()

    # ── Introspection ───────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "requests": self._database.get_stats(),
            "verification": self._database.get_verification_stats(),
            "worker": self._worker.stats,
            "feedback": self._feedback.summary(),
            "policies": self._policies.all_policies(),
        }

    @property
    def database(self) -> Database:
        return self._database

    @property
    def worker(self) -> VerificationWorker:
        return self._worker

    @property
    def feedback(self) -> FeedbackLoop:
        return self._feedback

    @property
    def engine(self) -> RoutingEngine:
        return self._engine
