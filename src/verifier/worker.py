"""Async background verification worker.

Pulls completed requests off an asyncio.Queue, compares each against the
highest-tier model under its use-case policy, escalates routing failures, and
feeds every failure back to the classifier as training data.

This is the loop that ties Phase 3 together:
    verify (3.2) → escalate (3.3) → feed back (3.4), all under per-use-case
    thresholds (3.1).
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

from src.classifier.tiers import ComplexityTier
from src.logging.database import Database
from src.logging.logger import RequestLogger
from src.models.providers.base import BaseProvider
from src.models.registry import ModelRegistry
from src.models.response import LLMResponse
from src.router.escalation import EscalationManager, EscalationResult
from src.verifier.feedback import FeedbackLoop
from src.verifier.judge import QualityJudge
from src.verifier.use_cases import PolicyBook

logger = logging.getLogger(__name__)

EscalationCallback = Callable[[int, EscalationResult], Awaitable[None]]


@dataclass
class VerificationJob:
    """A completed request awaiting quality verification."""

    log_id: int
    prompt: str
    response: LLMResponse
    tier: ComplexityTier
    model_name: str
    latency_budget_ms: Optional[float] = None


class VerificationWorker:
    """Background worker that processes the verification queue.

    Uses asyncio.Queue — no external dependencies like Redis.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        providers: dict[str, BaseProvider],
        database: Database,
        request_logger: RequestLogger,
        routing_config: dict,
        policy_book: Optional[PolicyBook] = None,
        feedback_loop: Optional[FeedbackLoop] = None,
        sample_rate: float = 0.3,
        on_escalated: Optional[EscalationCallback] = None,
    ) -> None:
        verification_cfg = routing_config.get("verification", {}) or {}

        # Created lazily inside the running loop. On Python 3.9 an
        # asyncio.Queue binds to whatever loop exists when it is constructed,
        # so building the worker outside the loop that later runs it would
        # attach its futures to the wrong loop.
        self._queue: Optional[asyncio.Queue] = None
        self._judge = QualityJudge(
            registry, providers, judge_model=verification_cfg.get("judge_model")
        )
        self._escalation = EscalationManager(registry, providers, routing_config)
        self._policies = policy_book or PolicyBook(routing_config)
        self._feedback = feedback_loop
        self._db = database
        self._logger = request_logger
        self._sample_rate = sample_rate
        self._on_escalated = on_escalated

        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Stats
        self._verified_count = 0
        self._passed_count = 0
        self._failed_count = 0
        self._escalated_count = 0
        self._served_count = 0
        self._feedback_count = 0

    # ── Queue management ────────────────────────────────────────────────

    @property
    def _q(self) -> asyncio.Queue:
        """The work queue, bound to the loop that first touches it."""
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    async def enqueue(self, job: VerificationJob) -> bool:
        """Add a job to the verification queue.

        Only a fraction of requests are sampled (per sample_rate).
        Returns True if the job was actually queued.
        """
        if random.random() > self._sample_rate:
            return False

        await self._q.put(job)
        return True

    async def enqueue_now(self, job: VerificationJob) -> None:
        """Queue a job unconditionally, bypassing sampling."""
        await self._q.put(job)

    async def drain(self, timeout: float = 30.0) -> None:
        """Wait until every queued job has been processed."""
        await asyncio.wait_for(self._q.join(), timeout=timeout)

    # ── Worker lifecycle ────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background verification loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        logger.info(
            f"Verification worker started (sample_rate={self._sample_rate})"
        )

    async def stop(self) -> None:
        """Gracefully stop the worker."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(
            f"Verification worker stopped. "
            f"Verified={self._verified_count}, "
            f"Passed={self._passed_count}, "
            f"Failed={self._failed_count}, "
            f"Escalated={self._escalated_count}"
        )

    async def _process_loop(self) -> None:
        """Main processing loop — runs until stopped."""
        while self._running:
            try:
                job = await asyncio.wait_for(self._q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            try:
                await self._process_job(job)
            except Exception as e:
                logger.error(f"Verification error on log_id={job.log_id}: {e}")
            finally:
                self._q.task_done()

    # ── Job processing ──────────────────────────────────────────────────

    async def _process_job(self, job: VerificationJob) -> None:
        """Verify one job, escalating and learning from it if it failed."""
        self._verified_count += 1

        # 3.1 + 3.2 — grade against the reference under this use case's policy.
        policy = self._policies.for_prompt(job.prompt)
        outcome = await self._judge.verify(job.prompt, job.response, policy)

        self._logger.record_verification(job.log_id, outcome.to_dict())

        if outcome.passed:
            self._passed_count += 1
            return

        self._failed_count += 1

        # 3.3 — escalate to the next tier up.
        result = await self._escalation.escalate(
            prompt=job.prompt,
            original_response=job.response,
            original_tier=job.tier,
            quality_score=outcome.score,
            quality_threshold=policy.threshold,
            use_case=outcome.use_case.value,
            trigger_reason=outcome.detail,
            latency_budget_ms=job.latency_budget_ms,
        )

        if result is not None:
            self._escalated_count += 1
            self._logger.log_escalation_event(job.log_id, result.to_dict())

            if result.returned_to_user:
                self._served_count += 1
                if self._on_escalated is not None:
                    await self._on_escalated(job.log_id, result)

        # 3.4 — the failure becomes a training example regardless of whether
        # escalation succeeded; the classifier still misjudged this prompt.
        if self._feedback is not None:
            entry = self._feedback.record_failure(
                prompt=job.prompt,
                original_tier=int(job.tier),
                quality_score=outcome.score,
                quality_gap=outcome.quality_gap,
                use_case=outcome.use_case.value,
            )
            if entry is not None:
                self._feedback_count += 1

    # ── Stats ───────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "verified": self._verified_count,
            "passed": self._passed_count,
            "failed": self._failed_count,
            "escalated": self._escalated_count,
            "escalations_served": self._served_count,
            "feedback_recorded": self._feedback_count,
            "queue_size": self._queue.qsize() if self._queue is not None else 0,
            "pass_rate": (
                round(self._passed_count / self._verified_count * 100, 1)
                if self._verified_count > 0
                else 0
            ),
        }
