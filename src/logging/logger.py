"""Structured request logger — bridges routing responses to the database."""

from __future__ import annotations

import hashlib
import logging
from typing import Optional

import structlog

from src.logging.database import Database
from src.router.engine import RoutedResponse

logger = structlog.get_logger(__name__)


class RequestLogger:
    """Logs every routed request to both SQLite and structured stdout."""

    def __init__(self, database: Database) -> None:
        self._db = database

    def log_request(
        self,
        prompt: str,
        routed_response: RoutedResponse,
        use_case: Optional[str] = None,
    ) -> int:
        """Log a completed request and return the log ID."""
        response = routed_response.response
        decision = routed_response.decision

        # Build log entry
        entry = {
            "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
            "prompt_preview": prompt[:200],
            "complexity_tier": decision.tier.value,
            "routed_model": decision.selected_model,
            "provider": response.provider,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost": response.cost,
            "latency_ms": response.latency_ms,
            "hypothetical_cost": routed_response.hypothetical_cost,
            "was_escalated": False,
            "was_verified": False,
            "use_case": use_case,
        }

        # Write to database
        log_id = self._db.log_request(entry)

        # Write structured log to stdout
        logger.info(
            "request_routed",
            log_id=log_id,
            tier=decision.tier.label,
            model=decision.selected_model,
            cost=f"${response.cost:.6f}",
            hypothetical=f"${routed_response.hypothetical_cost:.6f}",
            savings=f"${routed_response.hypothetical_cost - response.cost:.6f}",
            latency_ms=f"{response.latency_ms:.0f}",
            tokens=response.total_tokens,
        )

        return log_id

    def update_quality(self, log_id: int, quality_score: float) -> None:
        """Update quality score after async verification."""
        self._db.update_quality_score(log_id, quality_score)
        logger.info("quality_verified", log_id=log_id, score=quality_score)

    def record_verification(self, log_id: int, verification: dict) -> None:
        """Persist a full verification outcome (Phase 3.2)."""
        self._db.record_verification(log_id, verification)

        event = "quality_verified" if verification["passed"] else "routing_failure"
        emit = logger.info if verification["passed"] else logger.warning
        emit(
            event,
            log_id=log_id,
            use_case=verification["use_case"],
            strategy=verification["strategy"],
            score=f"{verification['score']:.2f}",
            threshold=verification["threshold"],
            quality_gap=verification["quality_gap"],
            divergence=f"{verification['divergence']:.0%}",
            reference_model=verification["reference_model"],
            detail=verification["detail"],
        )

    def mark_escalated(
        self, log_id: int, escalated_model: str, escalation_cost: float
    ) -> None:
        """Record an escalation event."""
        self._db.mark_escalated(log_id, escalated_model, escalation_cost)
        logger.warning(
            "request_escalated",
            log_id=log_id,
            escalated_model=escalated_model,
            escalation_cost=f"${escalation_cost:.6f}",
        )

    def log_escalation_event(self, log_id: int, event: dict) -> int:
        """Write the full escalation audit row (Phase 3.3).

        Captures original model, escalated model, cost delta, and the quality
        gap that triggered the escalation.
        """
        self._db.mark_escalated(
            log_id, event["escalated_model"], event["escalated_cost"]
        )
        event_id = self._db.log_escalation(log_id, event)

        logger.warning(
            "request_escalated",
            log_id=log_id,
            event_id=event_id,
            use_case=event["use_case"],
            original_model=event["original_model"],
            escalated_model=event["escalated_model"],
            cost_delta=f"${event['cost_delta']:+.6f}",
            quality_gap=event["quality_gap"],
            trigger=event["trigger_reason"],
            served_to_user=event["returned_to_user"],
            latency_ms=f"{event['escalation_latency_ms']:.0f}",
        )
        return event_id
