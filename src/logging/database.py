"""SQLite database schema and ORM models for the audit trail."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    case, create_engine, func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


class RequestLog(Base):
    """Every routed request gets a row in this table."""

    __tablename__ = "request_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    prompt_hash = Column(String(64), nullable=False, index=True)
    prompt_preview = Column(String(200), nullable=False)
    complexity_tier = Column(Integer, nullable=False, index=True)
    routed_model = Column(String(100), nullable=False, index=True)
    provider = Column(String(50), nullable=False)
    input_tokens = Column(Integer, nullable=False)
    output_tokens = Column(Integer, nullable=False)
    cost = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    quality_score = Column(Float, nullable=True)  # Filled by verifier
    was_escalated = Column(Boolean, default=False, nullable=False)
    escalated_model = Column(String(100), nullable=True)
    escalation_cost = Column(Float, nullable=True)
    hypothetical_cost = Column(Float, nullable=False)  # GPT-4o baseline cost

    # ── Phase 3: verification detail ────────────────────────────────────
    use_case = Column(String(40), nullable=True, index=True)
    verification_strategy = Column(String(40), nullable=True)
    quality_threshold = Column(Float, nullable=True)
    quality_gap = Column(Float, nullable=True)     # >0 means it missed the bar
    divergence = Column(Float, nullable=True)      # 0.0 identical - 1.0 divergent
    was_verified = Column(Boolean, default=False, nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "prompt_hash": self.prompt_hash,
            "prompt_preview": self.prompt_preview,
            "complexity_tier": self.complexity_tier,
            "routed_model": self.routed_model,
            "provider": self.provider,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost": self.cost,
            "latency_ms": self.latency_ms,
            "quality_score": self.quality_score,
            "was_escalated": self.was_escalated,
            "escalated_model": self.escalated_model,
            "escalation_cost": self.escalation_cost,
            "hypothetical_cost": self.hypothetical_cost,
            "use_case": self.use_case,
            "verification_strategy": self.verification_strategy,
            "quality_threshold": self.quality_threshold,
            "quality_gap": self.quality_gap,
            "divergence": self.divergence,
            "was_verified": self.was_verified,
        }


class EscalationEvent(Base):
    """One row per escalation — the Phase 3.3 audit trail.

    Kept separate from request_logs so the cost delta and the quality gap that
    triggered each escalation are preserved in full, not flattened.
    """

    __tablename__ = "escalation_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    request_log_id = Column(Integer, index=True, nullable=False)

    original_model = Column(String(100), nullable=False)
    escalated_model = Column(String(100), nullable=False)
    original_cost = Column(Float, nullable=False)
    escalated_cost = Column(Float, nullable=False)
    cost_delta = Column(Float, nullable=False)
    quality_gap = Column(Float, nullable=False)

    original_tier = Column(Integer, nullable=False)
    escalated_tier = Column(Integer, nullable=False)
    use_case = Column(String(40), nullable=True)
    trigger_reason = Column(Text, nullable=True)
    returned_to_user = Column(Boolean, default=False, nullable=False)
    latency_budget_ms = Column(Float, nullable=True)
    escalation_latency_ms = Column(Float, nullable=False, default=0.0)
    attempts = Column(Integer, nullable=False, default=1)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "request_log_id": self.request_log_id,
            "original_model": self.original_model,
            "escalated_model": self.escalated_model,
            "original_cost": self.original_cost,
            "escalated_cost": self.escalated_cost,
            "cost_delta": self.cost_delta,
            "quality_gap": self.quality_gap,
            "original_tier": self.original_tier,
            "escalated_tier": self.escalated_tier,
            "use_case": self.use_case,
            "trigger_reason": self.trigger_reason,
            "returned_to_user": self.returned_to_user,
            "latency_budget_ms": self.latency_budget_ms,
            "escalation_latency_ms": self.escalation_latency_ms,
            "attempts": self.attempts,
        }


class Database:
    """Manages the SQLite connection and provides query helpers."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            db_url = os.getenv("DATABASE_URL", "sqlite:///./data/autopilot.db")

        # Ensure data directory exists
        if db_url.startswith("sqlite:///"):
            db_path = db_url.replace("sqlite:///", "")
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

        self._engine = create_engine(db_url, echo=False)

        if db_url.startswith("sqlite"):
            self._configure_sqlite()

        self._SessionLocal = sessionmaker(bind=self._engine)
        Base.metadata.create_all(self._engine)

    def _configure_sqlite(self) -> None:
        """Make SQLite safe for more than one process.

        The API container writes request rows while the worker container
        writes retrain state, and the dashboard reads throughout. Default
        SQLite locking would surface that as 'database is locked'; WAL plus a
        busy timeout lets readers and one writer proceed concurrently.
        """
        from sqlalchemy import event

        @event.listens_for(self._engine, "connect")
        def set_pragmas(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.close()

    def get_session(self) -> Session:
        return self._SessionLocal()

    # ── Write operations ────────────────────────────────────────────────

    def log_request(self, log_entry: dict) -> int:
        """Insert a request log entry and return its ID."""
        with self.get_session() as session:
            record = RequestLog(**log_entry)
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    def update_quality_score(self, log_id: int, quality_score: float) -> None:
        """Update the quality score after verification."""
        with self.get_session() as session:
            record = session.query(RequestLog).filter_by(id=log_id).first()
            if record:
                record.quality_score = quality_score
                record.was_verified = True
                session.commit()

    def record_verification(self, log_id: int, verification: dict) -> None:
        """Persist the full verification outcome for a request."""
        with self.get_session() as session:
            record = session.query(RequestLog).filter_by(id=log_id).first()
            if not record:
                return
            record.quality_score = verification.get("score")
            record.use_case = verification.get("use_case")
            record.verification_strategy = verification.get("strategy")
            record.quality_threshold = verification.get("threshold")
            record.quality_gap = verification.get("quality_gap")
            record.divergence = verification.get("divergence")
            record.was_verified = True
            session.commit()

    def mark_escalated(
        self, log_id: int, escalated_model: str, escalation_cost: float
    ) -> None:
        """Mark a request as escalated after quality failure."""
        with self.get_session() as session:
            record = session.query(RequestLog).filter_by(id=log_id).first()
            if record:
                record.was_escalated = True
                record.escalated_model = escalated_model
                record.escalation_cost = escalation_cost
                session.commit()

    def log_escalation(self, log_id: int, event: dict) -> int:
        """Insert a full escalation audit row and return its ID."""
        with self.get_session() as session:
            record = EscalationEvent(
                request_log_id=log_id,
                original_model=event["original_model"],
                escalated_model=event["escalated_model"],
                original_cost=event["original_cost"],
                escalated_cost=event["escalated_cost"],
                cost_delta=event["cost_delta"],
                quality_gap=event["quality_gap"],
                original_tier=event["original_tier"],
                escalated_tier=event["escalated_tier"],
                use_case=event.get("use_case"),
                trigger_reason=event.get("trigger_reason"),
                returned_to_user=event.get("returned_to_user", False),
                latency_budget_ms=event.get("latency_budget_ms"),
                escalation_latency_ms=event.get("escalation_latency_ms", 0.0),
                attempts=event.get("attempts", 1),
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record.id

    # ── Read operations (for dashboard) ─────────────────────────────────

    def get_stats(self) -> dict:
        """Aggregate statistics for the dashboard."""
        with self.get_session() as session:
            total_requests = session.query(func.count(RequestLog.id)).scalar() or 0
            total_cost = session.query(func.sum(RequestLog.cost)).scalar() or 0.0
            total_hypothetical = (
                session.query(func.sum(RequestLog.hypothetical_cost)).scalar() or 0.0
            )
            avg_latency = session.query(func.avg(RequestLog.latency_ms)).scalar() or 0.0
            avg_quality = (
                session.query(func.avg(RequestLog.quality_score))
                .filter(RequestLog.quality_score.isnot(None))
                .scalar()
                or 0.0
            )
            escalation_count = (
                session.query(func.count(RequestLog.id))
                .filter(RequestLog.was_escalated == True)
                .scalar()
                or 0
            )

            # An escalated request is paid for twice: once on the original
            # model and again on the higher tier. The additional spend is the
            # re-run's FULL cost, not its delta against the original — a delta
            # can even be negative, which would credit us for failing.
            escalation_cost = (
                session.query(func.sum(EscalationEvent.escalated_cost)).scalar()
                or 0.0
            )
            true_cost = total_cost + escalation_cost

            savings = total_hypothetical - true_cost
            savings_pct = (
                (savings / total_hypothetical * 100) if total_hypothetical > 0 else 0
            )

            # Model distribution
            model_dist = (
                session.query(
                    RequestLog.routed_model,
                    func.count(RequestLog.id).label("count"),
                )
                .group_by(RequestLog.routed_model)
                .all()
            )

            # Tier distribution
            tier_dist = (
                session.query(
                    RequestLog.complexity_tier,
                    func.count(RequestLog.id).label("count"),
                )
                .group_by(RequestLog.complexity_tier)
                .all()
            )

            return {
                "total_requests": total_requests,
                # Routed-model spend only, kept separate so the escalation
                # overhead stays visible rather than buried in one number.
                "routed_cost": round(total_cost, 6),
                "escalation_cost": round(escalation_cost, 6),
                # What the system actually cost, escalations included.
                "total_cost": round(true_cost, 6),
                "total_hypothetical_cost": round(total_hypothetical, 6),
                "total_savings": round(savings, 6),
                "savings_percentage": round(savings_pct, 2),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_quality_score": round(avg_quality, 2),
                "escalation_count": escalation_count,
                "escalation_rate": (
                    round(escalation_count / total_requests * 100, 2)
                    if total_requests > 0
                    else 0
                ),
                "model_distribution": {row[0]: row[1] for row in model_dist},
                "tier_distribution": {str(row[0]): row[1] for row in tier_dist},
            }

    def get_recent_logs(self, limit: int = 50) -> list[dict]:
        """Get recent request logs for display."""
        with self.get_session() as session:
            logs = (
                session.query(RequestLog)
                .order_by(RequestLog.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [log.to_dict() for log in logs]

    def _escalation_cost_by_period(self, period_expr) -> dict[str, float]:
        """Escalation spend keyed by the same period expression as the costs."""
        with self.get_session() as session:
            rows = (
                session.query(
                    period_expr(EscalationEvent.timestamp),
                    func.sum(EscalationEvent.escalated_cost),
                )
                .group_by(period_expr(EscalationEvent.timestamp))
                .all()
            )
            return {str(key): (value or 0.0) for key, value in rows}

    def get_daily_costs(self, days: int = 30) -> list[dict]:
        """Daily cost aggregates for the cost chart.

        `actual_cost` includes escalation spend so the chart agrees with the
        headline savings figure rather than contradicting it.
        """
        escalations = self._escalation_cost_by_period(func.date)

        with self.get_session() as session:
            results = (
                session.query(
                    func.date(RequestLog.timestamp).label("date"),
                    func.sum(RequestLog.cost).label("routed_cost"),
                    func.sum(RequestLog.hypothetical_cost).label("hypothetical_cost"),
                    func.count(RequestLog.id).label("request_count"),
                )
                .group_by(func.date(RequestLog.timestamp))
                .order_by(func.date(RequestLog.timestamp).desc())
                .limit(days)
                .all()
            )
            out = []
            for date, routed, hypothetical, count in results:
                escalation = escalations.get(str(date), 0.0)
                out.append(
                    {
                        "date": str(date),
                        "routed_cost": round(routed or 0.0, 6),
                        "escalation_cost": round(escalation, 6),
                        "actual_cost": round((routed or 0.0) + escalation, 6),
                        "hypothetical_cost": round(hypothetical or 0.0, 6),
                        "request_count": count,
                    }
                )
            return out

    def get_quality_distribution(self) -> list[float]:
        """Get all quality scores for distribution chart."""
        with self.get_session() as session:
            scores = (
                session.query(RequestLog.quality_score)
                .filter(RequestLog.quality_score.isnot(None))
                .all()
            )
            return [s[0] for s in scores]

    def get_all_logs(self) -> list[dict]:
        """Get all logs (for export/analysis)."""
        with self.get_session() as session:
            logs = session.query(RequestLog).order_by(RequestLog.timestamp).all()
            return [log.to_dict() for log in logs]

    # ── Phase 3 read operations ─────────────────────────────────────────

    def get_escalation_events(self, limit: int = 50) -> list[dict]:
        """Recent escalation events, newest first."""
        with self.get_session() as session:
            events = (
                session.query(EscalationEvent)
                .order_by(EscalationEvent.timestamp.desc())
                .limit(limit)
                .all()
            )
            return [e.to_dict() for e in events]

    def get_weekly_costs(self, weeks: int = 12) -> list[dict]:
        """Weekly cost aggregates against the all-top-tier baseline.

        As with the daily view, `actual_cost` includes escalation spend.
        """
        def week_of(column):
            return func.strftime("%Y-W%W", column)

        escalations = self._escalation_cost_by_period(week_of)

        with self.get_session() as session:
            results = (
                session.query(
                    week_of(RequestLog.timestamp).label("week"),
                    func.sum(RequestLog.cost),
                    func.sum(RequestLog.hypothetical_cost),
                    func.count(RequestLog.id),
                )
                .group_by("week")
                .order_by(func.min(RequestLog.timestamp).desc())
                .limit(weeks)
                .all()
            )
            out = []
            for week, routed, hypothetical, count in reversed(results):
                escalation = escalations.get(str(week), 0.0)
                actual = (routed or 0.0) + escalation
                out.append(
                    {
                        "week": week,
                        "routed_cost": round(routed or 0.0, 6),
                        "escalation_cost": round(escalation, 6),
                        "actual_cost": round(actual, 6),
                        "hypothetical_cost": round(hypothetical or 0.0, 6),
                        "savings": round((hypothetical or 0.0) - actual, 6),
                        "request_count": count,
                    }
                )
            return out

    def get_escalation_rate_over_time(self, days: int = 30) -> list[dict]:
        """Daily escalation rate — the trend that shows routing improving."""
        with self.get_session() as session:
            results = (
                session.query(
                    func.date(RequestLog.timestamp).label("date"),
                    func.count(RequestLog.id),
                    func.sum(
                        case((RequestLog.was_escalated == True, 1), else_=0)  # noqa: E712
                    ),
                    func.sum(
                        case((RequestLog.was_verified == True, 1), else_=0)  # noqa: E712
                    ),
                )
                .group_by(func.date(RequestLog.timestamp))
                .order_by(func.date(RequestLog.timestamp).desc())
                .limit(days)
                .all()
            )
            out = []
            for date, total, escalated, verified in reversed(results):
                escalated = escalated or 0
                verified = verified or 0
                out.append(
                    {
                        "date": str(date),
                        "request_count": total,
                        "escalated_count": escalated,
                        "verified_count": verified,
                        # Rate is over VERIFIED requests: unverified ones were
                        # never given a chance to escalate, so including them
                        # would understate the true failure rate.
                        "escalation_rate": (
                            round(escalated / verified * 100, 2) if verified else 0.0
                        ),
                    }
                )
            return out

    def get_verification_stats(self) -> dict:
        """Aggregates over the verification and escalation tables."""
        with self.get_session() as session:
            verified = (
                session.query(func.count(RequestLog.id))
                .filter(RequestLog.was_verified == True)  # noqa: E712
                .scalar()
                or 0
            )
            failed = (
                session.query(func.count(RequestLog.id))
                .filter(RequestLog.quality_gap > 0)
                .scalar()
                or 0
            )
            escalations = session.query(func.count(EscalationEvent.id)).scalar() or 0
            escalation_spend = (
                session.query(func.sum(EscalationEvent.cost_delta)).scalar() or 0.0
            )
            avg_gap = (
                session.query(func.avg(EscalationEvent.quality_gap)).scalar() or 0.0
            )
            served = (
                session.query(func.count(EscalationEvent.id))
                .filter(EscalationEvent.returned_to_user == True)  # noqa: E712
                .scalar()
                or 0
            )

            by_use_case = (
                session.query(
                    RequestLog.use_case,
                    func.count(RequestLog.id),
                    func.avg(RequestLog.quality_score),
                )
                .filter(RequestLog.was_verified == True)  # noqa: E712
                .group_by(RequestLog.use_case)
                .all()
            )

            return {
                "verified_count": verified,
                "failed_count": failed,
                "pass_rate": (
                    round((verified - failed) / verified * 100, 1) if verified else 0.0
                ),
                "escalation_count": escalations,
                "escalations_served": served,
                "escalation_cost_delta": round(escalation_spend, 6),
                "avg_quality_gap": round(avg_gap, 3),
                "by_use_case": {
                    (row[0] or "unknown"): {
                        "count": row[1],
                        "avg_quality": round(row[2], 2) if row[2] else None,
                    }
                    for row in by_use_case
                },
            }
