"""Request and response schemas for the public API."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ── POST /v1/completions ────────────────────────────────────────────────

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class CompletionRequest(BaseModel):
    """A standard chat-completion request.

    Deliberately has no `model` field: choosing the model is the whole point
    of this service. Callers describe what they want, the router decides how
    to serve it.
    """

    messages: list[Message] = Field(min_length=1)
    max_tokens: Optional[int] = Field(default=None, gt=0, le=32_000)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("messages")
    @classmethod
    def must_contain_a_user_message(cls, messages: list[Message]) -> list[Message]:
        if not any(m.role == "user" for m in messages):
            raise ValueError("at least one message must have role 'user'")
        return messages

    def to_prompt(self) -> str:
        """Flatten the conversation into the single prompt the router sees."""
        if len(self.messages) == 1:
            return self.messages[0].content
        return "\n\n".join(f"{m.role}: {m.content}" for m in self.messages)


class RoutingMetadata(BaseModel):
    """Why this request went where it went."""

    tier: str
    tier_confidence: dict[str, float]
    selected_model: str
    provider: str
    is_fallback: bool
    reasoning: str
    use_case: str


class UsageMetadata(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class CostMetadata(BaseModel):
    actual_cost: float
    hypothetical_cost: float = Field(
        description="What this request would have cost on the most expensive model"
    )
    savings: float
    savings_percentage: float


class CompletionResponse(BaseModel):
    id: int = Field(description="Request log id — use it to poll for escalations")
    content: str
    latency_ms: float
    routing: RoutingMetadata
    usage: UsageMetadata
    cost: CostMetadata


# ── GET /v1/models ──────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name: str
    provider: str
    display_name: str
    cost_per_1m_input_tokens: float
    cost_per_1m_output_tokens: float
    context_window: int
    quality_tier: str
    avg_latency_ms: int


class ModelsResponse(BaseModel):
    models: list[ModelInfo]
    count: int
    cheapest: str
    most_expensive: str


# ── GET /v1/stats ───────────────────────────────────────────────────────

class SavingsSummary(BaseModel):
    """The headline number."""

    total_requests: int
    routed_cost: float = Field(description="Spend on the initially routed models")
    escalation_cost: float = Field(
        description="Extra spend re-running requests that failed verification"
    )
    total_cost: float = Field(description="routed_cost + escalation_cost")
    baseline_cost: float = Field(
        description="Cost if every request had gone to the most expensive model"
    )
    total_savings: float
    cost_reduction_percentage: float = Field(
        description="Net of escalation overhead, not before it"
    )


class StatsResponse(BaseModel):
    savings: SavingsSummary
    model_distribution: dict[str, int]
    tier_distribution: dict[str, int]
    avg_latency_ms: float
    avg_quality_score: float
    verification: dict
    feedback: dict


# ── PUT /v1/routing-config ──────────────────────────────────────────────

class TierRouting(BaseModel):
    primary: str
    fallback: Optional[str] = None
    description: Optional[str] = None


class RoutingConfigUpdate(BaseModel):
    """Remap tiers to models without a redeploy.

    Keys must be tier_1 / tier_2 / tier_3; model names are validated against
    the live registry before anything is applied.
    """

    routing: dict[str, TierRouting]

    @field_validator("routing")
    @classmethod
    def validate_tier_keys(
        cls, routing: dict[str, TierRouting]
    ) -> dict[str, TierRouting]:
        if not routing:
            raise ValueError("routing must not be empty")
        allowed = {"tier_1", "tier_2", "tier_3"}
        unknown = set(routing) - allowed
        if unknown:
            raise ValueError(
                f"unknown tier keys: {sorted(unknown)}; allowed: {sorted(allowed)}"
            )
        return routing


class RoutingConfigResponse(BaseModel):
    routing: dict
    updated: bool = False


# ── Escalation polling ──────────────────────────────────────────────────

class EscalationResponse(BaseModel):
    """The improved answer, when async verification produced one."""

    request_log_id: int
    escalated: bool
    content: Optional[str] = None
    original_model: Optional[str] = None
    escalated_model: Optional[str] = None
    cost_delta: Optional[float] = None
    quality_gap: Optional[float] = None
    served_within_latency_budget: Optional[bool] = None
    reason: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
