"""FastAPI service — the router exposed over HTTP.

    POST /v1/completions          route a prompt, get back the answer + why
    GET  /v1/models               available models and their costs
    GET  /v1/stats                cost savings summary
    PUT  /v1/routing-config       remap tiers to models without a redeploy

Run it with:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.api.schemas import (
    CompletionRequest,
    CompletionResponse,
    CostMetadata,
    EscalationResponse,
    ModelInfo,
    ModelsResponse,
    RoutingConfigResponse,
    RoutingConfigUpdate,
    RoutingMetadata,
    SavingsSummary,
    StatsResponse,
    UsageMetadata,
)
from src.api.security import (
    check_startup_policy,
    enforce_context_window,
    require_api_key,
)
from src.models.providers.mode import ProviderConfigurationError
from src.models.providers.tokens import count_tokens
from src.service import AutopilotService

logger = logging.getLogger(__name__)

#: Populated on startup. Module-level so tests can swap it out.
service: Optional[AutopilotService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Bring the service up with the worker and retrain scheduler running."""
    global service
    # Refuses to start if live providers would be exposed without auth.
    check_startup_policy()
    service = AutopilotService()

    # Verification always runs in-process: the queue is an in-memory
    # asyncio.Queue, so it cannot be consumed from another container. The
    # retrain scheduler is separable, and under docker-compose it lives in the
    # worker container instead — hence the env switch.
    run_scheduler = os.getenv("RUN_RETRAIN_SCHEDULER", "true").lower() == "true"
    await service.start(with_scheduler=run_scheduler)
    logger.info(
        f"LLM Cost AutoPilot API ready (retrain scheduler: "
        f"{'on' if run_scheduler else 'off — handled by the worker'})"
    )
    try:
        yield
    finally:
        await service.stop()
        logger.info("LLM Cost AutoPilot API stopped")


app = FastAPI(
    title="LLM Cost AutoPilot",
    description=(
        "An intelligent routing layer that sends each prompt to the cheapest "
        "model capable of handling it, verifies the result against a "
        "high-tier reference, and escalates when quality falls short."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _require_service() -> AutopilotService:
    if service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is still starting up",
        )
    return service


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# ── Health ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["meta"])
async def health() -> dict:
    ready = service is not None
    return {
        "status": "ok" if ready else "starting",
        "classifier_trained": (
            service._classifier.is_trained if ready else False
        ),
        "mock_providers": os.getenv("MOCK_PROVIDERS", "true").lower() == "true",
    }


# ── 5.1 Completions ─────────────────────────────────────────────────────

@app.post(
    "/v1/completions",
    dependencies=[Depends(require_api_key)],
    response_model=CompletionResponse,
    tags=["completions"],
    summary="Route a completion request to the cheapest capable model",
)
async def create_completion(body: CompletionRequest) -> CompletionResponse:
    """Route a prompt and return the answer with full routing metadata.

    The caller does not pick a model. The classifier assigns a complexity
    tier, the router maps that tier to a model, and the response explains
    what was chosen and why.
    """
    svc = _require_service()
    prompt = body.to_prompt()

    # Reject a prompt no model could serve, before paying to find out. The
    # tier isn't known until classification, so this checks the tightest
    # window in the registry — a backstop beneath the payload size limits,
    # and the check that starts mattering if a small-context model is added.
    prompt_tokens = count_tokens(prompt)
    tightest = min(
        svc.engine.registry.list_all(),
        key=lambda m: m.context_window - m.max_tokens,
    )
    enforce_context_window(prompt_tokens, tightest)

    try:
        routed = await svc.ask(
            prompt=prompt,
            max_tokens=body.max_tokens,
            temperature=body.temperature,
        )
    except ProviderConfigurationError:
        # A deployment problem, not a caller problem — and its message is safe
        # to surface because it is written by us, not by a provider SDK.
        logger.exception("Provider misconfigured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model providers are not correctly configured.",
        )
    except Exception:
        # Deliberately opaque: provider exceptions routinely embed request
        # URLs, org identifiers and occasionally key fragments.
        logger.exception("Routing failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream model providers are unavailable.",
        )

    response, decision = routed.response, routed.decision
    savings = routed.hypothetical_cost - response.cost
    savings_pct = (
        (savings / routed.hypothetical_cost * 100)
        if routed.hypothetical_cost > 0
        else 0.0
    )

    return CompletionResponse(
        id=routed.log_id or 0,
        content=response.text,
        latency_ms=round(response.latency_ms, 2),
        routing=RoutingMetadata(
            tier=decision.tier.label,
            tier_confidence=decision.confidence,
            selected_model=decision.selected_model,
            provider=response.provider,
            is_fallback=decision.is_fallback,
            reasoning=decision.reasoning,
            use_case=routed.use_case or svc.detect_use_case(prompt).value,
        ),
        usage=UsageMetadata(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        ),
        cost=CostMetadata(
            actual_cost=round(response.cost, 8),
            hypothetical_cost=round(routed.hypothetical_cost, 8),
            savings=round(savings, 8),
            savings_percentage=round(savings_pct, 2),
        ),
    )


@app.get(
    "/v1/completions/{log_id}/escalation",
    dependencies=[Depends(require_api_key)],
    response_model=EscalationResponse,
    tags=["completions"],
    summary="Collect the improved answer, if verification escalated this request",
)
async def get_escalation(log_id: int) -> EscalationResponse:
    """Verification runs after the response is returned, so an escalated
    (better) answer arrives slightly later. Poll here to collect it.
    """
    svc = _require_service()
    result = svc.get_escalated_result(log_id)

    if result is None:
        return EscalationResponse(
            request_log_id=log_id,
            escalated=False,
            reason=(
                "No escalation recorded — the response passed verification, "
                "was not sampled for it, or verification has not finished yet."
            ),
        )

    return EscalationResponse(
        request_log_id=log_id,
        escalated=True,
        content=result.new_response.text,
        original_model=result.original_model,
        escalated_model=result.escalated_model,
        cost_delta=round(result.cost_delta, 8),
        quality_gap=result.quality_gap,
        served_within_latency_budget=result.returned_to_user,
        reason=result.trigger_reason,
    )


# ── 5.2 Configuration endpoints ─────────────────────────────────────────

@app.get(
    "/v1/models",
    dependencies=[Depends(require_api_key)],
    response_model=ModelsResponse,
    tags=["config"],
    summary="List available models and their costs",
)
async def list_models() -> ModelsResponse:
    svc = _require_service()
    registry = svc.engine.registry
    models = [ModelInfo(**m.to_dict()) for m in registry.list_all()]

    return ModelsResponse(
        models=sorted(models, key=lambda m: m.cost_per_1m_output_tokens),
        count=len(models),
        cheapest=registry.get_cheapest().name,
        most_expensive=registry.get_highest_quality().name,
    )


@app.get(
    "/v1/stats",
    dependencies=[Depends(require_api_key)],
    response_model=StatsResponse,
    tags=["config"],
    summary="Cost savings summary — the headline number",
)
async def get_stats() -> StatsResponse:
    svc = _require_service()
    stats = svc.stats()
    requests = stats["requests"]

    return StatsResponse(
        savings=SavingsSummary(
            total_requests=requests["total_requests"],
            routed_cost=requests["routed_cost"],
            escalation_cost=requests["escalation_cost"],
            total_cost=requests["total_cost"],
            baseline_cost=requests["total_hypothetical_cost"],
            total_savings=requests["total_savings"],
            cost_reduction_percentage=requests["savings_percentage"],
        ),
        model_distribution=requests["model_distribution"],
        tier_distribution=requests["tier_distribution"],
        avg_latency_ms=requests["avg_latency_ms"],
        avg_quality_score=requests["avg_quality_score"],
        verification=stats["verification"],
        feedback=stats["feedback"],
    )


@app.get(
    "/v1/routing-config",
    dependencies=[Depends(require_api_key)],
    response_model=RoutingConfigResponse,
    tags=["config"],
)
async def get_routing_config() -> RoutingConfigResponse:
    svc = _require_service()
    return RoutingConfigResponse(routing=svc.engine.get_routing_config()["routing"])


@app.put(
    "/v1/routing-config",
    dependencies=[Depends(require_api_key)],
    response_model=RoutingConfigResponse,
    tags=["config"],
    summary="Remap tiers to models without redeploying",
)
async def update_routing_config(
    body: RoutingConfigUpdate,
) -> RoutingConfigResponse:
    """Swap the model behind a tier at runtime.

    Every referenced model is validated against the registry first, so a typo
    cannot leave the router pointing at a model that does not exist.
    """
    svc = _require_service()
    registry = svc.engine.registry

    unknown = []
    for tier_key, tier in body.routing.items():
        for role, name in (("primary", tier.primary), ("fallback", tier.fallback)):
            if name is not None and name not in registry:
                unknown.append(f"{tier_key}.{role}='{name}'")

    if unknown:
        raise HTTPException(
            # Literal rather than the status constant, whose name differs
            # across Starlette versions.
            status_code=422,
            detail=(
                f"Unknown models: {', '.join(unknown)}. "
                f"Available: {registry.model_names}"
            ),
        )

    # Merge over the existing config so a partial update keeps untouched tiers.
    current = svc.engine.get_routing_config()
    merged = dict(current.get("routing", {}))
    for tier_key, tier in body.routing.items():
        existing = dict(merged.get(tier_key, {}))
        existing.update(tier.model_dump(exclude_none=True))
        merged[tier_key] = existing

    svc.engine.update_routing_config({"routing": merged})
    logger.info(f"Routing config updated for tiers: {sorted(body.routing)}")

    return RoutingConfigResponse(routing=merged, updated=True)
