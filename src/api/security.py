"""API authentication and request limits.

This service spends money on the caller's behalf, and a single request can fan
out to three model calls (routed, verification reference, escalation). An
unauthenticated endpoint is therefore not just an access problem — it is a
billing one.

Policy:
  * AUTOPILOT_API_KEY set          → every /v1 route requires it.
  * unset, MOCK_PROVIDERS=true     → open, with a startup warning (local dev,
                                     where no real spend is possible).
  * unset, MOCK_PROVIDERS=false    → refuse to start. Serving live, paid
                                     traffic with no auth is never the intent.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException, status

from src.models.providers.mode import mock_requested

logger = logging.getLogger(__name__)

API_KEY_ENV = "AUTOPILOT_API_KEY"

#: Total prompt characters accepted in one request. Roughly 60k tokens, so it
#: fits every model in the registry while refusing the multi-megabyte payloads
#: that turn one request into dollars of fan-out.
MAX_PROMPT_CHARS = 250_000
MAX_MESSAGES = 50


def configured_key() -> str:
    return os.getenv(API_KEY_ENV, "").strip()


def check_startup_policy() -> None:
    """Refuse to serve live traffic without authentication."""
    if configured_key():
        logger.info("API authentication enabled")
        return

    if mock_requested():
        logger.warning(
            "No %s set — the API is UNAUTHENTICATED. Acceptable with mock "
            "providers; set a key before pointing this at real models.",
            API_KEY_ENV,
        )
        return

    raise RuntimeError(
        f"Refusing to start: MOCK_PROVIDERS is false and {API_KEY_ENV} is not "
        f"set, which would expose paid model calls to anyone who can reach "
        f"this service. Set {API_KEY_ENV}, or set MOCK_PROVIDERS=true."
    )


async def require_api_key(
    authorization: str = Header(default=""),
    x_api_key: str = Header(default=""),
) -> None:
    """Authenticate a request when a key is configured.

    Accepts `Authorization: Bearer <key>` or `X-API-Key: <key>`.
    """
    expected = configured_key()
    if not expected:
        return  # open mode; check_startup_policy already decided this is ok

    presented = x_api_key.strip()
    if not presented and authorization.lower().startswith("bearer "):
        presented = authorization[7:].strip()

    # Constant-time comparison so a wrong key can't be recovered by timing.
    if not presented or not secrets.compare_digest(presented, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def enforce_context_window(prompt_tokens: int, model_config) -> None:
    """Reject a prompt that cannot fit the model chosen for it.

    The registry declares a context window per model but nothing used to check
    it, so an oversized prompt was billed on an attempt that could only fail.
    """
    budget = model_config.context_window - model_config.max_tokens
    if prompt_tokens > budget:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Prompt is {prompt_tokens:,} tokens; {model_config.name} allows "
                f"{budget:,} once its {model_config.max_tokens:,}-token reply is "
                f"reserved from a {model_config.context_window:,}-token window."
            ),
        )
