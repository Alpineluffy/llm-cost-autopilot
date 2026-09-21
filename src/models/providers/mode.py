"""Explicit resolution of mock vs. live provider mode.

Mock mode must be *asked for*, never arrived at by accident. The previous
behaviour inferred it — a missing key, an uninstalled SDK, or an unreachable
Ollama all silently flipped a provider to mock. A deployment with a typo in
an environment variable would then serve invented answers to users, labelled
with a real model id and a plausible cost, with nothing in the logs to say so.

The rule here: if MOCK_PROVIDERS is not set to true, a provider that cannot
reach its backend raises. Failing at startup is recoverable; fabricating
answers is not.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_TRUTHY = {"1", "true", "yes", "on"}

#: Values that mean "this is the .env.example placeholder, not a real key".
_PLACEHOLDER_PREFIXES = ("your-", "your_", "changeme", "xxx", "<")


class ProviderConfigurationError(RuntimeError):
    """A live provider was requested but cannot be used as configured."""


def mock_requested() -> bool:
    """True when the operator explicitly asked for mock providers."""
    return os.getenv("MOCK_PROVIDERS", "true").strip().lower() in _TRUTHY


def is_placeholder(credential: str) -> bool:
    lowered = credential.strip().lower()
    return any(lowered.startswith(p) for p in _PLACEHOLDER_PREFIXES)


def resolve_mock_mode(
    provider: str,
    credential: str = "",
    credential_env: str = "",
    requires_credential: bool = True,
) -> bool:
    """Decide whether `provider` runs mocked, or refuse to start.

    Returns True for mock mode. Raises ProviderConfigurationError when live
    mode was requested but the provider is not usable — rather than quietly
    degrading to fabricated output.
    """
    if mock_requested():
        logger.debug(f"{provider}: mock mode (MOCK_PROVIDERS is set)")
        return True

    if requires_credential:
        if not credential.strip():
            raise ProviderConfigurationError(
                f"{provider}: MOCK_PROVIDERS is false but {credential_env} is not "
                f"set. Set {credential_env}, or set MOCK_PROVIDERS=true to run "
                f"against simulated responses on purpose."
            )
        if is_placeholder(credential):
            raise ProviderConfigurationError(
                f"{provider}: {credential_env} still holds the placeholder value "
                f"from .env.example. Replace it with a real key, or set "
                f"MOCK_PROVIDERS=true."
            )

    return False


def require_sdk(provider: str, module: str, package: str):
    """Import a provider SDK, or explain exactly what to install."""
    try:
        return __import__(module, fromlist=["*"])
    except ImportError as exc:
        raise ProviderConfigurationError(
            f"{provider}: live mode needs the '{package}' package "
            f"(pip install {package}), or set MOCK_PROVIDERS=true."
        ) from exc
