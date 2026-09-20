"""Phase 5 — the HTTP surface."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client backed by a throwaway database, with everything verified."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/api.db")
    monkeypatch.setenv("MOCK_PROVIDERS", "true")
    monkeypatch.setenv("VERIFICATION_SAMPLE_RATE", "1.0")

    from src.api import main

    with TestClient(main.app) as c:
        yield c


def _complete(client, content: str = "What is a variable in Python?"):
    return client.post(
        "/v1/completions", json={"messages": [{"role": "user", "content": content}]}
    )


# ── Health ──────────────────────────────────────────────────────────────

def test_health_reports_ready(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["classifier_trained"] is True


# ── 5.1 POST /v1/completions ────────────────────────────────────────────

def test_completion_returns_an_answer(client):
    r = _complete(client)
    assert r.status_code == 200
    body = r.json()
    assert body["content"]
    assert body["id"] > 0


def test_completion_explains_which_model_and_why(client):
    body = _complete(client).json()["routing"]
    assert body["selected_model"]
    assert body["tier"] in ("simple", "moderate", "complex")
    assert body["reasoning"]
    assert body["use_case"]
    assert 0.0 <= max(body["tier_confidence"].values()) <= 1.0


def test_completion_reports_the_savings(client):
    cost = _complete(client).json()["cost"]
    assert cost["hypothetical_cost"] >= cost["actual_cost"]
    assert cost["savings"] == pytest.approx(
        cost["hypothetical_cost"] - cost["actual_cost"], abs=1e-9
    )


def test_caller_cannot_choose_the_model(client):
    """A `model` field is ignored — routing is the service's job."""
    r = client.post(
        "/v1/completions",
        json={
            "messages": [{"role": "user", "content": "What is a list?"}],
            "model": "gpt-4o",
        },
    )
    assert r.status_code == 200
    # The router picked for itself rather than honoring the hint.
    assert "model" not in r.json()


def test_multi_turn_conversation_is_accepted(client):
    r = client.post(
        "/v1/completions",
        json={
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "What is a tuple?"},
            ]
        },
    )
    assert r.status_code == 200


def test_empty_messages_rejected(client):
    assert client.post("/v1/completions", json={"messages": []}).status_code == 422


def test_assistant_only_conversation_rejected(client):
    r = client.post(
        "/v1/completions",
        json={"messages": [{"role": "assistant", "content": "hi"}]},
    )
    assert r.status_code == 422


def test_out_of_range_temperature_rejected(client):
    r = client.post(
        "/v1/completions",
        json={"messages": [{"role": "user", "content": "hi"}], "temperature": 9.0},
    )
    assert r.status_code == 422


# ── Escalation polling ──────────────────────────────────────────────────

def test_escalation_endpoint_reports_when_there_is_none(client):
    log_id = _complete(client).json()["id"]
    body = client.get(f"/v1/completions/{log_id}/escalation").json()
    assert body["request_log_id"] == log_id
    assert isinstance(body["escalated"], bool)


def test_escalation_endpoint_handles_unknown_id(client):
    body = client.get("/v1/completions/999999/escalation").json()
    assert body["escalated"] is False


# ── 5.2 GET /v1/models ──────────────────────────────────────────────────

def test_models_lists_costs(client):
    body = client.get("/v1/models").json()
    assert body["count"] == len(body["models"])
    assert body["cheapest"] and body["most_expensive"]
    for m in body["models"]:
        assert m["cost_per_1m_input_tokens"] >= 0
        assert m["provider"] in ("openai", "anthropic", "ollama")


def test_models_sorted_cheapest_first(client):
    models = client.get("/v1/models").json()["models"]
    costs = [m["cost_per_1m_output_tokens"] for m in models]
    assert costs == sorted(costs)


# ── 5.2 GET /v1/stats ───────────────────────────────────────────────────

def test_stats_reports_the_headline_number(client):
    _complete(client)
    savings = client.get("/v1/stats").json()["savings"]
    assert savings["total_requests"] >= 1
    assert savings["baseline_cost"] >= savings["total_cost"]
    assert 0 <= savings["cost_reduction_percentage"] <= 100


def test_stats_includes_distributions(client):
    _complete(client)
    body = client.get("/v1/stats").json()
    assert body["model_distribution"]
    assert body["tier_distribution"]
    assert "verification" in body and "feedback" in body


# ── 5.2 PUT /v1/routing-config ──────────────────────────────────────────

def test_routing_config_can_be_read(client):
    routing = client.get("/v1/routing-config").json()["routing"]
    assert {"tier_1", "tier_2", "tier_3"} <= set(routing)


def test_routing_config_update_takes_effect(client):
    r = client.put(
        "/v1/routing-config",
        json={"routing": {"tier_1": {"primary": "claude-haiku-4-5"}}},
    )
    assert r.status_code == 200
    assert r.json()["updated"] is True
    assert r.json()["routing"]["tier_1"]["primary"] == "claude-haiku-4-5"

    # And it survives into subsequent reads.
    after = client.get("/v1/routing-config").json()["routing"]
    assert after["tier_1"]["primary"] == "claude-haiku-4-5"


def test_partial_update_leaves_other_tiers_alone(client):
    before = client.get("/v1/routing-config").json()["routing"]
    client.put(
        "/v1/routing-config",
        json={"routing": {"tier_1": {"primary": "claude-haiku-4-5"}}},
    )
    after = client.get("/v1/routing-config").json()["routing"]
    assert after["tier_3"] == before["tier_3"]


def test_partial_update_keeps_untouched_fields_within_a_tier(client):
    before = client.get("/v1/routing-config").json()["routing"]["tier_1"]
    client.put(
        "/v1/routing-config",
        json={"routing": {"tier_1": {"primary": "claude-haiku-4-5"}}},
    )
    after = client.get("/v1/routing-config").json()["routing"]["tier_1"]
    assert after["fallback"] == before["fallback"]


def test_unknown_model_is_rejected(client):
    """A typo must not leave the router pointing at a nonexistent model."""
    r = client.put(
        "/v1/routing-config",
        json={"routing": {"tier_1": {"primary": "gpt-5-turbo-ultra"}}},
    )
    assert r.status_code == 422
    assert "gpt-5-turbo-ultra" in r.json()["detail"]


def test_unknown_model_in_fallback_is_rejected(client):
    r = client.put(
        "/v1/routing-config",
        json={
            "routing": {
                "tier_1": {"primary": "llama3.2", "fallback": "nonexistent-model"}
            }
        },
    )
    assert r.status_code == 422


def test_unknown_tier_key_is_rejected(client):
    r = client.put(
        "/v1/routing-config",
        json={"routing": {"tier_9": {"primary": "gpt-4o"}}},
    )
    assert r.status_code == 422


def test_rejected_update_does_not_change_anything(client):
    before = client.get("/v1/routing-config").json()["routing"]
    client.put(
        "/v1/routing-config",
        json={"routing": {"tier_1": {"primary": "does-not-exist"}}},
    )
    assert client.get("/v1/routing-config").json()["routing"] == before
