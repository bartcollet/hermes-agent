"""Anthropic org spend limit (HTTP 400 "specified API usage limits"): the sole key is benched
until the reset time Anthropic states, and the next turn goes straight to the fallback with
zero primary calls. Offline: mocked client, tmp HERMES_HOME, no network."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from agent.error_classifier import FailoverReason, classify_api_error
from run_agent import AIAgent

# Dynamic future instant (the parser ignores past times), minute precision.
_FUTURE = (datetime.now(UTC) + timedelta(days=30)).replace(second=0, microsecond=0)
_MSG = (
    "You have reached your specified API usage limits. "
    f"You will regain access on {_FUTURE:%Y-%m-%d} at {_FUTURE:%H:%M} UTC."
)
_RESET_AT = int(_FUTURE.timestamp())


class _Err(Exception):
    def __init__(self, message, status_code, body):
        super().__init__(message)
        self.status_code = status_code
        self.body = body
        self.response = MagicMock(headers={})


def _spend_limit_error():
    body = {"type": "error", "error": {"type": "invalid_request_error", "message": _MSG}}
    return _Err(_MSG, 400, body)


def _load_sole_anthropic_pool(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir(parents=True)
    entry = {
        "id": "cred-1", "label": "cred-1", "auth_type": "api_key", "priority": 0,
        "source": "manual", "access_token": "sk-ant-api03-test",
        "base_url": "https://api.anthropic.com",
    }
    (home / "auth.json").write_text(json.dumps({"version": 1, "credential_pool": {"anthropic": [entry]}}))
    monkeypatch.setenv("HERMES_HOME", str(home))
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    from agent.credential_pool import load_pool
    return load_pool("anthropic")


def _bench(pool, error_context):
    classified = classify_api_error(_spend_limit_error(), provider="anthropic", model="claude-sonnet-5-5")
    assert classified.reason == FailoverReason.billing
    ctx = error_context if error_context is not None else classified.error_context
    pool.mark_exhausted_and_rotate(
        status_code=400, error_context=ctx, failure_reason=classified.reason.value, model="claude-sonnet-5-5",
    )


def test_billing_with_reset_at_benches_sole_credential_until_reset(tmp_path, monkeypatch):
    pool = _load_sole_anthropic_pool(tmp_path, monkeypatch)
    _bench(pool, None)  # reset_at comes from the classifier, as in the live path
    assert pool.has_available() is False
    assert pool.select() is None
    assert pool.next_available_at() == _RESET_AT


def test_billing_without_reset_at_gets_the_one_hour_bench(tmp_path, monkeypatch):
    pool = _load_sole_anthropic_pool(tmp_path, monkeypatch)
    before = time.time()
    _bench(pool, {})
    until = pool.next_available_at()
    assert until is not None
    assert before + 3600 - 5 <= until <= time.time() + 3600 + 5
    assert pool.has_available() is False


def test_next_turn_makes_zero_primary_calls_and_uses_first_fallback(tmp_path, monkeypatch):
    """Benched sole credential + fallback_activated: turn-start restore stays on the fallback."""
    pool = _load_sole_anthropic_pool(tmp_path, monkeypatch)
    _bench(pool, None)

    primary_client = MagicMock(name="primary_anthropic_client")
    agent = AIAgent.__new__(AIAgent)
    agent.model, agent.provider = "fallback-model", "fallback-provider"
    agent.base_url = "https://fallback.example.com/v1"
    agent._credential_pool = pool
    agent._fallback_activated = True
    agent._fallback_index = 1
    agent._rate_limited_until = 0
    agent._primary_runtime = {
        "model": "claude-sonnet-5-5", "provider": "anthropic", "base_url": "https://api.anthropic.com",
        "api_mode": "anthropic_messages", "api_key": "sk-ant-api03-test",
        "client_kwargs": {"api_key": "sk-ant-api03-test", "base_url": "https://api.anthropic.com"},
    }
    agent._create_openai_client = MagicMock(return_value=primary_client)
    agent._replace_primary_openai_client = MagicMock(return_value=True)
    agent._apply_client_headers_for_base_url = MagicMock()

    assert agent._restore_primary_runtime() is False
    assert agent.provider == "fallback-provider"
    assert agent._fallback_activated is True
    assert agent._fallback_index == 1
    agent._create_openai_client.assert_not_called()
    agent._replace_primary_openai_client.assert_not_called()
    primary_client.messages.create.assert_not_called()
    primary_client.chat.completions.create.assert_not_called()
