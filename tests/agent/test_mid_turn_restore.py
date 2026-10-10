"""Mid-turn primary-runtime restore after a transient failover.

``restore_primary_runtime`` only ran at the start of a turn, so one long
agentic turn that failed over on a connection blip rode out the whole turn on
the fallback runtime even after the primary recovered (2026-10-09: three
connection errors at 09:58 → 75 Opper calls until 10:30 while OpenRouter was
healthy). ``fallback.mid_turn_restore_seconds`` makes the retry loop re-attempt
the primary before an API call once that many seconds passed since a TRANSIENT
activation. Billing (402) / rate-limit / entitlement activations and a window
of 0 keep the legacy turn-scoped behaviour.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _LoopState, _run_api_retry_loop
import agent.conversation_loop as _cl
from agent.error_classifier import FailoverReason
from agent.mid_turn_restore import (
    maybe_restore_primary_mid_turn,
    record_fallback_activation,
)
from run_agent import AIAgent


def _fallback_agent(*, window: int = 300, transient: bool = True,
                    active: bool = True, restore_returns: bool = True) -> SimpleNamespace:
    """Minimal agent carrying only the mid-turn restore provenance slots."""
    agent = SimpleNamespace(
        _mid_turn_restore_seconds=window,
        _provider_fallback_active=active,
        _provider_fallback_transient=transient,
        _provider_fallback_reason=FailoverReason.timeout if transient else FailoverReason.billing,
        _provider_fallback_activated_at=time.monotonic(),
        _restore_primary_runtime=MagicMock(),
    )

    def _restore():
        if restore_returns:
            # The real restore clears the provenance flags on success.
            agent._provider_fallback_active = False
        return restore_returns

    agent._restore_primary_runtime.side_effect = _restore
    return agent


class TestMaybeRestorePrimaryMidTurn:
    def test_transient_restores_once_window_elapsed(self):
        agent = _fallback_agent(window=300)
        agent._provider_fallback_activated_at = time.monotonic() - 301

        assert maybe_restore_primary_mid_turn(agent) is True
        agent._restore_primary_runtime.assert_called_once_with()

        # The restore cleared the provenance: a later API call in the same turn
        # does not re-attempt (and does not loop re-attempting every call).
        assert maybe_restore_primary_mid_turn(agent) is False
        agent._restore_primary_runtime.assert_called_once_with()

    def test_before_window_no_restore(self):
        agent = _fallback_agent(window=300)
        agent._provider_fallback_activated_at = time.monotonic() - 10

        assert maybe_restore_primary_mid_turn(agent) is False
        agent._restore_primary_runtime.assert_not_called()

    def test_billing_rate_limit_entitlement_never_restored_mid_turn(self):
        # 402 / 429 / entitlement: the account, not the transport, is the problem —
        # restoring mid-turn would replay the original failure every window.
        for reason in (FailoverReason.billing, FailoverReason.rate_limit,
                       FailoverReason.model_entitlement, FailoverReason.auth_permanent):
            agent = _fallback_agent(window=300, transient=False)
            agent._provider_fallback_reason = reason
            agent._provider_fallback_activated_at = time.monotonic() - 10_000

            assert maybe_restore_primary_mid_turn(agent) is False, reason
            agent._restore_primary_runtime.assert_not_called()

    def test_window_zero_keeps_legacy_behaviour(self):
        agent = _fallback_agent(window=0)
        agent._provider_fallback_activated_at = time.monotonic() - 10_000

        assert maybe_restore_primary_mid_turn(agent) is False
        agent._restore_primary_runtime.assert_not_called()

    def test_gated_restore_rearms_timer(self):
        # ``restore_primary_runtime`` returns False when its gates deflect
        # (rate-limit cooldown, entitlement, reset gate): stay on the fallback
        # and wait another full window instead of retrying on every API call.
        agent = _fallback_agent(window=300, restore_returns=False)
        armed_at = time.monotonic() - 301
        agent._provider_fallback_activated_at = armed_at

        assert maybe_restore_primary_mid_turn(agent) is False
        agent._restore_primary_runtime.assert_called_once_with()
        assert agent._provider_fallback_activated_at > armed_at

        # Re-armed: no second attempt until the window elapses again.
        assert maybe_restore_primary_mid_turn(agent) is False
        agent._restore_primary_runtime.assert_called_once_with()

    def test_model_once_temporary_restore_not_mid_turn_restored(self):
        # ``/model --once`` reuses ``_fallback_activated`` without provider-fallback
        # provenance; the mid-turn restore must not touch those sessions.
        agent = _fallback_agent(active=False)
        agent._fallback_activated = True
        agent._provider_fallback_activated_at = time.monotonic() - 10_000

        assert maybe_restore_primary_mid_turn(agent) is False
        agent._restore_primary_runtime.assert_not_called()

    def test_missing_provenance_slots_are_inert(self):
        agent = SimpleNamespace()
        assert maybe_restore_primary_mid_turn(agent) is False


class TestReasonVerdict:
    @pytest.mark.parametrize("reason,expected", [
        (None, True),  # unclassified/unknown is retryable by definition
        (FailoverReason.timeout, True),      # connection blibs / read timeouts
        (FailoverReason.overloaded, True),   # 503 / 529
        (FailoverReason.server_error, True),  # 500 / 502
        (FailoverReason.unknown, True),
        (FailoverReason.billing, False),
        (FailoverReason.rate_limit, False),
        (FailoverReason.upstream_rate_limit, False),
        (FailoverReason.model_entitlement, False),
        (FailoverReason.auth_permanent, False),
        (FailoverReason.ssl_cert_verification, False),
    ])
    def test_reason_is_transient(self, reason, expected):
        agent = SimpleNamespace()
        record_fallback_activation(agent, reason)
        assert agent._provider_fallback_transient is expected, reason
        assert agent._provider_fallback_reason is reason
        assert agent._provider_fallback_activated_at <= time.monotonic()


# Integration: the real ``try_activate_fallback`` stamps the provenance on
# every successful switch (fixture mirrors tests/agent/test_primary_runtime_restore.py).
def _make_tool_defs(*names: str) -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": f"{n} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for n in names
    ]


def _make_real_agent():
    with (
        patch("model_tools.get_tool_definitions", return_value=_make_tool_defs("web_search")),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
        patch("agent.context_compressor.get_model_context_length", return_value=200_000),
        patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
    ):
        agent = AIAgent(
            api_key="test-key-12345678",
            base_url="https://my-llm.example.com/v1",
            provider="custom",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model={"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
        )
        agent.client = MagicMock()
        return agent


def _mock_resolve(base_url="https://openrouter.ai/api/v1", api_key="fallback-key-1234"):
    mock_client = MagicMock()
    mock_client.api_key = api_key
    mock_client.base_url = base_url
    return mock_client


class TestActivationRecording:
    @pytest.mark.parametrize("reason,transient_expected,restore_expected", [
        (None, True, True),
        (FailoverReason.timeout, True, True),
        (FailoverReason.server_error, True, True),
        # Billing / rate-limit arm the primary rate-limit cooldown
        # (``_arm_rate_limit_cooldown``), which gates the start-of-turn restore.
        (FailoverReason.billing, False, False),
        (FailoverReason.rate_limit, False, False),
        # Entitlement stays out of the cooldown arming set (its rejection
        # markers are separate); the start-of-turn restore still runs.
        (FailoverReason.model_entitlement, False, True),
    ])
    def test_try_activate_fallback_stamps_provenance(self, reason, transient_expected, restore_expected):
        agent = _make_real_agent()
        agent.model = "primary-model"
        agent._primary_runtime["model"] = "primary-model"
        before = time.monotonic()
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_resolve(), None),
        ):
            assert agent._try_activate_fallback(reason=reason) is True

        assert agent._provider_fallback_active is True
        assert agent._provider_fallback_transient is transient_expected, reason
        assert agent._provider_fallback_reason is reason
        assert before <= agent._provider_fallback_activated_at <= time.monotonic()

        if not restore_expected:
            return
        # A transient (or non-cooldown) fallback turns over after a restore (the
        # entry's runtime is now current, so a re-walk would skip it): restore,
        # then a fresh activation re-stamps from its own reason.
        with patch("agent.process_bootstrap.OpenAI", return_value=MagicMock()):
            assert agent._restore_primary_runtime() is True
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_resolve(), None),
        ):
            assert agent._try_activate_fallback(reason=FailoverReason.billing) is True
        assert agent._provider_fallback_transient is False
        assert agent._provider_fallback_reason is FailoverReason.billing


# Wiring: the retry loop must attempt the mid-turn restore before each API call.
@dataclass
class _Fallthrough:
    action: str = "fallthrough"


@dataclass
class _Break:
    action: str = "break"


class TestRetryLoopWiring:
    def test_retry_loop_attempts_mid_turn_restore_before_api_call(self):
        agent = SimpleNamespace(provider="lmstudio", client=MagicMock())
        s = _LoopState(
            user_message="hi", system_message=None, moa_config=None,
            original_user_message="hi", conversation_history=[],
            effective_task_id=None, turn_id="t1", _should_review_memory=False,
            _plugin_user_context=None, _ext_prefetch_cache=None, messages=[],
            active_system_prompt="sys", current_turn_user_idx=0,
            _preflight_compression_blocked=False, max_compression_attempts=3,
        )
        s.retry_count, s.max_retries = 0, 1
        with (
            patch.object(_cl, "maybe_restore_primary_mid_turn") as mid,
            patch.object(_cl, "build_api_request", lambda agent: _Fallthrough()),
            patch.object(_cl, "perform_api_call", lambda agent: _Break()),
        ):
            assert _run_api_retry_loop(agent, s) is None
        mid.assert_called_once_with(agent)