"""Mid-turn primary-runtime restore after a transient failover.

``restore_primary_runtime`` only runs at the start of a turn, so one long
agentic turn that fails over on a connection blip rides out the whole turn on
the fallback runtime even after the primary recovers (2026-10-09: three
connection errors at 09:58 → 75 Opper calls until 10:30 while OpenRouter was
healthy). ``fallback.mid_turn_restore_seconds`` (default 300, 0 = legacy off)
makes the retry loop re-attempt the primary before an API call once that many
seconds passed since a TRANSIENT activation (connection error / timeout / 5xx).
Billing (402), rate-limit and entitlement activations keep the legacy
turn-scoped behaviour — restoring mid-turn would replay the original failure.
The gates inside ``restore_primary_runtime`` (rate-limit cooldown, entitlement,
reset gate) still apply; a gated or failed re-attempt re-arms the timer so the
next try waits another full window instead of firing on every API call.
"""

from __future__ import annotations

import logging
import time

from agent.error_classifier import FailoverReason

logger = logging.getLogger(__name__)

# Failures worth re-attempting the primary mid-turn: the transport or the
# endpoint hiccuped, not the account. 402 / rate-limit / entitlement stay
# turn-scoped. ``unknown`` and reason-less activations (``try_activate_fallback()``
# callers that pass no reason) are retryable by definition.
TRANSIENT_FAILOVER_REASONS = frozenset({
    FailoverReason.timeout,       # connection / read timeouts incl. APIConnectionError
    FailoverReason.overloaded,   # 503 / 529
    FailoverReason.server_error,  # 500 / 502
    FailoverReason.unknown,       # unclassified, but retryable
})

DEFAULT_MID_TURN_RESTORE_SECONDS = 300


def reason_is_transient(reason) -> bool:
    """Would a fallback activated for this reason be worth re-attempting mid-turn?

    ``None`` (reason-less activation) counts as transient: the classifier's
    ``unknown`` verdict is retryable by definition.
    """
    return reason is None or reason in TRANSIENT_FAILOVER_REASONS


def record_fallback_activation(agent, reason) -> None:
    """Stamp a real fallback activation with its time and transient verdict.

    Called from ``try_activate_fallback`` on every successful switch, so the
    mid-turn restore can later decide — within the same turn — whether the
    primary is worth re-attempting. ``/model --once`` restorations reuse
    ``_fallback_activated`` but never pass through here, so they carry no
    provenance and are never touched mid-turn.
    """
    agent._provider_fallback_reason = reason
    agent._provider_fallback_transient = reason_is_transient(reason)
    agent._provider_fallback_activated_at = time.monotonic()


def maybe_restore_primary_mid_turn(agent) -> bool:
    """Re-attempt the primary runtime before an API call, once per elapsed window.

    Returns True only when the primary runtime was restored. Called from the
    top of the API retry loop (``agent/conversation_loop.py``), so each API
    call attempt in a turn checks the window. A gated (rate-limit cooldown,
    entitlement, reset gate) or failed re-attempt re-arms
    ``_provider_fallback_activated_at`` and stays on the fallback; a successful
    restore clears ``_provider_fallback_active`` itself, so no repeated firing.
    """
    window = getattr(agent, "_mid_turn_restore_seconds", DEFAULT_MID_TURN_RESTORE_SECONDS) or 0
    if window <= 0:
        return False  # legacy: the next turn's start-of-turn restore handles it
    # Only a real provider failover carries provenance (``/model --once`` also
    # reuses ``_fallback_activated``).
    if not getattr(agent, "_provider_fallback_active", False):
        return False
    if not getattr(agent, "_provider_fallback_transient", False):
        # 402 / rate-limit / entitlement: restoring mid-turn would replay the
        # original failure every window. Ride the turn out on the fallback.
        return False
    activated_at = getattr(agent, "_provider_fallback_activated_at", None)
    if activated_at is None or time.monotonic() - activated_at < window:
        return False
    restore = getattr(agent, "_restore_primary_runtime", None)
    if not callable(restore):
        return False
    logger.info(
        "Mid-turn primary restore re-attempt: fallback active %ss (activation reason: %s)",
        window, getattr(getattr(agent, "_provider_fallback_reason", None), "value", None),
    )
    try:
        restored = bool(restore())
    except Exception as exc:  # never let the restore attempt kill the API call
        logger.warning("Mid-turn primary restore re-attempt raised: %s", exc)
        restored = False
    # Re-arm whether the gates deflected us (stay on the fallback, retry next
    # window) or the restore raised: a re-attempt must never fire per API call.
    agent._provider_fallback_activated_at = time.monotonic()
    return restored