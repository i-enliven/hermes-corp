"""Tests for the /topup command — shared view core + gateway handler.

``/topup`` is the focused money surface (balance in, top-up out) — the rehaul
folded the old ``/credits`` + ``/billing`` surfaces into it. These tests
exercise the surface-agnostic ``build_credits_view()`` fail-open contract via
the gateway handler and lock the command registry state (``/topup`` is the
only billing surface, alias-free, on every platform). The live portal path is
the pruned ``hermes_cli.nous_account`` client, so logged-in rendering is
fail-open offline (``logged_in=False``) and is covered by the gateway
not-logged-in test plus manual verification.
"""

from __future__ import annotations

import asyncio

import pytest

import agent.account_usage as account_usage
from agent.account_usage import CreditsView


# ── gateway _handle_topup_command (the messaging billing surface) ────────────


class _FakeEvent:
    pass


def _make_gateway_stub():
    """Minimal object exposing the mixin's _handle_topup_command."""
    from gateway.slash_commands import GatewaySlashCommandsMixin

    class _Stub(GatewaySlashCommandsMixin):
        def __init__(self):
            pass

    return _Stub()


def test_gateway_topup_not_logged_in(monkeypatch):
    monkeypatch.setattr(
        account_usage, "build_credits_view", lambda *a, **kw: CreditsView(logged_in=False)
    )
    stub = _make_gateway_stub()
    out = asyncio.run(stub._handle_topup_command(_FakeEvent()))
    assert "Not logged into Nous Portal" in out


# ── command registry ────────────────────────────────────────────────────────


def test_credits_command_fully_removed():
    """`/credits` and the old `/billing` are gone entirely — not commands, not
    aliases. Billing lives only on /topup, with NO aliases, on every platform."""
    from hermes_cli.commands import resolve_command, COMMAND_REGISTRY

    # Both old names resolve to nothing.
    assert resolve_command("credits") is None
    assert resolve_command("billing") is None
    # No standalone command for either remains in the registry.
    assert not any(c.name in ("credits", "billing") for c in COMMAND_REGISTRY)
    # And no command carries either as an alias.
    for c in COMMAND_REGISTRY:
        assert "credits" not in (c.aliases or ())
        assert "billing" not in (c.aliases or ())
    # /topup is the billing surface, on every surface, and carries no aliases.
    entry = next(c for c in COMMAND_REGISTRY if c.name == "topup")
    assert entry.cli_only is False
    assert entry.gateway_only is False
    assert not entry.aliases
