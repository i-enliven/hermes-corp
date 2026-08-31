"""Unit tests for the Remote Spending core (agent/billing_view.py).

Covers:
- Decimal money parsing/formatting (server emits decimal strings, not 2dp).
- BillingState payload parsing (role tiering, presets, bounds, sub-structs).
- Fail-open builder behavior.
- Idempotency key generation.
- Custom-amount validation against bounds + multipleOf 0.01.
- HERMES_DEV_BILLING_FIXTURE offline scaffolding.

No network: HTTP-layer tests monkeypatch the request function for the builder.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

import agent.billing_view as bv
from agent.billing_view import (
    AutoReload,
    AutoReloadCard,
    BillingState,
    CardInfo,
    MonthlyCap,
    PaymentMethodInfo,
    billing_state_from_payload,
    build_billing_state,
    format_money,
    new_idempotency_key,
    parse_money,
    validate_charge_amount,
)

# ---------------------------------------------------------------------------
# Decimal money
# ---------------------------------------------------------------------------


def test_parse_money_decimal_string():
    assert parse_money("142.5") == Decimal("142.5")
    assert parse_money("100") == Decimal("100")
    assert parse_money(None) is None
    assert parse_money("abc") is None
    assert parse_money("") is None


def test_format_money_display_rules():
    assert format_money(Decimal("142.5")) == "$142.50"
    assert format_money(Decimal("100")) == "$100"
    assert format_money(Decimal("0.01")) == "$0.01"
    assert format_money(None) == "—"


# ---------------------------------------------------------------------------
# BillingState payload parsing
# ---------------------------------------------------------------------------


def _member_payload() -> dict:
    return {
        "org": {"id": "o1", "slug": "acme", "name": "Acme", "role": "MEMBER"},
        "balanceUsd": "142.5",
        "cliBillingEnabled": True,
        "chargePresets": ["100", "250", "500"],
        "bounds": {"minUsd": "10", "maxUsd": "10000"},
        "card": None,
        "monthlyCap": None,
        "autoReload": None,
    }


def _owner_payload() -> dict:
    p = _member_payload()
    p["org"]["role"] = "OWNER"
    p["card"] = {"brand": "visa", "last4": "4242"}
    p["monthlyCap"] = {
        "limitUsd": "1000",
        "spentThisMonthUsd": "180",
        "isDefaultCeiling": True,
    }
    p["autoReload"] = {"enabled": True, "thresholdUsd": "20", "reloadToUsd": "100"}
    return p


def test_state_member_tier_parse():
    s = billing_state_from_payload(_member_payload())
    assert s.logged_in
    assert s.role == "MEMBER"
    assert s.balance_usd == Decimal("142.5")
    assert s.cli_billing_enabled is True
    assert s.charge_presets == (Decimal("100"), Decimal("250"), Decimal("500"))
    assert s.min_usd == Decimal("10") and s.max_usd == Decimal("10000")
    assert s.card is None and s.monthly_cap is None and s.auto_reload is None
    assert s.is_admin is False
    assert s.can_charge is False  # not admin


@pytest.mark.parametrize(
    "role,can_change_plan_raw,is_admin,can_change_plan",
    [
        ("OWNER", None, True, True),
        ("ADMIN", None, True, True),
        ("FINANCE_ADMIN", True, False, True),
        ("SECURITY_ADMIN", None, False, False),
        ("MEMBER", None, False, False),
    ],
)
def test_state_five_roles(role, can_change_plan_raw, is_admin, can_change_plan):
    payload = _member_payload()
    payload["org"]["role"] = role
    if can_change_plan_raw is not None:
        payload["canChangePlan"] = can_change_plan_raw

    state = billing_state_from_payload(payload)

    assert state.is_admin is is_admin
    assert state.can_change_plan_raw is can_change_plan_raw
    assert state.can_change_plan is can_change_plan


def test_state_owner_tier_parse():
    s = billing_state_from_payload(_owner_payload())
    assert s.is_admin is True
    assert s.can_charge is True  # admin + kill-switch on
    assert s.card == CardInfo(brand="visa", last4="4242")
    assert s.card is not None and s.card.masked == "visa ····4242"
    assert s.monthly_cap == MonthlyCap(
        limit_usd=Decimal("1000"),
        spent_this_month_usd=Decimal("180"),
        is_default_ceiling=True,
    )
    assert s.auto_reload == AutoReload(
        enabled=True, threshold_usd=Decimal("20"), reload_to_usd=Decimal("100")
    )


def test_state_payment_method_card_kind():
    p = _member_payload()
    p["paymentMethod"] = {"kind": "card", "brand": "visa", "last4": "4242"}
    s = billing_state_from_payload(p)
    pm = s.payment_method
    assert pm is not None and pm.kind == "card"
    assert pm.brand == "visa" and pm.last4 == "4242"

def test_state_payment_method_link_and_unknown_kinds():
    p = _member_payload()
    p["paymentMethod"] = {"kind": "link", "email": "x@y.com"}
    s = billing_state_from_payload(p)
    assert s.payment_method.kind == "link"
    p["paymentMethod"] = {"kind": "weird_thing", "brand": "visa"}
    s = billing_state_from_payload(p)
    pm = s.payment_method
    assert pm is not None and pm.kind == "unknown" and pm.raw_kind == "weird_thing"


def test_state_auto_reload_card_distinct():
    p = _member_payload()
    p["autoReload"] = {
        "enabled": True,
        "thresholdUsd": "20",
        "reloadToUsd": "100",
        "card": {"kind": "distinct", "paymentMethodId": "pm_1", "brand": "visa", "last4": "0002"},
    }
    s = billing_state_from_payload(p)
    ar = s.auto_reload
    assert ar is not None and ar.card == AutoReloadCard(
        kind="distinct", payment_method_id="pm_1", brand="visa", last4="0002"
    )


def test_card_provenance_label():
    c = CardInfo(brand="visa", last4="4242", resolved_via="subPin")
    assert c.provenance == "the card on your subscription"
    assert "the card on your subscription" in c.display
    assert CardInfo(brand="visa", last4="4242").provenance is None


# ---------------------------------------------------------------------------
# Fail-open builder
# ---------------------------------------------------------------------------


def test_build_billing_state_fail_open_when_client_missing(monkeypatch):
    """hermes_cli.nous_billing was pruned — the builder fail-opens cleanly."""
    monkeypatch.delenv("HERMES_DEV_BILLING_FIXTURE", raising=False)
    s = build_billing_state()
    assert s.logged_in is False
    assert (s.error or "") == "billing client unavailable"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_new_idempotency_key_unique_and_uuid_shaped():
    a, b = new_idempotency_key(), new_idempotency_key()
    assert a != b
    assert len(a) == 36 and a.count("-") == 4


# ---------------------------------------------------------------------------
# Amount validation (Screen 3 custom input)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,err_substr",
    [
        ("", "dollar amount"),
        ("0", "greater than"),
        ("-5", "greater than"),
        ("10.005", "cent"),       # multipleOf 0.01 — sub-cent rejected
        ("5", "Minimum"),         # below bounds.minUsd
        ("99999", "Maximum"),     # above bounds.maxUsd
    ],
)
def test_validate_amount_rejections(raw, err_substr):
    v = validate_charge_amount(raw, min_usd=Decimal("10"), max_usd=Decimal("10000"))
    assert not v.ok
    assert err_substr.lower() in (v.error or "").lower()


def test_validate_amount_accepts_in_bounds_value():
    v = validate_charge_amount("250", min_usd=Decimal("10"), max_usd=Decimal("10000"))
    assert v.ok
    assert v.amount == Decimal("250")
    assert v.error is None


# ---------------------------------------------------------------------------
# HERMES_DEV_BILLING_FIXTURE — offline card/scope state scaffolding (T0)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,has_card,is_admin,billing_on",
    [
        ("nocard", False, True, True),
        ("card", True, True, True),
        ("card-autoreload", True, True, True),
        ("notadmin", True, False, True),
        ("billing-off", False, True, False),
    ],
)
def test_billing_fixture_card_and_gate_invariants(monkeypatch, name, has_card, is_admin, billing_on):
    """Each fixture state honors the card/admin/kill-switch contract the gate reads."""
    monkeypatch.setenv("HERMES_DEV_BILLING_FIXTURE", name)
    s = build_billing_state()
    assert s.logged_in is True
    assert (s.card is not None) is has_card
    assert s.is_admin is is_admin
    assert s.cli_billing_enabled is billing_on
