"""WhatsApp JID alias helpers for pairing/allowlist matching (restored).

The whatsapp platform adapter (which owned JID aliasing) was pruned, but
``gateway/pairing.py`` still needs the alias semantics documented in
#23778: a phone number, ``phone@s.whatsapp.net`` and ``phone:device@s...``
JID forms must all resolve to the same principal.
"""

from __future__ import annotations


def _is_whatsapp_like(platform: str) -> bool:
    return platform in ("whatsapp", "whatsapp_cloud")


def normalize_whatsapp_identifier(user_id: str) -> str:
    """Normalize a WhatsApp user id to the bare phone number form.

    ``15551234567``, ``15551234567@s.whatsapp.net`` and
    ``15551234567:47@s.whatsapp.net`` all normalize to ``15551234567``.
    """
    raw = str(user_id or "").strip()
    if not raw:
        return raw
    host, sep, _suffix = raw.partition("@")
    if sep:
        host = host.split(":", 1)[0]
    return host.strip() or raw


def expand_whatsapp_aliases(user_id: str) -> set:
    """Return the set of WhatsApp identifier forms equivalent to ``user_id``."""
    raw = str(user_id or "").strip()
    if not raw:
        return set()
    phone = normalize_whatsapp_identifier(raw)
    aliases = {raw, phone}
    if phone and phone != raw:
        aliases.add(f"{phone}@s.whatsapp.net")
    return aliases
