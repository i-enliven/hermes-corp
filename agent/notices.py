from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class AgentNotice:
    """A structured, driver-agnostic out-of-band notice.

    The agent fires these via ``AIAgent.notice_callback`` (and clears them via
    ``notice_clear_callback``); each driver renders it its own way — the TUI as a
    status-bar override, the CLI as a console line, etc.
    """

    text: str
    level: str = "info"  # info | warn | error | success
    kind: str = "sticky"  # sticky | ttl
    ttl_ms: Optional[int] = None  # honored only when kind == "ttl"
    key: Optional[str] = None  # dedupe / fired-once-latch / clear key
    id: Optional[str] = None
