"""Codex/Responses prompt-cache helpers (extracted from the pruned transport).

The full Responses API transport (``ResponsesApiTransport``) was pruned; the
helpers below remain live because ``agent/auxiliary_client.py`` (Codex aux
path) and ``agent/transports/chat_completions.py`` (``_add_prompt_cache_key``)
both import them. They preserve the exact pre-prune semantics:

- ``_cache_scope_from_session_id`` — normalize a physical session id into a
  stable logical cache scope. Cron fires build ``session_id`` as
  ``cron_<job_id>_<YYYYMMDD_HHMMSS>`` (see cron/scheduler.py); the trailing
  per-fire timestamp is stripped so repeat fires of the same job share a cache
  scope (#51395/#52295). Every non-cron session id already identifies one
  conversation/agent instance and is used unchanged.
- ``_content_cache_key`` — content-address the prompt cache key within a
  logical scope: ``pck_<sha256[:24]>`` of (scope + instructions + sorted tool
  schemas), ``None`` when there is nothing static to key on (#78941 keeps
  unrelated sessions from bucket-sharing; sorting keeps the hash
  insertion-order independent).
- ``_default_prompt_cache_retention_for_request`` — ``"24h"`` only for hosts/
  models with an opt-in prompt-cache retention contract (Meta Model API,
  Bedrock Mantle); ``None`` elsewhere.
"""

import hashlib
import json
import re
from typing import Any, Dict, List, Optional

# Cron fires build session_id as ``cron_<job_id>_<YYYYMMDD_HHMMSS>`` (see
# cron/scheduler.py). The trailing timestamp is per-fire noise; stripped so
# repeat fires of the same job share a cache scope (see #51395/#52295).
_CRON_SESSION_ID_RE = re.compile(r"^(cron_.+)_\d{8}_\d{6}$")


def _cache_scope_from_session_id(session_id: Optional[str]) -> str:
    """Normalize a physical session_id into a stable logical cache scope.

    Every non-cron session_id already identifies one conversation/agent
    instance (main run, a specific child/subagent, a sibling child, ...),
    so it is used unchanged. Only cron's per-fire timestamp needs stripping.
    """
    sid = str(session_id or "")
    match = _CRON_SESSION_ID_RE.match(sid)
    return match.group(1) if match else sid


_EXTENDED_PROMPT_CACHE_MODELS = (
    "gpt-5.5-pro",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5.1-chat-latest",
    "gpt-5.1-codex",
    "gpt-5.1",
    "gpt-5-codex",
    "gpt-5",
    "gpt-4.1",
)
_EXTENDED_PROMPT_CACHE_MODEL_RE = re.compile(
    rf"(?:^|[./:])(?:{'|'.join(re.escape(name) for name in _EXTENDED_PROMPT_CACHE_MODELS)})"
    r"(?:-\d{4}-\d{2}-\d{2})?$"
)


def _default_prompt_cache_retention_for_request(
    model: str,
    base_url: Any,
) -> Optional[str]:
    """Return ``24h`` for supported hosts/models (Bedrock Mantle, Meta)."""
    from utils import base_url_hostname

    hostname = base_url_hostname(str(base_url or "")).lower()
    # Meta Model API: prompt caching is opt-in via prompt_cache_retention.
    # Measured 0% hits on /chat/completions vs 93-99% on /responses with 24h.
    if hostname == "api.meta.ai":
        return "24h"

    hostname_parts = hostname.split(".")
    is_bedrock_mantle = (
        len(hostname_parts) == 4
        and hostname_parts[0] == "bedrock-mantle"
        and bool(hostname_parts[1])
        and hostname_parts[2:] == ["api", "aws"]
    )
    if not is_bedrock_mantle:
        return None

    normalized = str(model or "").strip().lower().replace("_", "-")
    if _EXTENDED_PROMPT_CACHE_MODEL_RE.search(normalized):
        return "24h"
    return None


def _content_cache_key(
    instructions: str,
    tools: Optional[List[Dict[str, Any]]],
    scope_id: str = "",
) -> Optional[str]:
    """Content-address the prompt cache key within a logical cache scope.

    Returns ``pck_<sha256[:24]>`` of (scope_id + instructions + sorted tool
    schemas), or None when there is nothing static to key on. The cache key
    is a routing hint only — never a correctness boundary — so two requests
    sharing a scope, system prompt, and tool set intentionally resolve to the
    same warm prefix bucket.

    ``scope_id`` (pass ``_cache_scope_from_session_id(session_id)``) keeps
    unrelated sessions — independent conversations, main vs. child/subagent,
    sibling children — from concentrating onto the same bucket merely because
    their static prefix matches (see #78941), while still letting recurring
    cron fires of one job share a stable key across their timestamped
    session_ids (the original #51395/#52295 fix this built on). Sorting tools
    by name keeps the hash insertion-order independent.
    """
    if not instructions and not tools:
        return None
    tools_part = ""
    if tools:
        sorted_tools = sorted(
            (t for t in tools if isinstance(t, dict)),
            key=lambda t: str(t.get("name") or t.get("type") or ""),
        )
        tools_part = json.dumps(
            sorted_tools, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    # \x00 separators so a scope/instructions/tools boundary can't be forged
    # by content that happens to contain the same bytes.
    content = f"{scope_id}\x00{instructions or ''}\x00{tools_part}"
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"pck_{digest}"
