# Hermes Agent - Development & Contributor Guide

Hermes is a personal AI agent running across CLI, TUI, messaging gateway (Telegram, Discord, Slack, etc.), Desktop, and ACP. It learns across sessions (memory + skills), delegates to subagents, executes scheduled cron jobs, and drives terminal/browser environments.

---

## 1. Core Architecture & Philosophy

- **Prompt Caching is Sacred**: Long-lived conversations rely on byte-stable prefix caching. Never mutate past context, swap toolsets dynamically mid-conversation, or rebuild the system prompt during a session.
- **Narrow Waist, Broad Edges**: Every core tool schema is sent on every API call. The core agent is intentionally minimal. Most capability arrives at the edges via CLI commands, skills, service-gated tools, or plugins.
- **The Footprint Ladder** (prefer highest rung first):
  1. Extend existing code (0 new schema footprint).
  2. CLI command + skill (`hermes <subcmd>` guided by skill).
  3. Service-gated tool (`check_fn` in registry; visible only when configured).
  4. Plugin / MCP Server (modular, discovered at runtime).
  5. New core tool (last resort for universal primitives: `terminal`, `read_file`, `web_search`).
- **Session-Scoped Capabilities**: Capabilities dependent on client/platform (GUI, desktop panes, reactions) resolve availability from session metadata/platform, never from process environment variables like `HERMES_DESKTOP=1`.

---

## 2. Key Codebase Map

```
run_agent.py          # AIAgent — core conversation loop, execution budget, turn coordination
model_tools.py        # Tool orchestration & function call dispatch
toolsets.py           # Toolset definitions & platform mappings
tools/registry.py     # Tool registry (@registry.register with check_fn gating)
tools/                # Built-in tool implementations (terminal, file, web, etc.)
cli.py                # HermesCLI — interactive CLI orchestrator
hermes_state.py       # SessionDB — SQLite session/message persistence (FTS5 search)
hermes_constants.py   # get_hermes_home(), display_hermes_home() — profile-aware paths
hermes_logging.py     # Logging setup (agent.log, errors.log, gateway.log)
gateway/              # Messaging gateway (run.py, session.py, platforms/)
plugins/              # Plugin definitions (memory, context_engine, model-providers, etc.)
skills/               # Built-in skill library
cron/                 # Scheduler and scheduled jobs
```

---

## 3. Hard Invariants & Ground Rules

1. **Config vs. Secrets**:
   - `.env` is exclusively for credentials and secrets (API keys, tokens).
   - All behavioral settings, timeouts, models, and flags live in `config.yaml`.
2. **Profile-Safe Paths**:
   - Never hardcode `~/.hermes/`. Always resolve directories using `hermes_constants.get_hermes_home()`.
3. **Message Role Alternation**:
   - Maintain strict alternating roles (`system` → `user` → `assistant` → `tool` → `assistant` ...).
   - Never inject synthetic user turns mid-loop.
4. **Test Isolation**:
   - Tests must never touch the user's real `~/.hermes/` directory; always use a temporary `HERMES_HOME` via fixtures.
   - Behavior contracts over snapshots: test invariants and relationships, not frozen literals or enum counts (no change-detector tests).

---

## 4. Operational & Anti-Loop Discipline

When executing tasks or investigating code:
- **Empty Output = Complete / No Results**:
  In `terminal`, `search_files`, and `read_file`, receiving empty output (`""` or 0 lines) signals EOF or empty match sets. It is **not** a prompt to repeat the command or re-query with identical parameters.
- **Guardrail Triggers Demand Strategy Shifts**:
  If a loop guardrail warns or halts (`sequence_repeat_warning`, `idempotent_no_progress_block`), do **not** repeat the same call or attempt trivial syntax escapes (e.g. switching between `terminal` and `execute_code` with the same underlying shell command). Stop, synthesize what has been learned, report the finding, and proceed to the next distinct phase or ask for user guidance.
- **Targeted Reading Over Blind Pagination**:
  Do not paginate sequentially through massive god-files (such as `gateway/run.py` or `run_agent.py`) using multiple small-offset `read_file` calls. Instead, use targeted searches (`grep`, ripgrep, AST analysis) to locate specific symbols and read only relevant function boundaries.

---

## 5. Development & Testing Workflow

```bash
# Prefer local project venv managed by uv
source .venv/bin/activate

# Running tests
uv run pytest tests/path/to/test_file.py -q

# Format & Lint
uv run ruff check .
uv run ruff format .
```
