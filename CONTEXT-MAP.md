# Context Map

This repo uses a multi-context domain-docs layout. Each context owns a `CONTEXT.md` (glossary + domain notes) and a `docs/adr/` (context-scoped decisions). Files are created lazily by `/domain-modeling` when terms or decisions actually get resolved — don't create them upfront.

| Context | Directories | CONTEXT.md | docs/adr/ |
| --- | --- | --- | --- |
| agent-core | `run_agent.py`, `agent/`, `model_tools.py`, `toolsets.py` | `CONTEXT.md` (repo root) | `docs/adr/` (repo root) |
| tools | `tools/`, `tools/environments/` | `tools/CONTEXT.md` | `tools/docs/adr/` |
| gateway | `gateway/`, `gateway/platforms/` | `gateway/CONTEXT.md` | `gateway/docs/adr/` |
| cli | `cli.py`, `hermes_cli/` | `hermes_cli/CONTEXT.md` | `hermes_cli/docs/adr/` |
| plugins-and-skills | `plugins/`, `skills/`, `optional-skills/` | `plugins/CONTEXT.md` | `plugins/docs/adr/` |
| cron | `cron/` | `cron/CONTEXT.md` | `cron/docs/adr/` |
| adapters | `acp_adapter/`, `tui_gateway/`, `ui-tui/`, `apps/*` | `ui-tui/CONTEXT.md` | `ui-tui/docs/adr/` |

System-wide decisions that span contexts go in the root `docs/adr/`.

Consumer rules: see `docs/agents/domain.md`.
