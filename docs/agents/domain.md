# Domain Docs

How the engineering skills should consume this repo's domain documentation when exploring the codebase.

## Before exploring, read these

- **`CONTEXT-MAP.md`** at the repo root: it points at one `CONTEXT.md` per context. Read each one relevant to the topic.
- **`docs/adr/`**: read ADRs that touch the area you're about to work in. In multi-context repos, also check `<context>/docs/adr/` for context-scoped decisions.

If any of these files don't exist, **proceed silently**. Don't flag their absence; don't suggest creating them upfront. The `/domain-modeling` skill (reached via `/grill-with-docs` and `/improve-codebase-architecture`) creates them lazily when terms or decisions actually get resolved.

## File structure

This repo uses the **multi-context** layout (presence of `CONTEXT-MAP.md` at the root):

```
/
├── CONTEXT-MAP.md
├── docs/adr/                          ← system-wide decisions
├── agent-core/                        ← run_agent.py, agent/, model_tools.py, toolsets.py
│   ├── CONTEXT.md
│   └── docs/adr/                      ← context-specific decisions
├── tools/                             ← tools/, tools/environments/
│   ├── CONTEXT.md
│   └── docs/adr/
├── gateway/                           ← gateway/, gateway/platforms/
│   ├── CONTEXT.md
│   └── docs/adr/
├── hermes_cli/                        ← cli.py, hermes_cli/
│   ├── CONTEXT.md
│   └── docs/adr/
├── plugins/                           ← plugins/, skills/, optional-skills/
│   ├── CONTEXT.md
│   └── docs/adr/
├── cron/                              ← cron/
│   ├── CONTEXT.md
│   └── docs/adr/
└── adapters/                          ← acp_adapter/, tui_gateway/, ui-tui/, apps/*
    ├── CONTEXT.md
    └── docs/adr/
```

Note: the Python agent core occupies the repo root, so the `agent-core` context's `CONTEXT.md` and `docs/adr/` live at the root level (`CONTEXT.md`, `docs/adr/`), while the other contexts keep theirs inside their directories.

## Use the glossary's vocabulary

When your output names a domain concept (in an issue title, a refactor proposal, a hypothesis, a test name), use the term as defined in the relevant context's `CONTEXT.md`. Don't drift to synonyms the glossary explicitly avoids.

If the concept you need isn't in the glossary yet, that's a signal: either you're inventing language the project doesn't use (reconsider) or there's a real gap (note it for `/domain-modeling`).

## Flag ADR conflicts

If your output contradicts an existing ADR, surface it explicitly rather than silently overriding:

> _Contradicts ADR-0007 (event-sourced orders), but worth reopening because…_
