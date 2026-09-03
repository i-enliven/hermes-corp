<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤ (Corporate Edition)

<p align="center">
  <a href="https://github.com/i-enliven/hermes-agent"><img src="https://img.shields.io/badge/Release-Corporate-blue?style=for-the-badge" alt="Corporate Edition"></a>
  <a href="https://github.com/i-enliven/hermes-agent/blob/feature/hermes-corp/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
</p>

**Hermes Agent Corporate Edition** is an autonomous, self-improving AI agent tailored for enterprise workflows, developer productivity, and corporate communication pipelines. Featuring a built-in learning loop, persistent memory, subagent delegation, and scheduled automations, Hermes operates across terminal sessions, background tasks, and corporate messaging platforms.

Bring your own models and infrastructure — integrate with enterprise OpenAI endpoints, Azure OpenAI, GitHub Copilot, Anthropic, OpenRouter, or self-hosted LLMs. Switch anytime with `hermes model` with zero vendor lock-in.

---

## Key Capabilities

<table>
<tr>
  <td width="30%"><b>Modern Terminal Interface (TUI & REPL)</b></td>
  <td>Interactive terminal UI (<code>hermes --tui</code>) and classic REPL with syntax highlighting, streaming tool execution, slash-command autocompletion, interrupt-and-redirect, and persistent session navigation.</td>
</tr>
<tr>
  <td><b>Enterprise Messaging Integration</b></td>
  <td>Direct gateway integrations with <b>Microsoft Teams</b> and <b>Corporate Email (IMAP/SMTP)</b>. Deliver scheduled reports, trigger agent runs via chat or email, and maintain cross-platform context continuity.</td>
</tr>
<tr>
  <td><b>Autonomous Learning & Skills</b></td>
  <td>Persistent cross-session memory with automatic compaction. Autonomous skill creation from completed tasks, skill versioning, and full-text search (FTS5) over historical conversations. Compatible with the open <a href="https://agentskills.io">Agent Skills</a> specification.</td>
</tr>
<tr>
  <td><b>Delegation & Subagent Trees</b></td>
  <td>Spawn isolated subagents for parallel research and task breakdown. Persistent Jupyter notebook toolsets, code execution sandboxes, and background subprocess monitoring.</td>
</tr>
<tr>
  <td><b>Scheduled Automations (Cron)</b></td>
  <td>Built-in natural-language cron scheduler for unattended tasks: daily progress summaries, health audits, automated backups, and pipeline validations.</td>
</tr>
<tr>
  <td><b>Flexible Model Architecture</b></td>
  <td>Native support for Azure OpenAI, GitHub Copilot, Anthropic, Bedrock, Vertex AI, OpenRouter, and any OpenAI-compatible custom inference gateway. Configure fallback chains with <code>hermes fallback</code>.</td>
</tr>
</table>

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- [`uv`](https://github.com/astral-sh/uv) (recommended) or `pip`
- Git

### Quickstart

Clone the repository and install dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/i-enliven/hermes-agent.git
cd hermes-agent

# Create virtual environment and install
uv venv
source .venv/bin/activate
uv pip install -e ".[all]"
```

Run first-time setup:

```bash
# Interactive setup wizard
hermes setup

# Or launch the interactive terminal interface directly
hermes --tui
```

---

## Common CLI Commands

```bash
hermes              # Start interactive chat (classic REPL)
hermes --tui        # Start modern Terminal User Interface (TUI)
hermes -c           # Resume your most recent session
hermes model        # Configure model provider and default inference model
hermes fallback     # View and configure fallback model providers
hermes tools        # Enable or disable toolsets
hermes config       # View or edit configuration (config.yaml)
hermes gateway      # Start corporate messaging gateway (Teams, Email)
hermes doctor       # Run self-diagnostic environment and credential check
hermes sessions     # Manage, search, export, or prune session histories
hermes logs         # View agent logs and diagnostic traces
```

---

## Core Toolsets

Hermes Agent ships with a comprehensive set of built-in toolsets:

- **File Operations**: `read_file`, `write_file`, `patch`, `search_files` with atomic updates and diff validation.
- **Terminal & Execution**: `terminal`, `process`, `execute_code`, and persistent `jupyter_execute` environments.
- **Web & Retrieval**: DuckDuckGo search (`web_search`), page extraction (`web_extract`), and headless browser automation (`browser_*`).
- **Delegation**: `delegate_task` for spawning isolated subagents with structured output contracts and loop guardrails.
- **Scheduling**: `cronjob` for managing background recurring jobs.
- **Corporate Platforms**: Microsoft Teams (`plugins/platforms/teams`) and Email (`plugins/platforms/email`).

---

## Configuration

Configuration is stored in `~/.hermes/config.yaml` and environment variables in `~/.hermes/.env`.

Example `config.yaml`:

```yaml
model:
  provider: custom
  model: gpt-4o
  custom:
    api_base: https://your-enterprise-gateway.company.com/v1
    api_key_env: ENTERPRISE_LLM_KEY

display:
  interface: tui     # 'tui' for the modern interface, or 'cli' for classic REPL

agent:
  reasoning_effort: medium
  max_subagents: 4
```

---

## Health & Diagnostics

Verify your environment, API keys, and tool dependencies at any time:

```bash
hermes doctor
```

`hermes doctor` checks provider credentials, database health, binary tools (ripgrep, git, python runtime), and platform configurations to ensure stable operation.

---

## License

This project is licensed under the [MIT License](LICENSE).
