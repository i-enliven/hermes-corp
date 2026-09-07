"""Shared fixtures for the tool-call guardrail tests.

Lives here rather than in either test module because the fact it builds -- a
decision that stopped a turn -- has exactly one meaning, and the two suites that
need it (the pure decision/state tests and the runtime journey tests) must agree
on what it says. A copy in each file would drift the first time
``ToolGuardrailDecision`` gains a field.
"""

from typing import Any

from agent.tool_guardrails import ToolGuardrailDecision


def halt_decision(**overrides: Any) -> ToolGuardrailDecision:
    """A decision that stopped a turn, with the fields every rendering reads.

    Field defaults are the ones a real ``sequence_repeat_halt`` carries, so a
    test that only cares about precedence still renders sensibly. Override only
    what the test is about.
    """
    fields: dict[str, Any] = {
        "action": "halt",
        "code": "sequence_repeat_halt",
        "message": "repeating without progress",
        "tool_name": "terminal",
        "count": 4,
    }
    fields.update(overrides)
    return ToolGuardrailDecision(**fields)
