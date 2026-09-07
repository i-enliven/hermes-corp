"""The turn state owns the guardrail half; the guardrail half stands alone.

Criterion 6 of #4: the guardrail half must stay independently constructible, so
the pure renderers and the one-shot handoff stay testable without assembling a
whole turn. If these fail, nesting has leaked a dependency downward.
"""

from agent.tool_guardrails import TurnGuardrailState
from agent.turn_state import AgentTurnState
from tests.guardrail_test_helpers import halt_decision as _halt_decision


def test_the_guardrail_half_can_be_built_without_a_turn():
    guardrail_only = TurnGuardrailState()
    assert guardrail_only.halt_decision is None
    assert guardrail_only.pending_resumption is None


def test_a_turn_state_carries_its_own_guardrail_account():
    turn = AgentTurnState()
    assert isinstance(turn.guardrails, TurnGuardrailState)
    assert turn.guardrails.halt_decision is None


def test_two_turns_do_not_share_one_guardrail_account():
    first = AgentTurnState()
    first.guardrails.record_halt(_halt_decision())
    assert first.guardrails.halt_decision is not None

    second = AgentTurnState()
    assert second.guardrails.halt_decision is None, "a turn state is not per-turn"


def test_opening_a_turn_forgets_the_previous_turns_halt():
    """A halt is current only for the turn that was stopped by it."""
    previous = AgentTurnState()
    previous.guardrails.record_halt(_halt_decision())

    current = AgentTurnState.begin_turn(previous)
    assert current.guardrails.halt_decision is None, (
        "a stale halt outlived its turn, so the next turn would report a halt it never had"
    )


def test_opening_a_turn_keeps_the_handoff_owed_to_the_model():
    """The resumption handoff crosses the boundary; nothing else does.

    The finalizer arms it at the end of the halted turn and the next turn's
    prologue spends it. Constructing the new turn state must not silently drop
    the debt, or the strategy shift is never delivered.
    """
    previous = AgentTurnState()
    decision = _halt_decision()
    previous.guardrails.record_halt(decision)
    previous.guardrails.arm_resumption(decision)

    current = AgentTurnState.begin_turn(previous)
    assert current.guardrails.pending_resumption is decision
    assert current.guardrails.halt_decision is None


def test_opening_a_turn_arms_nothing_when_no_halt_was_had():
    current = AgentTurnState.begin_turn(AgentTurnState())
    assert current.guardrails.pending_resumption is None
    assert current.guardrails.halt_decision is None


def test_opening_the_first_turn_works_without_a_predecessor():
    current = AgentTurnState.begin_turn(None)
    assert current.guardrails.halt_decision is None
    assert current.guardrails.pending_resumption is None


def test_the_reported_counters_are_per_turn_and_start_empty():
    turn = AgentTurnState()
    assert turn.api_call_count == 0
    assert turn.interrupted is False
    assert turn.failed is False
    assert turn.turn_exit_reason == "unknown"


def test_opening_a_turn_resets_the_reported_counters():
    previous = AgentTurnState()
    previous.api_call_count = 7
    previous.interrupted = True
    previous.failed = True
    previous.turn_exit_reason = "guardrail_halt"

    current = AgentTurnState.begin_turn(previous)
    assert current.api_call_count == 0
    assert current.interrupted is False
    assert current.failed is False
    assert current.turn_exit_reason == "unknown", (
        "the previous turn's exit reason would be reported against this one"
    )
