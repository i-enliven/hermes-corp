# The one way a test opens a turn.
#
# finalize_turn used to take the turn's reported facts as four loose keyword
# arguments, so a test could claim any combination -- including combinations the
# loop could never produce. It now takes the turn state itself, positionally, so
# the facts a test asserts against are the facts the loop would have written.
#
# Lives at tests/ root, importable as `turn_state_test_helpers`, matching
# guardrail_test_helpers.
from agent.turn_state import AgentTurnState

__all__ = ["turn_state"]


def turn_state(
    *,
    api_call_count: int = 0,
    interrupted: bool = False,
    failed: bool = False,
    turn_exit_reason: str = "unknown",
) -> AgentTurnState:
    """Open a turn carrying the facts a test wants the finalizer to see.

    Prefer leaving a field at its default: a turn that did not fail should not
    say it did, and the point of moving these onto one object is that they can
    no longer disagree with each other.
    """
    turn = AgentTurnState()
    turn.api_call_count = api_call_count
    turn.interrupted = interrupted
    turn.failed = failed
    turn.turn_exit_reason = turn_exit_reason
    return turn
