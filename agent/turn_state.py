"""The one owner of what is true only for the current turn.

A *turn* is one invocation of ``run_conversation``: a user message arriving and
the agent's final response leaving. A handful of facts are true for exactly that
span and for no longer -- which guardrail stopped the turn, whether the model has
been told to change strategy, how many provider calls the turn has spent, whether
the user interrupted it, whether the turn failed, and why the loop ended.

Before this existed, those facts were loose locals in ``run_conversation`` passed
into the turn finalizer as a long list of keyword arguments, while the guardrail
facts sat on the agent under a private name. Adding one per-turn fact therefore
meant editing a signature and every site that named it, and the boundary of a
turn was split across two lifetimes. This object is the single home for them,
built at the one place the per-turn reset already happens, so a turn has one
owner and one boundary.

Deliberately narrow, in two ways:

* It holds only the guardrail account and the tallies the turn result already
  reports. The remaining per-turn locals stay where they are and follow field by
  field, per the staging recorded in ADR-0002 -- moving them all at once was
  rejected there as a release-sized change that would bury a reviewable seam.
* The guardrail half is a separate object (:class:`~agent.tool_guardrails.TurnGuardrailState`)
  reachable through :attr:`guardrails`, so it stays constructible and testable
  without assembling a whole turn.

Not to be confused with the two other ``TurnState`` types in this repository:
``gateway/session_state.py``'s tracks which agent holds the running slot and its
lease, and the TUI's ``turnStore`` holds render state for display. Neither is
agent-core per-turn truth; this is the only one that is.
"""

from agent.tool_guardrails import TurnGuardrailState

__all__ = ["AgentTurnState"]


class AgentTurnState:
    """Everything true of the current turn, and of no turn before it.

    Construct one per turn at the per-turn reset site via :meth:`begin_turn`,
    which is what makes the turn boundary a single event in one place.
    """

    def __init__(self) -> None:
        # The guardrail account: what stopped this turn, and the resumption debt.
        self.guardrails: TurnGuardrailState = TurnGuardrailState()
        # How many provider calls this turn has made. Reported as ``api_calls``.
        self.api_call_count: int = 0
        # The user interrupted this turn. Reported as ``interrupted``.
        self.interrupted: bool = False
        # This turn ended in failure. Reported as ``failed``.
        self.failed: bool = False
        # Why the loop ended, for diagnostics and for the failure explanation
        # built from it. Reported as ``turn_exit_reason``.
        self.turn_exit_reason: str = "unknown"

    @classmethod
    def begin_turn(cls, previous: "AgentTurnState | None" = None) -> "AgentTurnState":
        """Open a turn, carrying forward only what is owed across the boundary.

        Everything a halt consisted of dies with the turn that was halted. The
        one thing that must survive is the resumption note still owed to the
        model, whose whole purpose is to arrive in the turn *after* the halt --
        so it is copied, not consumed. Spending it stays the prologue's job,
        which is what records that the model was actually told.
        """
        turn = cls()
        if previous is not None:
            owed = previous.guardrails.pending_resumption
            if owed is not None:
                turn.guardrails.arm_resumption(owed)
        return turn

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(api_call_count={self.api_call_count},"
            f" interrupted={self.interrupted}, failed={self.failed},"
            f" turn_exit_reason={self.turn_exit_reason!r},"
            f" halted={'yes' if self.guardrails.halt_decision is not None else 'no'},"
            f" resumption_owed={'yes' if self.guardrails.pending_resumption is not None else 'no'})"
        )
