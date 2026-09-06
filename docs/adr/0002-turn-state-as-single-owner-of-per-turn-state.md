---
status: accepted
---

# Adopt turn state as the single owner of per-turn state, staged

One decision — "a guardrail halted this turn" — had no home: it was set in the agent,
read by the loop, exported by the finalizer, re-consumed by the turn prologue, and
initialised at construction, with two stores holding it under opposite precedence rules.
We decided to give it a home: a turn guardrail state object, nested inside a turn state
object that owns everything true only for the current turn, so that the finalizer takes
the turn state as its one argument instead of a long list of keyword arguments.

## Considered options

- **Give the guardrail its own owner and leave the rest alone.** Rejected: it fixes the
  reported symptom while leaving the same ping-ponging pattern in place for the next
  decision to fall into.
- **Move all per-turn state in one step.** Rejected as a release-sized change that would
  bury a small, reviewable seam.
- **Keep the two owners as siblings.** Rejected: the finalizer would still take two
  arguments and the turn boundary would still be split across two lifetimes.

## Consequences

- Migration is staged: the guardrail fields and the counters the result already reports
  move first; the remaining per-turn locals follow field by field behind them. Each step
  stays reviewable, and the guardrail payoff lands first.
- The compatibility shims that let a half-built agent stand in for a full one in tests are
  removed as the fields move, not kept: they are the reason the state had no owner in the
  first place, and freezing them as public accessors would make the defect permanent.
- Tests that pre-set a field to its empty value rather than leaving it unset can pass
  without exercising anything. Those sites are updated deliberately, not by default.
