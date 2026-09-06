---
status: accepted
---

# Persist the guardrail halt record on the session row

A guardrail halt must change the behaviour of the turn that *follows* it, but the only
thing that survived the trip between turns was a sentence of prose in the previous
assistant message, recovered by string-matching it. We decided to store the halt record
as structured data on the session row, written when the turn ends and read once when the
next turn begins, and to delete the prose-matching recovery entirely. A missing record
degrades to exactly the pre-existing behaviour — no instruction on the next turn — and a
failed write is logged, never raised.

## Considered options

- **Harden the prose protocol instead** (keep matching text, but against a machine-readable
  sentinel). Rejected: it keeps string-matching as the source of truth for a mandate the
  model must obey, which is the defect being removed, only tidier.
- **Reuse the existing JSON grab-bag on the session row.** Rejected: it is the model-runtime
  column, and parking turn-control state in it would mislead the next reader.
- **Reuse the presentation metadata column on the message row.** Rejected outright: the
  message row keeps provider-replay content and presentation metadata as *separate*
  columns, and this is a mandate the provider must see — the inverse of what that column is
  for.
- **Declare cross-process resumption out of scope.** Rejected: process restarts, idle
  reaping, and cache eviction are the ordinary ways a session loses its in-memory state,
  so the uncovered case is the common case.

## Consequences

- The recovery-by-prose path was not merely unreliable, it was a false-positive generator:
  it fired on any assistant message containing the halt phrasing, so an assistant turn that
  merely *discussed* guardrails caused the next user message to be delivered with a
  fabricated "the previous turn was halted by a tool-call guardrail" system instruction.
  Any future recovery path must key on structured data, never on prose.
- The halt record is scoped to one session, so a cron firing — which mints a fresh session
  each time — cannot inherit one. Cross-firing inheritance is a separate decision for the
  cron context, not an oversight here.
- Adding the column is declarative: the schema reconciler adds any column declared in the
  schema but absent from the live table on next startup. The cost is not the change, it is
  that the column is then permanent in every user's database.
