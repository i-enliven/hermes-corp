# Agent Core

The conversation engine: it turns a user message into a model response plus tool
calls, and owns what the model is told about what just happened.

## Language

### Turns

**Turn**:
One pass of the agent from a user message to the final response. The unit that a
guardrail halt stops.
_Avoid_: request, message, run

**Firing**:
One scheduled execution of a cron job, which mints its own session and therefore its
own turns.
_Avoid_: run, execution, tick

**Turn state**:
The account of everything that is true only for the current turn. It is discarded
when the turn ends.
_Avoid_: session state, context, scratch pad

### Guardrails

**Guardrail**:
A standing check that may stop the agent from repeating an action that is not making
progress.
_Avoid_: guard, safety check, limiter

**Guardrail halt**:
The event in which a guardrail stops the current turn.
_Avoid_: pause, stop, abort, interrupt

**Halt decision**:
The single guardrail decision that caused a guardrail halt.
_Avoid_: trigger, violation, block

**Halt record**:
The durable account of a guardrail halt. It is scoped to one session and outlives the
turn that raised it.
_Avoid_: flag, marker, state

**Resumption handoff**:
The transfer of a halt decision to the turn following the halt. It is held in memory
for the duration of one turn boundary; the durable halt record is what carries the same
fact across a process boundary.
_Avoid_: pending, carry-over, smuggle

**Resumption note**:
The one-shot instruction delivered on the first message of the turn after a halt.
_Avoid_: reminder, nudge, prologue, sidecar

**Strategy shift**:
The behaviour a resumption note mandates: summarise what was learned, name the
blocker, then ask or propose before acting again.
_Avoid_: pivot, recovery

**Turn guardrail state**:
The guardrail half of the turn state: what stopped the current turn, whether a
resumption note is still owed to the model, and whether one was delivered.
_Avoid_: context, session state
