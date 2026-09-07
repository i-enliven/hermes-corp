# The durable half of a guardrail halt's account.
#
# A halt stops one turn, but its consequence is felt by the next, and the gap
# between those two turns is where processes die. The in-memory handoff on
# `TurnGuardrailState` covers the gap when the agent object survives it -- which
# it usually does, because the gateway and the terminal UI both reuse one agent
# across a session. It does not cover a restart, a gateway cache eviction, or an
# idle reap, and when it misses, the user gets no strategy-shift instruction and
# no hint that one was owed.
#
# This module is the only route either end takes to the stored account. Three
# functions, each keyed by session, each failure-tolerant: a halt record is a
# nudge, and losing one must cost the user a nudge and not a turn.
#
# The record is structured data, written by the turn finalizer and read once by
# the following turn's prologue. Nothing here decides that a halt happened by
# inspecting message text -- the recovery-by-prose scan that used to do exactly
# that fabricated instructions out of assistant prose which merely quoted guardrail
# wording, and it is gone for good.
import logging
from agent.tool_guardrails import ToolGuardrailDecision
logger = logging.getLogger(__name__)
def _reach(agent):
    '''Return the agent's session store and its session id, or (None, '').

    A persistence-isolated fork (a background review) must not reach the canonical
    store: it would publish a halt the user never experienced into a session they
    do recognise. Such an agent reads and writes nothing here.
    '''
    if getattr(agent, '_persist_disabled', False):
        return None, ''
    store = getattr(agent, '_session_db', None)
    session_id = getattr(agent, 'session_id', '') or ''
    if store is None or not session_id:
        return None, ''
    return store, session_id
def write_halt_record(agent, decision) -> bool:
    '''Store the account of a halt that stopped this turn. Never raises.

    Written undelivered: the account says a turn was stopped and the model has
    not been told yet. Whether it is told is the following turn's business, and
    that turn records the delivery back through `mark_halt_delivered`.

    Returns whether the account was actually stored, so a caller can log the
    difference between a turn that kept its nudge and one that lost it. Losing it
    is possible in two ways, and the second is easy to miss: the write can raise,
    or it can land on a session that has no row yet and therefore change nothing.
    The store reports the second as a false, which is why the value is passed
    through rather than assumed.
    '''
    store, session_id = _reach(agent)
    if store is None or decision is None:
        return False
    try:
        return bool(store.write_guardrail_halt_record(session_id, decision.halt_record()))
    except Exception:
        logger.warning(
            'could not store the halt record for session %s; the next turn will '
            'not be told to change strategy',
            session_id,
            exc_info=True,
        )
        return False
def take_halt_decision(agent):
    '''Return the halt decision this session is still owed, or None.

    Reads the stored account once. An account already marked delivered is nothing
    owed, which is also the answer for a session that predates the column and for
    one that never halted: nothing owed in every case, and never an error.

    The delivery is recorded back on the spot, so a turn which dies after reading
    but before finishing does not leave the same instruction owed twice. The
    caller renders the note from the returned decision, keeping the rendering in
    the one place it lives rather than duplicating it here.
    '''
    store, session_id = _reach(agent)
    if store is None:
        return None
    try:
        record = store.read_guardrail_halt_record(session_id)
        if not record:
            return None
        if record.get("delivered"):
            # Already told. The account outlives its own delivery so it can be
            # audited, which is exactly why reading it has to ask.
            return None
        decision = ToolGuardrailDecision.from_halt_record(record)
        if decision is None:
            return None
        if not decision.resumption_note():
            return None
        store.mark_guardrail_halt_record_delivered(session_id)
        return decision
    except Exception:
        logger.warning(
            'could not read the halt record for session %s; proceeding without a '
            'strategy-shift instruction',
            session_id,
            exc_info=True,
        )
        return None
def mark_halt_delivered(agent) -> None:
    '''Record that a strategy-shift instruction reached the model.

    Needed because the in-memory handoff usually serves the next turn before the
    stored account is ever consulted, and an account left reading 'not delivered'
    would have the first cold turn after a later restart nudge again for a halt
    the model was told about hours earlier. Marked rather than deleted, so the
    account of a halt outlives its own delivery and can be audited.
    '''
    store, session_id = _reach(agent)
    if store is None:
        return
    try:
        store.mark_guardrail_halt_record_delivered(session_id)
    except Exception:
        logger.debug(
            'could not mark the halt record delivered for session %s',
            session_id,
            exc_info=True,
        )
