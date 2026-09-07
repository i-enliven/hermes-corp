"""Pure tool-call guardrail primitive tests."""

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from agent.tool_guardrails import (
    ToolCallGuardrailConfig,
    ToolCallGuardrailController,
    ToolCallSignature,
    ToolGuardrailDecision,
    TurnGuardrailState,
    canonical_tool_args,
    classify_tool_failure,
)


# ── The halt decision renders itself ────────────────────────────────────────
#
# Every user- and model-facing string a guardrail halt produces is a pure
# function of the decision that caused it, so all four are testable without
# constructing an agent, and there is exactly one place each is written.


def _halt_decision(**overrides: Any) -> ToolGuardrailDecision:
    fields: dict[str, Any] = {
        "action": "halt",
        "code": "sequence_repeat_halt",
        "message": "repeating without progress",
        "tool_name": "terminal",
        "count": 4,
    }
    fields.update(overrides)
    return ToolGuardrailDecision(**fields)


def test_halt_decision_renders_the_halt_prose_from_its_own_fields():
    prose = _halt_decision().halt_prose()

    assert prose == (
        "I stopped retrying terminal because it hit the tool-call guardrail "
        "(sequence_repeat_halt) after 4 repeated non-progressing attempts. "
        "The last tool result explains the blocker; the next step is to change "
        "strategy instead of repeating the same call."
    )


def test_halt_prose_names_a_tool_when_the_decision_does_not():
    # A decision can halt without attributing a tool; the sentence stays readable.
    assert "a tool" in _halt_decision(tool_name="").halt_prose()


def test_halt_decision_renders_the_resumption_note_naming_tool_and_code():
    note = _halt_decision().resumption_note()

    assert note == (
        "[System Instruction: The previous turn was halted by a tool-call guardrail "
        "on 'terminal' (sequence_repeat_halt) due to repeated unprogressing actions. "
        "MANDATORY STRATEGY SHIFT: Do NOT immediately emit another inspection or tool call. "
        "You must first summarize what you have learned so far from your previous attempts, "
        "explain the blocker, and ask the user for guidance or propose an alternative strategy "
        "before executing any more tools.]"
    )


def test_resumption_note_omits_the_tool_clause_when_no_tool_is_named():
    # One renderer serves both the named and unnamed case; the wording is shared,
    # which is what stops the two variants drifting apart again.
    note = _halt_decision(tool_name="").resumption_note()

    assert "on ''" not in note
    assert "guardrail (sequence_repeat_halt)" in note
    assert "MANDATORY STRATEGY SHIFT" in note


def test_halt_decision_renders_the_status_line():
    assert _halt_decision().status_line() == (
        "⚠️ Tool guardrail halted terminal: sequence_repeat_halt"
    )


def test_synthetic_tool_result_keeps_its_machine_readable_key_set():
    # The synthetic result is persisted into the transcript and replayed to the
    # provider, so its shape is a wire contract: the keys are frozen and only the
    # human-readable text inside them may ever change.
    decision = _halt_decision(
        action="block",
        code="repeated_exact_failure_block",
        message="boom keeps repeating",
        signature=ToolCallSignature(tool_name="terminal", args_hash="abc123"),
    )

    payload = json.loads(decision.synthetic_tool_result())

    assert set(payload) == {"error", "guardrail"}
    assert payload["error"] == "boom keeps repeating"
    assert set(payload["guardrail"]) == {
        "action",
        "code",
        "message",
        "tool_name",
        "count",
        "signature",
    }


# ── One owner for the turn's guardrail facts ────────────────────────────────


def test_turn_guardrail_state_reports_the_first_decision_that_stopped_the_turn():
    # Two guardrails can stop one turn. The cause reported to the user is the
    # first one that stopped it, not whichever was recorded last.
    state = TurnGuardrailState()
    first = _halt_decision(code="repeated_exact_failure_block", tool_name="terminal")
    second = _halt_decision(code="sequence_repeat_halt", tool_name="read_file")

    state.record_halt(first)
    state.record_halt(second)

    assert state.halt_decision is first


def test_turn_guardrail_state_ignores_a_decision_that_does_not_stop_the_turn():
    state = TurnGuardrailState()

    state.record_halt(_halt_decision(action="allow", code="allow"))
    state.record_halt(_halt_decision(action="warn", code="sequence_repeat_warning"))

    assert state.halt_decision is None


def test_turn_guardrail_state_hands_the_resumption_note_over_exactly_once():
    # The handoff crosses the turn boundary: armed when the halted turn ends,
    # read once by the following turn's prologue, then spent.
    state = TurnGuardrailState()
    decision = _halt_decision()
    state.arm_resumption(decision)

    assert state.take_pending_resumption() is decision
    assert state.take_pending_resumption() is None


def test_turn_guardrail_state_begins_a_turn_with_no_halt_but_keeps_the_handoff():
    # Clearing the halt at the turn boundary must not discard a handoff armed by
    # the previous turn: the prologue of the turn that begins has not read it yet.
    state = TurnGuardrailState()
    decision = _halt_decision()
    state.record_halt(decision)
    state.arm_resumption(decision)

    state.begin_turn()

    assert state.halt_decision is None
    assert state.take_pending_resumption() is decision


def test_a_delivered_note_stays_reportable_after_being_consumed():
    """Consuming the handoff must not erase the fact that it was delivered.

    An empty handoff means either "delivered" or "there was never a halt to
    deliver for", so the two have to be distinguishable or "was the model told
    to change strategy?" cannot be asked of the turn at all.
    """
    state = TurnGuardrailState()
    decision = _halt_decision()
    state.arm_resumption(decision)

    assert state.take_pending_resumption() is decision
    assert state.pending_resumption is None
    assert state.resumption_delivered is decision

    # A turn that never halted reports neither: absence, not a spent delivery.
    assert TurnGuardrailState().resumption_delivered is None


def test_beginning_a_turn_clears_the_previous_turns_delivery_record():
    state = TurnGuardrailState()
    state.arm_resumption(_halt_decision())
    state.take_pending_resumption()
    assert state.resumption_delivered is not None

    state.begin_turn()

    assert state.resumption_delivered is None


def test_halt_decision_is_immutable_once_produced():
    """The decision that stopped a turn cannot be rewritten by a later module.

    The halt fact used to be reachable as a plain attribute that any module in
    the journey could assign to, which is how the two stores drifted apart. The
    decision is a frozen record, so the cause reported to the user is the one the
    guardrail produced.
    """
    decision = _halt_decision()

    with pytest.raises(FrozenInstanceError):
        decision.code = "rewritten_by_a_downstream_module"

    with pytest.raises(FrozenInstanceError):
        decision.action = "allow"


def test_controller_keeps_no_halt_store_of_its_own():
    # The halt fact has exactly one home. The controller counts and decides; it
    # must not also keep a copy, which is how two stores came to disagree about
    # which guardrail stopped a turn.
    controller = ToolCallGuardrailController(ToolCallGuardrailConfig(hard_stop_enabled=True))
    args = {"query": "same"}
    for _ in range(3):
        controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    controller.before_call("web_search", args)

    assert not hasattr(controller, "_halt_decision")
    assert not hasattr(controller, "halt_decision")


def test_tool_call_signature_hashes_canonical_nested_unicode_args_without_exposing_raw_args():
    args_a = {
        "z": [{"β": "☤", "a": 1}],
        "a": {"y": 2, "x": "secret-token-value"},
    }
    args_b = {
        "a": {"x": "secret-token-value", "y": 2},
        "z": [{"a": 1, "β": "☤"}],
    }

    assert canonical_tool_args(args_a) == canonical_tool_args(args_b)
    sig_a = ToolCallSignature.from_call("web_search", args_a)
    sig_b = ToolCallSignature.from_call("web_search", args_b)

    assert sig_a == sig_b
    assert len(sig_a.args_hash) == 64
    metadata = sig_a.to_metadata()
    assert metadata == {"tool_name": "web_search", "args_hash": sig_a.args_hash}
    assert "secret-token-value" not in json.dumps(metadata)
    assert "☤" not in json.dumps(metadata)




def test_config_parses_nested_warn_and_hard_stop_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warnings_enabled": False,
            "hard_stop_enabled": True,
            "warn_after": {
                "exact_failure": 3,
                "same_tool_failure": 4,
                "idempotent_no_progress": 5,
            },
            "hard_stop_after": {
                "exact_failure": 6,
                "same_tool_failure": 7,
                "idempotent_no_progress": 8,
            },
        }
    )

    assert cfg.warnings_enabled is False
    assert cfg.hard_stop_enabled is True
    assert cfg.exact_failure_warn_after == 3
    assert cfg.same_tool_failure_warn_after == 4
    assert cfg.no_progress_warn_after == 5
    assert cfg.exact_failure_block_after == 6
    assert cfg.same_tool_failure_halt_after == 7
    assert cfg.no_progress_block_after == 8


def test_default_repeated_identical_failed_call_warns_without_blocking():
    controller = ToolCallGuardrailController()
    args = {"query": "same"}

    decisions = []
    for _ in range(5):
        assert controller.before_call("web_search", args).action == "allow"
        decisions.append(
            controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
        )

    assert decisions[0].action == "allow"
    assert [d.action for d in decisions[1:]] == ["warn", "warn", "warn", "warn"]
    assert {d.code for d in decisions[1:]} == {"repeated_exact_failure_warning"}
    assert controller.before_call("web_search", args).action == "allow"
    # Nothing escalated to a stop: every decision above allowed execution, which
    # is the whole of what the controller reports. It keeps no halt record of
    # its own — the turn's guardrail state is the single home of that fact.


def test_hard_stop_enabled_blocks_repeated_exact_failure_before_next_execution():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            exact_failure_warn_after=2,
            exact_failure_block_after=2,
            same_tool_failure_halt_after=99,
        )
    )
    args = {"query": "same"}

    assert controller.before_call("web_search", args).action == "allow"
    first = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert first.action == "allow"

    assert controller.before_call("web_search", args).action == "allow"
    second = controller.after_call("web_search", args, '{"error":"boom"}', failed=True)
    assert second.action == "warn"
    assert second.code == "repeated_exact_failure_warning"

    blocked = controller.before_call("web_search", args)
    assert blocked.action == "block"
    assert blocked.code == "repeated_exact_failure_block"
    assert blocked.count == 2














def test_mutating_or_unknown_tools_are_not_blocked_for_repeated_identical_success_output_by_default():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            no_progress_warn_after=2,
            no_progress_block_after=2,
            sequence_repeat_warn_after=99,
        )
    )
    for _ in range(3):
        assert controller.before_call("write_file", {"path": "/tmp/x", "content": "x"}).action == "allow"
        assert controller.after_call("write_file", {"path": "/tmp/x", "content": "x"}, "ok", failed=False).action == "allow"
        assert controller.before_call("custom_tool", {"x": 1}).action == "allow"
        assert controller.after_call("custom_tool", {"x": 1}, "ok", failed=False).action == "allow"






# ── Per-turn runaway-loop caps (Claude Code v2.1.212, Week 29) ──────────────

from agent.tool_guardrails import LoopCapConfig  # noqa: E402






def test_loop_cap_zero_disables_and_junk_falls_back():
    # 0 is a legitimate "unlimited" value; negatives / junk fall back to default.
    assert LoopCapConfig.from_mapping({"max_web_searches": 0}).max_web_searches == 0
    assert LoopCapConfig.from_mapping({"max_web_searches": -5}).max_web_searches == 50
    assert LoopCapConfig.from_mapping({"max_subagents": "nope"}).max_subagents == 50


def test_web_search_cap_blocks_after_limit_regardless_of_hard_stop():
    # Loop caps fire even with hard_stop_enabled=False (the per-turn loop
    # detector's flag). Each distinct query avoids the loop detector so we know
    # the block came from the loop cap, not exact-failure repetition.
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=False,
            loop_caps=LoopCapConfig(max_web_searches=3),
        )
    )
    for i in range(3):
        assert controller.before_call("web_search", {"query": f"q{i}"}).action == "allow"
    decision = controller.before_call("web_search", {"query": "q4"})
    assert decision.action == "block"
    assert decision.code == "loop_web_search_cap"
    assert decision.should_halt is True


# ── Semantic paging loop tests ──────────────────────────────────────────────


def test_config_parses_semantic_paging_thresholds():
    cfg = ToolCallGuardrailConfig.from_mapping(
        {
            "warn_after": {
                "semantic_paging": 3,
            },
            "hard_stop_after": {
                "semantic_paging": 6,
            },
        }
    )
    assert cfg.paging_loop_warn_after == 3
    assert cfg.paging_loop_block_after == 6


def test_consecutive_offset_reads_trigger_paging_warning():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(paging_loop_warn_after=4)
    )
    path = "gateway/run.py"
    offsets = [1, 51, 101, 151]

    for i, offset in enumerate(offsets):
        args = {"path": path, "offset": offset, "limit": 50}
        assert controller.before_call("read_file", args).action == "allow"
        decision = controller.after_call(
            "read_file", args, f"content for chunk {i}", failed=False
        )
        if i < 3:
            assert decision.action == "allow", f"Call {i+1} should be allowed"
        else:
            assert decision.action == "warn"
            assert decision.code == "semantic_paging_loop_warning"
            assert "Semantic paging loop detected" in decision.message
            assert "4 consecutive offset-increment reads" in decision.message
            assert "search_files" in decision.message


def test_interleaved_non_read_tool_resets_paging_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(paging_loop_warn_after=3)
    )
    path = "gateway/run.py"

    # Read 1
    controller.after_call("read_file", {"path": path, "offset": 1}, "c1", failed=False)
    # Read 2
    controller.after_call("read_file", {"path": path, "offset": 51}, "c2", failed=False)

    # Interleaved diagnostic tool call (search_files)
    controller.after_call(
        "search_files", {"path": path, "pattern": "def foo"}, "match", failed=False
    )

    # Read 3 (offset advances, but streak was reset)
    decision = controller.after_call(
        "read_file", {"path": path, "offset": 101}, "c3", failed=False
    )
    assert decision.action == "allow"

    # Read 4 (streak is now 2)
    decision = controller.after_call(
        "read_file", {"path": path, "offset": 151}, "c4", failed=False
    )
    assert decision.action == "allow"

    # Read 5 (streak is now 3 -> triggers warning)
    decision = controller.after_call(
        "read_file", {"path": path, "offset": 201}, "c5", failed=False
    )
    assert decision.action == "warn"
    assert decision.code == "semantic_paging_loop_warning"


def test_switching_target_file_resets_paging_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(paging_loop_warn_after=3)
    )
    # 2 reads on file A
    controller.after_call("read_file", {"path": "a.py", "offset": 1}, "c1", failed=False)
    controller.after_call("read_file", {"path": "a.py", "offset": 51}, "c2", failed=False)

    # Switch to file B
    controller.after_call("read_file", {"path": "b.py", "offset": 1}, "c3", failed=False)
    decision = controller.after_call("read_file", {"path": "b.py", "offset": 51}, "c4", failed=False)
    assert decision.action == "allow"


def test_backward_offset_jump_resets_paging_streak():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(paging_loop_warn_after=3)
    )
    path = "a.py"
    controller.after_call("read_file", {"path": path, "offset": 1}, "c1", failed=False)
    controller.after_call("read_file", {"path": path, "offset": 51}, "c2", failed=False)

    # Jump backwards to offset 10 -> resets streak to 1
    decision = controller.after_call("read_file", {"path": path, "offset": 10}, "c3", failed=False)
    assert decision.action == "allow"

    # Advance from 10 to 30 -> streak is 2
    decision = controller.after_call("read_file", {"path": path, "offset": 30}, "c4", failed=False)
    assert decision.action == "allow"


def test_paging_loop_halt_and_block_when_hard_stop_enabled():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            hard_stop_enabled=True,
            paging_loop_warn_after=2,
            paging_loop_block_after=4,
        )
    )
    path = "large.py"

    # Read 1: allow
    assert controller.before_call("read_file", {"path": path, "offset": 1}).action == "allow"
    d1 = controller.after_call("read_file", {"path": path, "offset": 1}, "c1", failed=False)
    assert d1.action == "allow"

    # Read 2: warn
    assert controller.before_call("read_file", {"path": path, "offset": 51}).action == "allow"
    d2 = controller.after_call("read_file", {"path": path, "offset": 51}, "c2", failed=False)
    assert d2.action == "warn"
    assert d2.code == "semantic_paging_loop_warning"

    # Read 3: warn
    assert controller.before_call("read_file", {"path": path, "offset": 101}).action == "allow"
    d3 = controller.after_call("read_file", {"path": path, "offset": 101}, "c3", failed=False)
    assert d3.action == "warn"

    # Read 4: halt
    assert controller.before_call("read_file", {"path": path, "offset": 151}).action == "allow"
    d4 = controller.after_call("read_file", {"path": path, "offset": 151}, "c4", failed=False)
    assert d4.action == "halt"
    assert d4.code == "semantic_paging_loop_halt"
    assert d4.should_halt is True

    # Read 5: before_call blocks execution
    d5 = controller.before_call("read_file", {"path": path, "offset": 201})
    assert d5.action == "block"
    assert d5.code == "semantic_paging_loop_block"
    assert d5.allows_execution is False


def test_tool_call_signature_canonicalizes_file_paths():
    # Relative path vs dot-slash
    sig_relative = ToolCallSignature.from_call("read_file", {"path": "foo.py"})
    sig_dot_slash = ToolCallSignature.from_call("read_file", {"path": "./foo.py"})
    assert sig_relative == sig_dot_slash

    # Redundant slashes and parent traversal
    sig_nested = ToolCallSignature.from_call("read_file", {"path": "dir//sub/../foo.py"})
    sig_clean = ToolCallSignature.from_call("read_file", {"path": "dir/foo.py"})
    assert sig_nested == sig_clean

    # Whitespace stripping
    sig_padded = ToolCallSignature.from_call("read_file", {"path": "  foo.py  \n"})
    assert sig_padded == sig_relative

    # Alternate path keys
    assert ToolCallSignature.from_call("edit", {"file_path": "./a.py"}) == ToolCallSignature.from_call("edit", {"file_path": "a.py"})
    assert ToolCallSignature.from_call("write", {"filepath": "./a.py"}) == ToolCallSignature.from_call("write", {"filepath": "a.py"})
    assert ToolCallSignature.from_call("patch", {"target_file": "./a.py"}) == ToolCallSignature.from_call("patch", {"target_file": "a.py"})

    # Non-string or empty values are preserved safely
    assert canonical_tool_args({"path": ""}) == '{"path":""}'
    assert canonical_tool_args({"path": None}) == '{"path":null}'


def test_idempotent_no_progress_catches_aliased_file_paths():
    controller = ToolCallGuardrailController(
        ToolCallGuardrailConfig(
            warnings_enabled=True,
            no_progress_warn_after=2,
        )
    )

    # First call: ./foo.py
    d1_before = controller.before_call("read_file", {"path": "./foo.py"})
    assert d1_before.action == "allow"
    d1_after = controller.after_call("read_file", {"path": "./foo.py"}, "content of foo.py", failed=False)
    assert d1_after.action == "allow"

    # Second call: foo.py (aliased path, identical content returned)
    d2_before = controller.before_call("read_file", {"path": "foo.py"})
    assert d2_before.action == "allow"
    d2_after = controller.after_call("read_file", {"path": "foo.py"}, "content of foo.py", failed=False)
    assert d2_after.action == "warn"
    assert d2_after.code == "idempotent_no_progress_warning"
