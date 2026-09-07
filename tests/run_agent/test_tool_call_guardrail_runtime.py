"""Runtime tests for tool-call loop guardrails."""

import json
import uuid
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from tests.guardrail_test_helpers import halt_decision as _halt_decision
from agent.tool_guardrails import TurnGuardrailState
from run_agent import AIAgent


def _make_tool_defs(*names: str) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} tool",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for name in names
    ]


def _mock_tool_call(name="web_search", arguments="{}", call_id=None):
    return SimpleNamespace(
        id=call_id or f"call_{uuid.uuid4().hex[:8]}",
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _mock_response(content="Hello", finish_reason="stop", tool_calls=None):
    msg = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=msg, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], model="test/model", usage=None)


def _make_agent(*tool_names: str, max_iterations: int = 10, config: dict | None = None) -> AIAgent:
    with (
        patch("run_agent.get_tool_definitions", return_value=_make_tool_defs(*tool_names)),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("hermes_cli.config.load_config", return_value=config or {}),
        patch("hermes_cli.config.load_config_readonly", return_value=config or {}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            max_iterations=max_iterations,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    agent._cached_system_prompt = "You are helpful."
    agent._use_prompt_caching = False
    agent.compression_enabled = False
    agent.save_trajectories = False
    return agent


def _seed_exact_failures(agent: AIAgent, tool_name: str, args: dict, count: int = 2) -> None:
    for _ in range(count):
        agent._tool_guardrails.after_call(
            tool_name,
            args,
            json.dumps({"error": "boom"}),
            failed=True,
        )


def _hard_stop_config(**overrides) -> dict:
    cfg = {
        "tool_loop_guardrails": {
            "warnings_enabled": True,
            "hard_stop_enabled": True,
            "hard_stop_after": {
                "exact_failure": 2,
                "same_tool_failure": 8,
                "idempotent_no_progress": 5,
            },
        }
    }
    cfg["tool_loop_guardrails"].update(overrides)
    return cfg


def test_default_sequential_path_warns_repeated_exact_failure_without_blocking_execution():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-soft")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_called_once()
    assert len(starts) == 1
    assert any(event[0][0] == "tool.completed" for event in progress)
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-soft"
    assert "repeated_exact_failure_warning" in messages[0]["content"]
    assert "repeated_exact_failure_block" not in messages[0]["content"]
    # Nothing stopped the turn: the guardrail account of the turn is empty.
    assert agent._turn_state.guardrails.halt_decision is None


def test_config_enabled_hard_stop_blocks_repeated_exact_failure_before_execution():
    agent = _make_agent("web_search", config=_hard_stop_config())
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args)
    starts = []
    progress = []
    agent.tool_start_callback = lambda *a, **k: starts.append((a, k))
    agent.tool_progress_callback = lambda *a, **k: progress.append((a, k))
    tc = _mock_tool_call("web_search", json.dumps(args), "c-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc:
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert starts == []
    assert progress == []
    assert len(messages) == 1
    assert messages[0]["role"] == "tool"
    assert messages[0]["tool_call_id"] == "c-block"
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_sequential_after_call_appends_guidance_to_tool_result_without_extra_messages():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    _seed_exact_failures(agent, "web_search", args, count=1)
    tc = _mock_tool_call("web_search", json.dumps(args), "c-warn")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    assert [m["role"] for m in messages] == ["tool"]
    assert messages[0]["tool_call_id"] == "c-warn"
    assert "Tool loop warning" in messages[0]["content"]
    assert "repeated_exact_failure_warning" in messages[0]["content"]


def test_same_tool_failure_warning_tells_model_to_recover_with_tools():
    agent = _make_agent("terminal")
    guardrails = getattr(agent, "_tool_guardrails")
    guardrails.after_call(
        "terminal",
        {"command": "bad-1"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    guardrails.after_call(
        "terminal",
        {"command": "bad-2"},
        json.dumps({"exit_code": 1}),
        failed=True,
    )
    tc = _mock_tool_call("terminal", json.dumps({"command": "bad-3"}), "c-recover")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with patch("run_agent.handle_function_call", return_value=json.dumps({"exit_code": 1})):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    content = messages[0]["content"]
    assert "same_tool_failure_warning" in content
    assert "Do not switch to text-only replies" in content
    assert "keep using tools" in content
    assert "pwd && ls -la" in content
    assert "absolute path" in content
    assert "different tool" in content


def test_config_enabled_hard_stop_concurrent_path_does_not_submit_blocked_calls_and_preserves_result_order():
    agent = _make_agent("web_search", config=_hard_stop_config())
    blocked_args = {"query": "blocked"}
    allowed_args = {"query": "allowed"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    starts = []
    progress_events = []
    agent.tool_start_callback = lambda tool_call_id, name, args: starts.append((tool_call_id, name, args))
    agent.tool_progress_callback = lambda event, name, preview, args, **kw: progress_events.append((event, name, args, kw))
    calls = [
        _mock_tool_call("web_search", json.dumps(blocked_args), "c-block"),
        _mock_tool_call("web_search", json.dumps(allowed_args), "c-allow"),
    ]
    msg = SimpleNamespace(content="", tool_calls=calls)
    messages = []
    executed = []

    def fake_handle(name, args, task_id, **kwargs):
        executed.append((name, args, kwargs["tool_call_id"]))
        return json.dumps({"ok": args["query"]})

    with patch("run_agent.handle_function_call", side_effect=fake_handle):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    assert executed == [("web_search", allowed_args, "c-allow")]
    assert [m["tool_call_id"] for m in messages] == ["c-block", "c-allow"]
    assert "repeated_exact_failure_block" in messages[0]["content"]
    assert json.loads(messages[1]["content"]) == {"ok": "allowed"}
    assert starts == [("c-allow", "web_search", allowed_args)]
    started_events = [event for event in progress_events if event[0] == "tool.started"]
    completed_events = [event for event in progress_events if event[0] == "tool.completed"]
    assert started_events == [("tool.started", "web_search", allowed_args, {})]
    assert len(completed_events) == 1
    assert completed_events[0][1] == "web_search"


def test_relay_rewrite_precedes_sequential_policy_approval_checkpoint_and_dispatch():
    agent = _make_agent("write_file")
    original_args = {"path": "/original/path", "content": "old"}
    final_args = {"path": "/approved/path", "content": "new"}
    tc = _mock_tool_call("write_file", json.dumps(original_args), "c-rewrite")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    observed = {
        "plugin": [],
        "guardrail": [],
        "approval": [],
        "checkpoint": [],
        "start": [],
        "dispatch": [],
    }

    original_before_call = agent._tool_guardrails.before_call

    def observe_guardrail(name, args):
        observed["guardrail"].append((name, dict(args)))
        return original_before_call(name, args)

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(final_args)), dict(final_args)

    def observe_plugin(name, args, **kwargs):
        del kwargs
        observed["plugin"].append((name, dict(args)))
        return (None, None)

    def observe_approval(name, args):
        observed["approval"].append((name, dict(args)))
        return None

    def dispatch(name, args, task_id, **kwargs):
        del task_id, kwargs
        observed["dispatch"].append((name, dict(args)))
        return json.dumps({"ok": True})

    agent._checkpoint_mgr = SimpleNamespace(
        enabled=True,
        get_working_dir_for_path=lambda path: path,
        ensure_checkpoint=lambda path, reason: observed["checkpoint"].append(
            (path, reason)
        ),
    )
    agent.tool_start_callback = lambda _call_id, name, args: observed["start"].append(
        (name, dict(args))
    )

    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            side_effect=observe_plugin,
        ),
        patch.object(agent._tool_guardrails, "before_call", side_effect=observe_guardrail),
        patch(
            "acp_adapter.edit_approval.maybe_require_edit_approval",
            side_effect=observe_approval,
        ),
        patch("model_tools.registry.dispatch", side_effect=dispatch),
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    expected = [("write_file", final_args)]
    assert observed["plugin"] == expected
    assert observed["guardrail"] == expected
    assert observed["approval"] == expected
    assert observed["start"] == expected
    assert observed["dispatch"] == expected
    assert observed["checkpoint"] == [
        ("/approved/path", "before write_file")
    ]


def test_relay_rewrite_is_guarded_before_dispatch_in_concurrent_path():
    agent = _make_agent("web_search", config=_hard_stop_config())
    original_args = {"query": "original"}
    blocked_args = {"query": "blocked"}
    _seed_exact_failures(agent, "web_search", blocked_args)
    tc = _mock_tool_call("web_search", json.dumps(original_args), "c-rewrite-block")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []
    starts = []

    def relay_execute(name, args, callback, **kwargs):
        del name, args, kwargs
        return callback(dict(blocked_args)), dict(blocked_args)

    agent.tool_start_callback = lambda *args: starts.append(args)
    with (
        patch("agent.relay_tools.execute", side_effect=relay_execute),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as dispatch,
    ):
        agent._execute_tool_calls_concurrent(msg, messages, "task-1")

    dispatch.assert_not_called()
    assert starts == []
    assert "repeated_exact_failure_block" in messages[0]["content"]


def test_plugin_pre_tool_block_wins_without_counting_as_toolguard_block():
    agent = _make_agent("web_search")
    args = {"query": "same"}
    tc = _mock_tool_call("web_search", json.dumps(args), "c-plugin")
    msg = SimpleNamespace(content="", tool_calls=[tc])
    messages = []

    with (
        patch(
            "hermes_cli.plugins._dispatch_pre_tool_call_hooks",
            return_value=("plugin policy", None),
        ),
        patch("run_agent.handle_function_call", return_value="SHOULD_NOT_RUN") as mock_hfc,
    ):
        agent._execute_tool_calls_sequential(msg, messages, "task-1")

    mock_hfc.assert_not_called()
    assert "plugin policy" in messages[0]["content"]
    assert agent._tool_guardrails.before_call("web_search", args).action == "allow"


def test_a_halted_turn_reports_its_guardrail_record_and_arms_the_resumption():
    """The turn result's guardrail record has never had a positive assertion.

    Every stub pinned the halt field to empty and the only assertion in this
    file was the negative one above, so the whole shape of what a halted turn
    reports was uncaught. Assert presence, the reported fields, and that the
    following turn is owed the strategy-shift note.
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    # The record the user-facing surfaces read from.
    assert "guardrail" in result
    assert result["guardrail"]["action"] in {"block", "halt"}
    assert result["guardrail"]["tool_name"] == "web_search"
    assert result["guardrail"]["count"] >= 1
    # And the handoff to the turn that follows is armed, exactly once.
    assert agent._turn_state.guardrails.pending_resumption is not None


def test_the_first_decision_to_stop_the_turn_is_the_one_the_user_is_told_about():
    """Two guardrails stop one turn, through the public entry point; the cause reported is the first.

    The controller used to keep its own halt, last-writer-wins, while the agent
    kept one that was first-writer-wins, so which guardrail the user was told
    about depended on the order the two stores happened to be written. Both
    writes now go through one owner, and nothing in the pipeline may quietly
    replace the cause once it is set.

    Driven through ``run_conversation`` rather than by poking the owner directly.
    Both decisions come from the per-turn runaway-loop caps, which are consulted
    before a call runs and count from zero at the turn boundary -- so neither
    depends on a tool result being classified as a failure, and neither needs
    seeding. The batch is::

        web_search(a)  terminal  web_search(b)  delegate_task(a)  delegate_task(b)

    with both caps set to one. ``terminal`` is not parallel-safe and the two
    searches carry distinct arguments, so the segment planner puts every call in
    a segment of its own and they run strictly in the order the model asked for.
    The two commands succeed, so no failure counter is involved and the caps are
    the only guardrails that can fire:

      * the second ``web_search``  -> ``loop_web_search_cap``
      * the second ``delegate_task`` -> ``loop_subagent_cap``

    Which of the two the turn blames is the whole of what this is about. The
    assertion is deliberately about *arrival order* rather than about a named
    code: the segment planner is free to run the calls in whatever order it
    likes, and pinning a code here would test that policy instead of the one
    that matters. What must hold however it orders them is that the cause
    reported is the first decision that stopped the turn, and that every surface
    the user and the model read agrees on that one.
    """
    config = {
        "tool_loop_guardrails": {
            "warnings_enabled": False,
            "hard_stop_enabled": False,
            # A cap of zero disables a limit outright, so both allowances are one:
            # spent by the first call of each kind, refused on the second.
            "loop_caps": {"max_subagents": 1, "max_web_searches": 1},
        }
    }
    agent = _make_agent(
        "web_search", "terminal", "delegate_task", max_iterations=10, config=config
    )

    # Every command succeeds, so no failure counter advances and the caps are the
    # only guardrails able to stop the turn.
    def _execute(tool_name: str, args: dict) -> str:
        return json.dumps({"ok": True})

    # Observe the arrival order at the owner's single write path, without
    # changing what it does. Every decision that reaches it is noted, whether or
    # not the owner accepts it as the cause -- first-wins rejects the later one,
    # and that rejection is the behaviour under test.
    arrived: list = []
    accepted: list = []
    real_record_halt = TurnGuardrailState.record_halt

    def _spy_record_halt(self, decision):
        was_accepted = real_record_halt(self, decision)
        if decision is not None and decision.should_halt:
            arrived.append(decision)
        if was_accepted:
            accepted.append(decision)
        return was_accepted

    search_a = {"query": "alpha"}
    search_b = {"query": "beta"}
    spawn_a = {"goal": "investigate"}
    spawn_b = {"goal": "verify"}

    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[
                _mock_tool_call("web_search", json.dumps(search_a), "c-search-a"),
                _mock_tool_call("terminal", json.dumps({"command": "true"}), "c-command"),
                _mock_tool_call("web_search", json.dumps(search_b), "c-search-b"),
                _mock_tool_call("delegate_task", json.dumps(spawn_a), "c-delegate-a"),
                _mock_tool_call("delegate_task", json.dumps(spawn_b), "c-delegate-b"),
            ],
        ),
        _mock_response(content="I will change approach.", finish_reason="stop"),
    ]
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", side_effect=_execute),
        patch.object(TurnGuardrailState, "record_halt", _spy_record_halt),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search, run a command, then delegate twice")

    assert result["turn_exit_reason"] == "guardrail_halt"

    # Non-vacuity: two *different* guardrails must really have reached the owner,
    # or "the first one wins" would be asserting nothing at all.
    assert len(arrived) >= 2, f"expected two competing halts, got {[d.code for d in arrived]}"
    assert len({d.code for d in arrived}) >= 2, "both halts carried the same code"
    # Only one of them may be the cause.
    assert len(accepted) == 1, f"expected exactly one accepted cause, got {[d.code for d in accepted]}"

    cause = agent._turn_state.guardrails.halt_decision
    assert cause is not None

    # The cause is the first decision that stopped the turn, not the last.
    assert cause is arrived[0]
    assert cause is accepted[0]

    # And every surface the user and the model read agrees on that one cause.
    assert result["guardrail"]["code"] == cause.code
    assert result["guardrail"]["tool_name"] == cause.tool_name
    assert cause.code in result["final_response"]
    # The note owed to the following turn names the same guardrail, so the model
    # is told to change strategy about the thing that actually stopped it.
    assert agent._turn_state.guardrails.pending_resumption is cause




def test_default_run_conversation_warns_without_guardrail_halt():
    agent = _make_agent("web_search", max_iterations=10)
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 4)
    ]
    responses.append(_mock_response(content="done", finish_reason="stop", tool_calls=None))
    agent.client.chat.completions.create.side_effect = responses

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})) as mock_hfc,
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert mock_hfc.call_count == 3
    assert result["turn_exit_reason"].startswith("text_response")
    assert "guardrail" not in result
    # The other half of "exactly when". Absence of the halt key says the turn
    # was not stopped; the delivery flag being unset says no note was handed
    # over. A finalizer that set it unconditionally would pass every positive
    # test in this file and fail only here.
    assert "guardrail_resumption_delivered" not in result
    assert result["final_response"] == "done"
    tool_contents = [m["content"] for m in result["messages"] if m.get("role") == "tool"]
    assert any("repeated_exact_failure_warning" in content for content in tool_contents)




def test_guardrail_halt_emits_final_response_through_stream_delta_callback():
    """Regression for #30770: when the guardrail halts the loop, the
    synthesized halt message must be pushed through ``stream_delta_callback``
    so SSE/TUI clients see why the agent stopped instead of a silent stream
    close.  Without this the chat-completions SSE writer drains an empty
    queue and emits a finish chunk with zero content (indistinguishable
    from a crash for Open WebUI and similar clients).
    """
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    responses = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses

    deltas: list = []
    agent.stream_delta_callback = lambda d: deltas.append(d)
    # The mocked client returns SimpleNamespace responses which aren't
    # iterable as streaming chunks; force the non-streaming code path so
    # the guardrail-halt branch is reached without engaging the real
    # streaming machinery.
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result = agent.run_conversation("search repeatedly")

    assert result["turn_exit_reason"] == "guardrail_halt"
    halt_text = result["final_response"]
    assert "stopped retrying" in halt_text

    # The halt message must have been pushed through the callback at least
    # once.  Empty-queue SSE writers were the bug — clients saw no content
    # delta before the finish chunk.
    text_deltas = [d for d in deltas if isinstance(d, str)]
    assert halt_text in text_deltas, (
        f"halt message was never streamed; callback only saw {deltas!r}"
    )


def test_sequence_loop_warns_on_successful_repeated_commands():
    cfg = {"tool_loop_guardrails": {"warn_after": {"sequence_repeat": 2}}}
    agent = _make_agent("terminal", config=cfg)
    guardrails = getattr(agent, "_tool_guardrails")

    # First call - clean execution
    d1 = guardrails.after_call("terminal", {"command": "git status"}, json.dumps({"exit_code": 0, "stdout": "clean"}), failed=False)
    assert d1.action == "allow"

    # Second call - same clean execution
    d2 = guardrails.after_call("terminal", {"command": "git status"}, json.dumps({"exit_code": 0, "stdout": "clean"}), failed=False)
    assert d2.action == "warn"
    assert d2.code == "sequence_repeat_warning"
    assert "Tool sequence loop detected" in d2.message


def test_sequence_loop_ping_pong_detection():
    cfg = {"tool_loop_guardrails": {"warn_after": {"sequence_repeat": 2}}}
    agent = _make_agent("terminal", "read_file", config=cfg)
    guardrails = getattr(agent, "_tool_guardrails")

    # Sequence: terminal -> read_file -> terminal -> read_file
    guardrails.after_call("terminal", {"command": "ls"}, "ok", failed=False)
    guardrails.after_call("read_file", {"path": "a.txt"}, "content", failed=False)
    guardrails.after_call("terminal", {"command": "ls"}, "ok", failed=False)
    d4 = guardrails.after_call("read_file", {"path": "a.txt"}, "content", failed=False)

    assert d4.action == "warn"
    assert d4.code == "sequence_repeat_warning"
    assert "2-step tool call pattern" in d4.message

def test_sequence_loop_hard_stop_blocks_execution():
    config = _hard_stop_config(
        hard_stop_after={
            "exact_failure": 2,
            "same_tool_failure": 8,
            "idempotent_no_progress": 5,
            "sequence_repeat": 3,
        }
    )
    agent = _make_agent("terminal", config=config)
    guardrails = getattr(agent, "_tool_guardrails")

    # Simulate 2 repeats of terminal command
    guardrails.after_call("terminal", {"command": "check"}, "ok", failed=False)
    guardrails.after_call("terminal", {"command": "check"}, "ok", failed=False)

    # 3rd repeat in before_call should block
    d_before = guardrails.before_call("terminal", {"command": "check"})
    assert d_before.action == "block"
    assert d_before.code == "sequence_repeat_block"


def test_turn_resuming_after_a_discussion_of_guardrails_does_not_fabricate_a_strategy_shift():
    """A turn that only discusses guardrails must not earn the next turn a strategy shift.

    The recovery-by-prose scan decided a halt had happened by matching halt
    wording in the most recent assistant message. An assistant turn quoting that
    wording while explaining the mechanism therefore caused the following user
    message to reach the model carrying a fabricated "the previous turn was
    halted by a tool-call guardrail" system instruction, on a turn where no
    guardrail had fired.

    Driven through the public conversation entry point with the prior exchange
    supplied as the session transcript, the way a resumed session presents it,
    so the guarantee covers the whole journey and not just the prologue.
    """
    agent = _make_agent("web_search")
    agent._disable_streaming = True

    # The halt wording, assembled from parts so this fixture can quote it without
    # embedding it as one contiguous literal.
    halt_wording = "hit the " "tool-call " "guardrail"

    # Turn one: an ordinary prose answer that happens to quote the halt wording.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content=(
                f"I stopped retrying terminal because it {halt_wording} "
                "(sequence_repeat_halt) when a command kept failing identically."
            ),
            finish_reason="stop",
        )
    ]
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("how do the tool loop guardrails work?")

    # Nothing halted: the quoted wording was prose, not an event.
    assert first["turn_exit_reason"] != "guardrail_halt"
    assert halt_wording in (first["final_response"] or "")

    # Turn two: an ordinary follow-up on the same live agent, with the first
    # turn's exchange threaded forward the way any caller threads a transcript.
    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="Glad that helped.", finish_reason="stop")

    agent.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        agent.run_conversation(
            "thanks, that is clear",
            conversation_history=[
                {"role": "user", "content": "how do the tool loop guardrails work?"},
                {"role": "assistant", "content": first["final_response"]},
            ],
        )

    user_msgs = [m for m in captured if m.get("role") == "user"]
    assert user_msgs, "the follow-up turn never reached the model"
    assert "MANDATORY STRATEGY SHIFT" not in (user_msgs[-1].get("content") or "")


def test_turn_resumption_after_guardrail_halt_injects_strategy_shift():
    agent = _make_agent("web_search", max_iterations=10, config=_hard_stop_config())
    same_args = {"query": "same"}
    # Turn 1: tool calls repeat and trigger guardrail halt
    responses_t1 = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent.client.chat.completions.create.side_effect = responses_t1
    agent._disable_streaming = True

    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result1 = agent.run_conversation("search repeatedly")

    assert result1["turn_exit_reason"] == "guardrail_halt"
    # The halt was handed over to the turn that follows.
    assert agent._turn_state.guardrails.pending_resumption is not None

    # Turn 2: user replies "what should we do next?"
    captured_messages = []

    def _capture_call(**kwargs):
        msgs = kwargs.get("messages", [])
        captured_messages.extend(msgs)
        return _mock_response(content="I will summarize what I learned.", finish_reason="stop")

    agent.client.chat.completions.create.side_effect = _capture_call

    with (
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        result2 = agent.run_conversation("what should we do next?")

    user_msgs = [m for m in captured_messages if m.get("role") == "user"]
    assert len(user_msgs) >= 1
    last_user_content = user_msgs[-1].get("content", "")
    assert "MANDATORY STRATEGY SHIFT: Do NOT immediately emit another inspection or tool call." in last_user_content
    assert "summarize what you have learned so far" in last_user_content
    # The key means "this turn halted", full stop. A turn that merely delivered a
    # note from the turn before it must not carry one, or the same key would mean
    # two different things and a reader could not tell a halt from a delivery.
    assert "guardrail" not in result2
    # The turn that received the note records the delivery on the one object that
    # holds the guardrail facts, so the handoff's whole journey is observable: the
    # halted turn armed it, this turn spent it.
    assert agent._turn_state.guardrails.resumption_delivered is not None
    assert agent._turn_state.guardrails.pending_resumption is None




# ── The halt record: the account that outlives the agent ────────────────────
#
# Everything above tests a halt within one live agent. These test the gap between
# two turns, which is where the agent object is not guaranteed to survive: a
# restart, a gateway cache eviction, an idle reap, a cron firing that mints a
# fresh agent every time. The in-memory handoff cannot cross any of those, and
# the fallback that used to -- greping the previous assistant's text for the
# phrase "hit the tool-call guardrail" -- was deleted in ticket #2 because it
# fabricated strategy shifts out of prose that merely discussed guardrails.
#
# So the account of a halt is stored, on the session, and read back through one
# narrow port. These are the tests the ticket says do not exist today.
# ─────────────────────────────────────────────────────────────────────────────

def _persisted_agent(session_id: str, *tool_names: str, config: dict | None = None,
                     max_iterations: int = 10):
    """An agent wired to a real session store, the way production wires one.

    `_make_agent` leaves persistence off, which is right for every test above and
    wrong for these: a halt record only matters across the gap between two turns,
    and that gap is only real if the store is.
    """
    from hermes_state import SessionDB

    agent = _make_agent(*tool_names, max_iterations=max_iterations, config=config)
    agent._persist_disabled = False
    agent._session_db = SessionDB()
    agent.session_id = session_id
    return agent


def _halt_a_turn(agent, prompt: str = "search repeatedly"):
    """Drive one turn to a guardrail halt and hand back its result.

    The halt is produced the way the journey test produces it -- repeated
    identical calls under a hard-stop config -- so the record under test is one a
    real halt made, not one a test assembled.
    """
    same_args = {"query": "same"}
    agent.client.chat.completions.create.side_effect = [
        _mock_response(
            content="",
            finish_reason="tool_calls",
            tool_calls=[_mock_tool_call("web_search", json.dumps(same_args), f"c{i}")],
        )
        for i in range(1, 10)
    ]
    agent._disable_streaming = True
    with (
        patch("run_agent.handle_function_call", return_value=json.dumps({"error": "boom"})),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        return agent.run_conversation(prompt)


def test_a_halt_survives_a_restart_and_still_buys_the_strategy_shift(tmp_path):
    """The ticket's premise, tested as stated: a halt the model was never told
    about still buys the strategy shift after the process comes back cold.

    Two agents, one session, one store between them. The second is constructed
    fresh, so it holds no in-memory handoff -- which is the whole point: the note
    it delivers can only have come from the stored account. A test that reused the
    first agent would pass whether or not the record was ever written.
    """
    session_id = "restart-halt-session"
    agent = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    result = _halt_a_turn(agent)
    assert result["turn_exit_reason"] == "guardrail_halt"

    # The account exists, undelivered: the halt stopped the turn and the model
    # has not been told yet.
    stored = agent._session_db.read_guardrail_halt_record(session_id)
    assert stored is not None
    assert stored["delivered"] is False

    # A fresh agent over the same session and the same store: the cold start.
    survivor = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    survivor.client.close = MagicMock()

    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="I will change approach.", finish_reason="stop")

    survivor.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(survivor, "_save_trajectory"),
        patch.object(survivor, "_cleanup_task_resources"),
    ):
        survivor.run_conversation("what should we do next?")

    user_msgs = [m for m in captured if m.get("role") == "user"]
    assert user_msgs, "the resumed turn never reached the model"
    content = user_msgs[-1].get("content", "")
    assert "MANDATORY STRATEGY SHIFT: Do NOT immediately emit another inspection or tool call." in content
    assert "summarize what you have learned so far" in content
    # The note names the halt it answers, so the model can act on the right one.
    assert "web_search" in content


def test_a_halt_survives_the_agent_being_released_from_the_cache(tmp_path):
    """Drive the gateway's real release path, not a stand-in for it.

    Eviction and idle reap are two triggers of one shared method,
    ``GatewayRunner._release_evicted_agent_soft``, so this exercises that method
    directly and thereby covers both. Driving it rather than dropping a local
    reference matters because the method does not merely let the object go: it
    actively empties ``_session_messages`` and drops ``_db_flush_scan_prefix``,
    and its whole stated purpose is reclaiming memory from a cached session.

    That purpose is the regression this test exists for. The plausible next edit
    to a method whose job is freeing memory is to free the halt record with the
    rest of it -- and no other test here would notice, because the restart test
    discards the agent instead of releasing it. So the assertions below first
    prove the real method ran and really mutated the agent, and only then that
    the account outlived it.
    """
    from gateway.run import GatewayRunner

    session_id = "released-halt-session"
    agent = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    assert _halt_a_turn(agent)["turn_exit_reason"] == "guardrail_halt"

    # Precondition for the proof below: there has to be something to reclaim, or
    # "the list is empty afterwards" would pass on an agent that never ran.
    assert getattr(agent, "_session_messages", None), "the halted turn kept no history to free"

    released_clients = MagicMock()
    agent.release_clients = released_clients
    # The method needs only this from `self`, and only on the fallback branch that
    # a real AIAgent never takes because it has `release_clients`.
    runner_self = SimpleNamespace(_cleanup_agent_resources=MagicMock())

    GatewayRunner._release_evicted_agent_soft(runner_self, agent)

    # The real method really ran, and really did its reclaiming. Without these the
    # test would still pass if the call were commented out, which is the whole
    # difference between driving a path and naming it.
    released_clients.assert_called_once()
    assert agent._session_messages == []
    assert agent._db_flush_scan_prefix is None
    runner_self._cleanup_agent_resources.assert_not_called()

    # And the account outlived the reclamation, because it lives in the session
    # row and not in the memory the method exists to free.
    stored = agent._session_db.read_guardrail_halt_record(session_id)
    assert stored is not None
    assert stored["delivered"] is False

    replacement = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    replacement.client.close = MagicMock()
    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="Understood.", finish_reason="stop")

    replacement.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(replacement, "_save_trajectory"),
        patch.object(replacement, "_cleanup_task_resources"),
    ):
        replacement.run_conversation("carry on")

    user_msgs = [m for m in captured if m.get("role") == "user"]
    assert user_msgs, "the turn after the release never reached the model"
    assert "MANDATORY STRATEGY SHIFT" in user_msgs[-1].get("content", "")


def test_a_store_that_refuses_the_record_costs_the_nudge_and_not_the_turn(tmp_path):
    """A halt record is a nudge, not a checkpoint.

    The ticket asks for exactly this asymmetry, and it is the reason the write
    sits inside its own guard rather than beside the ones that decide the turn's
    outcome: a database that will not take the account must not be allowed to
    lose the user their answer. The turn still halts, the same-process handoff is
    still armed, and the failure is said out loud.
    """
    from agent import guardrail_record as record_port

    session_id = "refusing-store-session"
    agent = _persisted_agent(session_id, "web_search", config=_hard_stop_config())

    complaints = []

    class _Stubborn(type(agent._session_db)):
        # Subclassing the real store rather than stubbing it: the point is that
        # ONE write refuses, not that the database is absent. Every other call the
        # turn makes on the store still works, so a failure here can only be the
        # record's own.
        def write_guardrail_halt_record(self, _sid, _record):
            raise RuntimeError("the disk is full")
    agent._session_db = _Stubborn()

    with patch.object(record_port.logger, "warning", side_effect=lambda *a, **k: complaints.append(a)):
        result = _halt_a_turn(agent)

    assert result["turn_exit_reason"] == "guardrail_halt"
    assert "stopped retrying" in (result["final_response"] or "")
    # The user still gets the halt's own explanation, and the same-process handoff
    # is still armed: only the durable copy was lost.
    assert agent._turn_state.guardrails.pending_resumption is not None
    assert complaints, "a lost halt record was swallowed without a word"

    # Criterion 5's third clause. The durable copy never landed, so the route a
    # cold restart would take comes up empty: the nudge lives only in memory, and
    # no later turn is nudged twice for a halt already explained.
    assert record_port.take_halt_decision(agent) is None, (
        "a record the store refused must not resurface on a later turn")


def test_the_halt_record_never_reaches_the_provider_payload(tmp_path):
    """The account is the agent's own business; the provider sees a sentence.

    The stored account is structured data about a guardrail -- its code, its
    count, the hash of the arguments that tripped it. None of that belongs in a
    request. What the model is told is the rendered instruction, in prose, and
    nothing else. This is the leak test the ticket asks for, and it is worth
    stating plainly because the cheapest way to make the durable route work would
    be to hand the model the record and let it read its own history.
    """
    session_id = "no-leak-session"
    agent = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    assert _halt_a_turn(agent)["turn_exit_reason"] == "guardrail_halt"
    stored = agent._session_db.read_guardrail_halt_record(session_id)
    assert stored is not None

    survivor = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    survivor.client.close = MagicMock()
    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="Noted.", finish_reason="stop")

    survivor.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(survivor, "_save_trajectory"),
        patch.object(survivor, "_cleanup_task_resources"),
    ):
        survivor.run_conversation("and now?")

    payload = json.dumps(captured)
    # The stored account as a whole stays out. This is the leak that would matter:
    # the cheap way to wire the durable route is to hand the model the record and
    # let it read its own history, which is the design the ticket rejects.
    assert json.dumps(stored) not in payload
    # The agent's own bookkeeping about whether it has told the model yet. The
    # model has no business knowing that, and it is not part of any rendering.
    assert '"delivered"' not in payload
    # What the model IS told, it is told as prose. The code and the count appear
    # inside the rendered sentence on purpose -- a refusal the model cannot act on
    # is a worse outcome than a named one -- so the assertion here is about shape,
    # not about absence.
    user_msgs = [m for m in captured if m.get("role") == "user"]
    assert "MANDATORY STRATEGY SHIFT" in user_msgs[-1].get("content", "")


def test_nothing_reads_a_halt_out_of_prose_that_merely_discussed_guardrails(tmp_path):
    """The durable route must not become the deleted one by another name.

    Ticket #2 removed a fallback that recovered a halt by searching the previous
    assistant message for the phrase "hit the tool-call guardrail", because it
    fabricated strategy shifts whenever an assistant merely talked about
    guardrails. Storing the account is the answer to that, so the durable route
    has to fail the same way the deleted route would have: an assistant turn full
    of halt wording, and no stored account, must buy no instruction.
    """
    session_id = "prose-only-session"
    agent = _persisted_agent(session_id, "web_search", config=_hard_stop_config())

    halt_wording = (
        "I stopped retrying web_search because it hit the tool-call guardrail "
        "(repeated_exact_failure_block) after 3 repeated non-progressing attempts."
    )
    survivor = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    survivor.client.close = MagicMock()
    assert survivor._session_db.read_guardrail_halt_record(session_id) is None

    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="Right, continuing.", finish_reason="stop")

    survivor.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(survivor, "_save_trajectory"),
        patch.object(survivor, "_cleanup_task_resources"),
    ):
        survivor.run_conversation(
            "what next?",
            conversation_history=[
                {"role": "user", "content": "tell me about the guardrails"},
                {"role": "assistant", "content": halt_wording},
            ],
        )

    user_msgs = [m for m in captured if m.get("role") == "user"]
    assert user_msgs, "the turn never reached the model"
    assert "MANDATORY STRATEGY SHIFT" not in user_msgs[-1].get("content", "")


def test_a_delivered_note_is_not_delivered_twice_across_a_restart(tmp_path):
    """Delivery is recorded, so a cold start cannot repeat an instruction.

    The account outlives its own delivery so it can be audited, which means
    reading it has to ask whether it was already acted on. Without that the
    first cold turn after a later restart would nudge a second time for a halt
    the model was told about hours earlier -- the same class of bug as the prose
    fallback, arriving from the opposite direction.
    """
    session_id = "delivered-once-session"
    agent = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    assert _halt_a_turn(agent)["turn_exit_reason"] == "guardrail_halt"

    # The same-process route serves the next turn, and records the delivery.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="I will change approach.", finish_reason="stop")
    ]
    agent._disable_streaming = True
    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        told = agent.run_conversation("what now?")
    assert told["guardrail_resumption_delivered"] is True
    assert agent._session_db.read_guardrail_halt_record(session_id)["delivered"] is True

    # And a cold start over the same session owes nothing: it was already told.
    survivor = _persisted_agent(session_id, "web_search", config=_hard_stop_config())
    survivor.client.close = MagicMock()
    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="Carrying on.", finish_reason="stop")

    survivor.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(survivor, "_save_trajectory"),
        patch.object(survivor, "_cleanup_task_resources"),
    ):
        survivor.run_conversation("and then?")

    user_msgs = [m for m in captured if m.get("role") == "user"]
    assert user_msgs
    assert "MANDATORY STRATEGY SHIFT" not in user_msgs[-1].get("content", "")


def test_a_session_predating_the_change_carries_a_null_record_and_no_fabricated_nudge(
    tmp_path,
):
    """A session that existed before the record shipped must behave, not merely survive.

    The column arrives NULL for every such session, so the durable route has to read
    'nothing owed' out of an absent account rather than fall through to something
    invented. A null read that defaulted to a nudge would hand every old session a
    strategy shift it never earned, which is the opposite of the ticket's promise.
    """
    session_id = "predating-session"
    agent = _persisted_agent(session_id, "web_search")

    # One ordinary turn, so the row comes into being with history and still no account.
    agent.client.chat.completions.create.side_effect = [
        _mock_response(content="all good", finish_reason="stop", tool_calls=None),
    ]
    with (
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
    ):
        first = agent.run_conversation("hi there")
    assert first["turn_exit_reason"].startswith("text_response")

    stored = agent._session_db.read_guardrail_halt_record(session_id)
    assert stored is None, "a session that never halted has no account"

    # A fresh agent on the same session, the way a cold resume arrives at it.
    survivor = _persisted_agent(session_id, "web_search")
    survivor.client.close = MagicMock()
    captured: list = []

    def _capture(**kwargs):
        captured.extend(kwargs.get("messages", []))
        return _mock_response(content="fine", finish_reason="stop", tool_calls=None)

    survivor.client.chat.completions.create.side_effect = _capture
    with (
        patch.object(survivor, "_save_trajectory"),
        patch.object(survivor, "_cleanup_task_resources"),
    ):
        survivor.run_conversation("and now?")

    payload = json.dumps(captured)
    assert "MANDATORY STRATEGY SHIFT" not in payload
    assert "summarize what you have learned" not in payload
    assert survivor._session_db.read_guardrail_halt_record(session_id) is None


def test_one_firing_neither_reads_nor_writes_another_firings_record(tmp_path):
    """Two sessions sharing one store keep two separate accounts.

    A cron firing mints a fresh agent every time, and the gateway reuses agents
    across sessions, so the account has to be keyed by session and nothing else.
    A record that bled across would nudge a run that never halted, and a write
    that landed in the wrong row would spend another firing's nudge before it
    arrived. Both are checked here, on one shared store.
    """
    first = _persisted_agent("firing-one", "web_search", config=_hard_stop_config())
    assert _halt_a_turn(first)["turn_exit_reason"] == "guardrail_halt"
    record_one = first._session_db.read_guardrail_halt_record("firing-one")
    assert record_one is not None

    # A different session, same store, same tools: it has halted nothing.
    other = _persisted_agent("firing-two", "web_search", config=_hard_stop_config())
    assert other._session_db.read_guardrail_halt_record("firing-two") is None

    # And when the second one halts in its own right, it spends its own account
    # and leaves the first one's exactly as it was.
    assert _halt_a_turn(other)["turn_exit_reason"] == "guardrail_halt"
    record_two = other._session_db.read_guardrail_halt_record("firing-two")
    assert record_two is not None
    assert other._session_db.read_guardrail_halt_record("firing-one") == record_one

    # Each cold resume collects its own nudge, and only its own.
    resume_one = _persisted_agent("firing-one", "web_search")
    resume_one.client.close = MagicMock()
    seen_one: list = []

    def _capture_one(**kwargs):
        seen_one.extend(kwargs.get("messages", []))
        return _mock_response(content="ok", finish_reason="stop", tool_calls=None)

    resume_one.client.chat.completions.create.side_effect = _capture_one
    with (
        patch.object(resume_one, "_save_trajectory"),
        patch.object(resume_one, "_cleanup_task_resources"),
    ):
        resume_one.run_conversation("and now?")

    assert any(
        "MANDATORY STRATEGY SHIFT" in m.get("content", "")
        for m in seen_one if m.get("role") == "user"
    )
    assert json.dumps(seen_one).count("MANDATORY STRATEGY SHIFT") == 1
