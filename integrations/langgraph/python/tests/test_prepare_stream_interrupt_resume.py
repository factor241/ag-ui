"""Regression tests for strict interrupt-resume ordering (#1743)."""

import unittest
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ag_ui.core import EventType, ResumeEntry, UserMessage

from tests._helpers import make_agent


@dataclass
class FakeInterrupt:
    value: Any
    id: str = "fake-interrupt"


@dataclass
class FakeTask:
    interrupts: List[FakeInterrupt] = field(default_factory=list)


def _make_state(messages, tasks=None):
    state = MagicMock()
    state.values = {"messages": messages}
    state.tasks = tasks or []
    state.next = []
    state.metadata = {"writes": {}}
    return state


def _make_input(messages, *, forwarded_props=None, resume=None):
    value = MagicMock()
    value.thread_id = "t1"
    value.run_id = "run-1"
    value.messages = messages
    value.state = {}
    value.tools = []
    value.context = []
    value.forwarded_props = forwarded_props or {}
    value.resume = resume
    return value


async def _empty_stream():
    if False:
        yield None


def _checkpoint_signature(messages):
    return [
        (
            type(message).__name__,
            getattr(message, "id", None),
            deepcopy(getattr(message, "content", None)),
            deepcopy(getattr(message, "tool_calls", None)),
        )
        for message in messages
    ]


class TestPrepareStreamInterruptResumeOrdering(unittest.IsolatedAsyncioTestCase):
    async def test_handle_stream_events_uses_forwarded_node_name_for_continue_mode(self):
        agent = make_agent()
        checkpoint_messages = [HumanMessage(id="h1", content="do something")]
        agent.graph.aget_state = AsyncMock(
            return_value=_make_state(messages=checkpoint_messages)
        )
        agent.graph.astream_events.return_value = _empty_stream()
        input_data = _make_input(
            [
                UserMessage(id="h1", role="user", content="do something"),
                UserMessage(id="h2", role="user", content="follow up"),
            ],
            forwarded_props={"node_name": "approval_node"},
        )

        async for _ in agent._handle_stream_events(input_data):
            pass

        agent.graph.aupdate_state.assert_awaited_once()
        self.assertEqual(
            agent.graph.aupdate_state.await_args.kwargs.get("as_node"),
            "approval_node",
        )

    async def test_standard_resume_with_interrupt_bypasses_regenerate(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}
        checkpoint_messages = [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
        state = _make_state(
            checkpoint_messages,
            [FakeTask(interrupts=[FakeInterrupt(value="Approve?", id="int-1")])],
        )
        input_data = _make_input(
            [UserMessage(id="h1", role="user", content="do something")],
            resume=[
                ResumeEntry(
                    interrupt_id="int-1",
                    status="resolved",
                    payload="yes",
                )
            ],
        )
        agent.prepare_regenerate_stream = AsyncMock()
        before = _checkpoint_signature(checkpoint_messages)

        result = await agent.prepare_stream(
            input_data,
            state,
            {"configurable": {"thread_id": "t1"}},
        )

        agent.prepare_regenerate_stream.assert_not_awaited()
        self.assertIsNotNone(result["stream"])
        self.assertEqual(before, _checkpoint_signature(checkpoint_messages))
        stream_input = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(stream_input, Command)
        self.assertEqual(stream_input.resume, {"int-1": "yes"})

    async def test_interrupt_without_resume_emits_standard_outcome(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}
        state = _make_state(
            [HumanMessage(id="h1", content="do something")],
            [FakeTask(interrupts=[FakeInterrupt(value="Approve?", id="int-1")])],
        )
        input_data = _make_input(
            [UserMessage(id="h1", role="user", content="do something")]
        )

        result = await agent.prepare_stream(
            input_data,
            state,
            {"configurable": {"thread_id": "t1"}},
        )

        events = result["events_to_dispatch"]
        self.assertEqual(
            [event.type for event in events],
            [EventType.RUN_STARTED, EventType.RUN_FINISHED],
        )
        self.assertEqual(events[-1].outcome.type, "interrupt")

    async def test_no_interrupt_normal_flow_produces_stream(self):
        agent = make_agent()
        agent.active_run = {"id": "run-1", "mode": "start"}
        state = _make_state(
            [HumanMessage(id="h1", content="hello")],
            [FakeTask()],
        )
        input_data = _make_input(
            [
                UserMessage(id="h1", role="user", content="hello"),
                UserMessage(id="h2", role="user", content="follow up"),
            ]
        )

        result = await agent.prepare_stream(
            input_data,
            state,
            {"configurable": {"thread_id": "t1"}},
        )

        self.assertIsNotNone(result["stream"])


class TestCheckpointSignature(unittest.TestCase):
    def test_checkpoint_signature_does_not_retain_mutable_message_references(self):
        messages = [
            AIMessage(
                id="ai1",
                content=[{"type": "text", "text": "before"}],
                tool_calls=[
                    {
                        "id": "tc-1",
                        "name": "approval",
                        "args": {"approved": False},
                    }
                ],
            )
        ]

        before = _checkpoint_signature(messages)
        messages[0].content[0]["text"] = "after"  # type: ignore[index]
        messages[0].tool_calls[0]["args"]["approved"] = True

        self.assertNotEqual(before, _checkpoint_signature(messages))


if __name__ == "__main__":
    unittest.main()
