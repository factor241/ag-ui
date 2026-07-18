"""Strict AG-UI resume contract for the LangGraph Python integration."""

import unittest
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from ag_ui.core import CustomEvent, EventType, ResumeEntry, UserMessage
from ag_ui_langgraph.agent import _ResumeClaimRegistry

from tests._helpers import make_agent


@dataclass
class FakeInterrupt:
    value: Any
    id: str


@dataclass
class FakeTask:
    interrupts: List[FakeInterrupt] = field(default_factory=list)


def _state(*interrupt_ids: str):
    state = MagicMock()
    state.values = {
        "messages": [
            HumanMessage(id="h1", content="do something"),
            AIMessage(
                id="ai1",
                content="",
                tool_calls=[{"id": "tc-1", "name": "approval", "args": {}}],
            ),
        ]
    }
    state.tasks = [
        FakeTask(
            interrupts=[
                FakeInterrupt(
                    value={"reason": "confirmation", "message": interrupt_id},
                    id=interrupt_id,
                )
                for interrupt_id in interrupt_ids
            ]
        )
    ]
    state.next = []
    state.metadata = {"writes": {}}
    return state


def _state_with_expiry(expires_at: str):
    state = _state()
    state.tasks = [
        FakeTask(
            interrupts=[
                FakeInterrupt(
                    value={
                        "reason": "confirmation",
                        "message": "int-1",
                        "expiresAt": expires_at,
                    },
                    id="int-1",
                )
            ]
        )
    ]
    return state


def _input(*, resume=None, forwarded_props=None):
    value = MagicMock()
    value.thread_id = "thread-1"
    value.run_id = "run-1"
    value.messages = [UserMessage(id="h1", role="user", content="do something")]
    value.state = {}
    value.tools = []
    value.context = []
    value.forwarded_props = forwarded_props or {}
    value.resume = resume
    return value


async def _prepare(agent, state, input_data):
    agent.active_run = {"id": "run-1", "mode": "start"}
    return await agent.prepare_stream(
        input_data,
        state,
        {"configurable": {"thread_id": "thread-1"}},
    )


def _event_types(result):
    return [event.type for event in result.get("events_to_dispatch", [])]


class TestStrictResumeContract(unittest.IsolatedAsyncioTestCase):
    async def test_interrupted_run_uses_only_standard_outcome_with_all_open_interrupts(self):
        agent = make_agent()

        result = await _prepare(agent, _state("int-1", "int-2"), _input())

        events = result["events_to_dispatch"]
        self.assertEqual(_event_types(result), [EventType.RUN_STARTED, EventType.RUN_FINISHED])
        self.assertFalse(any(isinstance(event, CustomEvent) for event in events))
        self.assertEqual(events[-1].outcome.type, "interrupt")
        self.assertEqual(
            [interrupt.id for interrupt in events[-1].outcome.interrupts],
            ["int-1", "int-2"],
        )
        agent.graph.astream_events.assert_not_called()

    async def test_full_resume_set_builds_one_native_command_and_dispatches_once(self):
        agent = make_agent()
        resume = [
            ResumeEntry(
                interrupt_id="int-2",
                status="resolved",
                payload={"approved": False},
            ),
            ResumeEntry(
                interrupt_id="int-1",
                status="resolved",
                payload={"approved": True},
            ),
        ]

        result = await _prepare(agent, _state("int-1", "int-2"), _input(resume=resume))

        self.assertIsNotNone(result["stream"])
        agent.graph.astream_events.assert_called_once()
        command = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertIsInstance(command, Command)
        self.assertEqual(
            command.resume,
            {
                "int-2": {"approved": False},
                "int-1": {"approved": True},
            },
        )

    async def test_cancelled_entry_uses_native_interrupt_id_mapping_without_sentinel(self):
        agent = make_agent()
        resume = [
            ResumeEntry(interrupt_id="int-1", status="cancelled", payload=None),
        ]

        await _prepare(agent, _state("int-1"), _input(resume=resume))

        command = agent.graph.astream_events.call_args.kwargs["input"]
        self.assertEqual(command.resume, {"int-1": None})

    async def test_cancelled_entry_rejects_every_non_null_payload(self):
        for payload in (False, 0, "", {}, [], True):
            with self.subTest(payload=payload):
                agent = make_agent()
                resume = [
                    ResumeEntry(
                        interrupt_id="int-1",
                        status="cancelled",
                        payload=payload,
                    ),
                ]

                result = await _prepare(agent, _state("int-1"), _input(resume=resume))

                self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
                agent.graph.astream_events.assert_not_called()

    async def test_resume_before_utc_expiry_is_accepted(self):
        agent = make_agent()
        expires_at = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload=True)]

        result = await _prepare(
            agent,
            _state_with_expiry(expires_at),
            _input(resume=resume),
        )

        self.assertIsNotNone(result["stream"])
        agent.graph.astream_events.assert_called_once()

    async def test_expired_or_equal_utc_expiry_fails_closed(self):
        for expires_at in (
            (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            datetime.now(timezone.utc).isoformat(),
        ):
            with self.subTest(expires_at=expires_at):
                agent = make_agent()
                resume = [
                    ResumeEntry(interrupt_id="int-1", status="resolved", payload=True)
                ]

                result = await _prepare(
                    agent,
                    _state_with_expiry(expires_at),
                    _input(resume=resume),
                )

                self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
                agent.graph.astream_events.assert_not_called()

    async def test_expiry_exactly_equal_to_frozen_utc_now_fails_closed(self):
        frozen_now = datetime(2030, 1, 1, 12, 0, tzinfo=timezone.utc)

        class FrozenDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return frozen_now if tz is not None else frozen_now.replace(tzinfo=None)

        agent = make_agent()
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload=True)]

        with patch("ag_ui_langgraph.agent.datetime", FrozenDateTime):
            result = await _prepare(
                agent,
                _state_with_expiry(frozen_now.isoformat()),
                _input(resume=resume),
            )

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_malformed_or_naive_expiry_fails_closed(self):
        for expires_at in ("not-a-timestamp", "2030-01-01T00:00:00"):
            with self.subTest(expires_at=expires_at):
                agent = make_agent()
                resume = [
                    ResumeEntry(interrupt_id="int-1", status="resolved", payload=True)
                ]

                result = await _prepare(
                    agent,
                    _state_with_expiry(expires_at),
                    _input(resume=resume),
                )

                self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
                agent.graph.astream_events.assert_not_called()

    async def test_claim_capacity_exhaustion_fails_closed_without_dispatch(self):
        agent = make_agent()
        registry = _ResumeClaimRegistry(max_claims=1)
        registry.try_claim(("other-thread", "checkpoint", ("other-interrupt",)))
        agent._resume_claim_registry = registry
        resume = [ResumeEntry(interrupt_id="int-1", status="resolved", payload=True)]

        result = await _prepare(agent, _state("int-1"), _input(resume=resume))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        self.assertIn(
            "capacity",
            result["events_to_dispatch"][0].message,
        )
        agent.graph.astream_events.assert_not_called()

    async def test_partial_resume_is_run_error_without_graph_dispatch(self):
        agent = make_agent()
        resume = [
            ResumeEntry(interrupt_id="int-1", status="resolved", payload=True),
        ]

        result = await _prepare(agent, _state("int-1", "int-2"), _input(resume=resume))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_explicit_empty_resume_is_partial_and_fails_closed(self):
        agent = make_agent()

        result = await _prepare(agent, _state("int-1"), _input(resume=[]))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_duplicate_resume_id_is_run_error_without_graph_dispatch(self):
        agent = make_agent()
        resume = [
            ResumeEntry(interrupt_id="int-1", status="resolved", payload=True),
            ResumeEntry(interrupt_id="int-1", status="resolved", payload=False),
        ]

        result = await _prepare(agent, _state("int-1", "int-2"), _input(resume=resume))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_unknown_or_stale_resume_id_is_run_error_without_graph_dispatch(self):
        agent = make_agent()
        resume = [
            ResumeEntry(interrupt_id="old-int", status="resolved", payload=True),
        ]

        result = await _prepare(agent, _state("current-int"), _input(resume=resume))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_reused_resume_after_interrupt_closed_is_run_error_without_graph_dispatch(self):
        agent = make_agent()
        resume = [
            ResumeEntry(interrupt_id="int-1", status="resolved", payload=True),
        ]

        result = await _prepare(agent, _state(), _input(resume=resume))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_malformed_resume_entry_is_run_error_without_graph_dispatch(self):
        agent = make_agent()
        malformed = ResumeEntry.model_construct(
            interrupt_id="",
            status="not-a-status",
            payload=True,
        )

        result = await _prepare(agent, _state("int-1"), _input(resume=[malformed]))

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_legacy_resume_is_run_error_without_graph_dispatch(self):
        agent = make_agent()

        result = await _prepare(
            agent,
            _state("int-1"),
            _input(forwarded_props={"command": {"resume": True}}),
        )

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()

    async def test_mixed_standard_and_legacy_resume_is_run_error_without_graph_dispatch(self):
        agent = make_agent()
        resume = [
            ResumeEntry(interrupt_id="int-1", status="resolved", payload=True),
        ]

        result = await _prepare(
            agent,
            _state("int-1"),
            _input(
                resume=resume,
                forwarded_props={"command": {"resume": False}},
            ),
        )

        self.assertEqual(_event_types(result), [EventType.RUN_ERROR])
        agent.graph.astream_events.assert_not_called()


if __name__ == "__main__":
    unittest.main()
