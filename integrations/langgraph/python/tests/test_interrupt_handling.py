"""Tests for parallel interrupt collection and standard outcome mapping."""

import unittest
from dataclasses import dataclass, field
from typing import Any, List
from unittest.mock import MagicMock

import pytest

from ag_ui.core import EventType, RunFinishedEvent

from ag_ui_langgraph.agent import LangGraphAgent
from ag_ui_langgraph.interrupts import lg_interrupt_to_agui


@dataclass
class FakeInterrupt:
    value: Any
    id: Any = None


@dataclass
class FakeTask:
    interrupts: List[FakeInterrupt] = field(default_factory=list)


def make_agent():
    return LangGraphAgent(name="test", graph=MagicMock())


class TestCollectInterrupts(unittest.TestCase):
    def test_collects_interrupts_from_every_task(self):
        tasks = [
            FakeTask(interrupts=[FakeInterrupt(value="A")]),
            FakeTask(interrupts=[]),
            FakeTask(interrupts=[FakeInterrupt(value="B")]),
        ]

        interrupts = make_agent()._collect_interrupts(tasks)

        self.assertEqual([interrupt.value for interrupt in interrupts], ["A", "B"])

    def test_handles_empty_none_and_malformed_tasks(self):
        class BareTask:
            pass

        agent = make_agent()
        self.assertEqual(agent._collect_interrupts(None), [])
        self.assertEqual(agent._collect_interrupts([]), [])
        self.assertEqual(agent._collect_interrupts([BareTask(), {}]), [])


class TestEmitInterruptFinish(unittest.TestCase):
    def test_emits_only_standard_interrupt_outcome(self):
        agent = make_agent()
        events = agent._emit_interrupt_finish(
            thread_id="t1",
            run_id="run-1",
            lg_interrupts=[
                FakeInterrupt(
                    value={"reason": "confirmation", "message": "ok?"},
                    id="int-1",
                ),
                FakeInterrupt(value="second", id="int-2"),
            ],
        )

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], RunFinishedEvent)
        self.assertEqual(events[0].type, EventType.RUN_FINISHED)
        self.assertEqual(events[0].outcome.type, "interrupt")
        self.assertEqual(
            [interrupt.id for interrupt in events[0].outcome.interrupts],
            ["int-1", "int-2"],
        )

    def test_deprecated_flags_cannot_disable_standard_outcome(self):
        with self.assertWarns(DeprecationWarning):
            agent = LangGraphAgent(
                name="test",
                graph=MagicMock(),
                enable_legacy_on_interrupt_event=True,
                emit_interrupt_outcome=False,
            )

        events = agent._emit_interrupt_finish(
            thread_id="t1",
            run_id="run-1",
            lg_interrupts=[FakeInterrupt(value="confirm", id="int-1")],
        )

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], RunFinishedEvent)
        self.assertEqual(events[0].outcome.type, "interrupt")


class TestInterruptMappingHardening(unittest.TestCase):
    def test_missing_langgraph_id_raises(self):
        with pytest.raises(ValueError, match="missing `id`"):
            lg_interrupt_to_agui(FakeInterrupt(value="confirm", id=None))

    def test_real_id_and_falsy_fields_are_preserved(self):
        result = lg_interrupt_to_agui(
            FakeInterrupt(
                value={
                    "reason": "",
                    "toolCallId": "",
                    "responseSchema": {},
                },
                id="lg-real-42",
            )
        )

        self.assertEqual(result.id, "lg-real-42")
        self.assertEqual(result.reason, "")
        self.assertEqual(result.tool_call_id, "")
        self.assertEqual(result.response_schema, {})


if __name__ == "__main__":
    unittest.main()
