"""Tests for interrupt mapping hooks and native resume commands."""

import unittest
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import MagicMock

from ag_ui.core import Interrupt as AGUIInterrupt, ResumeEntry, RunFinishedEvent
from langgraph.types import Command

from ag_ui_langgraph.agent import LangGraphAgent
from ag_ui_langgraph.interrupts import lg_interrupt_to_agui, lg_interrupts_to_agui
from tests._helpers import make_agent


@dataclass
class FakeInterrupt:
    value: Any
    id: str = "int-default"


class TestDefaultHooks(unittest.TestCase):
    def test_vectorized_interrupt_hook_matches_module(self):
        agent = make_agent()
        interrupts = [
            FakeInterrupt(value="string value", id="int-1"),
            FakeInterrupt(value={"reason": "r2"}, id="int-2"),
        ]

        self.assertEqual(
            agent._interrupts_to_agui(interrupts),
            lg_interrupts_to_agui(interrupts),
        )

    def test_resume_builder_uses_native_interrupt_id_map(self):
        agent = make_agent()
        entries = [
            ResumeEntry(
                interrupt_id="i1",
                status="resolved",
                payload={"approved": True},
            ),
            ResumeEntry(interrupt_id="i2", status="cancelled", payload=None),
        ]

        command = agent._build_command_from_agui_resume(entries)

        self.assertIsInstance(command, Command)
        self.assertEqual(
            command.resume,
            {"i1": {"approved": True}, "i2": None},
        )


class FanOutAgent(LangGraphAgent):
    def _interrupts_to_agui(self, lg_interrupts) -> List[AGUIInterrupt]:
        result: List[AGUIInterrupt] = []
        for interrupt in lg_interrupts:
            value = interrupt.value
            if isinstance(value, dict) and "action_requests" in value:
                for request in value["action_requests"]:
                    result.append(
                        AGUIInterrupt(
                            id=f"fan-{request['id']}",
                            reason=request.get("reason", "langgraph:interrupt"),
                            metadata={"langgraph": {"raw": value}},
                        )
                    )
            else:
                result.append(lg_interrupt_to_agui(interrupt))
        return result


class TestSubclassFanOut(unittest.TestCase):
    def test_standard_outcome_contains_every_fanned_out_interrupt(self):
        agent = FanOutAgent(name="test", graph=MagicMock())
        events = agent._emit_interrupt_finish(
            thread_id="t1",
            run_id="run-1",
            lg_interrupts=[
                FakeInterrupt(
                    value={
                        "action_requests": [
                            {"id": "a1", "reason": "approve A"},
                            {"id": "a2", "reason": "approve B"},
                        ]
                    },
                    id="int-1",
                )
            ],
        )

        self.assertEqual(len(events), 1)
        self.assertIsInstance(events[0], RunFinishedEvent)
        self.assertEqual(events[0].outcome.type, "interrupt")
        self.assertEqual(
            [interrupt.id for interrupt in events[0].outcome.interrupts],
            ["fan-a1", "fan-a2"],
        )


if __name__ == "__main__":
    unittest.main()
