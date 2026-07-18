"""Concurrent resumes are serialized per thread inside one backend process."""

import asyncio
import unittest
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from ag_ui.core import EventType, ResumeEntry, RunAgentInput
from ag_ui_langgraph import LangGraphAgent


class ApprovalState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def _resume_input(*, run_id: str, interrupt_id: str) -> RunAgentInput:
    return RunAgentInput(
        thread_id="shared-thread",
        run_id=run_id,
        state={},
        messages=[],
        tools=[],
        context=[],
        forwarded_props={},
        resume=[
            ResumeEntry(
                interrupt_id=interrupt_id,
                status="resolved",
                payload={"approved": True},
            )
        ],
    )


class TestConcurrentResumeLock(unittest.IsolatedAsyncioTestCase):
    async def test_two_clones_resume_one_real_checkpoint_exactly_once(self):
        side_effects = []

        async def approval_node(state):
            answer = interrupt(
                {"reason": "confirmation", "message": "Approve this action?"}
            )
            # Keep the winning stream open long enough for an unlocked loser to
            # read the same checkpoint and dispatch a second Command(resume=...).
            await asyncio.sleep(0.05)
            side_effects.append(answer)
            return {}

        builder = StateGraph(ApprovalState)
        builder.add_node("approval", approval_node)
        builder.add_edge(START, "approval")
        builder.add_edge("approval", END)
        graph = builder.compile(checkpointer=InMemorySaver())
        config = {"configurable": {"thread_id": "shared-thread"}}
        await graph.ainvoke({"messages": []}, config)
        checkpoint = await graph.aget_state(config)
        interrupt_id = checkpoint.tasks[0].interrupts[0].id

        # Widen the read/dispatch race deterministically. With the per-thread
        # lock, only the winner reaches this delay while the checkpoint is open.
        original_aget_state = graph.aget_state

        async def delayed_aget_state(run_config, *args, **kwargs):
            state = await original_aget_state(run_config, *args, **kwargs)
            if any(task.interrupts for task in state.tasks):
                await asyncio.sleep(0.05)
            return state

        graph.aget_state = delayed_aget_state

        dispatches = 0
        original_astream_events = graph.astream_events

        def counted_astream_events(*args, **kwargs):
            nonlocal dispatches
            dispatches += 1
            return original_astream_events(*args, **kwargs)

        graph.astream_events = counted_astream_events

        template = LangGraphAgent(name="approval", graph=graph)
        first = template.clone()
        second = template.clone()

        async def collect(agent, input_data):
            return [event async for event in agent.run(input_data)]

        first_events, second_events = await asyncio.gather(
            collect(first, _resume_input(run_id="run-1", interrupt_id=interrupt_id)),
            collect(second, _resume_input(run_id="run-2", interrupt_id=interrupt_id)),
        )

        event_types = [
            [event.type for event in first_events],
            [event.type for event in second_events],
        ]
        self.assertEqual(dispatches, 1)
        self.assertEqual(len(side_effects), 1)
        self.assertEqual(
            sum(EventType.RUN_ERROR in events for events in event_types),
            1,
        )
        self.assertEqual(
            sum(EventType.RUN_FINISHED in events for events in event_types),
            1,
        )


if __name__ == "__main__":
    unittest.main()
