"""Concurrent resumes are serialized per thread inside one backend process."""

import asyncio
import json
import unittest
from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from ag_ui.core import EventType, ResumeEntry, RunAgentInput
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint


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


class FreshRegistryCloneAgent(LangGraphAgent):
    """A valid subclass override that accidentally creates fresh guards."""

    def __init__(self, *, name, graph, api_key, description=None, config=None):
        super().__init__(
            name=name,
            graph=graph,
            description=description,
            config=config,
        )
        self.api_key = api_key

    def clone(self):
        return type(self)(
            name=self.name,
            graph=self.graph,
            api_key=self.api_key,
            description=self.description,
            config=dict(self.config),
        )


async def _make_paused_graph():
    side_effects = []

    async def approval_node(state):
        answer = interrupt(
            {"reason": "confirmation", "message": "Approve this action?"}
        )
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

    original_aget_state = graph.aget_state

    async def delayed_aget_state(run_config, *args, **kwargs):
        state = await original_aget_state(run_config, *args, **kwargs)
        if any(task.interrupts for task in state.tasks):
            await asyncio.sleep(0.05)
        return state

    graph.aget_state = delayed_aget_state

    metrics = {"dispatches": 0}
    original_astream_events = graph.astream_events

    def counted_astream_events(*args, **kwargs):
        metrics["dispatches"] += 1
        return original_astream_events(*args, **kwargs)

    graph.astream_events = counted_astream_events
    return graph, interrupt_id, side_effects, metrics


async def _make_side_effect_blocked_graph():
    side_effects = []
    side_effect_reached = asyncio.Event()
    release_node = asyncio.Event()

    async def approval_node(state):
        answer = interrupt(
            {"reason": "confirmation", "message": "Approve this action?"}
        )
        side_effects.append(answer)
        side_effect_reached.set()
        await release_node.wait()
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

    metrics = {"dispatches": 0}
    original_astream_events = graph.astream_events

    def counted_astream_events(*args, **kwargs):
        metrics["dispatches"] += 1
        return original_astream_events(*args, **kwargs)

    graph.astream_events = counted_astream_events
    return (
        graph,
        interrupt_id,
        side_effects,
        side_effect_reached,
        release_node,
        metrics,
    )


class TestConcurrentResumeLock(unittest.IsolatedAsyncioTestCase):
    async def test_two_clones_resume_one_real_checkpoint_exactly_once(self):
        graph, interrupt_id, side_effects, metrics = await _make_paused_graph()

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
        self.assertEqual(metrics["dispatches"], 1)
        self.assertEqual(len(side_effects), 1)
        self.assertEqual(
            sum(EventType.RUN_ERROR in events for events in event_types),
            1,
        )
        self.assertEqual(
            sum(EventType.RUN_FINISHED in events for events in event_types),
            1,
        )

    async def test_cancelled_winner_claim_is_not_released_for_waiting_duplicate(self):
        (
            graph,
            interrupt_id,
            side_effects,
            side_effect_reached,
            release_node,
            metrics,
        ) = await _make_side_effect_blocked_graph()
        template = LangGraphAgent(name="approval", graph=graph)
        winner = template.clone()
        duplicate = template.clone()

        async def collect(agent, *, run_id):
            return [
                event
                async for event in agent.run(
                    _resume_input(run_id=run_id, interrupt_id=interrupt_id)
                )
            ]

        winner_task = asyncio.create_task(collect(winner, run_id="run-winner"))
        await asyncio.wait_for(side_effect_reached.wait(), timeout=1)
        self.assertEqual(len(side_effects), 1)

        async def collect_duplicate():
            return await collect(duplicate, run_id="run-duplicate")

        duplicate_task = asyncio.create_task(collect_duplicate())
        await asyncio.sleep(0.02)
        self.assertFalse(duplicate_task.done())

        winner_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await winner_task
        try:
            duplicate_events = await asyncio.wait_for(duplicate_task, timeout=1)
        finally:
            release_node.set()

        self.assertEqual(metrics["dispatches"], 1)
        self.assertEqual(len(side_effects), 1)
        self.assertEqual(
            [event.type for event in duplicate_events],
            [EventType.RUN_ERROR],
        )

    async def test_http_disconnect_after_side_effect_keeps_resume_claim(self):
        (
            graph,
            interrupt_id,
            side_effects,
            side_effect_reached,
            release_node,
            metrics,
        ) = await _make_side_effect_blocked_graph()
        template = LangGraphAgent(name="approval", graph=graph)
        app = FastAPI()
        add_langgraph_fastapi_endpoint(app, template, "/agent")
        body = _resume_input(
            run_id="run-disconnected",
            interrupt_id=interrupt_id,
        ).model_dump(by_alias=True)
        body_bytes = json.dumps(body).encode()
        request_delivered = False

        async def receive():
            nonlocal request_delivered
            if not request_delivered:
                request_delivered = True
                return {"type": "http.request", "body": body_bytes, "more_body": False}
            await side_effect_reached.wait()
            return {"type": "http.disconnect"}

        sent_messages = []

        async def send(message):
            sent_messages.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/agent",
            "raw_path": b"/agent",
            "query_string": b"",
            "root_path": "",
            "headers": [
                (b"content-type", b"application/json"),
                (b"accept", b"text/event-stream"),
            ],
            "client": ("127.0.0.1", 12345),
            "server": ("test", 80),
        }

        await asyncio.wait_for(app(scope, receive, send), timeout=1)
        self.assertTrue(any(item["type"] == "http.response.start" for item in sent_messages))
        self.assertEqual(len(side_effects), 1)

        duplicate_body = _resume_input(
            run_id="run-duplicate",
            interrupt_id=interrupt_id,
        ).model_dump(by_alias=True)
        try:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                duplicate = await client.post("/agent", json=duplicate_body)
        finally:
            release_node.set()

        self.assertEqual(duplicate.status_code, 200)
        self.assertIn("RUN_ERROR", duplicate.text)
        self.assertEqual(metrics["dispatches"], 1)
        self.assertEqual(len(side_effects), 1)

    async def test_endpoint_overrides_fresh_subclass_clone_guards(self):
        graph, interrupt_id, side_effects, metrics = await _make_paused_graph()
        template = FreshRegistryCloneAgent(
            name="approval",
            graph=graph,
            api_key="required-secret",
        )
        app = FastAPI()
        add_langgraph_fastapi_endpoint(app, template, "/agent")
        first_body = _resume_input(
            run_id="run-1",
            interrupt_id=interrupt_id,
        ).model_dump(by_alias=True)
        second_body = _resume_input(
            run_id="run-2",
            interrupt_id=interrupt_id,
        ).model_dump(by_alias=True)

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first, second = await asyncio.gather(
                client.post("/agent", json=first_body),
                client.post("/agent", json=second_body),
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(metrics["dispatches"], 1)
        self.assertEqual(len(side_effects), 1)
        self.assertEqual(
            sum("RUN_ERROR" in response.text for response in (first, second)),
            1,
        )


if __name__ == "__main__":
    unittest.main()
