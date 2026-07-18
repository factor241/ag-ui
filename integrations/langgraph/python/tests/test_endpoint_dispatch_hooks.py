"""Contract tests for FastAPI dependency and pre-dispatch hooks."""

import asyncio
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

from ag_ui_langgraph.endpoint import add_langgraph_fastapi_endpoint
from ag_ui_langgraph import LangGraphAgent
from tests._helpers import make_agent


def _request_body():
    return {
        "threadId": "thread-1",
        "runId": "run-1",
        "state": {},
        "messages": [],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }


def _strict_agent_with_open_interrupt():
    agent = make_agent()
    state = MagicMock()
    state.values = {"messages": []}
    state.tasks = [
        SimpleNamespace(
            interrupts=[
                SimpleNamespace(
                    id="int-1",
                    value={"reason": "confirmation", "message": "approve?"},
                )
            ]
        )
    ]
    state.next = []
    state.metadata = {"writes": {}}
    agent.graph.aget_state = AsyncMock(return_value=state)
    return agent


class RecordingAgent:
    def __init__(self, calls, *, label="template"):
        self.calls = calls
        self.label = label
        self.name = "recording-agent"
        self.bound_actor = None
        self._thread_lock_registry = object()
        self._resume_claim_registry = object()

    def clone(self):
        self.calls.append(("clone", self.label))
        return RecordingAgent(self.calls, label="request")

    async def run(self, input_data):
        self.calls.append(("run", self.label, self.bound_actor, input_data.run_id))
        if False:
            yield None


class ConfigRecordingAgent(LangGraphAgent):
    async def run(self, input_data):
        actor = self.config["configurable"]["request_context"]["actor"]
        self.graph.recorded_actors.append((input_data.run_id, actor))
        if False:
            yield None


class TestEndpointDispatchHooks(unittest.TestCase):
    def test_new_options_are_keyword_only(self):
        signature = inspect.signature(add_langgraph_fastapi_endpoint)

        self.assertEqual(
            signature.parameters["dependencies"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )
        self.assertEqual(
            signature.parameters["before_dispatch"].kind,
            inspect.Parameter.KEYWORD_ONLY,
        )

    def test_openapi_preserves_run_agent_input_request_schema(self):
        app = FastAPI()
        add_langgraph_fastapi_endpoint(
            app,
            RecordingAgent([]),
            "/agent",
        )

        schema = app.openapi()
        request_schema = schema["paths"]["/agent"]["post"]["requestBody"][
            "content"
        ]["application/json"]["schema"]

        self.assertEqual(
            request_schema,
            {"$ref": "#/components/schemas/RunAgentInput"},
        )
        run_input_schema = schema["components"]["schemas"]["RunAgentInput"]
        self.assertIn("resume", run_input_schema["properties"])
        self.assertTrue(
            {
                "threadId",
                "runId",
                "state",
                "messages",
                "tools",
                "context",
                "forwardedProps",
            }.issubset(run_input_schema["required"])
        )

    def test_dependency_deny_happens_before_clone_and_run(self):
        calls = []
        agent = RecordingAgent(calls)
        app = FastAPI()

        async def deny():
            calls.append(("dependency", "deny"))
            raise HTTPException(status_code=401, detail="not authenticated")

        add_langgraph_fastapi_endpoint(
            app,
            agent,
            "/agent",
            dependencies=[Depends(deny)],
        )

        response = TestClient(app).post("/agent", json=_request_body())

        self.assertEqual(response.status_code, 401)
        self.assertEqual(calls, [("dependency", "deny")])

    def test_before_dispatch_receives_request_agent_and_runs_before_agent(self):
        calls = []
        agent = RecordingAgent(calls)
        app = FastAPI()

        async def before_dispatch(input_data, request, request_agent):
            calls.append(
                (
                    "before_dispatch",
                    input_data.run_id,
                    request.url.path,
                    request_agent.label,
                )
            )
            request_agent.bound_actor = "actor-1"

        add_langgraph_fastapi_endpoint(
            app,
            agent,
            "/agent",
            before_dispatch=before_dispatch,
        )

        response = TestClient(app).post("/agent", json=_request_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            calls,
            [
                ("clone", "template"),
                ("before_dispatch", "run-1", "/agent", "request"),
                ("run", "request", "actor-1", "run-1"),
            ],
        )

    def test_sync_before_dispatch_is_supported(self):
        calls = []
        agent = RecordingAgent(calls)
        app = FastAPI()

        def before_dispatch(input_data, request, request_agent):
            calls.append(("before_dispatch", input_data.run_id, request.url.path))
            request_agent.bound_actor = "sync-actor"

        add_langgraph_fastapi_endpoint(
            app,
            agent,
            "/agent",
            before_dispatch=before_dispatch,
        )

        response = TestClient(app).post("/agent", json=_request_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            calls,
            [
                ("clone", "template"),
                ("before_dispatch", "run-1", "/agent"),
                ("run", "request", "sync-actor", "run-1"),
            ],
        )

    def test_before_dispatch_exception_prevents_run(self):
        calls = []
        agent = RecordingAgent(calls)
        app = FastAPI()

        def before_dispatch(input_data, request, request_agent):
            calls.append(("before_dispatch", input_data.run_id))
            raise HTTPException(status_code=403, detail="forbidden")

        add_langgraph_fastapi_endpoint(
            app,
            agent,
            "/agent",
            before_dispatch=before_dispatch,
        )

        response = TestClient(app).post("/agent", json=_request_body())

        self.assertEqual(response.status_code, 403)
        self.assertEqual(
            calls,
            [("clone", "template"), ("before_dispatch", "run-1")],
        )

    def test_before_dispatch_nested_binding_does_not_leak_between_clones(self):
        graph = MagicMock()
        graph.nodes = {}
        graph.recorded_actors = []
        agent = ConfigRecordingAgent(
            name="config-agent",
            graph=graph,
            config={
                "configurable": {
                    "request_context": {"actor": "template"},
                }
            },
        )
        app = FastAPI()

        def before_dispatch(input_data, request, request_agent):
            request_agent.config["configurable"]["request_context"]["actor"] = (
                input_data.run_id
            )

        add_langgraph_fastapi_endpoint(
            app,
            agent,
            "/agent",
            before_dispatch=before_dispatch,
        )
        client = TestClient(app)

        first = client.post("/agent", json=_request_body())
        second = client.post(
            "/agent",
            json={**_request_body(), "runId": "run-2"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(
            graph.recorded_actors,
            [("run-1", "run-1"), ("run-2", "run-2")],
        )
        self.assertEqual(
            agent.config["configurable"]["request_context"]["actor"],
            "template",
        )

    def test_malformed_resume_wire_shapes_return_run_error_without_clone(self):
        malformed_resumes = (
            {"interruptId": "int-1", "status": "resolved"},
            [{"interruptId": "int-1", "status": "invalid"}],
            ["not-an-entry"],
            [{"status": "resolved", "payload": True}],
        )

        for resume in malformed_resumes:
            with self.subTest(resume=resume):
                calls = []
                app = FastAPI()
                add_langgraph_fastapi_endpoint(
                    app,
                    RecordingAgent(calls),
                    "/agent",
                )
                body = {**_request_body(), "resume": resume}

                response = TestClient(app).post("/agent", json=body)

                self.assertEqual(response.status_code, 200)
                self.assertIn("RUN_ERROR", response.text)
                self.assertEqual(calls, [])

    def test_unrelated_malformed_field_preserves_fastapi_422(self):
        calls = []
        app = FastAPI()
        add_langgraph_fastapi_endpoint(
            app,
            RecordingAgent(calls),
            "/agent",
        )
        body = {**_request_body(), "messages": "not-an-array"}

        response = TestClient(app).post("/agent", json=body)

        self.assertEqual(response.status_code, 422)
        self.assertEqual(calls, [])

    def test_extra_legacy_and_mixed_resume_forms_are_run_error_sse(self):
        cases = (
            {
                "resume": [
                    {"interruptId": "int-1", "status": "resolved", "payload": True},
                    {"interruptId": "extra", "status": "resolved", "payload": True},
                ]
            },
            {"forwardedProps": {"command": {"resume": True}}},
            {
                "resume": [
                    {"interruptId": "int-1", "status": "resolved", "payload": True}
                ],
                "forwardedProps": {"command": {"resume": True}},
            },
        )

        for overrides in cases:
            with self.subTest(overrides=overrides):
                agent = _strict_agent_with_open_interrupt()
                app = FastAPI()
                add_langgraph_fastapi_endpoint(app, agent, "/agent")

                response = TestClient(app).post(
                    "/agent",
                    json={**_request_body(), **overrides},
                )

                self.assertEqual(response.status_code, 200)
                self.assertIn("RUN_ERROR", response.text)
                agent.graph.astream_events.assert_not_called()


class TestConcurrentEndpointIsolation(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_nested_auth_bindings_stay_request_local(self):
        graph = MagicMock()
        graph.nodes = {}
        graph.recorded_actors = []
        agent = ConfigRecordingAgent(
            name="config-agent",
            graph=graph,
            config={
                "configurable": {
                    "request_context": {"actor": "template"},
                }
            },
        )
        app = FastAPI()
        both_hooks_entered = asyncio.Event()
        hook_arrivals = 0

        async def before_dispatch(input_data, request, request_agent):
            nonlocal hook_arrivals
            request_agent.config["configurable"]["request_context"]["actor"] = (
                input_data.run_id
            )
            hook_arrivals += 1
            if hook_arrivals == 2:
                both_hooks_entered.set()
            await both_hooks_entered.wait()

        add_langgraph_fastapi_endpoint(
            app,
            agent,
            "/agent",
            before_dispatch=before_dispatch,
        )

        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            first, second = await asyncio.gather(
                client.post("/agent", json=_request_body()),
                client.post(
                    "/agent",
                    json={**_request_body(), "runId": "run-2"},
                ),
            )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertCountEqual(
            graph.recorded_actors,
            [("run-1", "run-1"), ("run-2", "run-2")],
        )
        self.assertEqual(
            agent.config["configurable"]["request_context"]["actor"],
            "template",
        )


if __name__ == "__main__":
    unittest.main()
