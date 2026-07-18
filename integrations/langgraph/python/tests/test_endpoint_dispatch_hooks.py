"""Contract tests for FastAPI dependency and pre-dispatch hooks."""

import inspect
import unittest

from fastapi import Depends, FastAPI, HTTPException
from fastapi.testclient import TestClient

from ag_ui_langgraph.endpoint import add_langgraph_fastapi_endpoint


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


class RecordingAgent:
    def __init__(self, calls, *, label="template"):
        self.calls = calls
        self.label = label
        self.name = "recording-agent"
        self.bound_actor = None

    def clone(self):
        self.calls.append(("clone", self.label))
        return RecordingAgent(self.calls, label="request")

    async def run(self, input_data):
        self.calls.append(("run", self.label, self.bound_actor, input_data.run_id))
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


if __name__ == "__main__":
    unittest.main()
