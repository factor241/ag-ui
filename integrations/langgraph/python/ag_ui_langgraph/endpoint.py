import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder

from .agent import LangGraphAgent

BeforeDispatch = Callable[
    [RunAgentInput, Request, LangGraphAgent],
    Optional[Awaitable[None]],
]


def add_langgraph_fastapi_endpoint(
    app: FastAPI,
    agent: LangGraphAgent,
    path: str = "/",
    *,
    dependencies: Optional[Sequence[Any]] = None,
    before_dispatch: Optional[BeforeDispatch] = None,
):
    """Add a LangGraph AG-UI endpoint to a FastAPI application.

    ``dependencies`` are installed on the POST route itself, so FastAPI runs
    them before the handler can clone or invoke the agent. ``before_dispatch``
    runs after the request-local clone is created and before ``run`` starts;
    it can bind authenticated request context to that isolated clone.
    """

    @app.post(path, dependencies=list(dependencies or ()))
    async def langgraph_agent_endpoint(input_data: RunAgentInput, request: Request):
        # Clone the agent so each request gets its own isolated state.
        # LangGraphAgent stores per-request state in self.active_run; sharing a
        # single instance across concurrent requests corrupts that state.
        request_agent = agent.clone()

        if before_dispatch is not None:
            hook_result = before_dispatch(input_data, request, request_agent)
            if inspect.isawaitable(hook_result):
                await hook_result

        # Get the accept header from the request
        accept_header = request.headers.get("accept")

        # Create an event encoder to properly format SSE events
        encoder = EventEncoder(accept=accept_header)

        async def event_generator():
            async for event in request_agent.run(input_data):
                yield encoder.encode(event)

        return StreamingResponse(
            event_generator(),
            media_type=encoder.get_content_type()
        )

    @app.get(f"{path.rstrip('/')}/health")
    def health():
        """Health check."""
        return {
            "status": "ok",
            "agent": {
                "name": agent.name,
            }
        }
