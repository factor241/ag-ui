import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.params import Depends as DependsParameter
from fastapi.routing import APIRoute
from fastapi.responses import StreamingResponse

from ag_ui.core import EventType, RunErrorEvent
from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder

from .agent import LangGraphAgent

BeforeDispatch = Callable[
    [RunAgentInput, Request, LangGraphAgent],
    Awaitable[None] | None,
]


def _is_resume_only_validation_error(error: RequestValidationError) -> bool:
    errors = error.errors()
    if not errors:
        return False
    for item in errors:
        location = item.get("loc") or ()
        if location and location[0] == "body":
            location = location[1:]
        if not location or location[0] != "resume":
            return False
    return True


def _invalid_resume_stream(request: Request) -> StreamingResponse:
    encoder = EventEncoder(accept=request.headers.get("accept"))

    async def invalid_resume_generator():
        yield encoder.encode(
            RunErrorEvent(
                type=EventType.RUN_ERROR,
                code="INVALID_RESUME",
                message="Invalid resume request: resume payload is malformed.",
            )
        )

    return StreamingResponse(
        invalid_resume_generator(),
        media_type=encoder.get_content_type(),
    )


class _ResumeValidationRoute(APIRoute):
    """Translate only resume-field body validation errors into AG-UI SSE."""

    def get_route_handler(self):
        original_handler = super().get_route_handler()

        async def resume_aware_handler(request: Request):
            try:
                return await original_handler(request)
            except RequestValidationError as exc:
                if (
                    isinstance(exc.body, dict)
                    and "resume" in exc.body
                    and _is_resume_only_validation_error(exc)
                ):
                    return _invalid_resume_stream(request)
                raise

        return resume_aware_handler


def add_langgraph_fastapi_endpoint(
    app: FastAPI,
    agent: LangGraphAgent,
    path: str = "/",
    *,
    dependencies: Optional[Sequence[DependsParameter]] = None,
    before_dispatch: Optional[BeforeDispatch] = None,
) -> None:
    """Add a LangGraph AG-UI endpoint to a FastAPI application.

    ``dependencies`` are installed on the POST route itself, so FastAPI runs
    them before the handler can clone or invoke the agent. ``before_dispatch``
    runs after the request-local clone is created and before ``run`` starts;
    it can bind authenticated request context to that isolated clone.
    """

    async def langgraph_agent_endpoint(
        request: Request,
        input_data: RunAgentInput,
    ) -> StreamingResponse:
        # Clone the agent so each request gets its own isolated state.
        # LangGraphAgent stores per-request state in self.active_run; sharing a
        # single instance across concurrent requests corrupts that state.
        request_agent = agent.clone()
        # Subclasses with required constructor arguments legitimately override
        # clone(). Force the template-owned guards onto every returned clone so
        # an override cannot accidentally reopen the replay race.
        request_agent._thread_lock_registry = agent._thread_lock_registry
        request_agent._resume_claim_registry = agent._resume_claim_registry

        if before_dispatch is not None:
            hook_result = before_dispatch(input_data, request, request_agent)
            if inspect.isawaitable(hook_result):
                await hook_result
        # Keep the invariant even if a hook bound broader runtime state.
        request_agent._thread_lock_registry = agent._thread_lock_registry
        request_agent._resume_claim_registry = agent._resume_claim_registry

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

    app.router.add_api_route(
        path,
        langgraph_agent_endpoint,
        methods=["POST"],
        dependencies=list(dependencies or ()),
        route_class_override=_ResumeValidationRoute,
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
