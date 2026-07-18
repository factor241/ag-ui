import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, Optional

from fastapi import Body, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.params import Depends as DependsParameter
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from ag_ui.core import EventType, RunErrorEvent
from ag_ui.core.types import RunAgentInput
from ag_ui.encoder import EventEncoder

from .agent import LangGraphAgent

BeforeDispatch = Callable[
    [RunAgentInput, Request, LangGraphAgent],
    Awaitable[None] | None,
]


def _is_resume_only_validation_error(error: ValidationError) -> bool:
    errors = error.errors()
    return bool(errors) and all(
        item.get("loc") and item["loc"][0] == "resume"
        for item in errors
    )


def _validation_error_with_body_location(
    error: ValidationError,
    *,
    body: Any,
) -> RequestValidationError:
    errors = []
    for item in error.errors():
        errors.append({**item, "loc": ("body", *item.get("loc", ()))})
    return RequestValidationError(errors, body=body)


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

    @app.post(path, dependencies=list(dependencies or ()))
    async def langgraph_agent_endpoint(
        request: Request,
        raw_input: Annotated[Any, Body()],
    ) -> StreamingResponse:
        try:
            input_data = RunAgentInput.model_validate(raw_input)
        except ValidationError as exc:
            # Resume is a streamed protocol operation. Malformed resume wire
            # shapes therefore terminate as the same standard RUN_ERROR SSE as
            # semantic resume validation. Unrelated malformed request fields
            # retain FastAPI's ordinary 422 contract.
            if (
                isinstance(raw_input, dict)
                and "resume" in raw_input
                and _is_resume_only_validation_error(exc)
            ):
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
            raise _validation_error_with_body_location(exc, body=raw_input) from exc

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
