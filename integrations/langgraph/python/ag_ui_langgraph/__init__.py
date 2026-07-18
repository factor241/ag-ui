from .agent import LangGraphAgent
from .types import (
    LangGraphEventTypes,
    CustomEventNames,
    State,
    SchemaKeys,
    ThinkingProcess,
    MessageInProgress,
    RunMetadata,
    MessagesInProgressRecord,
    ToolCall,
    BaseLangGraphPlatformMessage,
    LangGraphPlatformResultMessage,
    LangGraphPlatformActionExecutionMessage,
    LangGraphPlatformMessage,
    PredictStateTool,
    LangGraphReasoning,
)
from .utils import json_safe_stringify, make_json_safe
from .middlewares.state_streaming import StateStreamingMiddleware, StateItem
from .a2ui_tool import (
    get_a2ui_tools,
    A2UIToolParams,
    A2UIGuidelines,
    A2UI_OPERATIONS_KEY,
    BASIC_CATALOG_ID,
)

__all__ = [
    "LangGraphAgent",
    "get_a2ui_tools",
    "A2UIToolParams",
    "A2UIGuidelines",
    "A2UI_OPERATIONS_KEY",
    "BASIC_CATALOG_ID",
    "LangGraphEventTypes",
    "CustomEventNames",
    "State",
    "SchemaKeys",
    "ThinkingProcess",
    "MessageInProgress",
    "RunMetadata",
    "MessagesInProgressRecord",
    "ToolCall",
    "BaseLangGraphPlatformMessage",
    "LangGraphPlatformResultMessage",
    "LangGraphPlatformActionExecutionMessage",
    "LangGraphPlatformMessage",
    "PredictStateTool",
    "LangGraphReasoning",
    "add_langgraph_fastapi_endpoint",
    "StateStreamingMiddleware",
    "StateItem",
    "json_safe_stringify",
    "make_json_safe"
]


def __getattr__(name: str):
    """Load the optional FastAPI endpoint export only when requested."""
    if name != "add_langgraph_fastapi_endpoint":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        from .endpoint import add_langgraph_fastapi_endpoint
    except ModuleNotFoundError as exc:
        if exc.name == "fastapi" or (exc.name or "").startswith("fastapi."):
            raise ImportError(
                "add_langgraph_fastapi_endpoint requires the optional "
                "FastAPI dependencies; install ag-ui-langgraph[fastapi]."
            ) from exc
        raise
    globals()[name] = add_langgraph_fastapi_endpoint
    return add_langgraph_fastapi_endpoint
