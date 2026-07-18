# ag-ui-langgraph

Implementation of the AG-UI protocol for LangGraph.

Provides a complete Python integration for LangGraph agents with the AG-UI protocol, including FastAPI endpoint creation and comprehensive event streaming.

## Installation

```bash
pip install ag-ui-langgraph
```

## Usage

```python
from langgraph.graph import StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI
from my_langgraph_workflow import graph

# Add to FastAPI
app = FastAPI()
agent = LangGraphAgent(name="my-agent", graph=graph)
add_langgraph_fastapi_endpoint(app, agent, "/agent")
```

The endpoint registrar also accepts two keyword-only integration hooks:

```python
from fastapi import Depends

add_langgraph_fastapi_endpoint(
    app,
    agent,
    "/agent",
    dependencies=[Depends(require_authenticated_user)],
    before_dispatch=bind_request_context,
)
```

FastAPI evaluates `dependencies` before the handler can clone or run the
agent. After the dependency gate succeeds, the registrar creates a
request-local agent clone and calls
`before_dispatch(input, request, request_agent)` before `request_agent.run`.
The hook may be synchronous or asynchronous; its return value is ignored.

If the body contains a malformed `resume` value (for example, a non-array,
an invalid status, or an entry without `interruptId`), the endpoint returns a
standard `RUN_ERROR` SSE without cloning or running the agent. Validation
errors in unrelated request fields keep FastAPI's normal `422` response.

## Features

- **Native LangGraph integration** – Direct support for LangGraph workflows and state management
- **FastAPI endpoint creation** – Automatic HTTP endpoint generation with proper event streaming
- **Advanced event handling** – Comprehensive support for all AG-UI events including thinking, tool calls, and state updates
- **Message translation** – Seamless conversion between AG-UI and LangChain message formats

## Standard interrupt and resume contract

An interrupted run always terminates with the standard
`RunFinishedEvent.outcome.type == "interrupt"`. The outcome contains every
currently-open AG-UI `Interrupt`, including its exact LangGraph interrupt ID.
No legacy `on_interrupt` custom event is emitted.

```python
# Read interrupts from the standard outcome.
if (
    event.type == EventType.RUN_FINISHED
    and event.outcome
    and event.outcome.type == "interrupt"
):
    for interrupt in event.outcome.interrupts:
        print(interrupt.id, interrupt.reason, interrupt.message)
```

### Resuming a run

Resume only through `RunAgentInput.resume`. Send exactly one `ResumeEntry` for
every interrupt in the latest open interrupt outcome:

```python
input = RunAgentInput(
    thread_id="t1",
    run_id="r2",
    state={},
    messages=[],
    tools=[],
    context=[],
    forwarded_props={},
    resume=[
        ResumeEntry(
            interrupt_id="int-abc",
            status="resolved",
            payload={"approved": True},
        ),
    ],
)
```

The adapter validates the resume array against the current checkpoint before
dispatching the graph. Partial, stale, duplicate, unknown, malformed, reused,
empty, and mixed resume arrays terminate with standard `RUN_ERROR` and do not
dispatch the graph. `forwardedProps.command.resume` is rejected. A valid full
set becomes one native LangGraph `Command(resume={interrupt_id: payload, ...})`
and is dispatched exactly once.

An interrupt with `expiresAt` can be resumed only before that timezone-aware
ISO-8601 timestamp. Expired or malformed timestamps fail closed. A
`status="cancelled"` entry must omit `payload` or set it to `None`; every
non-null payload, including falsey values, is rejected.

Clones created from the same agent template share a per-thread `asyncio` lock.
The lock covers checkpoint re-read, validation, graph dispatch, and complete
stream consumption, so two concurrent resumes in one FastAPI backend process
cannot both execute. This guarantee is deliberately process-local; deployments
with multiple workers or hosts require durable cross-process coordination in
their checkpoint/runtime layer.

The deprecated constructor arguments `enable_legacy_on_interrupt_event` and
`emit_interrupt_outcome` remain accepted for source compatibility and emit a
`DeprecationWarning`, but their values are ignored. The adapter always emits
the standard interrupt outcome and never emits the legacy `on_interrupt`
custom event.

### Capabilities

`LangGraphAgent.get_capabilities()` returns `{"humanInTheLoop": {"supported": True, "interrupts": True, "approveWithEdits": True}}`.

### Customising interrupt mapping

If a graph middleware carries multiple logical decisions inside one LangGraph
interrupt, a subclass may override `_interrupts_to_agui` to expose each open
decision with a stable ID:

```python
from ag_ui_langgraph import LangGraphAgent
from ag_ui_langgraph.interrupts import lg_interrupt_to_agui
from ag_ui.core import Interrupt as AGUIInterrupt

class HITLLangGraphAgent(LangGraphAgent):
    def _interrupts_to_agui(self, lg_interrupts):
        out = []
        for lg in lg_interrupts:
            value = lg.value
            if isinstance(value, dict) and "action_requests" in value:
                out.extend(my_action_requests_to_agui(value))
            else:
                out.append(lg_interrupt_to_agui(lg))
        return out

```

The strict exact-ID validation applies to the mapped open interrupt list.

## To run the dojo examples

```bash
cd python/ag_ui_langgraph/examples
poetry install
poetry run dev
```
