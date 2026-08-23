"""Generic tool protocol for EGGROLL rollouts.

The rollout engine owns orchestration: schemas, parse/validate, dispatch,
append, malformed calls, and turn limits. Environments supply tools (or
reuse these) and domain state. Tools themselves never own the agent loop.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


def function_tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str] | None = None,
) -> dict[str, Any]:
    """OpenAI-style tool schema, same shape Maple generate already parses."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required or []),
            },
        },
    }


def tool_name(schema: dict) -> str:
    fn = schema.get("function") or schema
    return str(fn.get("name") or schema.get("name") or "")


def parse_tool_args(call: dict) -> dict | None:
    fn = call.get("function") or {}
    args = fn.get("arguments", call.get("arguments", {}))
    if isinstance(args, dict):
        return args
    if not isinstance(args, str):
        return None
    try:
        obj = json.loads(args)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_tool_call(call: dict) -> tuple[str, dict | None, str | None]:
    """Return ``(name, args, error)``. ``error`` set means the call is malformed."""
    fn = call.get("function") or {}
    name = str(fn.get("name") or call.get("name") or "").strip()
    if not name:
        return "", None, "missing tool name"
    raw = fn.get("arguments", call.get("arguments", {}))
    if raw is None or raw == "":
        return name, {}, None
    if isinstance(raw, dict):
        return name, raw, None
    if not isinstance(raw, str):
        return name, None, "arguments must be JSON object"
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return name, None, "arguments must be JSON object"
    if not isinstance(obj, dict):
        return name, None, "arguments must be JSON object"
    return name, obj, None


def validate_tool_args(schema: dict, args: dict | None) -> str | None:
    """Return an error message, or None if the call is well-formed."""
    if args is None:
        return "arguments must be JSON object"
    fn = schema.get("function") or schema
    params = fn.get("parameters") or {}
    required = params.get("required") or []
    missing = [key for key in required if key not in args]
    if missing:
        return "missing required arguments: " + ", ".join(missing)
    return None


@dataclass
class ToolResult:
    content: str
    ok: bool
    malformed: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolContext:
    """Episode-scoped handle so tools can read the env and mutate state."""

    env: Any
    inst: Any
    state: Any = None


@runtime_checkable
class Tool(Protocol):
    name: str

    def schema(self) -> dict[str, Any]: ...

    def call(self, args: dict | None, ctx: ToolContext) -> ToolResult | tuple[str, bool]: ...


class SchemaTool:
    """Wrap an OpenAI schema; dispatch to ``env.handle_tool`` / ``env.step``."""

    def __init__(self, spec: dict[str, Any]):
        self._spec = spec
        self.name = tool_name(spec)

    def schema(self) -> dict[str, Any]:
        return self._spec

    def call(self, args: dict | None, ctx: ToolContext) -> ToolResult:
        handle = getattr(ctx.env, "handle_tool", None)
        if callable(handle):
            return as_result(handle(self.name, args, ctx.state))
        return as_result(ctx.env.step(self.name, args))


def as_result(value: ToolResult | tuple[str, bool]) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    content, ok = value
    return ToolResult(content=str(content), ok=bool(ok))


def as_tools(raw: list | None) -> list[Tool]:
    if not raw:
        return []
    out: list[Tool] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(SchemaTool(item))
        else:
            out.append(item)
    return out


def tool_schemas(tools: list[Tool] | None) -> list[dict[str, Any]]:
    return [t.schema() for t in tools or []]


def lookup_tool(tools: list[Tool], name: str) -> Tool | None:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def invoke_tool(tools: list[Tool], call: dict, ctx: ToolContext) -> ToolResult:
    """Parse, validate, and dispatch one model tool call."""
    name, args, err = parse_tool_call(call)
    if err:
        return ToolResult(content=f"ERROR: {err}", ok=False, malformed=True)
    tool = lookup_tool(tools, name)
    if tool is None:
        return ToolResult(content=f"ERROR: unknown tool {name}", ok=False, malformed=True)
    verr = validate_tool_args(tool.schema(), args)
    if verr:
        return ToolResult(content=f"ERROR: {verr}", ok=False, malformed=True)
    try:
        return as_result(tool.call(args, ctx))
    except Exception as exc:
        return ToolResult(content=f"ERROR: {exc}", ok=False)


def dispatch_named(
    tools: list[Tool], name: str, args: dict | None, ctx: ToolContext
) -> ToolResult:
    """Call a tool by name (tests / env.step). Still validates required args."""
    tool = lookup_tool(tools, name)
    if tool is None:
        return ToolResult(content=f"ERROR: unknown tool {name}", ok=False, malformed=True)
    verr = validate_tool_args(tool.schema(), args if isinstance(args, dict) else None)
    if verr:
        return ToolResult(content=f"ERROR: {verr}", ok=False, malformed=True)
    try:
        return as_result(tool.call(args, ctx))
    except Exception as exc:
        return ToolResult(content=f"ERROR: {exc}", ok=False)


class MachineTool:
    """Reusable tool whose state is a machine with ``execute(name, args)``."""

    def __init__(
        self,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str] | None = None,
    ):
        self.name = name
        self._schema = function_tool(name, description, properties, required)

    def schema(self) -> dict[str, Any]:
        return self._schema

    def call(self, args: dict | None, ctx: ToolContext) -> ToolResult:
        return as_result(ctx.state.execute(self.name, args))


WRITE_TOOL = MachineTool(
    "write",
    "Set a named register to a string or integer.",
    {"key": {"type": "string"}, "value": {}},
    ["key", "value"],
)
READ_TOOL = MachineTool(
    "read",
    "Read a named register.",
    {"key": {"type": "string"}},
    ["key"],
)
ADD_TOOL = MachineTool(
    "add",
    "Add an integer to a numeric register.",
    {"key": {"type": "string"}, "delta": {"type": "integer"}},
    ["key", "delta"],
)
SUBMIT_TOOL = MachineTool(
    "submit",
    "Submit the final value of a register to the verifier.",
    {"key": {"type": "string"}},
    ["key"],
)
REGISTER_TOOLS: list[Tool] = [WRITE_TOOL, READ_TOOL, ADD_TOOL, SUBMIT_TOOL]


class WebSearchTool:
    """Reusable frozen SERP search. Env must implement ``search(inst, query)``."""

    name = "web_search"

    def schema(self) -> dict[str, Any]:
        return function_tool(
            self.name,
            "Search the web. Returns a frozen list of {title, url, snippet} hits.",
            {"query": {"type": "string"}},
            ["query"],
        )

    def call(self, args: dict | None, ctx: ToolContext) -> ToolResult:
        state = ctx.state if isinstance(ctx.state, dict) else None
        if state is not None and int(state.get("search_count", 0)) >= 1:
            return ToolResult("ERROR: already searched; answer now from the hits", False)
        query = str((args or {}).get("query", ""))
        payload = ctx.env.search(ctx.inst, query)
        if state is not None:
            state["search_count"] = int(state.get("search_count", 0)) + 1
            state["hits"] = list((payload or {}).get("results") or [])
        return ToolResult(json.dumps(payload, ensure_ascii=False), True)


class WebFetchTool:
    """Reusable fetch of a frozen SERP hit. Env must implement ``fetch(inst, url)``."""

    name = "web_fetch"

    def schema(self) -> dict[str, Any]:
        return function_tool(
            self.name,
            "Fetch a URL from the current search hits (frozen page or snippet). No live network.",
            {"url": {"type": "string"}},
            ["url"],
        )

    def call(self, args: dict | None, ctx: ToolContext) -> ToolResult:
        url = str((args or {}).get("url", ""))
        text = ctx.env.fetch(ctx.inst, url)
        if text is None:
            return ToolResult("ERROR: url not in current search hits", False)
        return ToolResult(str(text), True)
