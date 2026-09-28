"""Reconstruct model inputs from captured OpenAI Responses API calls."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class CapturedCall:
    """One completed Responses API call and its persisted request payload."""

    sequence: int
    call_id: str
    call_dir: Path
    request: dict[str, Any]
    response: dict[str, Any] | None
    response_ids: tuple[str, ...]


@dataclass(frozen=True)
class ReconstructedContext:
    """Chat-template inputs reconstructed from a captured response chain."""

    messages: tuple[dict[str, Any], ...]
    tools: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    response_chain: tuple[str, ...]


# Read a JSON object and fail with an artifact-specific path in the error message.
def read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


# Select the completed response object from JSON or parsed SSE artifacts.
def load_response_object(call_dir: Path) -> dict[str, Any] | None:
    regular_path = call_dir / "response-body.json"
    if regular_path.exists():
        body = read_json_object(regular_path).get("body")
        return body if isinstance(body, dict) else None

    # Prefer the final response.completed event because it contains all output items.
    events_path = call_dir / "response-events.jsonl"
    if not events_path.exists():
        return None
    completed_response: dict[str, Any] | None = None
    created_response: dict[str, Any] | None = None
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        data = event.get("data")
        if not isinstance(data, dict):
            continue
        response = data.get("response")
        if data.get("type") == "response.completed" and isinstance(response, dict):
            completed_response = response
        elif data.get("type") == "response.created" and isinstance(response, dict):
            created_response = response
        elif data.get("object") == "response":
            completed_response = data
    return completed_response or created_response


# Discover successful POST /responses calls in deterministic request order.
def discover_response_calls(capture_dir: Path) -> list[CapturedCall]:
    calls_dir = capture_dir / "calls"
    if not calls_dir.is_dir():
        raise FileNotFoundError(f"capture calls directory does not exist: {calls_dir}")
    calls: list[CapturedCall] = []
    for call_dir in sorted(calls_dir.iterdir()):
        call_path = call_dir / "call.json"
        body_path = call_dir / "request-body.json"
        if not call_path.exists() or not body_path.exists():
            continue
        summary = read_json_object(call_path)
        target = str(summary.get("target", "")).split("?", 1)[0].rstrip("/")
        if summary.get("method") != "POST" or not target.endswith("/responses"):
            continue
        if summary.get("outcome") != "completed":
            continue
        body = read_json_object(body_path).get("body")
        if not isinstance(body, dict):
            continue
        response_ids = tuple(
            item for item in summary.get("response_ids", []) if isinstance(item, str)
        )
        calls.append(
            CapturedCall(
                sequence=int(summary["sequence"]),
                call_id=str(summary["call_id"]),
                call_dir=call_dir,
                request=body,
                response=load_response_object(call_dir),
                response_ids=response_ids,
            )
        )
    return sorted(calls, key=lambda item: item.sequence)


# Extract the stable output list returned by a completed Responses object.
def response_output_items(response: Mapping[str, Any] | None) -> list[Any]:
    if response is None:
        return []
    output = response.get("output", [])
    return list(output) if isinstance(output, list) else []


# Normalize a Responses input field into an ordered item list.
def request_input_items(request: Mapping[str, Any]) -> list[Any]:
    request_input = request.get("input", [])
    if isinstance(request_input, str):
        return [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": request_input}],
            }
        ]
    return list(request_input) if isinstance(request_input, list) else []


# Build a response-ID index for resolving previous_response_id chains offline.
def index_calls_by_response_id(
    calls: Iterable[CapturedCall],
) -> dict[str, CapturedCall]:
    indexed: dict[str, CapturedCall] = {}
    for call in calls:
        response_ids = set(call.response_ids)
        if call.response is not None and isinstance(call.response.get("id"), str):
            response_ids.add(call.response["id"])
        for response_id in response_ids:
            indexed[response_id] = call
    return indexed


# Resolve all prior input/output items that the Responses state chain makes visible.
def resolve_response_chain(
    call: CapturedCall,
    response_index: Mapping[str, CapturedCall],
) -> tuple[list[Any], list[str], list[str]]:
    warnings: list[str] = []
    response_chain: list[str] = []
    visited: set[str] = set()

    # Recursively materialize a parent before appending its generated output.
    def resolve(current: CapturedCall) -> list[Any]:
        previous_response_id = current.request.get("previous_response_id")
        items: list[Any] = []
        if isinstance(previous_response_id, str) and previous_response_id:
            if previous_response_id in visited:
                raise ValueError(
                    f"cycle detected in previous_response_id chain: {previous_response_id}"
                )
            visited.add(previous_response_id)
            parent = response_index.get(previous_response_id)
            if parent is None:
                warnings.append(
                    f"previous response {previous_response_id!r} was not captured; "
                    "the reconstructed context is incomplete"
                )
            else:
                items.extend(resolve(parent))
                items.extend(response_output_items(parent.response))
                response_chain.append(previous_response_id)
        items.extend(request_input_items(current.request))
        return items

    return resolve(call), warnings, response_chain


# Flatten Responses content blocks to the text consumed by a chat template.
def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    fragments: list[str] = []
    for block in content:
        if isinstance(block, str):
            fragments.append(block)
        elif isinstance(block, dict) and isinstance(block.get("text"), str):
            fragments.append(block["text"])
        elif isinstance(block, dict) and isinstance(block.get("output_text"), str):
            fragments.append(block["output_text"])
        elif isinstance(block, dict):
            fragments.append(json.dumps(block, ensure_ascii=False, sort_keys=True))
    return "\n".join(fragment for fragment in fragments if fragment)


# Convert a Responses function schema to the OpenAI chat-template tool schema.
def normalize_tool(tool: Any) -> dict[str, Any] | None:
    if not isinstance(tool, dict) or tool.get("type") != "function":
        return None
    if isinstance(tool.get("function"), dict):
        return tool
    name = tool.get("name")
    if not isinstance(name, str):
        return None
    function: dict[str, Any] = {"name": name}
    if isinstance(tool.get("description"), str):
        function["description"] = tool["description"]
    parameters = tool.get("parameters")
    if isinstance(parameters, dict):
        function["parameters"] = parameters
    else:
        function["parameters"] = {"type": "object", "properties": {}}
    if isinstance(tool.get("strict"), bool):
        function["strict"] = tool["strict"]
    return {"type": "function", "function": function}


# Translate Responses items into the message/tool-call representation Qwen expects.
def items_to_messages(items: Sequence[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    messages: list[dict[str, Any]] = []
    warnings: list[str] = []
    for item_index, item in enumerate(items):
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            warnings.append(f"ignored non-object item at index {item_index}")
            continue
        item_type = item.get("type")
        if item_type == "message" or ("role" in item and "content" in item):
            role = str(item.get("role", "user"))
            if role == "developer":
                role = "system"
            messages.append(
                {"role": role, "content": content_text(item.get("content"))}
            )
            continue
        if item_type == "function_call":
            arguments = item.get("arguments", "{}")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    warnings.append(
                        f"preserved malformed function arguments at item {item_index} "
                        "under the _raw_arguments key"
                    )
                    arguments = {"_raw_arguments": arguments}
            if not isinstance(arguments, dict):
                warnings.append(
                    f"wrapped non-object function arguments at item {item_index} "
                    "under the _value key"
                )
                arguments = {"_value": arguments}
            tool_call = {
                "id": str(
                    item.get("call_id") or item.get("id") or f"call_{item_index}"
                ),
                "type": "function",
                "function": {
                    "name": str(item.get("name", "unknown_tool")),
                    "arguments": arguments,
                },
            }
            messages.append(
                {"role": "assistant", "content": "", "tool_calls": [tool_call]}
            )
            continue
        if item_type == "function_call_output":
            output = content_text(item.get("output", ""))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": str(item.get("call_id") or item.get("id") or ""),
                    "content": output,
                }
            )
            continue
        if item_type == "reasoning":
            summary = content_text(item.get("summary", []))
            if summary:
                messages.append({"role": "assistant", "content": summary})
            else:
                warnings.append(f"omitted empty reasoning item at index {item_index}")
            continue
        warnings.append(
            f"serialized unsupported Responses item type {item_type!r} at index {item_index}"
        )
        messages.append(
            {
                "role": "user",
                "content": json.dumps(item, ensure_ascii=False, sort_keys=True),
            }
        )
    return messages, warnings


# Reconstruct the effective semantic context and retain every approximation warning.
def reconstruct_context(
    call: CapturedCall,
    all_calls: Sequence[CapturedCall],
) -> ReconstructedContext:
    response_index = index_calls_by_response_id(all_calls)
    items, chain_warnings, response_chain = resolve_response_chain(call, response_index)
    messages, item_warnings = items_to_messages(items)

    # Current instructions apply to this request even when a previous response is chained.
    instructions = call.request.get("instructions")
    if isinstance(instructions, str) and instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    # Normalize only function tools because other Responses tools need model-specific handling.
    tools: list[dict[str, Any]] = []
    tool_warnings: list[str] = []
    raw_tools = call.request.get("tools", [])
    if isinstance(raw_tools, list):
        for tool_index, raw_tool in enumerate(raw_tools):
            normalized = normalize_tool(raw_tool)
            if normalized is None:
                tool_warnings.append(
                    f"ignored unsupported tool at request tool index {tool_index}"
                )
            else:
                tools.append(normalized)
    return ReconstructedContext(
        messages=tuple(messages),
        tools=tuple(tools),
        warnings=tuple(chain_warnings + item_warnings + tool_warnings),
        response_chain=tuple(response_chain),
    )


# Render Qwen's own chat template rather than inventing a second prompt format.
def render_prompt(tokenizer: Any, context: ReconstructedContext) -> str:
    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if context.tools:
        template_kwargs["tools"] = list(context.tools)
    rendered = tokenizer.apply_chat_template(list(context.messages), **template_kwargs)
    if not isinstance(rendered, str):
        raise TypeError("tokenizer.apply_chat_template did not return text")
    return rendered
