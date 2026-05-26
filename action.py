"""Session 6 Action role: pure MCP dispatch."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import artifacts
from schemas import AgentToolCall

if TYPE_CHECKING:
    from mcp import ClientSession


ARTIFACT_THRESHOLD_BYTES = 4096


async def execute(
    session: "ClientSession",
    tool_call: AgentToolCall,
) -> tuple[str, str | None]:
    blocked = _artifact_handle_argument(tool_call.arguments)
    if blocked:
        return (
            f"ERROR: artifact handles are not paths or URLs: {blocked}. "
            "Read attached artifact bytes from the prompt instead.",
            None,
        )

    result = await session.call_tool(tool_call.name, arguments=tool_call.arguments)
    text = _collapse_content(result)
    data = text.encode("utf-8")

    if len(data) <= ARTIFACT_THRESHOLD_BYTES:
        return text, None

    artifact = artifacts.put(
        data,
        source=tool_call.name,
        content_type="text/plain",
        descriptor=f"{tool_call.name} result",
    )
    preview = text[:240].replace("\n", " ")
    return (
        f"[artifact {artifact.id}, {artifact.size_bytes} bytes] preview: {preview}",
        artifact.id,
    )


def _artifact_handle_argument(arguments: dict[str, Any]) -> str | None:
    for key in ("path", "url"):
        value = arguments.get(key)
        if isinstance(value, str) and value.startswith("art:"):
            return f"{key}={value}"
    return None


def _collapse_content(result: Any) -> str:
    blocks = getattr(result, "content", None) or []
    parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text is not None:
            parts.append(str(text))
            continue
        if hasattr(block, "model_dump"):
            parts.append(json.dumps(block.model_dump(mode="json")))
            continue
        parts.append(str(block))
    return "\n".join(parts)
