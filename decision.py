"""Session 6 Decision role."""
from __future__ import annotations

import json
import os

from client import LLM
from schemas import AgentHistoryItem, AgentToolCall, DecisionOutput, Goal, MemoryItem

ARTIFACT_DECISION_PROVIDER = os.getenv("ARTIFACT_DECISION_PROVIDER", "openai")
FINAL_RESPONSE_PROVIDER = os.getenv("FINAL_RESPONSE_PROVIDER", ARTIFACT_DECISION_PROVIDER)


SYSTEM_PROMPT = """Respond with exactly one of two outputs: answer in plain text or call a tool. Do not do both.

Strings beginning with art: are internal artifact handles. They reference the artifact store. MCP tools accept real file paths and URLs as their arguments and reject the art: prefix at dispatch time. When a goal requires the bytes of an artifact, those bytes appear in the prompt under ATTACHED ARTIFACTS. Read them there. Do not pass art: handles to read_file or fetch_url.

When the goal asks for an extraction, a list, a comparison, or a selection, the answer must be substantive: at least three sentences or a list of items.

If memory or history already contains relevant tool results for the current goal, synthesize an answer from those results instead of calling another tool. Do not repeat the same kind of tool call for the same goal unless the previous result was empty or irrelevant.

For weather forecast goals, first resolve relative dates from get_time if needed, then search using the exact calendar date plus city, forecast, high, low, rain, and conditions. If web_search returns only links or vague snippets but includes a useful forecast URL, call fetch_url on the best forecast page before giving up.

If force_answer is true, answer from the provided memory and history. Do not call a tool.
When force_answer is true and the evidence is incomplete, give a best-effort answer with the limitation clearly stated. Do not ask the user to paste more data. If a later recommendation depends on uncertain weather, say that and prefer the safer indoor option.
"""


def next_step(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[AgentHistoryItem],
    mcp_tools: list[dict],
    force_answer: bool = False,
) -> DecisionOutput:
    provider = ARTIFACT_DECISION_PROVIDER if attached else None
    auto_route = None if attached else "decision"
    tools = None if force_answer else mcp_tools
    response = LLM().chat(
        prompt=_prompt(goal, hits, attached, history, force_answer),
        system=SYSTEM_PROMPT,
        provider=provider,
        tools=tools,
        tool_choice="none" if force_answer else "auto",
        auto_route=auto_route,
        temperature=0,
        max_tokens=2048,
    )

    tool_calls = response.get("tool_calls") or []
    if tool_calls:
        first = tool_calls[0]
        return DecisionOutput(
            tool_call=AgentToolCall(
                name=first["name"],
                arguments=first.get("arguments") or {},
            )
        )

    return DecisionOutput(answer=(response.get("text") or "").strip())


def final_response(query: str, history: list[AgentHistoryItem], draft_answer: str) -> str:
    """Compress completed per-goal answers into one concise user-facing answer."""
    if not draft_answer.strip():
        return ""

    payload = {
        "user_query": query,
        "completed_answers": [
            {
                "goal_id": item.goal_id,
                "answer": item.text,
            }
            for item in history
            if item.kind == "answer" and item.text and item.text.strip()
        ],
        "draft_answer": draft_answer,
    }
    response = LLM().chat(
        prompt=json.dumps(payload, indent=2),
        system=(
            "Create the final answer for the user from the completed agent history. "
            "Be concise but complete. Preserve key dates, numbers, weather conditions, "
            "and the final recommendation. Do not mention internal goals, agents, memory, "
            "history, or tool calls. Prefer 4-8 bullets or 2-4 short paragraphs."
        ),
        provider=FINAL_RESPONSE_PROVIDER,
        temperature=0,
        max_tokens=900,
    )
    return (response.get("text") or "").strip() or draft_answer


def _prompt(
    goal: Goal,
    hits: list[MemoryItem],
    attached: list[tuple[str, bytes]],
    history: list[AgentHistoryItem],
    force_answer: bool,
) -> str:
    payload = {
        "goal": goal.model_dump(mode="json"),
        "force_answer": force_answer,
        "memory_hits": [_memory_hit_payload(hit) for hit in hits],
        "history": [item.model_dump(mode="json", exclude_none=True) for item in history],
    }
    return (
        json.dumps(payload, indent=2)
        + "\n\nATTACHED ARTIFACTS:\n"
        + _attached_payload(attached)
    )


def _memory_hit_payload(hit: MemoryItem) -> dict:
    return {
        "id": hit.id,
        "kind": hit.kind,
        "descriptor": hit.descriptor,
        "value": hit.value,
        "artifact_id": hit.artifact_id,
        "source": hit.source,
        "goal_id": hit.goal_id,
    }


def _attached_payload(attached: list[tuple[str, bytes]]) -> str:
    if not attached:
        return "(none)"

    parts = []
    for artifact_id, data in attached:
        text = data.decode("utf-8", "replace")
        parts.append(f"--- {artifact_id} ---\n{text}")
    return "\n\n".join(parts)
