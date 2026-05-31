"""Session 6 Perception role.

Perception owns the run's goal list across iterations. It decomposes the query
on the first pass, then preserves goal positions while updating only done flags
and the optional artifact attachment for the next unfinished goal.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

from client import LLM
from schemas import (
    AgentHistoryItem,
    Goal,
    MemoryItem,
    Observation,
    PerceptionGoalOutput,
    PerceptionOutput,
)


PERCEPTION_PROVIDER = os.getenv("PERCEPTION_PROVIDER", "openai")

SYSTEM_PROMPT = """You are the Perception role in a Session 6 agent loop.

You receive the original user query, current memory hits, run history, and the
prior goal list. Return a fresh Observation as JSON.

Rules:
- If prior_goals is empty, decompose the user query into one or more bounded
  goals. Each goal text must be a short imperative statement.
- If prior_goals is not empty, preserve the same number of goals, the same
  order, and the same goal texts. Only update done and artifact_index.
- Mark a goal done once the history contains an answer or action result that
  satisfies it. Done goals stay done.
- If history contains an answer for a goal that directly addresses the goal but
  explains the requested external data could not be determined after tool
  attempts, mark that goal done so the loop can progress.
- For the first unfinished goal only, decide whether it needs raw bytes from a
  previous artifact. If yes, set artifact_index to one of the visible indexes
  from memory_hits. If no, leave artifact_index null.
- Do not invent artifact indexes. Do not reference artifact handles.
- Do not add, drop, or reorder goals after the first iteration.

Multi-source fetch rule:
- When the query asks to read, fetch, or analyse N specific sources (e.g. "read
  the top 3 results", "check each of the 5 pages", "fetch both articles"), create
  one goal per source fetch — never a single bulk "read all N" goal.
- The per-fetch goals must come AFTER the search/identify goal and BEFORE the
  synthesis goal. Use ordinal labels so Decision knows which source to fetch:
  "Fetch and read the 1st result URL", "Fetch and read the 2nd result URL", etc.
- CRITICAL — history vs memory_hits:
  * `history` = actions taken in THIS run. This is the ONLY source of truth for
    whether a fetch goal is done.
  * `memory_hits` = results from PRIOR runs kept for context. They must NEVER be
    used to mark a fetch goal as done. A fetch_url entry in memory_hits does not
    count as satisfying a per-fetch goal in the current run.
- A per-fetch goal is done ONLY when the `history` array (not memory_hits)
  contains a fetch_url tool_call result for that ordinal position that returned
  SUBSTANTIAL content — meaning no error (no "403", no "Error executing tool")
  AND length_bytes > 500. A near-empty or blocked fetch does NOT satisfy the goal.
- If history shows a fetch_url for this ordinal that errored or returned < 500
  bytes, the goal stays OPEN so Decision can try a different URL from the search
  results.
- An answer listing URLs does NOT satisfy a fetch goal.
- The synthesis goal is done only after ALL per-fetch goals are done.
"""


def observe(
    query: str,
    hits: list[MemoryItem],
    history: list[AgentHistoryItem],
    prior_goals: list[Goal],
    run_id: str,
) -> Observation:
    """Return the current typed Observation for this iteration."""
    artifact_map = _artifact_index_map(hits)
    payload = {
        "query": query,
        "run_id": run_id,
        "memory_hits": _memory_hits_payload(hits, artifact_map),
        "history": _history_payload(history, prior_goals),
        "prior_goals": _prior_goals_payload(prior_goals),
        "artifact_indexes_available": sorted(artifact_map),
    }
    output = _call_gateway(payload)
    return _to_observation(output, prior_goals, artifact_map)


def _call_gateway(payload: dict) -> PerceptionOutput:
    schema = PerceptionOutput.model_json_schema()
    response = LLM().chat(
        prompt=json.dumps(payload, indent=2),
        system=SYSTEM_PROMPT,
        provider=PERCEPTION_PROVIDER,
        auto_route="perception",
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "PerceptionOutput",
            "strict": True,
        },
        temperature=0,
        max_tokens=1024,
    )
    if response.get("parsed"):
        return PerceptionOutput.model_validate(response["parsed"])
    return PerceptionOutput.model_validate_json(response.get("text") or "{}")


def _to_observation(
    output: PerceptionOutput,
    prior_goals: list[Goal],
    artifact_map: dict[int, str],
) -> Observation:
    rows = output.goals or [PerceptionGoalOutput(text="Answer the user query")]

    if not prior_goals:
        goals = [
            Goal(
                id=f"g{i + 1}",
                text=_clean_goal_text(row.text),
                done=bool(row.done),
                attach_artifact_id=_artifact_for(row.artifact_index, artifact_map),
            )
            for i, row in enumerate(rows)
        ]
        return Observation(goals=goals)

    goals: list[Goal] = []
    for i, prior in enumerate(prior_goals):
        row = rows[i] if i < len(rows) else PerceptionGoalOutput(
            text=prior.text,
            done=prior.done,
        )
        goals.append(Goal(
            id=prior.id,
            text=prior.text,
            done=prior.done or bool(row.done),
            attach_artifact_id=None,
        ))

    first_open = next((i for i, goal in enumerate(goals) if not goal.done), None)
    if first_open is not None and first_open < len(rows):
        artifact_id = _artifact_for(rows[first_open].artifact_index, artifact_map)
        goals[first_open].attach_artifact_id = artifact_id or prior_goals[first_open].attach_artifact_id

    return Observation(goals=goals)


def _artifact_index_map(hits: Iterable[MemoryItem]) -> dict[int, str]:
    out: dict[int, str] = {}
    for hit in hits:
        if hit.artifact_id:
            out[len(out)] = hit.artifact_id
    return out


def _memory_hits_payload(hits: list[MemoryItem], artifact_map: dict[int, str]) -> list[dict]:
    by_artifact = {artifact_id: idx for idx, artifact_id in artifact_map.items()}
    payload = []
    for i, hit in enumerate(hits):
        artifact_index = by_artifact.get(hit.artifact_id or "")
        descriptor = hit.descriptor
        if hit.artifact_id:
            descriptor = descriptor.replace(hit.artifact_id, "[artifact]")
        row = {
            "i": i,
            "kind": hit.kind,
            "descriptor": descriptor,
            "source": hit.source,
            "has_artifact": hit.artifact_id is not None,
        }
        if artifact_index is not None:
            row["artifact_index"] = artifact_index
        payload.append(row)
    return payload


def _history_payload(history: list[AgentHistoryItem], prior_goals: list[Goal]) -> list[dict]:
    goal_positions = {goal.id: i for i, goal in enumerate(prior_goals)}
    payload = []
    for item in history:
        row = {
            "iter": item.iter,
            "kind": item.kind,
        }
        if item.goal_id in goal_positions:
            row["goal_position"] = goal_positions[item.goal_id]
        if item.text:
            row["text"] = item.text
        if item.tool:
            row["tool"] = item.tool
        if item.arguments:
            row["arguments"] = item.arguments
        if item.result_descriptor:
            row["result_descriptor"] = item.result_descriptor
        if item.artifact_id:
            row["has_artifact"] = True
        payload.append(row)
    return payload


def _prior_goals_payload(prior_goals: list[Goal]) -> list[dict]:
    return [
        {
            "position": i,
            "text": goal.text,
            "done": goal.done,
            "has_attachment": goal.attach_artifact_id is not None,
        }
        for i, goal in enumerate(prior_goals)
    ]


def _artifact_for(index: int | None, artifact_map: dict[int, str]) -> str | None:
    if index is None:
        return None
    return artifact_map.get(index)


def _clean_goal_text(text: str) -> str:
    text = " ".join((text or "").split())
    return text[:180] or "Answer the user query"
