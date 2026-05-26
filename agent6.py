from __future__ import annotations

import argparse
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time
import uuid

import httpx

import decision
import action
import artifacts
import memory
import perception
from schemas import AgentHistoryItem, Goal


MAX_ITERATIONS = 8
GATEWAY_URL = "http://localhost:8101"
ROOT = Path(__file__).parent
LLM_STAGE_TIMEOUT_SECONDS = int(os.getenv("AGENT6_LLM_STAGE_TIMEOUT", "120"))
ACTION_TIMEOUT_SECONDS = int(os.getenv("AGENT6_ACTION_TIMEOUT", "120"))
MAX_TOOL_CALLS_PER_GOAL = int(os.getenv("AGENT6_MAX_TOOL_CALLS_PER_GOAL", "3"))
TRACE_WIDTH = 86

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _hr(enabled: bool, title: str = "", ch: str = "-") -> None:
    if not enabled:
        return
    if title:
        pad = max(1, TRACE_WIDTH - len(title) - 6)
        print(f"\n{ch * 3} {title} {ch * pad}", flush=True)
    else:
        print(ch * TRACE_WIDTH, flush=True)


def _line(enabled: bool, text: str = "", indent: int = 0) -> None:
    if enabled:
        print(f"{' ' * indent}{text}", flush=True)


def _kv(enabled: bool, label: str, value, indent: int = 2) -> None:
    if enabled:
        print(f"{' ' * indent}{label:<18}: {value}", flush=True)


def _preview(value, limit: int = 180) -> str:
    text = str(value).replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + f"... <+{len(text) - limit} chars>"
    return text


def _json_preview(value, limit: int = 220) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except TypeError:
        text = str(value)
    return _preview(text, limit)


async def run(
    query: str,
    *,
    trace: bool = True,
    max_iterations: int = MAX_ITERATIONS,
    llm_timeout: int = LLM_STAGE_TIMEOUT_SECONDS,
    action_timeout: int = ACTION_TIMEOUT_SECONDS,
) -> str:
    started_at = time.time()
    _hr(trace, "AGENT6 RUN", "=")
    _kv(trace, "started_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    _kv(trace, "gateway", GATEWAY_URL)
    _kv(trace, "query", _preview(query, 320))
    _kv(trace, "max_iterations", max_iterations)
    _kv(trace, "llm_timeout_s", llm_timeout)
    _kv(trace, "action_timeout_s", action_timeout)

    _hr(trace, "BOOT")
    _line(trace, "[gateway] checking Gateway V3")
    ensure_gateway()
    run_id = uuid.uuid4().hex[:8]
    history: list[AgentHistoryItem] = []
    prior_goals: list[Goal] = []
    _kv(trace, "run_id", run_id)

    # Durable memory: classify the user's query so facts/preferences
    # in it survive into future runs.
    memory.remember(query, source="user_query", run_id=run_id)
    _line(trace, "[memory.write] remembered user query if it carried a durable fact/preference")

    _hr(trace, "MCP HANDSHAKE")
    _kv(trace, "server_script", ROOT / "mcp_server.py")
    _line(trace, "[mcp] opening stdio session")
    async with mcp_session() as session:
        _line(trace, "[mcp] loading tools")
        mcp_tools = await load_tools(session)
        tools = mcp_tools_for_decision(mcp_tools)
        _kv(trace, "tools_found", len(tools))
        for tool in tools:
            _line(trace, f"- {tool['name']}: {_preview(tool.get('description', ''), 120)}", indent=4)

        for it in range(1, max_iterations + 1):
            _hr(trace, f"ITER {it} / {max_iterations}")
            hits = memory.read(query, history)
            _line(trace, "[memory.read]")
            _kv(trace, "hits", len(hits), indent=4)
            for idx, hit in enumerate(hits[:5]):
                art = f" artifact={hit.artifact_id}" if hit.artifact_id else ""
                _line(trace, f"hit[{idx}] kind={hit.kind}{art}", indent=4)
                _line(trace, f"descriptor: {_preview(hit.descriptor, 180)}", indent=6)

            _hr(trace, f"ITER {it} -> PERCEPTION")
            _kv(trace, "prior_goals", len(prior_goals))
            _kv(trace, "history_items", len(history))
            t0 = time.time()
            try:
                obs = await _with_timeout(
                    "perception",
                    asyncio.to_thread(perception.observe, query, hits, history, prior_goals, run_id),
                    llm_timeout,
                )
            except TimeoutError as exc:
                _line(trace, f"[perception] ERROR {exc}")
                return f"ERROR: {exc}"
            except Exception as exc:
                _line(trace, f"[perception] ERROR {exc}")
                return f"ERROR: perception failed: {exc}"
            _kv(trace, "latency_s", round(time.time() - t0, 2))
            prior_goals = obs.goals
            _line(trace, "[perception] goals")
            for goal in obs.goals:
                status = "done" if goal.done else "open"
                attach = f" attach={goal.attach_artifact_id}" if goal.attach_artifact_id else ""
                _line(trace, f"[{status}] {goal.id}: {goal.text}{attach}", indent=4)
            if obs.all_done:
                _line(trace, "[done] all goals satisfied")
                break

            goal = obs.next_unfinished()
            _hr(trace, f"ITER {it} -> SELECT")
            _kv(trace, "goal_id", goal.id)
            _kv(trace, "goal", goal.text)
            attached = []
            if goal.attach_artifact_id and artifacts.exists(goal.attach_artifact_id):
                data = artifacts.get_bytes(goal.attach_artifact_id)
                attached.append((
                    goal.attach_artifact_id,
                    data,
                ))
                _line(trace, "[attach]")
                _kv(trace, "artifact_id", goal.attach_artifact_id, indent=4)
                _kv(trace, "bytes", len(data), indent=4)
            elif goal.attach_artifact_id:
                _line(trace, "[attach]")
                _kv(trace, "missing", goal.attach_artifact_id, indent=4)

            _hr(trace, f"ITER {it} -> DECISION")
            _kv(trace, "mcp_tools", len(tools))
            _kv(trace, "attached", len(attached))
            goal_tool_calls = _tool_call_count(history, goal.id)
            force_answer = goal_tool_calls >= MAX_TOOL_CALLS_PER_GOAL
            _kv(trace, "goal_tool_calls", goal_tool_calls)
            _kv(trace, "force_answer", force_answer)
            t0 = time.time()
            try:
                out = await _with_timeout(
                    "decision",
                    asyncio.to_thread(
                        decision.next_step,
                        goal,
                        hits,
                        attached,
                        history,
                        tools,
                        force_answer=force_answer,
                    ),
                    llm_timeout,
                )
            except TimeoutError as exc:
                _line(trace, f"[decision] ERROR {exc}")
                return f"ERROR: {exc}"
            except Exception as exc:
                _line(trace, f"[decision] ERROR {exc}")
                return f"ERROR: decision failed: {exc}"
            _kv(trace, "latency_s", round(time.time() - t0, 2))

            if out.is_answer:
                _line(trace, "[decision] ANSWER")
                _line(trace, _preview(out.answer or "", 500), indent=4)
                history.append(AgentHistoryItem(
                    iter=it,
                    kind="answer",
                    goal_id=goal.id,
                    text=out.answer,
                ))
                continue

            _line(trace, "[decision] TOOL_CALL")
            _kv(trace, "tool", out.tool_call.name, indent=4)
            _kv(trace, "arguments", _json_preview(out.tool_call.arguments), indent=4)

            _hr(trace, f"ITER {it} -> ACTION")
            t0 = time.time()
            try:
                result_text, art_id = await _with_timeout(
                    "action",
                    action.execute(session, out.tool_call),
                    action_timeout,
                )
            except TimeoutError as exc:
                _line(trace, f"[action] ERROR {exc}")
                return f"ERROR: {exc}"
            except Exception as exc:
                _line(trace, f"[action] ERROR {exc}")
                return f"ERROR: action failed: {exc}"
            _kv(trace, "latency_s", round(time.time() - t0, 2))
            if art_id:
                _line(trace, "[action] artifact result")
                _kv(trace, "artifact_id", art_id, indent=4)
                _kv(trace, "descriptor", _preview(result_text, 260), indent=4)
            else:
                _line(trace, "[action] text result")
                _kv(trace, "descriptor", _preview(result_text, 300), indent=4)
            _hr(trace, f"ITER {it} -> MEMORY WRITE")
            memory.record_outcome(
                tool_call=out.tool_call,
                result_text=result_text,
                artifact_id=art_id,
                run_id=run_id,
                goal_id=goal.id,
            )
            _kv(trace, "tool", out.tool_call.name)
            _kv(trace, "goal_id", goal.id)
            _kv(trace, "artifact_id", art_id or "none")
            history.append(AgentHistoryItem(
                iter=it,
                kind="action",
                goal_id=goal.id,
                tool=out.tool_call.name,
                arguments=out.tool_call.arguments,
                result_descriptor=result_text[:300],
                artifact_id=art_id,
            ))
            _kv(trace, "history_items", len(history))

    draft_answer = final_answer_from(history)
    answer = draft_answer
    if draft_answer:
        _hr(trace, "FINAL SYNTHESIS")
        t0 = time.time()
        try:
            answer = await _with_timeout(
                "final synthesis",
                asyncio.to_thread(decision.final_response, query, history, draft_answer),
                llm_timeout,
            )
            _kv(trace, "latency_s", round(time.time() - t0, 2))
            _kv(trace, "draft_chars", len(draft_answer))
            _kv(trace, "final_chars", len(answer))
        except Exception as exc:
            _line(trace, f"[final synthesis] ERROR {exc}")
            _line(trace, "[final synthesis] using unsummarized answer")
            answer = draft_answer

    _hr(trace, "TRACE SUMMARY")
    _kv(trace, "run_id", run_id)
    _kv(trace, "iterations_seen", len({item.iter for item in history}))
    _kv(trace, "history_items", len(history))
    _kv(trace, "answer_chars", len(answer))
    _kv(trace, "wall_clock_s", round(time.time() - started_at, 2))
    _hr(trace, "FINAL", "=")
    if trace:
        print(answer or "(no final answer)", flush=True)
        print("=" * TRACE_WIDTH, flush=True)
    return answer


def final_answer_from(history: list[AgentHistoryItem]) -> str:
    ordered_keys: list[str] = []
    answers_by_key: dict[str, str] = {}
    for item in history:
        if item.kind != "answer" or not item.text or not item.text.strip():
            continue
        key = item.goal_id or f"iter:{item.iter}"
        if key not in answers_by_key:
            ordered_keys.append(key)
        answers_by_key[key] = item.text.strip()
    return "\n\n".join(answers_by_key[key] for key in ordered_keys)


def _tool_call_count(history: list[AgentHistoryItem], goal_id: str | None) -> int:
    return sum(
        1
        for item in history
        if item.kind == "action" and item.goal_id == goal_id
    )


def ensure_gateway() -> None:
    try:
        r = httpx.get(f"{GATEWAY_URL}/v1/providers", timeout=3)
        r.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"Gateway V3 is not reachable at {GATEWAY_URL}. Start main.py first."
        ) from exc


async def _with_timeout(stage: str, awaitable, timeout_seconds: int):
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"{stage} timed out after {timeout_seconds}s") from exc


@asynccontextmanager
async def mcp_session():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server.py")],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


async def load_tools(session):
    return (await session.list_tools()).tools


def mcp_tools_for_decision(mcp_tools) -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema or {
                "type": "object",
                "properties": {},
            },
        }
        for tool in mcp_tools
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Session 6 agent loop.")
    parser.add_argument("query", nargs="+", help="User query to run through the agent.")
    parser.add_argument("--quiet", action="store_true", help="Only print the final answer.")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    parser.add_argument("--llm-timeout", type=int, default=LLM_STAGE_TIMEOUT_SECONDS)
    parser.add_argument("--action-timeout", type=int, default=ACTION_TIMEOUT_SECONDS)
    args = parser.parse_args()
    query = " ".join(args.query)
    answer = asyncio.run(
        run(
            query,
            trace=not args.quiet,
            max_iterations=args.max_iterations,
            llm_timeout=args.llm_timeout,
            action_timeout=args.action_timeout,
        )
    )
    if args.quiet or answer.startswith("ERROR:"):
        print(answer)


if __name__ == "__main__":
    main()
