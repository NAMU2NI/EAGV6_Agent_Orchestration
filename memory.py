"""Session 6 memory service.

Memory is deliberately boring in S6: one JSON file under ``state/``, typed
records, pure-Python keyword reads, and structured writes. The storage backend
is easy to delete between assignment attempts and easy to replace later.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from client import LLM
from schemas import AgentToolCall, MemoryItem


ROOT = Path(__file__).parent
STATE_DIR = ROOT / "state"
MEMORY_PATH = STATE_DIR / "memory.json"

MemoryKind = Literal["fact", "preference", "tool_outcome", "scratchpad"]

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "his", "her", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "our", "please", "that", "the", "their", "them", "then",
    "there", "this", "to", "was", "we", "with", "you", "your",
}


class RelevantMemoryId(BaseModel):
    id: str
    reason: str = ""


class RelevantMemoryResult(BaseModel):
    hits: list[RelevantMemoryId] = Field(default_factory=list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_state() -> None:
    STATE_DIR.mkdir(exist_ok=True)
    if not MEMORY_PATH.exists():
        MEMORY_PATH.write_text("[]", encoding="utf-8")


def _load() -> list[MemoryItem]:
    _ensure_state()
    raw = json.loads(MEMORY_PATH.read_text(encoding="utf-8-sig") or "[]")
    return [MemoryItem.model_validate(item) for item in raw]


def _save(items: list[MemoryItem]) -> None:
    _ensure_state()
    payload = [item.model_dump(mode="json") for item in items]
    MEMORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stable_id(kind: str, descriptor: str, run_id: str, goal_id: str | None = None) -> str:
    h = hashlib.sha256()
    h.update(kind.encode("utf-8"))
    h.update(b"\0")
    h.update(descriptor.encode("utf-8"))
    h.update(b"\0")
    h.update(run_id.encode("utf-8"))
    h.update(b"\0")
    h.update((goal_id or "").encode("utf-8"))
    return f"mem:{h.hexdigest()[:16]}"


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization used by read(). Kept intentionally simple."""
    tokens = re.findall(r"[a-z0-9][a-z0-9_:/.-]*", (text or "").lower())
    return [tok for tok in tokens if len(tok) > 1 and tok not in STOPWORDS]


def keywords_for(*parts: object, limit: int = 24) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for part in parts:
        for tok in tokenize(str(part)):
            if tok not in seen:
                seen.add(tok)
                out.append(tok)
            if len(out) >= limit:
                return out
    return out


def write(item: MemoryItem) -> MemoryItem:
    """Append or replace a typed memory item."""
    items = [old for old in _load() if old.id != item.id]
    items.append(item)
    _save(items)
    return item


def remember(query: str, *, source: str = "user_query", run_id: str = "manual") -> MemoryItem | None:
    """Persist durable user-stated facts/preferences when the query says one.

    This is intentionally heuristic in S6. The memory service should not spend
    an LLM call on every run just to decide whether a sentence is memorable.
    """
    text = (query or "").strip()
    low = text.lower()
    kind: MemoryKind | None = None

    if any(marker in low for marker in ("remember that", "note that", "keep in mind")):
        kind = "fact"
    if any(marker in low for marker in ("i prefer", "my preference", "i like", "i dislike")):
        kind = "preference"

    if kind is None:
        return None

    descriptor = text[:220]
    item = MemoryItem(
        id=_stable_id(kind, descriptor, run_id),
        kind=kind,
        keywords=keywords_for(text),
        descriptor=descriptor,
        value={"text": text},
        source=source,
        run_id=run_id,
        confidence=0.85,
        created_at=_now(),
    )
    return write(item)


def record_outcome(
    *,
    tool_call: AgentToolCall,
    result_text: str,
    artifact_id: str | None,
    run_id: str,
    goal_id: str | None = None,
) -> MemoryItem:
    descriptor = _describe_tool_outcome(tool_call, result_text, artifact_id)
    item = MemoryItem(
        id=_stable_id("tool_outcome", descriptor, run_id, goal_id),
        kind="tool_outcome",
        keywords=keywords_for(tool_call.name, tool_call.arguments, descriptor),
        descriptor=descriptor,
        value={
            "tool": tool_call.name,
            "arguments": tool_call.arguments,
            "result_preview": (result_text or "")[:1000],
        },
        artifact_id=artifact_id,
        source="mcp",
        run_id=run_id,
        goal_id=goal_id,
        confidence=1.0,
        created_at=_now(),
    )
    return write(item)


def add_scratchpad(note: str, *, run_id: str, goal_id: str | None = None) -> MemoryItem:
    item = MemoryItem(
        id=_stable_id("scratchpad", note[:220], run_id, goal_id),
        kind="scratchpad",
        keywords=keywords_for(note),
        descriptor=note[:220],
        value={"text": note},
        source="agent",
        run_id=run_id,
        goal_id=goal_id,
        confidence=0.5,
        created_at=_now(),
    )
    return write(item)


def read(
    query: str,
    history: Iterable[object] | None = None,
    *,
    kinds: Iterable[MemoryKind] | None = None,
    top_k: int = 8,
) -> list[MemoryItem]:
    """Keyword overlap across item keywords plus descriptor tokens."""
    wanted = set(kinds or [])
    query_terms = set(tokenize(query))
    for turn in history or []:
        query_terms.update(tokenize(str(turn)))

    ranked: list[tuple[int, float, MemoryItem]] = []
    for item in _load():
        if wanted and item.kind not in wanted:
            continue
        item_terms = set(item.keywords) | set(tokenize(item.descriptor))
        overlap = query_terms & item_terms
        if not overlap:
            continue
        ranked.append((len(overlap), item.created_at.timestamp(), item))

    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [item for _, _, item in ranked[:top_k]]


def filter(
    *,
    kinds: Iterable[MemoryKind] | None = None,
    goal_id: str | None = None,
    recent: int | None = None,
) -> list[MemoryItem]:
    """Structured filtering by kind, goal, and recency."""
    wanted = set(kinds or [])
    items = _load()
    if wanted:
        items = [item for item in items if item.kind in wanted]
    if goal_id is not None:
        items = [item for item in items if item.goal_id == goal_id]
    items.sort(key=lambda item: item.created_at, reverse=True)
    if recent is not None:
        items = items[:recent]
    return items


def relevant(
    query: str,
    *,
    kinds: Iterable[MemoryKind] | None = None,
    top_k: int = 5,
    candidate_limit: int = 25,
) -> list[MemoryItem]:
    """LLM-scored relevance for cases where keyword recall is too weak."""
    candidates = filter(kinds=kinds, recent=candidate_limit)
    if not candidates:
        return []

    schema = RelevantMemoryResult.model_json_schema()
    candidate_text = "\n".join(
        f"- id={item.id} kind={item.kind} descriptor={item.descriptor}"
        for item in candidates
    )
    llm = LLM()
    response = llm.chat(
        prompt=(
            f"Query:\n{query}\n\n"
            f"Candidate memory items:\n{candidate_text}\n\n"
            f"Return up to {top_k} relevant memory item ids in priority order."
        ),
        system="You rank memory items for relevance. Return only the requested schema.",
        response_format={
            "type": "json_schema",
            "schema": schema,
            "name": "RelevantMemoryResult",
            "strict": True,
        },
        auto_route="memory",
        temperature=0,
        max_tokens=512,
    )
    parsed = RelevantMemoryResult.model_validate(response.get("parsed") or {})
    by_id = {item.id: item for item in candidates}
    hits = [by_id[hit.id] for hit in parsed.hits if hit.id in by_id]
    return hits[:top_k]


def clear_state() -> None:
    """Test helper: remove only the memory JSON file."""
    if MEMORY_PATH.exists():
        MEMORY_PATH.unlink()
    _ensure_state()


def _describe_tool_outcome(
    tool_call: AgentToolCall,
    result_text: str,
    artifact_id: str | None,
) -> str:
    args = ", ".join(f"{k}={v}" for k, v in sorted(tool_call.arguments.items()))
    result = (result_text or "").strip().replace("\n", " ")
    result = result[:120] + ("..." if len(result) > 120 else "")
    art = f" -> artifact {artifact_id}" if artifact_id else ""
    return f"{tool_call.name}({args}){art}: {result}"
