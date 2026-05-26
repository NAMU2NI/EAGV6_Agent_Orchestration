"""Pydantic v2 request/response models for llm_gatewayV3."""
from datetime import datetime
from typing import Any, Literal, Optional, Union
from pydantic import BaseModel, Field, ConfigDict, model_validator


class ToolDef(BaseModel):
    """Canonical tool definition. Schema is JSON-Schema (typically from Pydantic)."""
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    # Optional opaque per-provider metadata (e.g. Gemini thoughtSignature)
    # that must be echoed back when sending the assistant turn.
    provider_meta: Optional[dict[str, Any]] = None

    model_config = ConfigDict(extra="allow")


class CacheableSystemBlock(BaseModel):
    text: str
    cache: bool = False


class ResponseFormat(BaseModel):
    type: Literal["json_schema", "json_object"] = "json_schema"
    schema_: Optional[dict[str, Any]] = Field(default=None, alias="schema")
    name: str = "out"
    strict: bool = True

    model_config = ConfigDict(populate_by_name=True)


class ChatRequest(BaseModel):
    """Backward-compatible request — every new field is optional."""
    messages: Optional[list[dict[str, Any]]] = None
    prompt: Optional[str] = None
    system: Optional[Union[str, list[CacheableSystemBlock]]] = None
    provider: Optional[str] = None
    model: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    stream: bool = False

    # New in V2:
    tools: Optional[list[ToolDef]] = None
    tool_choice: Optional[Union[str, dict[str, Any]]] = None  # "auto" | "none" | {name}
    cache_system: Optional[bool] = None
    reasoning: Optional[Literal["off", "low", "medium", "high"]] = None
    response_format: Optional[ResponseFormat] = None

    # New in V3: when set, the gateway runs a router LLM first to pick a worker tier.
    # Role labels track which cognitive layer is asking. The worker is picked
    # from a tier-to-order table; router never sees system, tools, schemas.
    auto_route: Optional[Literal["perception", "memory", "decision"]] = None


class RouterDecision(BaseModel):
    """What the router agent decided. Echoed back on the worker response so the
    agentic-world caller can see which model was picked and why."""
    role: Literal["perception", "memory", "decision"]
    tier: Literal["TINY", "LARGE", "HUGE"]
    estimated_tokens: int
    router_provider: str
    router_model: str
    router_latency_ms: int
    chosen_worker_provider: Optional[str] = None
    chosen_worker_model: Optional[str] = None
    fallback_used: bool = False  # true if router LLM failed and tier was decided by token-count rule


class ChatResponse(BaseModel):
    provider: str
    model: str
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    stop_reason: Literal["tool_use", "end_turn", "max_tokens", "error"] = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    latency_ms: int = 0
    tool_call_dialect: Literal["native", "prompted_fallback", "none"] = "none"
    reasoning_applied: bool = False
    parsed: Optional[dict[str, Any]] = None  # set when response_format used
    attempted: list[dict[str, Any]] = Field(default_factory=list)
    # New in V3: present only when auto_route was used
    router_decision: Optional[RouterDecision] = None


class MemoryItem(BaseModel):
    id: str
    kind: Literal["fact", "preference", "tool_outcome", "scratchpad"]
    keywords: list[str]
    descriptor: str            # one short human-readable line
    value: dict                # structured payload
    artifact_id: str | None = None    # handle into the artifact store
    source: str
    run_id: str
    goal_id: str | None = None
    confidence: float
    created_at: datetime


class Artifact(BaseModel):
    id: str                    # "art:<sha256-prefix>"
    content_type: str
    size_bytes: int
    source: str
    descriptor: str


class Goal(BaseModel):
    id: str
    text: str                  # short imperative description
    done: bool
    attach_artifact_id: str | None = None


class PerceptionGoalOutput(BaseModel):
    """LLM-facing goal output.

    No goal id and no artifact handle by design. Identity is positional, and
    artifact references are mapped from integer indexes supplied in the prompt.
    """
    text: str
    done: bool = False
    artifact_index: int | None = None


class PerceptionOutput(BaseModel):
    goals: list[PerceptionGoalOutput]


class Observation(BaseModel):
    goals: list[Goal]

    @property
    def all_done(self) -> bool:
        return all(g.done for g in self.goals)

    def next_unfinished(self) -> Goal:
        for goal in self.goals:
            if not goal.done:
                return goal
        raise ValueError("no unfinished goals")


class AgentToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentHistoryItem(BaseModel):
    iter: int
    kind: Literal["answer", "action", "note"]
    goal_id: str | None = None
    text: str | None = None
    tool: str | None = None
    arguments: dict[str, Any] | None = None
    result_descriptor: str | None = None
    artifact_id: str | None = None


class DecisionOutput(BaseModel):
    answer: str | None = None         # exactly one of these two is populated
    tool_call: AgentToolCall | None = None

    @model_validator(mode="after")
    def exactly_one_action(self):
        if bool(self.answer) == bool(self.tool_call):
            raise ValueError("exactly one of answer or tool_call must be populated")
        return self

    @property
    def is_answer(self) -> bool:
        return self.answer is not None
