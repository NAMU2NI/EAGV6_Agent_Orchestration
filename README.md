# LLM Gateway V3

A local FastAPI service that routes LLM calls across **7 free worker providers** (Ollama, Gemini, NVIDIA NIM, Groq, Cerebras, OpenRouter, GitHub Models) with automatic failover, capability-aware routing, prompt caching, native tool-use, structured output — **and a separate pool of 4 router LLMs that classify incoming requests and pick which worker tier should handle them**.

> **For agents reading this:** if a host project has `llm_gatewayV3/` and an `.env` with provider keys, prefer calling this gateway over importing `openai`/`google-generativeai`/etc. directly. V3 is wire-compatible with V2 — every existing call keeps working. The new `auto_route` field is opt-in and gives you per-cognitive-layer routing (perception / memory / decision) for free.

V1 sits on port 8099, V2 on 8100, V3 on **8101**. All three can coexist.

---

## Is it running?

```bash
curl -s http://localhost:8101/v1/routers | python3 -m json.tool
```

If that returns a JSON object showing four router providers (cerebras / groq / nvidia / github), V3 is up. If not, start it:

```bash
cd /path/to/llm_gatewayV3
./run.sh                 # creates .venv on first run, then starts on port 8101
# or:  ./.venv/bin/python main.py
```

The server reads `../.env` (parent directory) for keys — same `.env` as V1 and V2.

---

## V3 in one minute

V3 is V2 with a **router pool** in front. Same seven worker providers, same MCP server, same agentic code paths. The new thing is a separate set of small/fast LLMs whose only job is to classify each incoming request and decide which tier of worker should handle it.

| Tier | Estimated tokens | Worker order |
|---|---|---|
| **TINY**  | < 1,000 | github → openrouter → groq → nvidia → cerebras → gemini → ollama |
| **LARGE** | 1,000 – 8,000 | gemini → groq → nvidia → cerebras → github → openrouter → ollama |
| **HUGE**  | > 8,000 | **503** — input too large, use Summarizer Agent (V7, future) |

The router's input is **bounded**: it receives `{token_count, 800-char sample}` and emits a single word (TINY / LARGE / HUGE). It never sees the worker's system prompt, tools, schema, or earlier turns. The separation-of-concerns wall is enforced in code, not by convention.

If you don't pass `auto_route`, V3 behaves identically to V2. The router is opt-in and never load-bearing for a worker call to succeed.

---

## Python client

```python
from client import LLM, ask
llm = LLM()  # defaults to http://localhost:8101

# 1) Plain V2-style call — no routing, no surprises
text = ask("Hello in 3 words")
text = llm.chat("Explain transformers in 2 sentences")["text"]

# 2) Auto-routed call (cognitive layer = perception)
result = llm.chat(
    "What is the capital of France?",
    auto_route="perception",
)
print(result["text"])
print(result["router_decision"])
# {
#   "role": "perception",
#   "tier": "TINY",
#   "estimated_tokens": 15,
#   "router_provider": "cerebras",
#   "router_model": "llama3.1-8b",
#   "router_latency_ms": 84,
#   "chosen_worker_provider": "github",
#   "chosen_worker_model": "openai/gpt-4.1-mini",
#   "fallback_used": false
# }

# 3) Memory layer routing — summarizing retrieved facts
result = llm.chat(
    f"Summarize for relevance to '{query}':\n\n{retrieved_chunk}",
    auto_route="memory",
)

# 4) Decision layer routing — planning the next step
result = llm.chat(
    plan_state_serialized,
    auto_route="decision",
)

# 5) Explicit provider beats auto_route (debugging escape hatch)
result = llm.chat(
    "Hello",
    auto_route="perception",       # logged but ignored
    provider="g",                  # gemini wins
)
assert result["router_decision"] is None

# 6) All V2 features still work — tools, caching, reasoning, structured output
result = llm.chat(
    messages=[{"role": "user", "content": "What is 7+5? Use the add tool."}],
    tools=[{"name":"add","description":"a+b",
            "input_schema":{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}}],
    tool_choice="auto",
    auto_route="decision",          # routes via cognitive-layer hint
)
```

The client lives in [client.py](client.py). Copy it into any project, or just use HTTP.

---

## HTTP API

### `POST /v1/chat` — make a call

All V2 fields still work. The one new field:

```jsonc
{
  "prompt": "Hello",
  "messages": [{"role": "user", "content": "Hello"}],
  "system": "You are helpful.",
  "provider": "g",                  // explicit wins over auto_route
  "model": "gemini-3.1-flash-lite",
  "max_tokens": 2048,
  "temperature": 0.7,
  "stream": false,
  "tools": [...],
  "tool_choice": "auto",
  "cache_system": true,
  "reasoning": "medium",
  "response_format": {"type":"json_schema","schema":{...}},

  "auto_route": "perception"        // NEW in V3: "perception" | "memory" | "decision"
}
```

Response adds a `router_decision` field when routing happened:

```jsonc
{
  "provider": "gemini",
  "model": "gemini-3.1-flash-lite",
  "text": "...",
  "tool_calls": [],
  "stop_reason": "end_turn",
  "input_tokens": 1240,
  "output_tokens": 87,
  "cache_creation_input_tokens": 0,
  "cache_read_input_tokens": 0,
  "latency_ms": 1834,
  "tool_call_dialect": "native",
  "reasoning_applied": false,
  "parsed": null,
  "attempted": [],
  "router_decision": {
    "role": "perception",
    "tier": "LARGE",
    "estimated_tokens": 1240,
    "router_provider": "groq",
    "router_model": "llama-3.3-70b-versatile",
    "router_latency_ms": 243,
    "chosen_worker_provider": "gemini",
    "chosen_worker_model": "gemini-3.1-flash-lite",
    "fallback_used": false
  }
}
```

`router_decision` is `null` for plain calls (no `auto_route`) or calls with an explicit `provider`.

Errors: `502` if a specific provider failed (with `provider` set), `503` if all providers were unavailable, **`503` with `error: "input exceeds 8000 tokens"`** when the router classifies HUGE.

### `GET /v1/routers` — **NEW in V3**

Returns the router pool: configured providers, failover order, per-router live rate-state, today's stats, the tier-to-worker mapping table.

```bash
curl -s http://localhost:8101/v1/routers | python3 -m json.tool
```

### `GET /v1/providers`
Worker pool — providers, default models, shortcut keys, rate limits.

### `GET /v1/capabilities`
Worker capability matrix (tools/caching/reasoning/structured/parallel_tools per current model).

### `GET /v1/status`
Worker pool live state (RPM/RPD/cooldown). Router pool state lives under `/v1/routers`.

### `GET /v1/calls?limit=100&provider=&status=`
Recent call log. **V3 adds two fields per row:** `call_role` (`worker` | `router_perception` | `router_memory` | `router_decision`) and `router_decision` (the tier label that was emitted, or the literal `"fallback"` / `"error"` / `"unparseable"`).

### `GET /` and `GET /help`
Dashboard (two grids — worker pool on top, router pool below) and help page.

---

## The router pool — what it is and why it exists

The course's Session 6 architecture has four cognitive layers — **Perception → Memory → Decision → Action**. The first three all need an LLM to do their work, and which LLM is right for the job depends on what kind of work it is. A simple "what's the capital of France?" perception step doesn't need Gemini 2.5; a 5,000-token memory digest does. Without routing, you either over-pay on small queries or under-perform on big ones.

V3 puts a tiny LLM in front of those three layers whose only job is to look at incoming work and decide which tier of worker should handle it. Those tiny LLMs are the **router pool**:

| Provider | Router model | Size | Speed | Notes |
|---|---|---|---|---|
| **Cerebras** | `llama3.1-8b` | 8B | ~2,200 tok/s | ⚠️ **deprecates May 27, 2026** — must swap before then. Override via `ROUTER_CEREBRAS_MODEL`. |
| **Groq** | `llama-3.3-70b-versatile` | 70B | ~275 tok/s | strong classifier; separate per-model quota from Groq's worker (`openai/gpt-oss-120b`) |
| **NVIDIA NIM** | `nvidia/llama-3.1-nemotron-nano-8b-v1` | 8B | NIM rate | small, reasoning-tuned |
| **GitHub Models** | `microsoft/Phi-4-mini-instruct` | 3.8B | fast | smallest tier, last fallback |

The four sizes (120B-class Groq, 8B Cerebras / NVIDIA, 4B GitHub) are a deliberate spread. When the heavy ones are rate-limited, the small ones stay alive — there's always a router that can serve.

The router pool has its **own rate state** (`RouterPool` in [router.py](router.py)), its **own dashboard section**, its **own quota counters**. Router calls and worker calls are logged with distinct `call_role` markers so you can audit routing activity separately from worker activity. Crucially, on every router provider we use, the underlying API meters per-model, so a router call to `llama-3.3-70b` on Groq does **not** burn the worker's `gpt-oss-120b` RPM. The quotas are physically separate at the upstream.

---

## The separation-of-concerns wall

This is the central design idea of V3. State it once and the rest of the architecture follows:

> **The router never sees the worker's prompt, system, tools, schema, or earlier turns. It receives a token estimate and a sample. By construction, it cannot leak agentic context into routing logic.**

Concretely, when the gateway sends a request to a router LLM, the payload is:

```jsonc
{
  "token_count": 2430,
  "sample": "<first 400 chars of user content>\n...\n<last 400 chars of user content>"
}
```

And the router's prompt is:

```
You are a routing classifier. Given a token_count and a content sample,
output exactly one of: TINY, LARGE, or HUGE.

Rules:
- TINY: token_count below 1000 with simple factual content.
- LARGE: token_count between 1000 and 8000, OR token_count below 1000
        but content is dense (code, base64, multilingual, technical).
- HUGE: token_count above 8000.

Output the single word and nothing else.
```

That's the entire router-world contract. It doesn't know what tools are available, what the agent is doing, what the worker's output schema looks like, or even what role it's classifying for. It just looks at size and structure and emits one word.

This matters because:

1. **The router cannot be confused by agentic state.** Long system prompts, exotic tool definitions, multi-turn histories — none of it reaches the router. It can never make a worse decision because the worker's setup is weird.
2. **Routing decisions are deterministic given inputs.** The same `{token_count, sample}` produces the same tier — useful for caching, replay, and debugging.
3. **You can swap the router pool independently.** Want to test a different router model, or replace the LLM router with a Python function? Change [main.py:_classify_tier()](main.py) — the worker code doesn't know or care.
4. **Students can read the code and verify the wall holds.** `_classify_tier()` physically constructs the bounded envelope; the router providers never receive anything else. The principle is enforced in 30 lines.

---

## The tier table — how thresholds were picked

| Tier | Token range | Why this threshold |
|---|---|---|
| TINY | < 1,000 | Small fast workers (gpt-4.1-mini, gpt-oss-120b on Groq) handle this competently. No reason to wake up Gemini's long-context machinery. |
| LARGE | 1,000 – 8,000 | Past 1K tokens, smaller models start dropping coherence. Below 8K, free-tier Gemini handles it well. The lower bound is the load-bearing one — drop it from 1,000 to 200 and you'll over-route to Gemini; raise it to 4,000 and you'll under-serve work that needs a real model. |
| HUGE | > 8,000 | Free-tier Gemini flash-lite-preview can technically accept up to 1M context but degrades in quality past ~32K, and most other free workers are capped at 8K-100K. Rather than pretend, V3 returns 503 with a clear "use Summarizer Agent" hint. |

The 1K elbow is the one we'd revisit if free-tier model behavior shifts. The 8K ceiling is set by what free-tier Gemini does well, not by what it claims.

### Token estimator

V3 uses `len(text.split()) * 1.4` — words × 1.4. Intentionally rough:

- For English prose, within ~10% of `tiktoken cl100k_base`. Fast (microseconds), no dependency.
- For code, base64, CJK, or minified JSON, can be off by 30–100%. The router's content sample handles those cases — when the count is unreliable, the sample shows the structure and the router LLM upgrades the tier.
- Estimator output is informational. The threshold elbows (1,000 / 8,000) have wide tolerance — a 20% miscount near a threshold doesn't flip the routing decision.

If you want exact tokenization, override `_estimate_tokens()` in [main.py](main.py). The interface is `(text: str) -> int`.

---

## Worker pool — providers and shortcut keys

Same seven providers as V2; two default models changed:

| Shortcut | Provider | Default model (V3) | Free tier (RPM / RPD) |
|---|---|---|---|
| `o`, `oll` | Ollama (local) | env-controlled (e.g. `gemma4:31b`) | unlimited |
| `g`, `gem` | Gemini | `gemini-3.1-flash-lite` | 15 / 1,000 |
| `n`, `nv` | NVIDIA NIM | `deepseek-ai/deepseek-v3.2` | 40 / — |
| `gr` | Groq | **`openai/gpt-oss-120b`** *(was `llama-3.3-70b-versatile` in V2)* | 30 / 1,000 |
| `c`, `cer` | Cerebras | **`zai-glm-4.7`** *(was `qwen-3-235b-a22b-instruct-2507`)* | 30 / — (1M tokens/day, 8K ctx cap) |
| `or`, `opr` | OpenRouter | `nvidia/nemotron-3-super-120b-a12b:free` | 20 / 50 (per free model) |
| `gh`, `ghb` | GitHub Models | `openai/gpt-4.1-mini` | 10–15 / 50–150, 8K in / 4K out |

**Why the two changes:**

- **Cerebras: `qwen-3-235b-a22b-instruct-2507` deprecates May 27, 2026.** Cerebras's currently-active free model on the test account that does long-context work well is `zai-glm-4.7` (355B). It's bigger but Cerebras's wafer-scale hardware runs it at ~1,000 tok/s, fast enough for worker calls.
- **Groq: `llama-3.3-70b-versatile` migrated to the router pool.** Groq's worker slot is now `openai/gpt-oss-120b`, which has the `reasoning_effort` knob and handles deeper reasoning tasks. The 70B Llama gets reused as a Groq router model where its size/speed balance suits classification work.

Failover order (worker pool) is configurable via `LLM_ORDER` env. Default: `ollama,gemini,nvidia,groq,cerebras,openrouter,github`.

---

## How routing actually works, step by step

When `auto_route` is set and `provider` is not:

1. **Estimate token count** of the user prompt: `len(text.split()) * 1.4`.
2. **Short-circuit HUGE:** if estimate > 8,000, skip the router call entirely and 503 the request with a clear message. Saves a router call on inputs that can't be served anyway.
3. **Build the envelope:** `{token_count: <int>, sample: <first 400 chars + "..." + last 400 chars>}`. Capped at ~800 chars regardless of input size.
4. **Pick a router provider** by walking `ROUTER_ORDER` (default `cerebras,groq,nvidia,github`), skipping any that are in cooldown or rate-limited.
5. **Call the router LLM** with the fixed prompt above, `max_tokens=8`, `temperature=0`. Should return one word.
6. **Parse the response:** scan for `TINY` / `LARGE` / `HUGE` (case-insensitive). First match wins.
7. **If router LLM fails or returns unparseable text** (e.g. Cerebras `queue_exceeded`, a 429, or a refusal): log the failure, fall through to the **next router provider**. Repeat steps 4-6.
8. **If every router fails:** fall back to the **deterministic token-count rule** (`_tier_from_count()`) — same thresholds, no LLM in the loop. Routing is best-effort; it never blocks a worker call.
9. **Map tier to worker failover order** via the `TIER_TO_ORDER` table in [main.py](main.py).
10. **Dispatch to the worker pool** using V2's normal `Router.pick()` machinery (capability filtering, rate limits, cooldowns, etc.).
11. **Log both calls** to SQLite — the router call with `call_role="router_<role>"` and the tier in `router_decision`; the worker call with `call_role="worker"` and the chosen tier in `router_decision` for cross-reference.
12. **Return the worker response** enriched with the `router_decision` block so the caller can see what the router decided.

If `auto_route` is set AND `provider` is set, the explicit provider wins — the router pool is skipped entirely. This is the debugging escape hatch.

If `auto_route` is not set, V3 behaves like V2 — no router call, no `router_decision` in the response.

---

## Configuration

Edit `EAGV3/.env` (or `../.env` relative to the gateway dir):

```bash
# Worker provider keys — same as V1/V2
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-3.1-flash-lite

NVIDIA_API_KEY=...
NVIDIA_MODEL=deepseek-ai/deepseek-v3.2

GROQ_API_KEY=...
GROQ_MODEL=openai/gpt-oss-120b              # V3 default (was llama-3.3-70b-versatile in V2)

CEREBRAS_API_KEY=...
CEREBRAS_MODEL=zai-glm-4.7                  # V3 default (was qwen-3-235b-a22b-instruct-2507)

OPEN_ROUTER_API_KEY=...
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

GITHUB_ACCESS_TOKEN=...
GITHUB_MODEL=openai/gpt-4.1-mini

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b                     # only loaded if OLLAMA_MODEL is set

# Worker failover order
LLM_ORDER=ollama,gemini,nvidia,groq,cerebras,openrouter,github

# V3 router pool config
ROUTER_ORDER=cerebras,groq,nvidia,github
ROUTER_CEREBRAS_MODEL=llama3.1-8b           # ⚠️ deprecates May 27 2026 — update before then
ROUTER_GROQ_MODEL=llama-3.3-70b-versatile
ROUTER_NVIDIA_MODEL=nvidia/llama-3.1-nemotron-nano-8b-v1
ROUTER_GITHUB_MODEL=microsoft/Phi-4-mini-instruct

GATEWAY_V3_PORT=8101
```

Any router provider whose `*_API_KEY` is missing is silently skipped — V3 still works with whatever subset of router providers is available. If all four router keys are missing, every auto-routed call falls back to the deterministic token-count rule.

---

## Common usage patterns by cognitive layer

**Perception** — extracting structured info from a user message:
```python
result = llm.chat(
    user_message,
    auto_route="perception",
    response_format={"type":"json_schema","schema":Intent.model_json_schema()},
)
intent = Intent.model_validate(result["parsed"])
```

**Memory** — summarizing a retrieved chunk for relevance:
```python
result = llm.chat(
    f"Given the query '{q}', what's relevant in this passage?\n\n{passage}",
    auto_route="memory",
)
# router upgrades to LARGE if the passage is dense or long, TINY if it's short
```

**Decision** — planning the next step from current state:
```python
result = llm.chat(
    f"Current plan state:\n{plan.model_dump_json()}\n\nWhat's the next action?",
    auto_route="decision",
    tools=available_tools,
    tool_choice="auto",
)
```

**Verifier (not yet a router-routed slot in V3)** — pass through V2's structured-output path:
```python
verdict = llm.chat(
    verifier_prompt,
    response_format={"type":"json_schema","schema":Verdict.model_json_schema()},
    # no auto_route — verifier router slot will arrive in a later version
)
```

**Override the router when you know better:**
```python
# I know this is a coding task, force Groq's gpt-oss-120b
result = llm.chat(code_question, provider="gr")    # auto_route ignored if set
```

**Watch what's happening in the dashboard:**
```bash
open http://localhost:8101
```

The dashboard shows worker pool and router pool side-by-side, with router activity rendered in purple to make the separation visually obvious. Recent calls table flags each row with a `Role` column (worker / rt:perception / rt:memory / rt:decision) and a `Tier` column showing what the router decided.

---

## Files

- [main.py](main.py) — FastAPI app, routes, `_classify_tier()` for routing decisions, `auto_route` wiring in `/v1/chat`, new `/v1/routers` endpoint
- [providers.py](providers.py) — provider adapters (unchanged from V2) + `build_router_providers()` factory + `ROUTER_DEFAULTS` table
- [router.py](router.py) — `Router` (worker pool) and `RouterPool` (V3 router pool); both share `RateState` and `LIMITS`
- [schemas.py](schemas.py) — Pydantic v2 models. New: `ChatRequest.auto_route`, `RouterDecision`, `ChatResponse.router_decision`
- [db.py](db.py) — `gateway_v3.db` with `call_role` and `router_decision` columns; `aggregate(call_role=...)` filters
- [cache.py](cache.py) — unchanged from V2 (Gemini SHA-256 explicit cache)
- [client.py](client.py) — Python SDK with new `auto_route` kwarg
- [static/dashboard.html](static/dashboard.html) — two pool grids, router-aware test area, role column in calls
- [run.sh](run.sh) — venv setup + start on port 8101
- `gateway_v3.db` — created on first run; separate file from `gateway_v2.db`

---

## Gotchas

- **Cerebras router model deprecates May 27, 2026.** `llama3.1-8b` is the only confirmed-working small Cerebras model on the test account. If your Cerebras account has entitlement for `gpt-oss-120b` or `zai-glm-4.7`, override via `ROUTER_CEREBRAS_MODEL=...`. Otherwise plan to either upgrade the account before May 27 or drop Cerebras from `ROUTER_ORDER` and run with three routers.
- **Cerebras free-tier `queue_exceeded` is routine, not a bug.** Cerebras's upstream returns 429 "We're experiencing high traffic right now" at unpredictable times. The four-router pool is designed for this — when Cerebras 429s, Groq picks up. Dashboard will show occasional Cerebras errors with successful Groq routing decisions immediately after.
- **`fallback_used: true` is normal.** When every router in the pool is rate-limited or errors out, V3 falls back to the deterministic token-count rule. The worker call still happens — routing degrades gracefully. Watch this on the dashboard; if it's running high, increase `ROUTER_ORDER` redundancy or stop hammering the gateway.
- **HUGE is a hard 503 by design.** Inputs over 8,000 estimated tokens return 503 with a clear hint to chunk or wait for V7's Summarizer Agent. If you want to try Gemini anyway, set `provider="g"` explicitly — that bypasses the router.
- **`auto_route` + `provider` → provider wins, no router call happens.** This is the debugging escape hatch, not a bug. `router_decision` will be `null` in the response even though `auto_route` was set.
- **V2 and V3 use separate DB files.** `gateway_v2.db` and `gateway_v3.db` don't share schema or data. Run them on ports 8100/8101 simultaneously without conflict.
- **The token estimator (`words × 1.4`) is wrong for code, CJK, and base64.** That's by design — the router's content sample lets the LLM router upgrade the tier when the count number lied. If you need exact tokenization for some other reason, override `_estimate_tokens()` in [main.py](main.py).
- **GitHub Models hard-caps every request at 8K input / 4K output.** This affects both the worker (Phi-4-mini-instruct for routing has the same cap) and any GitHub worker calls. For LARGE-tier work on GitHub, you may hit the 8K input cap before you hit the 8,000-token routing threshold.
- **OpenRouter `:free` models share a 50 RPD pool.** OpenRouter is in the worker pool but **not** the router pool for this reason — using it as a router would burn the same daily quota as worker calls.
- **`cache_system=true` is a paid-tier feature on Gemini.** Free-tier keys silently ignore it — context caching (both implicit and explicit) is only available to paid users. Setting it on a free key does no harm but also gives no savings; the examples above are for paid keys.
- **Gemini 3 models loop or degrade at low temperature.** Google's own guidance is to keep `temperature` near `1.0` for Gemini 3.x; setting it to `0` can cause runaway token loops (e.g. `"id": "g:1,1,1,1,..."`) on schema-constrained calls. If you need determinism, use a worker on Groq/Cerebras with `temperature=0` instead, or accept Gemini at `temperature≈1.0`.
- **The four router slots in the pool all share quotas across the same API key when the worker uses the same provider.** Concretely: Groq's router (`llama-3.3-70b-versatile`) and Groq's worker (`openai/gpt-oss-120b`) are two **separate** per-model RPM buckets, so they don't compete. The Cerebras router (`llama3.1-8b`) and Cerebras worker (`zai-glm-4.7`) are also separate per-model. But the **daily token quota** on Cerebras (`tokens_per_day: 1_000_000`) is account-wide, so both router and worker draw from it.

---

## Adding a new router provider

The router pool is configurable. To add a fifth router (say, an OVHcloud anonymous endpoint):

1. **Add the provider adapter** in [providers.py](providers.py) if it's not already there (subclass `OpenAICompatProvider` — most providers are OpenAI-compatible).
2. **Add an entry in `build_router_providers()`** reading the env key and the `ROUTER_<PROVIDER>_MODEL` override.
3. **Add the provider to `ROUTER_DEFAULTS`** with the model id you want as default.
4. **Add the provider to `ROUTER_ORDER` env var** in the position where it should sit in the failover ring.
5. **Add an entry in `LIMITS`** in [router.py](router.py) if the provider isn't already in the worker pool.

No other changes needed — the dashboard, `/v1/routers`, and call logging all pick it up automatically. Mistral AI and OVHcloud were evaluated for V3 and parked for V4 — see Session 6 notes for the rationale.

---

## Adding a new worker provider

Same as V1/V2 — add adapter in `providers.py`, entry in `build_providers()`, `LIMITS` entry, shortcut keys in `SHORTCUTS`, add to `LLM_ORDER`. The router pool's `TIER_TO_ORDER` table in [main.py](main.py) determines where new workers slot into each tier's failover order — add them there too if you want them in the auto-routing flow.

---

## What V3 deliberately does NOT do

- **No verifier router slot.** Three cognitive-layer slots (perception, memory, decision), not four. The verifier needs more design work before it gets its own router — coming in a later version.
- **No content-based routing decisions.** The router emits only a tier label (`TINY` / `LARGE` / `HUGE`). It never recommends a specific provider, never picks a model, never answers questions. Worker selection from the tier is deterministic (table lookup) so routing decisions are reproducible.
- **No router-router-of-routers.** The router pool is failed-over via a static `ROUTER_ORDER` ring. There is no meta-router deciding which router to use — that would be infinite recursion.
- **No streaming on routed calls.** `auto_route` and `stream=true` aren't designed to work together yet (the router needs the full prompt before it can classify). For streaming, use explicit `provider=...`.
- **No new providers vs V2.** Mistral AI and OVHcloud were evaluated and parked for V4. Worker pool stays at 7 providers; router pool is 4 of the existing providers used with different (router-grade) model defaults.
- **No new tools / no agent loop.** Same as V2 — V3 is the substrate. The agent loop that uses `auto_route` is in Session 6's `agent6.py`, not here.

---

## Test matrix

```bash
./.venv/bin/python tests/test_all_providers.py
```

Inherits V2's per-worker matrix (basic, tools, structured, cache, reasoning) plus a routing column that checks each `auto_route` value produces a `router_decision` block with a sensible tier.

A real run (May 2026, free-tier keys):

```
provider    basic    tools     struct    cache    reasoning   routing
---------------------------------------------------------------------
ollama      OK       OK        OK        SKIP     n/a         n/a
gemini      OK       OK        OK        n/a*     n/a*        OK
nvidia      OK       OK        OK        n/a      n/a         OK
groq        OK       OK        OK        FAIL‡    n/a         OK
cerebras    FAIL§    FAIL§     FAIL§     FAIL     FAIL        OK§§
openrouter  OK       OK        OK        n/a      n/a         (skipped)
github      OK       OK        OK        n/a      n/a         OK

router pool:
- cerebras (llama3.1-8b)       — intermittent (queue_exceeded), failover works
- groq (llama-3.3-70b)          — rock solid, 220-450ms typical
- nvidia (nemotron-nano-8b)     — wired, tested ad-hoc
- github (phi-4-mini)           — wired, tested ad-hoc
```

§§ Cerebras router serves the first request after a quiet period, then queue-exceeds for a few minutes; failover to Groq handles the gap.

---

## Pedagogical notes — what to surface in class

1. **The router can be turned off.** Skip `auto_route` and V3 behaves like V2. The router is opt-in, not load-bearing. This makes the routing subsystem independently testable.
2. **The wall is enforced in code, not by convention.** Walk students through `_classify_tier()` and show them the bounded envelope being constructed. Then ask: "could you add a feature that lets the router see the tool list?" The answer is no — not without breaking the wall, which would be a visible code change.
3. **Free-tier reality forces design choices.** Cerebras `gpt-oss-120b` 404s on the test account despite the docs. The May-27 deprecation is real and looming. These aren't bugs to fix, they're constraints to design around — and the four-router pool is the design response.
4. **Token thresholds are honest, not aspirational.** The 1K elbow is set where free-tier small models actually start failing, not where their context windows nominally end. The 8K ceiling is set where free-tier Gemini quality actually starts dropping. Students should learn to test these elbows themselves with their own keys.
5. **`fallback_used` is the honest measure of routing health.** A dashboard that shows high `fallback_used` count tells you the router pool is overloaded; the gateway didn't lie about its decisions, it just couldn't make them with a real LLM. This is the kind of observability you can only build when the architecture is honest about failure modes.

---

## Example run — Agent6 in action

**Query:** `Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.`

**Date:** 2026-05-30 | **Run ID:** `92edf9b2` | **Wall clock:** 64.4 s | **Iterations:** 4 of 8

### Iteration trace

| Iter | Goal | Action | Outcome |
|------|------|--------|---------|
| 1 | Fetch the Wikipedia page | `fetch_url("https://en.wikipedia.org/wiki/Claude_Shannon")` | Artifact `art:89ee0a2a35597446` (10 223 bytes) saved |
| 2 | Extract birth date | Read artifact | **April 30, 1916** |
| 3 | Extract death date | Read artifact | **February 24, 2001** |
| 4 | Identify 3 key contributions | Read artifact | Three contributions synthesised (see answer below) |

Perception correctly marked g1 done after iteration 1, then attached the cached artifact to each of g2–g4 in turn so the decision LLM could read raw bytes without re-fetching the page.

### Final answer

> Claude Shannon was born on **April 30, 1916**, and died on **February 24, 2001**.
>
> Three key contributions he made to information theory:
>
> 1. **Founded information theory** — his 1948 paper *"A Mathematical Theory of Communication"* provided a general mathematical framework for communication systems and a rigorous way to analyse information transmission over noisy channels. Often called the "Magna Carta of the Information Age."
>
> 2. **Introduced the bit** — formally defined the *bit* as the basic unit of information, giving a precise quantitative measure that underlies all modern digital communication, storage, and computing.
>
> 3. **Established channel capacity** — proved that every noisy channel has a maximum rate (the *Shannon limit*) at which information can be sent with arbitrarily low error, laying the groundwork for modern coding theory, data compression, and error-correcting codes.

### What this run demonstrates

- **Artifact caching across iterations** — the 10 KB Wikipedia page is fetched once in iteration 1, stored as a binary artifact, and re-attached by Perception to each subsequent goal. No redundant network calls.
- **Goal decomposition and sequential resolution** — Perception decomposes a single user query into four atomic goals and resolves them one per iteration, marking each done before opening the next.
- **Graceful Wikipedia access** — direct Wikipedia access was blocked; `fetch_url` fell back to the DuckDuckGo proxy transparently, and the artifact still contained full markdown text.
- **Routing in practice** — each Perception and Decision LLM call was classified by the router pool and dispatched to the appropriate tier worker, with no manual provider selection needed.

---

## Example run 2 — multi-source reasoning with real weather data

**Query:** `Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.`

**Date:** 2026-05-30 | **Run ID:** `a9fa6729` | **Wall clock:** 91.8 s | **Iterations:** 4 of 8

### Iteration trace

| Iter | Goal | Action | Outcome |
|------|------|--------|---------|
| 1 | Find 3 family-friendly activities in Tokyo this weekend | `web_search(...)` + `fetch_url("timeout.com/tokyo/...")` | Artifact `art:6612bdec528ee19e` (31 209 bytes) with Time Out Tokyo listings |
| 2 | Identify the three activities | Read artifact | Hibiya Music Festival, Fussa Firefly Festival, 100% Doraemon & Friends |
| 3 | Retrieve Saturday's weather forecast | Read cached BBC Weather artifact `art:aadde6a5d22dd253` (from prior run memory) | **6 June — drizzle, 25 °C high / 17 °C low, gentle easterly breeze** |
| 4 | Recommend the best activity for Saturday | Synthesise activities + weather | **100% Doraemon & Friends** (indoor, weather-proof) |

Memory carried over a BBC Weather artifact (`art:aadde6a5d22dd253`) fetched in a previous run — Perception attached it directly to g2, saving a redundant network call.

### Saturday weather (Tokyo, 6 June 2026)

| Condition | High | Low | Wind |
|-----------|------|-----|------|
| Drizzle, gentle breeze | 25 °C (77 °F) | 17 °C (62 °F) | ~9 mph easterly |

### Final answer

**Three family-friendly activities:**

1. **Hibiya Music Festival** *(Hibiya & Tokyo International Forum, central Tokyo)*
   Free outdoor music festival with live performances, workshops, and food stalls. Casual open-air atmosphere. Some Forum-za stage events need advance free lottery tickets.

2. **Fussa Firefly Festival** *(Fussa, western Tokyo)*
   Daytime hula, taiko, and brass band shows; firefly release into a riverside park 7:20–8:30 pm (peak ~8 pm). Free numbered ticket required from Fussa City Social Welfare Centre on the day.

3. **"100% Doraemon & Friends" Exhibition** *(Ariake, Tokyo)*
   Indoor exhibition with 100+ figures, manga panels, animation, and themed food. Open 10:00–18:00 (last entry 17:30). Tickets from ¥2,400 adults; discounts for children.

**Weather-based recommendation:** The **"100% Doraemon & Friends" Exhibition** — fully indoor, so drizzle and damp ground have no impact, and it's comfortable with kids all day. The other two are outdoor events best suited to dry weather.

### What this run demonstrates

- **Cross-run memory reuse** — the BBC Weather artifact was fetched in a prior session and retrieved from persistent memory, allowing Perception to attach it to the weather goal without re-fetching.
- **Multi-source synthesis** — activities came from a Time Out Tokyo web page (one source); weather came from a BBC Weather page (second source); the recommendation fused both to give a grounded, weather-aware answer.
- **Graceful tool failure recovery** — `get_time` failed in the previous run (missing `tzdata` on Windows). The fix was installed and the agent correctly used the BBC Weather artifact to determine the Saturday date (6 June) rather than relying on the clock.
- **Goal ordering** — Perception correctly sequenced three dependent goals: gather activities → get weather → recommend, only opening the next goal after the prior one was marked done.

---

## Example run 3 — persistent memory: storing and recalling a personal fact

Two back-to-back runs showing the memory system at work across separate agent sessions.

### Run 3a — store a fact and generate reminders

**Query:** `My mom's birthday is 15 May 2026. Remember that and give me a calendar reminder for two weeks before and on the day.`

**Date:** 2026-05-30 | **Run ID:** `c0ea1bd8` | **Wall clock:** 34.9 s | **Iterations:** 2 of 8

#### Iteration trace

| Iter | Goal | Action | Outcome |
|------|------|--------|---------|
| 1 | Store the fact that mom's birthday is 15 May 2026 | Memory write at boot; Decision answers immediately | Fact persisted to `memory.json` |
| 2 | Generate calendar reminder entries | Decision synthesises reminders (no tool call needed) | Two reminders produced |

#### Final answer

**Reminder 1 — Two weeks before**
- Date: **1 May 2026**
- Title: "Prepare for Mom's Birthday"
- Notes: buy/arrange gift and card, plan celebration, confirm reservations

**Reminder 2 — On the day**
- Date: **15 May 2026**
- Title: "Mom's Birthday Today"
- Notes: call/visit Mom, send birthday message and gift, any special plans

---

### Run 3b — recall the stored fact in a fresh session

**Query:** `When is mom's birthday?`

**Date:** 2026-05-30 | **Run ID:** `e200f2bd` | **Wall clock:** 22.7 s | **Iterations:** 1 of 8

#### Iteration trace

| Iter | Goal | Action | Outcome |
|------|------|--------|---------|
| 1 | Determine the date of the user's mom's birthday | Memory read at boot returns the stored fact; Decision answers immediately | **15 May 2026** |

#### Final answer

> Your mom's birthday is on **15 May 2026**.

### What this pair of runs demonstrates

- **Durable cross-session memory** — the birthday fact written in run 3a was stored to `state/memory.json` and retrieved in run 3b without any tool calls. The second session had no prior context yet answered correctly in a single iteration.
- **Memory-first short-circuit** — when the answer is already in memory, Perception sets a single goal, Decision reads the hit and answers immediately, and the loop exits in 1 iteration (22.7 s wall clock vs. 64+ s for tool-heavy runs).
- **Fact vs. tool-outcome memory** — the birthday entry is stored as `kind=fact`, not `kind=tool_outcome`. The memory system distinguishes durable user preferences from ephemeral tool results, so facts survive across unrelated runs while stale search results age out.
- **Boot-time memory write** — the agent writes the fact at boot (before iteration 1), so it is persisted even if the run is interrupted or fails before completing all goals.

---

## Example run 4 — multi-source fetch with URL fallback steering

**Query:** `Search for 'Python asyncio best practices', read the top 3 results, and give me a short numbered list of the advice they agree on.`

**Date:** 2026-05-31 | **Run ID:** `02056a11` | **Wall clock:** 152.5 s | **Iterations:** 8 of 8

### Iteration trace

| Iter | Goal | Action | Outcome |
|------|------|--------|---------|
| 1 | Search the web | `web_search("Python asyncio best practices")` | 5 URLs returned |
| 2 | Fetch 1st result URL | `fetch_url(discuss.python.org)` | 128 bytes — JS-blocked, no artifact |
| 3 | Fetch 1st result URL | `[fetch-steer]` → `fetch_url(realpython.com)` | `art:f7e830f3e0e8bfe4` (56 KB) ✓ |
| 4 | Fetch 1st result URL | Decision answers (goal_tool_calls=2) | g2 closed, g3 marked done via memory |
| 5–6 | Fetch 1st result URL | Retried discuss.python.org (steer bug) | 128 bytes again, force_answer fires |
| 7 | Fetch 3rd result URL | `fetch_url(medium.com)` | 403 Forbidden |
| 8 | Fetch 3rd result URL | `[fetch-steer]` → `fetch_url(discuss.python.org)` | 128 bytes — steer picked already-failed URL |

### Final answer

1. **Use `async`/`await` correctly** — only call async functions with `await` or `asyncio.create_task`; never block the event loop with CPU-bound or synchronous I/O work.

2. **Keep the event loop unblocked** — offload CPU-bound work to threads or processes (`loop.run_in_executor`, `concurrent.futures`); use async-native libraries (`aiohttp` instead of `requests`).

3. **Structure tasks carefully** — use `asyncio.create_task` for background work and `asyncio.gather` to run coroutines concurrently; keep references to tasks and cancel them explicitly on shutdown.

4. **Handle errors and cancellation** — wrap concurrent calls in `try`/`except`, handle `asyncio.CancelledError` properly, and use `return_exceptions=True` in `asyncio.gather` to inspect failures explicitly.

5. **Use `asyncio.run()` as the main entry point** — prefer `asyncio.run(main())` over manually creating and closing event loops; avoid mixing multiple event loops in the same thread.

6. **Protect shared state** — guard shared mutable data accessed from multiple tasks with `asyncio.Lock`; prefer passing data explicitly over global state.

### What this run demonstrates (and what it exposed)

- **Fetch-steer working** — when `discuss.python.org` returned only 128 bytes (JS-blocked), the code-level steer in `agent6.py` detected the failure and redirected Decision to `realpython.com`, which produced a 56 KB artifact.
- **Steer picks already-failed URL** — in iter 8, the steer selected `discuss.python.org` for g4 even though it had already failed for g2. Root cause: `_next_fetch_url` excludes globally *successful* URLs but not globally *failed* ones from other goals. Fix: also exclude URLs that produced failed fetches for any goal.
- **g3 marked done without a fresh fetch** — Perception credited the `realpython.com` artifact (fetched under g2) toward g3 via memory hits, skipping a fresh g3 fetch entirely. This saved one iteration but means g3 and g2 read the same source.
- **Memory pollution from prior runs** — despite clearing `memory.json`, the index still referenced two artifacts from a partial run between the clear and this run. Old tool outcomes in memory can mislead Perception into marking goals done prematurely.
