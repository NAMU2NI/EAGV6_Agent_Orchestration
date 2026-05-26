# Session 6 Agent Orchestration Learning Notes

Date: 2026-05-26

This note summarizes what we built, what failed during testing, how we debugged it, and what fixes were applied.

## Goal

Build a generic Session 6 agent orchestration, not a hard-coded solution for one query.

The target architecture is:

- `agent6.py`: outer loop that wires the roles together.
- `memory.py`: typed durable memory service.
- `perception.py`: goal-state orchestrator.
- `decision.py`: chooses either one answer or one MCP tool call.
- `action.py`: pure MCP dispatch.
- `schemas.py`: Pydantic v2 contracts across boundaries.
- `mcp_server.py`: stdio MCP tools.
- Gateway V3: all LLM calls go through the gateway, not direct provider SDKs.

## Modules Built Or Connected

### `memory.py`

Memory was implemented as a typed file-backed service under `state/memory.json`.

Supported memory kinds:

- `fact`
- `preference`
- `tool_outcome`
- `scratchpad`

Main methods:

- `read(query, history, kinds=None, top_k=8)`
- `filter(kinds=None, goal_id=None, recent=None)`
- `relevant(...)`
- `record_outcome(...)`
- `remember(...)`

Important learning:

Memory reads are pure keyword overlap by default. This keeps the loop cheap and avoids an LLM call on every memory read.

### `perception.py`

Perception owns the goal list across iterations.

Responsibilities:

- First iteration: decompose query into bounded goals.
- Later iterations: preserve goal count, order, and text.
- Only update `done` flags and artifact attachment for the next unfinished goal.
- Use positional goal identity instead of model-generated goal IDs.
- Use artifact indexes, not raw `art:` handles, in the LLM output.

Current provider:

- Originally pinned to Gemini.
- Later changed to `openai` so Perception uses GPT through Gateway V3.

### `decision.py`

Decision receives one goal and returns exactly one of:

- a final answer for that goal, or
- a single typed tool call.

Important rule:

If attached artifact bytes are present, Decision bypasses auto-routing and uses a pinned provider. This was originally Gemini and later changed to OpenAI.

Current behavior:

- No attachment: `auto_route="decision"`
- With attachment: `provider="openai"`, `auto_route=None`

### `action.py`

Action is pure dispatch. It contains no LLM call.

Responsibilities:

- Receives a typed `ToolCall`.
- Rejects `path` or `url` arguments that start with `art:`.
- Calls MCP using `session.call_tool(...)`.
- Collapses MCP content blocks into a text result.
- Stores large payloads as artifacts when they exceed `ARTIFACT_THRESHOLD_BYTES`.

Current artifact threshold:

```text
4096 bytes
```

### `agent6.py`

Agent6 wires the loop:

1. Read memory.
2. Run Perception.
3. Select first unfinished goal.
4. Attach artifact bytes if Perception requested them.
5. Run Decision.
6. If Decision answers, append answer to history.
7. If Decision calls a tool, run Action and record outcome in memory/history.
8. Repeat until Perception marks all goals done or max iterations is hit.

We added readable trace logging so each stage is visible:

- Gateway boot
- MCP handshake
- memory hits
- Perception goals
- selected goal
- attached artifact
- Decision answer/tool call
- Action result
- Memory write
- final summary

This became critical for debugging.

## Environment and Dependency Fixes

### Problem: Importing `main.py` failed because `python-dotenv` was missing

Symptom:

```text
python-dotenv is not installed in that interpreter
```

Fix:

- Added `python-dotenv` to `requirements.txt`.
- Added `env_loader.py` as a fallback `.env` parser.
- Updated `main.py` and `mcp_server.py` to use `env_loader.load_env(...)`.

Learning:

Smoke tests should not fail just because raw Python lacks optional env loading support.

### Problem: Parent `.env` was not consistently loaded

The `.env` lives here:

```text
C:\Users\40100123\OneDrive - Anheuser-Busch InBev\EAG\Assignment 6\llm_gatewayV3\.env
```

But code runs from:

```text
...\llm_gatewayV3\llm_gatewayV3
```

Fix:

- Load both `ROOT.parent / ".env"` and `ROOT / ".env"`.
- Applied this to Gateway and MCP server loading.

### Problem: `.env` file had a UTF-8 BOM

Symptom:

The first env key could be read as:

```text
\ufeffGEMINI_API_KEY
```

instead of:

```text
GEMINI_API_KEY
```

Fix:

- Updated `env_loader.py` to read using `utf-8-sig`.
- Rewrote `.env` without BOM.

Learning:

On Windows, PowerShell edits can introduce BOM-related surprises. Env loaders should tolerate this.

## Query 1: Claude Shannon Wikipedia Fetch

Test query:

```text
Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory.
```

Expected architecture behavior:

1. Decision calls `fetch_url`.
2. Action stores the large Wikipedia page as an artifact.
3. Memory records the artifact handle.
4. Perception attaches that artifact to the extraction goal.
5. Decision reads the attached bytes and answers.

### Failure 1: The command hung for around 30 minutes

Command:

```powershell
python -B agent6.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
```

Observation:

No final result for a long time.

Debugging added:

- Readable trace prints in `agent6.py`.
- Per-stage timeouts:
  - LLM stage timeout
  - Action timeout

Root cause:

The loop was stuck inside Action while dispatching `fetch_url`.

More specifically:

- `fetch_url` used `crawl4ai`.
- `crawl4ai` could hang while fetching/rendering the page.

Fix:

- Changed `mcp_server.py` `fetch_url` default backend to a bounded `httpx` fetch.
- Used `BeautifulSoup` to extract readable text from HTML.
- Kept `crawl4ai` as an optional backend with timeout protection.

Learning:

Tool dispatch must be bounded. A single MCP tool should not be able to freeze the whole agent loop.

### Failure 2: Gateway rejected attached artifact through auto-router

After `fetch_url` worked, the Wikipedia artifact was around 85 KB.

Symptom:

```text
input exceeds 8000 tokens
```

Root cause:

Decision with an attached artifact was still using:

```python
auto_route="decision"
```

The Gateway router classified the prompt as too large and rejected it as `HUGE`.

Fix:

For attached artifacts, bypass auto-routing and explicitly choose the long-context provider:

```python
provider = ARTIFACT_DECISION_PROVIDER if attached else None
auto_route = None if attached else "decision"
```

At first this provider was Gemini. Later it was changed to OpenAI.

Learning:

Large artifact extraction should not go through the tiny/large router classifier. The caller already knows this is a large-context extraction case.

### Failure 3: Gemini model was unavailable

Symptom:

```text
gemini HTTP 404
This model models/gemini-3.1-flash-lite-preview is no longer available
```

Root cause:

`.env` had:

```text
GEMINI_MODEL=gemini-3.1-flash-lite-preview
```

Fix:

Changed to:

```text
GEMINI_MODEL=gemini-2.5-flash
```

Learning:

Provider model names are time-sensitive. Gateway startup/provider health should show the active model.

### Failure 4: Gemini quota hit during multi-goal extraction

Symptom:

```text
gemini HTTP 429
Quota exceeded for metric: generate_content_free_tier_requests
```

Root cause:

Perception split the Claude Shannon extraction into multiple goals:

```text
g1: Fetch page
g2: Extract birth date
g3: Extract death date
g4: Extract three contributions
```

Each attached-artifact Decision call used Gemini, so one query burned many Gemini calls.

Fix:

Moved the pinned Session 6 calls from Gemini to OpenAI:

- `PERCEPTION_PROVIDER=openai`
- `ARTIFACT_DECISION_PROVIDER=openai`
- `OPENAI_MODEL=gpt-5.1`

Gateway changes:

- Added `OpenAIProvider` in `providers.py`.
- Added OpenAI shortcuts in `router.py`:
  - `openai`
  - `oa`
  - `oai`
  - `gpt`
- Added OpenAI to `LIMITS`.
- Added OpenAI to `DEFAULT_ORDER`.
- Updated LARGE tier to prefer OpenAI before Gemini.

Current Gateway provider status after restart:

```text
providers: openai, gemini, groq, github, ollama
models.openai: gpt-5.1
```

Learning:

Pinned provider calls should use a provider with enough quota and context window for the assignment workload.

### Failure 5: Final answer only showed the last part

Symptom:

When Perception split the extraction into several goals, the final answer could miss earlier answers such as birth date and death date.

Root cause:

`agent6.py` final answer assembly returned only the last answer:

```python
for item in reversed(history):
    if item.kind == "answer" and item.text:
        return item.text
```

Fix:

Changed final answer assembly to join all answer entries in order:

```python
answers = [
    item.text.strip()
    for item in history
    if item.kind == "answer" and item.text and item.text.strip()
]
return "\n\n".join(answers)
```

Learning:

If Perception decomposes one user request into multiple answer goals, final synthesis must combine the answers, not only return the last one.

## Query 2: Tokyo Family-Friendly Weekend Plan

Test query:

```text
Find 3 family-friendly things to do in Tokyo this weekend. Check Saturday's weather forecast there and tell me which one is most appropriate.
```

### What happened

Perception created reasonable goals:

```text
g1: Identify three family-friendly things to do in Tokyo this coming weekend
g2: Check the weather forecast for Tokyo on Saturday of this coming weekend
g3: Recommend which of the three activities is most appropriate given Saturday's weather
```

But the loop never moved past `g1`.

Decision repeatedly called tools:

```text
iter 1: web_search
iter 2: web_search
iter 3: get_time
iter 4: web_search
iter 5: get_time
iter 6: get_time
iter 7: get_time
iter 8: web_search
```

Final trace:

```text
answer_chars: 0
FINAL: (no final answer)
```

### Root cause

This was not an exception. The loop kept working, but Decision never produced an answer for `g1`.

Since Agent6 always selects the first unfinished goal, and Perception never marked `g1` done, the loop stayed on `g1` until max iterations.

Main issue:

Decision stayed in tool-gathering mode instead of synthesizing an answer from the search results already collected.

Supporting issues:

- Search results were noisy and generic.
- Some results were broad listing pages rather than direct family-friendly recommendations.
- Persistent memory brought older unrelated hits, including Claude Shannon memory.
- There is no loop guard to stop repeated tool calls for the same goal.

### Learning

The system needs a rule that says:

```text
If the current goal asks to identify/list/find items, and memory/history already contains relevant tool results, synthesize an answer instead of calling another tool.
```

It also needs a loop guard:

```text
After N tool calls for the same goal, force Decision to answer from the gathered evidence using tool_choice="none".
```

This fix has been identified but not yet implemented.

## Current Status

Implemented:

- Typed memory service.
- Perception role.
- Decision role.
- Action role.
- Artifact store.
- Agent6 orchestration loop.
- Gateway V3 OpenAI provider.
- OpenAI GPT 5.1 configured.
- Readable trace monitoring.
- Timeout protection around LLM and Action stages.
- `fetch_url` no longer hangs by default.
- Final answer now combines all answer history entries.
- Anti-repetition guard for repeated tool calls on the same goal.
- Weather search prompt behavior now uses an exact Saturday date when possible.
- Perception can mark best-effort external-data answers as terminal so the loop can progress.
- Windows console output is forced to UTF-8-safe printing.
- Final response synthesis compresses verbose per-goal answers into a concise user-facing response.

Still to improve:

- Memory recall should avoid unrelated prior runs when the current query does not need them.
- Weather forecast tooling is still not a first-class MCP tool. Decision currently uses `get_time`, `web_search`, and sometimes `fetch_url`.
- Activity/event recommendations can be made more precise by fetching specific event pages rather than relying on search snippets.

## Useful Commands

Start Gateway V3:

```powershell
python -B main.py
```

Run Agent6:

```powershell
python -B agent6.py "Fetch https://en.wikipedia.org/wiki/Claude_Shannon and tell me his birth date, death date, and three key contributions to information theory."
```

Run with custom limits:

```powershell
python -B agent6.py --max-iterations 4 --llm-timeout 120 --action-timeout 60 "your query here"
```

Check Gateway providers:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8101/v1/providers
```

Check OpenAI is loaded:

```text
"openai": "gpt-5.1"
```

## Main Takeaways

1. Good trace logging turned an invisible hang into a visible stage-level failure.
2. MCP tools need timeouts and simple fallbacks.
3. Large artifact extraction should bypass auto-routing.
4. Provider model names and quotas can break otherwise correct architecture.
5. Final answer assembly must match the goal decomposition strategy.
6. ReAct-style loops need anti-repetition guards.
7. Perception deciding `done` is central. If it never marks the current goal done, Agent6 will never move forward.
8. Final response synthesis is different from task execution. The full agent loop should still do the work; synthesis only makes the finished answer easier to read.

## ReAct Loop Mapping

Yes, this Session 6 orchestration is a ReAct-style agent loop.

Classic ReAct means:

```text
Reason -> Act -> Observe -> Reason -> Act -> Observe ...
```

In this implementation, the same pattern is split across typed roles:

```text
Perception / Decision -> Action -> Memory / History -> Perception / Decision ...
```

Role mapping:

- `Perception`: observes memory and history, maintains goal state, and marks goals done.
- `Decision`: reasons about the next unfinished goal and chooses either an answer or one tool call.
- `Action`: executes the selected MCP tool. It does not reason.
- `Memory`: stores tool outcomes and durable facts/preferences.
- `agent6.py`: controls the loop and moves data between roles.

For the Tokyo query, the ReAct loop looked like this:

```text
Iteration 1:
Perception creates goals.
Decision answers g1 with three family-friendly activities.
History stores the answer.

Iteration 2:
Perception marks g1 done and selects g2.
Decision calls get_time for Tokyo.
Action executes get_time.
Memory/history store the Tokyo date/time.

Iteration 3:
Perception keeps g2 open.
Decision calls web_search using the exact Saturday date.
Action executes web_search.
Memory/history store weather search results.

Iteration 4:
Decision answers the weather goal from gathered evidence.
History stores the answer.

Iteration 5:
Perception marks g2 done and selects g3.
Decision recommends the best activity using the activity options plus weather evidence.
History stores the recommendation.

Iteration 6:
Perception marks all goals done.
agent6.py assembles the final answer.
```

Important learning:

The loop is not just tool-use. It is repeated state update:

```text
observe current state -> choose next action -> execute -> record result -> observe again
```

The anti-repetition guard we added is a ReAct safety mechanism. Without it, Decision can keep doing:

```text
web_search -> web_search -> get_time -> web_search -> ...
```

With the guard, after enough tool attempts for the same goal, Agent6 forces Decision to synthesize:

```text
tool -> tool -> answer from gathered evidence
```

This is what allowed the Tokyo query to move from `g1` to `g2` and finally to `g3`.

## Latest Fixes After Tokyo Testing

### Problem: Goal 1 got stuck in tool mode

The first Tokyo run stayed on:

```text
g1: Identify three family-friendly things to do in Tokyo this coming weekend
```

Decision repeatedly called tools instead of answering:

```text
web_search -> web_search -> get_time -> web_search -> get_time -> ...
```

Root cause:

Decision did not have a strong enough instruction to synthesize from existing results once enough evidence existed.

Fixes:

- Added a Decision rule: if memory/history already contains relevant tool results, synthesize an answer instead of calling another tool.
- Added `force_answer` to `decision.next_step(...)`.
- Added `MAX_TOOL_CALLS_PER_GOAL` in `agent6.py`.
- Agent6 now sets `force_answer=True` after repeated tool calls for the same goal.

Learning:

For ReAct loops, repeated tool use is not always progress. A generic agent needs a pressure valve that forces synthesis after enough evidence has been collected.

### Problem: Weather goal gave repeated limitation answers

After the first fix, the agent moved from `g1` to `g2`, but it kept answering that the weather data was insufficient. Perception kept `g2` open, so the loop repeated weather answers.

Root cause:

The weather snippets were weak and not date-specific. Also, Perception did not treat a direct best-effort answer as terminal.

Fixes:

- Decision now resolves relative weather dates using `get_time` and searches with the exact Saturday date.
- Example exact weather search:

```text
Tokyo weather forecast Saturday 2026-05-30 high low rain conditions
```

- Perception now has a rule that if an answer directly addresses a goal but says external data could not be determined after tool attempts, the goal can be marked done.
- `MAX_TOOL_CALLS_PER_GOAL` was raised from `2` to `3` so weather goals can do:

```text
get_time -> web_search -> answer
```

Result:

The Tokyo run completed:

```text
g1 done: three family-friendly activities
g2 done: Saturday weather forecast
g3 done: recommendation
```

The final recommendation was to choose an indoor activity, because the forecast suggested warm, humid conditions with a moderate chance of showers.

### Problem: Windows console crashed on Unicode output

Symptom:

```text
UnicodeEncodeError: 'charmap' codec can't encode character
```

Root cause:

PowerShell was using a Windows code page that could not print a Unicode character from the model response.

Fix:

`agent6.py` now configures stdout/stderr for UTF-8-safe output:

```python
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
```

Learning:

LLM output can contain Unicode punctuation. Console output should be encoding-safe, especially on Windows.

### Problem: Final Agent6 response was too verbose

After the agent successfully completed all goals, the final response was technically correct but too long. It included multiple per-goal answers stitched together.

Root cause:

The final answer combiner preserved all completed answers, but did not compress them into a user-facing final response.

Fix:

Added `decision.final_response(...)`.

The architecture still uses all roles:

```text
Memory -> Perception -> Decision -> Action -> Memory/History -> Perception
```

After all goals are complete, Agent6 now performs one final Decision-based synthesis step:

```text
completed per-goal answers -> concise final answer
```

Trace now includes:

```text
--- FINAL SYNTHESIS ---
draft_chars : ...
final_chars : ...
```

Learning:

Execution and presentation are separate concerns.

- The full agent loop should gather evidence, execute tools, and satisfy goals.
- The final synthesis step should make the completed result concise and readable.
- This keeps the architecture intact while improving the user experience.
