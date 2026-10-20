# Phase 4: The JARVIS agent

**Goal:** turn "search + answer" into an agent that decides what to do. It searches
documents, runs read-only SQL for numbers, reads a URL you give it, does arithmetic,
remembers the conversation, cites everything and says "I don't know". And it has to
do all that on a 4B model that fits in 4 GB of VRAM.

```bash
make chat                                                  # REPL with streaming and visible tool calls
.venv/bin/python -m jarvis_rag.agent -q "How many support tickets are still open?"
.venv/bin/python -m jarvis_rag.agent --thread work         # same thread id = same memory, across restarts
make eval-agent                                            # 19 graded scenarios (~4 min)
.venv/bin/python eval/run_agent_eval.py --set auto_search=false   # the experiment below
```

## What we built

| File | Job |
|---|---|
| `jarvis_rag/tools.py` | The 6 (+1) tools, SQL safety (`check_select`, `run_select`), AST calculator (`safe_eval`), SSRF guard (`public_only_hook`), citation labels (`cite_label`) |
| `jarvis_rag/agent.py` | System prompt + knowledge-base catalogue, 5 middlewares, `build_agent`, checkpointers, `stream_events`, chat CLI |
| `eval/agent_set.jsonl` | 19 scenarios: SQL (7), PDF/DOCX/text/code (6), mixed search+SQL, calculator, follow-up memory, unanswerable (3), small talk |
| `eval/run_agent_eval.py` | Deterministic grading: tools called, numbers (± tolerance), required strings, citation, abstention |

New settings (all in `.env.example`): `MEMORY_BACKEND=sqlite`, `AUTO_SEARCH=true`,
`AGENT_MAX_MODEL_CALLS=8`, `HISTORY_MAX_TOKENS=3000`, `TOOL_OUTPUT_MAX_CHARS=4000`,
`SQL_MAX_ROWS=200`, `SQL_TIMEOUT_S=5`, `ALLOW_INGEST_TOOL=false`, `FETCH_ALLOW_PRIVATE=false`.
New dependency: `langgraph-checkpoint-sqlite` (installed into `.venv` only).

Verified against the installed versions before writing code: `langchain 1.4.6`
(`from langchain.agents import create_agent`), `langgraph 1.2.16`, `langgraph-checkpoint 4.3.0`.
Two facts that shaped the code:
- In this version, **an exception inside a tool crashes the whole run** (checked with a
  tool that raises `ValueError`). LangChain only turns *argument-validation* errors into
  messages. `ToolErrorMiddleware` is opt-in, so we add it.
- `create_agent`'s graph is `__start__ → model ⇄ tools → __end__`, and middleware hooks
  slot in around those nodes.

## The ReAct loop

`learn/05_agent_loop_from_scratch.py` wrote this loop by hand. `create_agent` builds the
same loop as a LangGraph graph:

```
            ┌──────────────────────── before_agent: auto_search ───────────────────────┐
user msg ─▶ │  [search_knowledge_base(question)] ─▶ ToolMessage (as if the model asked) │
            └──────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
          ┌────────── model node (wrapped by middleware, outermost first) ──────────┐
          │ kb_prompt (system prompt + catalogue) → sql_follow_through →            │
          │ history_window (trim) → ModelCallLimit → ChatOllama.bind_tools(...)     │
          └─────────────────────────────────────────────────────────────────────────┘
                 │ AIMessage with tool_calls?                      │ no
                 ▼ yes                                             ▼
          tools node (ToolErrorMiddleware turns                  answer ─▶ checkpoint saved
          exceptions into error ToolMessages)                              (thread_id)
                 │
                 └──── ToolMessage(s) appended ────▶ back to the model node
```

Each pass through the model node is one LLM call. The model never runs anything
itself. It outputs a structured request, for example
`{"name": "run_sql", "args": {"query": "SELECT ..."}}`. Ollama's chat template for qwen3
renders the tool schemas into the prompt and parses `<tool_call>` JSON back out. The
graph runs the function and appends a `ToolMessage` with the same `tool_call_id`. Then
the model is called again with the longer message list. The loop ends when the model
replies without tool calls, or when `ModelCallLimitMiddleware` stops it.

**Memory is a checkpointer.** After every node, LangGraph saves the whole state (the
message list) under `thread_id`. Next turn, it loads that state and appends the new
`HumanMessage`. `InMemorySaver` keeps it in a dict, so it's gone on exit.
`SqliteSaver` writes `storage/memory.db`, so `--thread work` continues tomorrow. It is
deliberately a different file from `storage/jarvis.db`: `run_sql` can read every table
in `jarvis.db`, and the model has no business querying chat histories.

## The tools, and what keeps each one safe

| Tool | What it does | Guard rails |
|---|---|---|
| `search_knowledge_base(query, k=5, source_type=None)` | Phase 3 retriever (`rerank`). Each passage is prefixed with its citation label. | `k` clamped to 1–10. `source_type` aliases (`markdown`→`text`, `csv`→`table`). Bad value → error that lists the valid ones. |
| `list_tables()` / `describe_table(table)` | Discover the schema: columns, ranges, distinct values, sample rows | The table name must match `sqlite_master` exactly; that check is also what makes the f-string SQL safe |
| `run_sql(query)` | One SELECT, ≤ 200 rows | **Three independent locks**, below |
| `fetch_url(url)` | Main text of a page or PDF, fenced as untrusted | http(s) only, robots.txt, private/LAN/metadata IPs refused (every redirect checked), size clipped |
| `calculator(expression)` | Exact arithmetic | AST walk, no `eval()`. Only numbers, operators and 14 math functions. Exponent ≤ 1000. |
| `ingest_text(title, text)` | Save a note as `note://slug` | Only exists when `ALLOW_INGEST_TOOL=true` |

**`run_sql`: defence in depth.** No single check is trusted:
1. `check_select`: comments stripped, one statement, must start with `SELECT`/`WITH`. It
   is a regex, so it is the weakest lock, there mainly to give the model a clear error.
2. `file:storage/jarvis.db?mode=ro`: SQLite itself refuses to write the file.
3. `conn.set_authorizer(...)`: SQLite asks this function about *every action* while
   compiling the statement. We allow only `SELECT`, `READ`, `FUNCTION` and `RECURSIVE`,
   so `PRAGMA`, `ATTACH`, temp tables and writes are denied even if (1) had a bug.
   `test_authorizer_and_readonly_hold_even_if_the_regex_is_bypassed` patches (1) out
   and checks that `DELETE` still fails.
4. Plus a progress handler that aborts after `SQL_TIMEOUT_S`: an infinite recursive CTE
   is a valid read-only SELECT.

**`fetch_url` and SSRF.** Once a model can fetch URLs, anyone who can put text in front
of it can try `http://169.254.169.254/` (cloud credentials) or `http://localhost:8080/admin`.
That includes a web page or document it reads (indirect prompt injection). An httpx
`request` event hook resolves the host before *every* request, redirects included, and
refuses non-global IPs. It does not stop DNS rebinding (resolve-then-connect race). For
a local tool that is an accepted limit, but it matters again in Phase 5, when this
becomes an API.

**Errors are written for the model.** Every fixable failure raises `ToolInputError` with
an instruction, for example: "No table 'sales'. Tables: robot_sales, support_tickets."
`ToolErrorMiddleware` returns that text as an error `ToolMessage`, so the model can
retry. Any *other* exception becomes "the tool failed (RuntimeError)", without its
message. Exception text can contain paths and secrets, and
`test_unknown_internal_errors_are_not_leaked` checks that it doesn't reach the model.

**Citations are labels, not numbers.** Phase 3's `[1]`, `[2]` style breaks in an agent:
two searches both have a passage "[1]". So every passage starts with the label to cite:
`[data/docs/safety_report_2026.pdf, page 1]`, `[data/code/jarvis_rag/loaders.py, lines 211-234]`
or `[table robot_sales]`. The code line range comes from the `line` metadata Phase 2
stored outside the chunk text.

## Getting a 4B model to use tools: what we measured

The first version had a careful system prompt with rules: "use tools", "cite", "say I
don't know". Asked "What was the total revenue per region?", JARVIS answered **"I don't
know based on the available sources."** without calling a single tool. Below is what
was tried, each step measured on the same 7 probe questions (the first model step only).

| Attempt | Result |
|---|---|
| No system prompt at all | Revenue → `list_tables` ✓, the rest answered from memory ("I don't have access…") |
| Rules prompt with the exact phrase "I don't know based on the available sources" | 4 of 7 replied with that phrase **before using any tool**. The abstain rule acted as a shortcut. |
| Same rules without the phrase | Still no tool calls. The model doesn't connect "safety incidents" with "your documents". |
| + **catalogue** of the knowledge base (titles, code folders, table names) | Safety and tickets questions now search and query ✓. Code questions are still answered from imagination. The CEO question is answered from memory. |
| + "when unsure, search" + a retry nudge when no tool was used | Barely helped: the model repeated its answer |
| + **auto-search**: the first `search_knowledge_base` call is made for it | Code ✓ (with line ranges), CEO → "not mentioned" ✓, revenue → SQL ✓ |
| + the rules repeated at the end of each search result (**point-of-use footer**) | Tickets: "describes the SQL" → calls `run_sql` ✓ |

What this shows about small models:

1. **Deciding *whether* to use a tool is the weak step.** Once the right context is in
   front of it, qwen3-4b reads, cites and abstains well. So we took the decision away:
   `auto_search` (a `before_agent` middleware) runs the search and records it as an
   ordinary tool call plus result. The model still chooses everything after that: SQL,
   another search, calculator, `fetch_url`. Cost: about 0.4 s and about 1,000 prompt
   tokens per turn, small talk included.
2. **The model can't use what it doesn't know exists.** The catalogue (`kb_prompt`, a
   `dynamic_prompt` middleware) lists what's indexed. It is rebuilt when `kb.count()`
   changes, so a document ingested mid-chat shows up on the next turn.
3. **Instructions work best where they're used.** A rule 1,500 tokens back in the
   system prompt loses to the last thing the model read. The search result now ends with
   "(… Numbers over a SQL table need run_sql: call it, don't describe it. Not in the
   results: say you don't know.)".
4. **Don't put the exact fallback phrase in the system prompt.** "If X, reply exactly
   'I don't know…'" made the phrase an easy shortcut. Now the model is told to say it
   *after* using the tools, and is reminded in the tool result.
5. **Check for known failure patterns in code.** `sql_follow_through` (a `wrap_model_call`
   middleware) catches two of them: answers that *talk about* a query ("you could run
   SELECT…"), and answers to "how many/total/average…" when the search found a table but
   `run_sql` never ran. It re-asks once with "call run_sql on <table> now". The nudge is
   sent to the model only. It is never saved to memory (tested).
6. **Persona leaks.** "In the style of Tony Stark's JARVIS" made it greet the user as
   "Tony". The fix was "Do not call the user by any name" (or `ADDRESS_USER_AS`).
7. Also in place: few, clearly named tools with short "use when…" docstrings; forgiving
   argument parsing (`source_type` aliases, `17%`, `^`, `×` in the calculator);
   `REASONING=false`; temperature 0.2; and a hard limit of 8 model calls per question.

Smaller things the eval caught on the way: the word "mean" in "what does E-217 *mean*"
triggered the SQL nudge as if it meant "average", so "mean" was removed from the
trigger words. And the grader's abstention regex didn't recognise "the documents do not
mention…", which is the exact wording we asked for.

## Results: `make eval-agent` (19 scenarios, deterministic grading)

Final configuration compared with the same agent with `AUTO_SEARCH=false` (the other
mitigations on). Raw logs are in `eval/results/agent_*.log`.

| | passed | answer right | right tools | cited | model tool calls/q | p50 latency |
|---|---|---|---|---|---|---|
| `AUTO_SEARCH=false` | **10/19** | 11/19 | 10/19 | 11/14 | 1.4 | 4.6 s |
| `AUTO_SEARCH=true`, before the SQL nudge | 16/19 | 16/19 | 17/19 | 14/14 | 0.6 | 10.7 s |
| **final** (+ `sql_follow_through`, absence rule) | **18/19** | 19/19 | 19/19 | 13/14 | 0.7 | 11.9 s |

**Without auto-search the model makes things up, with citations.** It said wheel motor
brushes need replacing every **15,000 hours** (the manual says 4,000) and cited
`kestrel_k2_maintenance.docx` for it. It said the K2 does not support 5G and cited a
"section 4.3" that doesn't exist. It reported "12 support tickets" with E-217 (actually
53) without running SQL. A citation proves nothing if the model writes it itself without
having read the source. All of those answers took under 2 s: the fastest answers were
the wrong ones.

**But it was perfect at SQL (7/7) without auto-search.** With search results already in
context, the model sometimes took a table's *schema* chunk as enough and described the
query instead of running it (a03), or skipped the count entirely (a13).
`sql_follow_through` fixed both. One mitigation can create the next failure, which is
why the eval is rerun after every change.

What still isn't right in the final run, read by hand rather than taken from the
grader:
- **a13**: the count is right (53, via `run_sql`), but the number is cited to
  `product_specs.md` instead of `[table support_tickets]`. The answer also adds "power
  to the wheel motors exceeds safe limits", which isn't in any document. That's a small
  invented explanation inside an otherwise grounded answer.
- **a07**: the model did `13434000 / 33547000 * 100` in its head and wrote **40.06 %**.
  The exact value is 40.045 %. That's within the eval's ±0.1, so it was graded a pass,
  but it is an arithmetic slip that `calculator` exists to prevent. A 4B model doesn't
  reach for the calculator when the division "looks easy".
- **a11**: "the function `ignored` in `iter_files`" is right (`ignored` is nested inside
  `iter_files`), but a reader might want `iter_files` named first.
- Formatting quirks: once `[cite table robot_sales]`, and citations repeated after every
  list item.
- "What's the capital of France?" (not in the eval) still gets "Paris", now followed by
  "this is not in the knowledge base". The rule against answering from memory is
  weaker than the model's urge to state the obvious.

Latency: about 6–17 s per question on the GTX 1650 (p50 11.9 s). Each tool round is
another model call over a ~2k-token prompt.

## Context budget on an 8k model

| Part of the prompt | tokens (measured: Ollama's `input_tokens`) |
|---|---|
| System prompt + catalogue + 6 tool schemas | 1,242 (small talk with `AUTO_SEARCH=false`) |
| Auto-search result (5 passages, clipped at 4,000 chars) | +660 here, ≤ ~1,000 |
| Conversation history (`HISTORY_MAX_TOKENS`) | ≤ 3,000 (approximate counter) |
| Room left for SQL results, more searches and the answer | ≈ 3,000 |

Every turn now starts at about 1,900 prompt tokens, even "Good evening.". On the GTX
1650 that is roughly 1–2 s of prompt processing before the first token, the price of
`AUTO_SEARCH`. llama.cpp's prompt cache (Phase 3's `LLAMA_ARG_CACHE_RAM`) reuses the
unchanged prefix: the system prompt and tool schemas.

`history_window` trims only what is **sent** (`request.override(messages=...)`); the
checkpoint keeps the whole thread. Trimming always starts at a user message. A
`ToolMessage` whose `AIMessage` tool call was cut off would be an orphan, and
chat templates reject it or the model gets confused. If the current turn alone is over
budget, it is sent anyway (`test_fit_history_keeps_an_oversized_current_turn`).
`SummarizationMiddleware` exists, but it costs an extra LLM call each time it fires,
and on this GPU that is several seconds. Phase 11 can revisit it.

## Streaming

`stream_events()` flattens LangGraph's `stream_mode=["messages", "updates"]` into one
event stream that the CLI uses now and Phase 5's SSE endpoint will reuse:

```
{"type": "tool_call",   "name": "search_knowledge_base", "args": {...}, "id": "auto-1a2b3c4d"}
{"type": "tool_result", "name": "search_knowledge_base", "content": "[...]", "error": false}
{"type": "token",       "text": "There are 13"}            ← "messages" mode: LLM chunks from the model node
{"type": "token",       "text": " open support tickets."}
{"type": "answer",      "text": "There are 13 open support tickets. [table support_tickets]"}
```

`messages` mode yields token chunks as the LLM produces them (LangGraph hooks the
model's callbacks, so `ChatOllama` streams with no extra code). `updates` mode yields
each node's finished output, and that is where complete tool calls and tool results come
from.

## Tests (52 new, 158 total, no Ollama)

- `tests/test_tools.py` (38):
  - SQL: rejected statements, the authorizer holding when the regex is bypassed, the
    timeout, the row cap.
  - Calculator: cases it accepts, and attacks it rejects (`__import__`, dunder access,
    `10**100000`).
  - Citations: label formats; search filters and aliases.
  - `ingest_text`: only exists behind its flag.
  - Web: the SSRF guard on 127.0.0.1, 169.254.169.254, 10.x and ::1; `fetch_url` against
    an `httpx.MockTransport` site (fencing, robots.txt, `file://`, 404).
- `tests/test_agent.py` (14): a `ScriptedModel` replays tool calls and records every
  prompt it was shown. The tests cover:
  - the persona and catalogue in the prompt;
  - auto-search happening before the first model call;
  - the SQL loop;
  - tool errors returned to the model;
  - internal errors not leaked;
  - memory kept per thread;
  - **SQLite memory surviving a rebuild**;
  - the model-call limit stopping a looping model;
  - trimming that leaves the checkpoint intact;
  - the SQL nudge, and that it isn't stored;
  - the catalogue refreshing after an ingest.

## Try it

```bash
make chat
you: How many safety incidents were there in 2026, and what was the main cause?
you: Which product has more open tickets? Use SQL.
you: /history
you: What is 15% of the APAC revenue?          # SQL, then calculator
you: Summarise https://en.wikipedia.org/wiki/Okapi_BM25 in two sentences.
you: Who founded Halcyon Robotics?             # should not know
.venv/bin/python eval/run_agent_eval.py --set history_max_tokens=500 --only a15   # does memory survive a tiny window?
ALLOW_INGEST_TOOL=true make chat   →  "Remember that the Lisbon office closes at 18:00 on Fridays."
```

## Quiz

1. Why is `auto_search` a middleware that *records* a tool call in the history, rather
   than pasting the search results into the system prompt?
2. `check_select` already rejects anything that isn't a SELECT. Why also open the
   database read-only *and* install an authorizer? Give a query that passes
   `check_select` but would still be dangerous without the progress handler.
3. Conversation memory lives in a different SQLite file from the CSV tables. What could
   go wrong if they shared `storage/jarvis.db`?

<details><summary>Answers</summary>

1. Three reasons. (a) The model sees results in the same format as when it calls the
   tool itself, so the citation labels, the footer and the "search again" rule all work
   the same way. (b) It becomes part of the saved history, so a follow-up such as "and
   the Wren W1?" can still see what was retrieved last turn (within the trim window).
   (c) The stream and the eval see an ordinary `tool_call` event (id `auto-…`), so the
   UI shows *what* was searched, and the eval can tell automatic calls from the model's
   own choices. A system prompt is cached and shared by every turn; per-question
   evidence doesn't belong there.
2. Each lock fails in a different way. The regex can be fooled: a missed comment
   syntax, a future SQLite feature, or a bug in our stripping. `mode=ro` stops writes to
   the file, but not `ATTACH` of another file or expensive `PRAGMA`s. The authorizer
   sees SQLite's own parse, so it can't be fooled by how the text is written. Several
   independent locks mean one bug isn't a breach. A dangerous query that is pure SELECT:
   `WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c`.
   It never finishes, and in Phase 5 it would hold an API worker forever.
3. `run_sql` would be able to `SELECT` from the checkpoint tables. A prompt-injected
   document could then make the model read and repeat other threads' conversations,
   which in Phase 5 means other users' chats. A large `checkpoints` table would also
   appear in `list_tables` and clutter the catalogue. Keeping them apart is least
   privilege: the SQL tool can reach only the data it exists for.

</details>
