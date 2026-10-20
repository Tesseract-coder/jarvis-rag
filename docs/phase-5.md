# Phase 5: HTTP API

**Goal:** put the Phase 4 agent behind an HTTP API that a browser, a script or the
Gradio UI (Phase 6) can call. It should be safe to leave running: authenticated,
CORS-restricted, careful with uploads, and honest about what it cited.

```bash
echo "API_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(32))')" >> .env
make api                                       # http://127.0.0.1:8000/docs for interactive docs
export JARVIS_API_KEY=...                      # same value
python examples/client.py "What does E-217 mean?"          # streams tokens and tool calls
node examples/client.js "Which robot has the longer battery life?"
python examples/client.py --upload report.pdf --sources
```

## What we built

| File | Job |
|---|---|
| `jarvis_rag/api.py` | `create_app()` factory: 7 routes, API-key dependency, CORS, upload handling, SSE, error handling |
| `jarvis_rag/agent.py` | + `check_citations()`: is each cited label one a tool actually returned this turn? |
| `jarvis_rag/tools.py` | `BlockedURL` (an httpx error, so a crawl skips the page), `make_web_loader(max_pages)`. `run_sql`/`describe_table` output now starts with `[table x]`. |
| `examples/client.py`, `examples/client.js` | Streaming clients with no SDK: httpx, and plain `fetch` (Node 18+ or browser) |
| `tests/test_api.py`, `tests/fakes.py` | 27 tests with `TestClient`; `ScriptedModel` moved to a shared module |

New settings: `API_HOST=127.0.0.1`, `API_PORT=8000`, `API_KEY`, `CORS_ORIGINS`
(comma-separated), `MAX_MESSAGE_CHARS=4000`, `MAX_CRAWL_DEPTH=2`.

New dependencies, checked before adding: `fastapi 0.142.2`, `starlette 1.7.0`,
`python-multipart 0.0.32` (`uvicorn 0.54.0` was already installed through chromadb).
On the day of locking, PyPI had **six FastAPI releases published the same day**
(0.151 → 0.156), and Starlette 1.9.0 was two days old. Pinning
`fastapi>=0.142,<0.142.3` and `starlette>=1.7,<1.8` picks releases at least two weeks
old. A burst of fresh releases might be fine, or might be a compromised or broken
publish; waiting costs nothing. This version of FastAPI ships SSE support
(`fastapi.sse.EventSourceResponse`, `ServerSentEvent`) with keep-alive pings, so no
`sse-starlette` is needed.

## The routes

```
GET    /health                 {"status": "ok"|"degraded", "ollama": bool, "model"}   no key; 503 if Ollama is down
POST   /chat                   {"message", "thread_id"?} → {"answer", "thread_id", "tool_calls", "citations"}
POST   /chat/stream            same request; text/event-stream
POST   /ingest/file            multipart "file" → {"sources": [{source, chunks, added, removed, unchanged}], "skipped"}
POST   /ingest/url             {"url", "crawl_depth": 0..2} → same
GET    /sources                [{source, type, chunks, ingested_at}]
DELETE /sources/{source:path}  {source, chunks_deleted, file_deleted, table_dropped}
```

A real response:

```json
{
  "answer": "There are 13 open support tickets. [table support_tickets]",
  "thread_id": "33acfebb6f7a44f0ba7f78b5ec21d97c",
  "tool_calls": [
    {"name": "search_knowledge_base", "args": {"query": "How many support tickets are still open?"}},
    {"name": "run_sql", "args": {"query": "SELECT COUNT(*) FROM support_tickets WHERE status = 'open'"}}
  ],
  "citations": [{"label": "table support_tickets", "verified": true}]
}
```

**Citations are checked, not trusted.** In Phase 4, without auto-search, the model cited
`kestrel_k2_maintenance.docx` for a number it made up, and cited a "section 4.3" that
doesn't exist. Brackets in the answer prove nothing. `check_citations` lists every
source-like `[label]` in the answer and marks it `verified` only if a tool returned that
label during this turn. That is why every tool's output now *starts* with its label,
`run_sql` included. A citation without the page still counts (`[r.pdf]` for
`[r.pdf, page 2]`). A UI can show unverified citations in red; both example clients
print them. What it does **not** check is whether the cited passage actually *supports*
the claim. That needs a judge model (Phase 3 showed a 4B judge isn't reliable).

**`thread_id`** is optional. Leave it out and the server makes a random 128-bit id and
returns it; send it back to continue. Ids are limited to `[A-Za-z0-9_.:-]{1,64}`.
There is one API key, so anyone holding it can read any thread whose id they know.
That makes this a single-user (or single-team) server. Multi-user isolation would mean
per-user keys, with the user id prefixed to `thread_id` on the server. The client must
not be able to choose that prefix.

## Server-Sent Events

SSE is the simplest streaming protocol that works through proxies: one long HTTP
response, `text/event-stream`, made of text blocks:

```
event: start
data: {"thread_id": "api-demo"}

event: tool_call
data: {"name": "search_knowledge_base", "args": {"query": "What is the battery life of the Wren W1?"}, "id": "auto-5485813c"}

event: tool_result
data: {"name": "search_knowledge_base", "content": "[data/docs/product_specs.md]\n[Halcyon Robotics — Product…", "error": false}

event: token
data: {"text": " battery"}
        … one event per token …
event: answer
data: {"text": "The battery life of the Wren W1 is 9 hours … [data/docs/product_specs.md].", "thread_id": "api-demo", "citations": [{"label": "data/docs/product_specs.md", "verified": true}]}
```

- **Why POST?** The browser's `EventSource` only does GET, and the question doesn't
  belong in a URL (it ends up in logs and history). So both clients read the stream with
  `fetch` and parse it themselves (about 20 lines). They buffer, because a network chunk
  can end in the middle of an event.
- **Errors after the first byte** can't change the HTTP status (200 is already sent). So
  the route catches them and sends `event: error` with only the exception *type*
  (tested: "ollama exploded at /secret/path" goes out as `internal error (RuntimeError)`).
- **The generator is synchronous.** `stream_events` iterates a sync LangGraph stream, and
  FastAPI runs sync generators in a worker thread (`iterate_in_threadpool`), so the
  event loop is never blocked. It also sends keep-alive comments (`: ping`) during long
  silences, such as a slow first token, so proxies don't close the connection.
- `tool_result` content is cut to 500 characters in the stream. The UI needs a preview,
  while the model gets the full 4,000.

## Security decisions

| Threat | What the API does |
|---|---|
| Exposed to the LAN by accident | `create_app` **refuses to start** when `API_HOST` isn't loopback and `API_KEY` is empty |
| Guessing the key | `secrets.compare_digest`, so comparison time doesn't reveal how many leading characters matched. Use a 32-byte random key. |
| Malicious web page calling the API from your browser | CORS lists the allowed origins (`CORS_ORIGINS`); preflight from anything else gets 400. `allow_credentials=False`. Auth is a header, not a cookie, so a cross-site page can't make your browser send it. |
| Upload named `../../.bashrc` | `safe_filename` keeps the last path component and `[A-Za-z0-9._-]`. Files only go into `data/uploads/` |
| Huge upload | Copied in 1 MB pieces, aborted past `MAX_FILE_MB` (413). Written to `.part`, then atomically renamed, so a failed upload never replaces a good file and never leaves junk (tested) |
| Executable / unknown types | Allowlist of extensions (pdf, docx, csv, text, code) → 415. Unreadable content (binary `.py`, empty file) → 422, and the file is removed |
| `POST /ingest/url {"url": "http://169.254.169.254/..."}` (SSRF) | Same guarded loader as `fetch_url`: every request and redirect resolved, non-global IPs refused (tested with 127.0.0.1). Crawl depth ≤ 2, ≤ 25 pages |
| `DELETE /sources/../../etc/passwd` | Deletes only chunks with exactly that `source`. A file is removed only if it resolves *inside* `data/uploads/` (tested: deleting `specs.md` leaves the file alone) |
| Deleted CSV still queryable | Deleting a `table` source also `DROP`s the SQL table; otherwise `run_sql` could still read data the index no longer lists |
| Stack traces in responses | Global exception handler returns `internal error (TypeName)` and logs the trace server-side |
| Concurrent ingests corrupting the BM25 or index state | One `threading.Lock` around ingest and delete. Chats keep running and see the new data on their next retrieval (the BM25 index rebuilds when the ids change) |

Not covered yet, on purpose: rate limiting, TLS and request-size limits at the proxy
(Phase 8, Docker). Without a body-size limit in a proxy in front of it, Starlette spools a
whole multipart body to a temp file *before* our handler runs, so `MAX_FILE_MB` stops
it from being ingested but not from filling the temp disk first.

## One GPU, one queue

Three questions sent one after another, then the same three at the same time:

| | sequential | concurrent |
|---|---|---|
| Q1 (safety incidents) | 8.2 s | 9.0 s |
| Q2 (E-217) | 11.3 s | **25.8 s** |
| Q3 (open tickets) | 9.1 s | 15.3 s |
| wall time for all three | 28.6 s | 25.8 s |

The API accepts all three at once (each sync route runs in a worker thread), but Ollama
processes one generation at a time on this 4 GB card. The overlap saves almost nothing
in total (only the CPU work overlaps: retrieval and reranking), and the unlucky request
waits for the others. That's fine for a personal assistant. For a team you'd want
either a bigger GPU with `OLLAMA_NUM_PARALLEL` > 1 (each parallel slot needs its own KV
cache: another ~1 GB at 8k context), or a hosted provider (Phase 7).

## Tests (27 new, 185 total)

`tests/test_api.py` drives the real app through `TestClient`, using a `ScriptedModel`
agent, fake embeddings and an `httpx.MockTransport` web site. Covered:
- refusing to start on a network host without a key;
- 401 on every route but `/health`;
- `/health` returning 503 when Ollama is down;
- chat: tool calls, verified and unverified citations, thread memory, input validation;
- the SSE event order, and errors sent in-band without leaking internals;
- markdown upload (traversal name) → list → delete (file removed);
- CSV upload → table → delete (table dropped);
- upload rejections (415, 400, 413, 422) with nothing left on disk;
- delete never touching files outside `uploads/`;
- URL ingestion, plus private addresses refused;
- `safe_filename`;
- CORS preflight.

## Try it

```bash
make api   # then, in another terminal:
curl -s localhost:8000/health
curl -s localhost:8000/sources -H "X-API-Key: $JARVIS_API_KEY" | python -m json.tool
curl -sN localhost:8000/chat/stream -H "X-API-Key: $JARVIS_API_KEY" -H 'content-type: application/json' \
     -d '{"message": "Which product has more open tickets?"}'
python examples/client.py --upload some.pdf && python examples/client.py "Summarise some.pdf"
python examples/client.py --thread mine "My favourite robot is the Wren." && \
python examples/client.py --thread mine "Which robot do I like, and what is its battery life?"
curl -s -X DELETE "localhost:8000/sources/data/uploads/some.pdf" -H "X-API-Key: $JARVIS_API_KEY"
```

Open http://127.0.0.1:8000/docs: FastAPI generated the OpenAPI schema from the Pydantic
models, including the API-key scheme (the "Authorize" button) and the SSE route.

## Quiz

1. `/chat/stream` hits an exception halfway through an answer. Why can't it return a
   500, and what does the client see instead?
2. Why does CORS block a malicious page from calling the API in your browser, and why
   does CORS do nothing against `curl` or a Python script?
3. `check_citations` reports every citation as `verified: true`. Name two ways the
   answer can still be wrong or unsupported.

<details><summary>Answers</summary>

1. The status line and headers (`200`, `text/event-stream`) went out with the first
   event; HTTP can't take them back. So the route catches the exception and sends a
   final `event: error` with a safe message, then ends the stream. The client has to
   handle that event, and should also treat "stream ended without `answer`" as a failure
   (for example when the connection dropped).
2. CORS is enforced *by the browser*. It protects the user from pages that try to read
   responses from another origin using the user's own network position (your
   localhost) or credentials. The page at evil.example can't make the browser accept the
   response, and with a custom header (`X-API-Key`) the preflight fails before the real
   request is sent. `curl` isn't a browser and ignores CORS completely. Against scripts
   the protection is the API key, plus binding to 127.0.0.1.
3. (a) The label is real but the passage doesn't say what the answer claims. Phase 4's
   a13 added "power to the wheel motors exceeds safe limits" next to a verified
   citation. The checker only verifies provenance, not support. (b) The fact comes from
   a different source than the one cited (a13 again: a SQL count cited to
   `product_specs.md`, which a tool *did* return, so it was "verified"). (c) A claim with
   no citation at all passes, because there is nothing to check. (d) The source itself is
   wrong or out of date.

</details>
