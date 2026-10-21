# Phase 8: Docker

**Goal:** one image that runs JARVIS (API + UI) anywhere Docker runs, built
reproducibly, running with the least privilege that still works, and without
installing anything new on this machine. That is option 1 from the planning question:
**no NVIDIA Container Toolkit**. The GPU stays with the host's Ollama.

```bash
make docker-build        # ~2 min cold, ~2 s when only code changed
make docker-up           # http://127.0.0.1:8000/ui (API docs: /docs), host's Ollama on the GPU
make docker-logs / make docker-down
docker compose --profile cpu up -d jarvis-cpu   # fully self-contained: FastEmbed + hosted LLM
```

## What we built

| File | Job |
|---|---|
| `Dockerfile` | Two stages. uv installs the hash-locked deps into `/opt/venv`; the runtime stage gets only the venv + `jarvis_rag/`. Non-root, healthcheck, all state under `/app/storage`, `/app/data`, `/tmp`. |
| `docker-compose.yml` | `jarvis` (host network → host Ollama) and `jarvis-cpu` (bridge network, `cpu` profile), sharing one hardening block |
| `.dockerignore` | Keeps `.env`, `storage/`, `data/`, `.venv/`, `.git/` out of the build context |
| `jarvis_rag/api.py` | + `BodySizeLimit` ASGI middleware (closes Phase 5's upload gap); + `classify_failure`: provider auth/rate-limit/outage → 502/503 with an actionable message |
| `requirements.txt` | Now locked **with hashes** (`make lock` adds `--generate-hashes`): 4,105 sha256 entries, the same versions as before |
| `tests/test_docker.py` + 5 API tests | Guards on the security-relevant lines; body limits; failure mapping |

## Reaching the host's Ollama without changing it

Ollama listens on `127.0.0.1:11434` only (checked with `ss -ltn`). From a container on
Docker's default bridge network, `127.0.0.1` is the *container*, and the host is
`172.17.0.1`, where Ollama isn't listening. The common fix, `OLLAMA_HOST=0.0.0.0`,
means editing a system service *and* exposing an unauthenticated model server to the
LAN. We did neither:

```
network_mode: host        container shares the host's network namespace
  container → 127.0.0.1:11434  = the host's Ollama (GPU, already running)
  app listens 127.0.0.1:8000   = the host's loopback, not the LAN
```

That's Linux-only (on Docker Desktop, use `host.docker.internal`), and it gives up
network isolation for this one container. That's acceptable here: the app only binds
loopback and has nothing else to reach.

The second service, `jarvis-cpu`, is the "no Ollama at all" shape (laptop without GPU,
a VPS): bridge network, `127.0.0.1:8000:8000`, `EMBED_PROVIDER=fastembed` and a hosted
LLM. Phase 7's fingerprint result makes this cheap. FastEmbed's nomic produces the
*same* vectors, so the container **reads the index the host built with Ollama**
(checked: correct top hit, `product_specs.md`, similarity 0.779). Inside the container
the app must listen on `0.0.0.0`, and it has no way of knowing the port is published on
loopback only. So our own guards fire, and it refuses to start without `API_KEY` and
`UI_PASSWORD` (both tested). Compose's `${API_KEY:?}` would have been a neater way to
enforce that, but Compose evaluates it for *every* service in the file, which blocked
`docker compose up jarvis` too.

## Reproducible, verifiable builds

| Layer | Pinned how | Why |
|---|---|---|
| Base image | `python:3.12-slim-trixie@sha256:2259…` | The tag was rebuilt *the day we pulled it* (OS patches). A tag is a moving pointer; a digest is the content. To take patches, pull and bump the digest. That is a deliberate change. |
| Build tool | `ghcr.io/astral-sh/uv:0.12.24@sha256:3af4…` | Same uv as the host |
| Python deps | `uv pip sync --require-hashes requirements.txt` | Every wheel must match the sha256 recorded at lock time. A swapped or tampered file on PyPI fails the build instead of shipping. |

We're applying two different policies on purpose. For *package versions* we wait two
weeks (Phases 5–7), so we're never the first to pull a bad release. For the *base OS
image* we want the newest patched build, since its purpose is to get security fixes.
The digest pin keeps both reproducible.

## Least privilege, verified

```
$ docker exec … id                      uid=1000 gid=1000            (non-root; your uid, so ./storage files stay yours)
$ docker exec … touch /app/x            Read-only file system        (read_only: true)
$ docker exec … touch /app/storage/.w   ok                           (only the volumes and /tmp are writable)
$ docker exec … grep CapEff /proc/1/status   0000000000000000        (cap_drop: [ALL])
```

Plus `no-new-privileges` (setuid binaries can't escalate) and `init: true` (tini as PID
1 forwards SIGTERM to uvicorn and reaps zombies). A read-only root filesystem forced us
to list every place the app writes:

| Writer | Where it goes now |
|---|---|
| Chroma, SQLite tables, conversation memory | `/app/storage` (bind mount) |
| Uploads | `/app/data/uploads` (bind mount) |
| FastEmbed / reranker / Hugging Face downloads | `/app/storage/models` (`HF_HOME` too), so they persist across containers |
| Gradio upload staging | `GRADIO_TEMP_DIR=/tmp/gradio` (tmpfs, 1 GB) |
| Anything that writes to `$HOME` | `HOME=/tmp` |
| `.pyc` files | `PYTHONDONTWRITEBYTECODE=1` (bytecode was compiled at build time: `UV_COMPILE_BYTECODE=1`) |

**`HEALTHCHECK`** calls `/health` with the standard library (slim images have no
curl). `/health` returns 503 when Ollama is unreachable, so `unhealthy` means "can't
answer questions", not just "process is alive". It became healthy about 15 s after
start.

## Closing Phase 5's upload gap: `BodySizeLimit`

Phase 5 noted that Starlette spools a whole multipart body to a temp file *before* any
handler runs. Inside a container whose `/tmp` is a 1 GB tmpfs, that matters even more.
Instead of adding a reverse proxy just for that, a 40-line pure-ASGI middleware sits in
front of the app:

1. A `Content-Length` over the limit gets a **413 immediately**, before a single body
   byte is read. Live: a 30 MB upload got `413`.
2. Chunked bodies declare no length, so `receive()` is wrapped to count bytes.

The first version of (2) raised an exception when the count passed the limit, and the
test failed with **400** instead of 413. FastAPI's body parser catches exceptions from
`receive()` and reports "error parsing the body". So the middleware now sends the 413
itself, tells the app the client disconnected, and discards anything the app tries to
send afterwards. The test was worth writing: the obvious implementation looked right
and wasn't.

## Errors a user can act on

The `jarvis-cpu` test with a deliberately fake Groq key first produced
`500 {"detail": "internal error (AuthenticationError)"}`. That was safe, but useless.
`classify_failure` maps failures by exception *name and status code*, because every SDK
(groq, openai, google-genai, httpx) has its own classes:

| Failure | Response |
|---|---|
| provider 401/403 (`AuthenticationError`, `PermissionDenied…`) | **502** "the LLM provider (groq) rejected the credentials: check its API key" |
| provider 429 (`RateLimitError`, `ResourceExhausted`) | **503** "… is rate-limiting requests: retry shortly, or set LLM_FALLBACK" |
| connection/timeout (`ConnectError`, `…Timeout…`) | **503** "a model service is unreachable: is Ollama running?" |
| anything else | 500 "internal error (TypeName)" |

The same message goes into the SSE `error` event. The exception's own text is never
echoed: provider errors can contain URLs and key prefixes (tested with
`"Invalid API Key gsk_secret"`).

## End-to-end runs

| Check | Result |
|---|---|
| `make docker-build` (cold) | 1 min 55 s; image 1.3 GB on disk, 279 MB compressed |
| rebuild after a code change | 1.5 s (dependency layers cached) |
| `make docker-up` → healthy | ~15 s |
| `/chat` "open tickets + E-217?" via the host's GPU Ollama | search → describe_table → run_sql, 13 tickets, citations verified, 32 s (first call: model warm-up) |
| restart the container, continue thread `docker-1` | 2 messages restored; "how many of those are Wren W1?" → 6 (correct) |
| `docker compose run … ingest` | works; files written as uid 1000 |
| `/ui` with `UI_PASSWORD` set | login page; `/ui/config` → 401; `gradio_client` without login → refused, with login → works |
| `jarvis-cpu` without `API_KEY` / without `UI_PASSWORD` | refuses to start, one-line reason each |
| `jarvis-cpu` with a fake Groq key | healthy (`"ollama": null`), sources visible, FastEmbed retrieval correct, `/chat` → 502 with the message above |
| container memory, idle after a chat | 375 MB |

Not done: a real hosted-LLM answer from `jarvis-cpu` (no provider key). Every other part
of that path was exercised.

### Where the 1.3 GB goes

`/opt/venv` is 843 MB. The biggest items: `kubernetes` 119 MB, gradio 83, pandas 72,
onnxruntime 68, chromadb's Rust bindings 57, numpy 70. `kubernetes` comes in through
`chromadb`, which imports it only for its distributed (cluster) mode. Deleting it would
save about 10%, but stripping files out of an installed package to shave an image is the
kind of hack that breaks on the next upgrade. Phase 11 can look at splitting
dev-only dependencies (`pytest`) out instead.

## A caveat worth repeating

`./storage` is shared between the host and the container. Chroma's on-disk format is
not designed for two processes writing at once. So run `make app` **or** `make
docker-up`, not both. While the server runs, ingest through the API
(`POST /ingest/file`, or the UI), not `make docker-ingest`.

## Tests (+9, 219 total)

- `tests/test_docker.py` (4): secrets and data excluded from the build context; base
  images pinned by digest; `--require-hashes` used and the lock has hashes; non-root
  `USER`, `read_only`, `cap_drop`, `no-new-privileges`, `init`; only `127.0.0.1:`
  ports published.
- `tests/test_api.py` (+5): the body limit for declared and chunked (`Transfer-Encoding`)
  bodies; four provider-failure mappings, for both `/chat` and `/chat/stream`, that
  never leak the exception text.

## Try it

```bash
make docker-build && make docker-up
open http://127.0.0.1:8000/ui
docker compose exec jarvis python -m jarvis_rag.check_env      # the doctor, from inside the container
docker inspect --format '{{json .State.Health}}' $(docker compose ps -q jarvis) | python -m json.tool
sudo systemctl stop ollama; sleep 40; docker compose ps            # → unhealthy; /health → 503
sudo systemctl start ollama
make docker-down
```

## Quiz

1. Why does `jarvis` use `network_mode: host` instead of publishing a port, and what
   would you change on macOS?
2. The Dockerfile pins the base image by digest, but we deliberately *want* the newest
   patched base image. Isn't that a contradiction?
3. The `jarvis-cpu` service publishes `127.0.0.1:8000:8000`, so only this machine can
   reach it. Why does the app still insist on `API_KEY` and `UI_PASSWORD`?

<details><summary>Answers</summary>

1. The host's Ollama only listens on 127.0.0.1. On a bridge network the container's
   127.0.0.1 is itself, so it can't reach Ollama without reconfiguring Ollama to
   listen on a reachable interface, which also exposes it to the LAN. Host networking
   shares the host's loopback, so both directions work and nothing new is exposed. On
   macOS, Docker runs inside a VM and host networking doesn't reach the Mac. Use the
   bridge network with `OLLAMA_BASE_URL=http://host.docker.internal:11434`.
2. No: pinning and updating are separate actions. The digest makes every build of a
   given commit produce the same image, so you can rebuild exactly what you tested and
   roll back. Taking patches is then a deliberate one-line change (pull the tag, copy
   the new digest), ideally automated by a bot that opens a pull request. Unpinned tags
   give you patches *and* surprise changes nobody reviewed.
3. The app decides at startup from what it can see: it is binding `0.0.0.0`. Port
   publishing happens outside the container, in Docker's configuration, which the app
   can't inspect. Someone could change it to `8000:8000` (all interfaces) later and get
   no warning at all. A check that relies on configuration elsewhere being right is a
   check that eventually fails silently. Requiring credentials whenever the app *might*
   be exposed keeps it safe regardless.

</details>
