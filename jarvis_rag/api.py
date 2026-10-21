"""HTTP API for JARVIS.

    make api                                  # uvicorn on API_HOST:API_PORT (default 127.0.0.1:8000)
    open http://127.0.0.1:8000/docs           # interactive OpenAPI docs ("Authorize" takes the key)

    GET    /health                 liveness + Ollama reachability (no key needed)
    POST   /chat                   {"message", "thread_id"?} -> answer, tool calls, checked citations
    POST   /chat/stream            same, as Server-Sent Events: tool_call, tool_result, token, answer
    POST   /ingest/file            multipart upload (pdf, docx, md/txt, code, csv)
    POST   /ingest/url             {"url", "crawl_depth"?}
    GET    /threads/{thread_id}    a conversation's messages (to redraw it after a page reload)
    GET    /sources                what is indexed
    DELETE /sources/{source}       remove a source (and its uploaded file / SQL table)

Every route except /health requires the X-API-Key header when API_KEY is set.
Without API_KEY the server refuses to bind to anything but localhost.
"""

import ipaddress
import json
import logging
import re
import secrets
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi import Path as FastApiPath
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from jarvis_rag import loaders
from jarvis_rag.agent import build_agent, check_citations, stream_events
from jarvis_rag.config import Settings, get_settings
from jarvis_rag.ingest import collect, ingest_documents
from jarvis_rag.llm import model_name
from jarvis_rag.tools import make_web_loader
from jarvis_rag.vectorstore import KnowledgeBase

log = logging.getLogger(__name__)

UPLOAD_EXTS = {".pdf", ".docx", ".csv", *loaders.TEXT_EXTS, *loaders.CODE_LANGS}
THREAD_ID = r"^[A-Za-z0-9_.:-]{1,64}$"


# --------------------------------------------------------------------------- request / response models

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, description="The user's question")
    thread_id: str | None = Field(default=None, pattern=THREAD_ID,
                                  description="Conversation id; omit to start a new one (returned in the reply)")


class ToolCall(BaseModel):
    name: str
    args: dict


class Citation(BaseModel):
    label: str
    verified: bool = Field(description="a tool really returned this source during this turn")


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    tool_calls: list[ToolCall]
    citations: list[Citation]


class ThreadMessage(BaseModel):
    role: str = Field(description="user | assistant")
    content: str
    tools: list[str] = Field(default=[], description="tools called before this assistant message")


class UrlRequest(BaseModel):
    url: str = Field(pattern=r"^https?://", max_length=2000)
    crawl_depth: int = Field(default=0, ge=0)


class IngestedSource(BaseModel):
    source: str
    chunks: int
    added: int
    removed: int
    unchanged: int


class IngestResponse(BaseModel):
    sources: list[IngestedSource]
    skipped: list[dict]


class SourceInfo(BaseModel):
    source: str
    type: str | None
    chunks: int
    ingested_at: str


class DeleteResponse(BaseModel):
    source: str
    chunks_deleted: int
    file_deleted: bool
    table_dropped: str | None


# --------------------------------------------------------------------------- helpers

def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def classify_failure(exc: Exception, provider: str) -> tuple[int, str]:
    """Turn a model/provider failure into a status and a message a user can act on,
    without echoing the exception text (which can carry URLs, keys, paths). Matched
    by name and status code, because every SDK has its own AuthenticationError class."""
    name = type(exc).__name__
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if "Authentication" in name or "PermissionDenied" in name or code in (401, 403):
        return 502, f"the LLM provider ({provider}) rejected the credentials: check its API key"
    if "RateLimit" in name or "ResourceExhausted" in name or code == 429:
        return 503, f"the LLM provider ({provider}) is rate-limiting requests: retry shortly, or set LLM_FALLBACK"
    if "Connect" in name or "Timeout" in name:
        where = "is Ollama running?" if provider == "ollama" else f"can this server reach {provider}?"
        return 503, f"a model service is unreachable ({name}): {where}"
    return 500, f"internal error ({name})"


def safe_filename(name: str | None) -> str:
    """Browser-supplied names can be "../../.bashrc" or "C:\\x\\y.pdf": keep the last
    path component, then only boring characters."""
    base = re.split(r"[\\/]", name or "")[-1]
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
    if not base or "." not in base:
        raise HTTPException(400, "file name must have an extension")
    return base[:120]


def save_upload(upload: UploadFile, dest: Path, max_bytes: int) -> None:
    """Copy in 1 MB pieces and stop at the limit, rather than trusting Content-Length."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    written = 0
    try:
        with tmp.open("wb") as f:
            while chunk := upload.file.read(1 << 20):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(413, f"file larger than {max_bytes // 1_000_000} MB")
                f.write(chunk)
        tmp.replace(dest)  # atomic: a half-written upload never replaces a good file
    finally:
        tmp.unlink(missing_ok=True)


def ingest_response(results, skipped) -> IngestResponse:
    return IngestResponse(
        sources=[IngestedSource(source=r.source, chunks=r.chunks, added=r.added, removed=r.removed,
                                unchanged=r.unchanged) for r in results],
        skipped=[{"what": w, "why": y} for w, y in skipped])


class BodySizeLimit:
    """Pure ASGI middleware: refuse request bodies over `max_bytes` *before* Starlette
    spools them to a temp file (Phase 5's open gap). Checks Content-Length up front,
    and counts the bytes of chunked uploads that don't declare one.

    For the counted case we can't raise: FastAPI's body parser would catch it and
    answer 400. So we answer 413 ourselves, tell the app the client disconnected,
    and drop whatever the app tries to send afterwards."""

    def __init__(self, app, max_bytes: int):
        self.app, self.max_bytes = app, max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        length = dict(scope["headers"]).get(b"content-length")
        if length and length.isdigit() and int(length) > self.max_bytes:
            return await self._reject(send)
        received, started, rejected = 0, False, False

        async def counting_receive():
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    rejected = True
                    if not started:
                        await self._reject(send)
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal started
            if rejected:
                return  # our 413 already went out
            started = started or message["type"] == "http.response.start"
            await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not rejected:
                raise  # a real error; a ClientDisconnect after our 413 is expected

    async def _reject(self, send):
        body = json.dumps({"detail": f"request body larger than {self.max_bytes // 1_000_000} MB"}).encode()
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})


# --------------------------------------------------------------------------- app

def create_app(settings: Settings | None = None, kb: KnowledgeBase | None = None, agent=None,
               web_factory=None) -> FastAPI:
    """App factory (`uvicorn --factory jarvis_rag.api:create_app`). Tests pass a fake
    agent, a knowledge base with fake embeddings, and a fake web loader."""
    s = settings or get_settings()
    if s.api_key is None and not is_loopback(s.api_host):
        raise RuntimeError(f"API_HOST={s.api_host} exposes the API to the network: set API_KEY in .env first.")
    if s.api_key is None:
        log.warning("API_KEY is not set: anyone who can reach %s:%s can use the API", s.api_host, s.api_port)
    kb = kb or KnowledgeBase(s)
    agent = agent or build_agent(s, kb=kb)
    web_factory = web_factory or (lambda: make_web_loader(s, max_pages=s.url_max_pages))
    write_lock = threading.Lock()  # one ingest/delete at a time; chats keep reading meanwhile

    app = FastAPI(title=f"{s.agent_name} API", version="0.5.0",
                  description="Local RAG agent: chat with your documents, code and tables.")
    # The largest legitimate body is one upload plus multipart framing.
    app.add_middleware(BodySizeLimit, max_bytes=int(s.max_file_mb * 1e6) + 1_000_000)
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST", "DELETE"],
                       allow_headers=["X-API-Key", "Content-Type"], allow_credentials=False)

    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def require_key(key: Annotated[str | None, Security(api_key_header)]) -> None:
        expected = s.api_key.get_secret_value() if s.api_key else None
        # compare_digest: time doesn't depend on how many leading characters match
        if expected and not (key and secrets.compare_digest(key.encode(), expected.encode())):
            raise HTTPException(401, "missing or wrong X-API-Key", headers={"WWW-Authenticate": "APIKey"})

    auth = [Depends(require_key)]

    @app.exception_handler(Exception)
    def internal_error(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error on %s %s", request.method, request.url.path)
        status, detail = classify_failure(exc, s.llm_provider)
        return JSONResponse({"detail": detail}, status_code=status)

    def writable() -> None:
        if s.kb_read_only:
            raise HTTPException(403, "the knowledge base is read-only on this deployment (KB_READ_ONLY=true)")

    def check_message(req: ChatRequest) -> str:
        if len(req.message) > s.max_message_chars:
            raise HTTPException(422, f"message longer than {s.max_message_chars} characters")
        return req.thread_id or uuid.uuid4().hex

    # ---- health
    @app.get("/health")
    def health() -> JSONResponse:
        ollama = None
        if "ollama" in (s.llm_provider, s.embed_provider):
            try:
                ollama = httpx.get(f"{s.ollama_base_url}/api/version", timeout=2).status_code == 200
            except httpx.HTTPError:
                ollama = False
        ok = ollama is not False
        return JSONResponse({"status": "ok" if ok else "degraded", "ollama": ollama, "provider": s.llm_provider,
                             "model": model_name(s), "read_only": s.kb_read_only},
                            status_code=200 if ok else 503)

    # ---- chat
    @app.post("/chat", response_model=ChatResponse, dependencies=auth)
    def chat(req: ChatRequest) -> ChatResponse:
        thread_id = check_message(req)
        calls, outputs, answer = [], [], ""
        for ev in stream_events(agent, req.message, thread_id):
            if ev["type"] == "tool_call":
                calls.append(ToolCall(name=ev["name"], args=ev["args"]))
            elif ev["type"] == "tool_result":
                outputs.append(ev["content"])
            elif ev["type"] == "answer":
                answer = ev["text"]
        return ChatResponse(answer=answer, thread_id=thread_id, tool_calls=calls,
                            citations=check_citations(answer, outputs))

    @app.post("/chat/stream", response_class=EventSourceResponse, dependencies=auth)
    def chat_stream(req: ChatRequest):
        """SSE events: `start` {thread_id}, `tool_call`, `tool_result`, `token` {text},
        `discard` {reason} (drop tokens since the last tool call), `answer` {text, thread_id,
        citations}, or `error` {message}."""
        thread_id = check_message(req)
        yield ServerSentEvent(event="start", data={"thread_id": thread_id})
        outputs = []
        try:
            for ev in stream_events(agent, req.message, thread_id):
                kind = ev.pop("type")
                if kind == "tool_result":
                    outputs.append(ev["content"])
                    ev["content"] = ev["content"][:500]  # the UI needs a preview, not 4,000 chars
                if kind == "answer":
                    ev |= {"thread_id": thread_id, "citations": check_citations(ev["text"], outputs)}
                yield ServerSentEvent(event=kind, data=ev)
        except Exception as e:  # headers are already sent: report in-band, don't drop the stream
            log.exception("chat stream failed")
            yield ServerSentEvent(event="error", data={"message": classify_failure(e, s.llm_provider)[1]})

    @app.get("/threads/{thread_id}", response_model=list[ThreadMessage], dependencies=auth)
    def thread(thread_id: Annotated[str, FastApiPath(pattern=THREAD_ID)]):
        """A conversation's user/assistant messages (tool traffic summarised as tool names),
        so a UI can redraw it after a reload. Unknown id = empty list."""
        messages = agent.get_state({"configurable": {"thread_id": thread_id}}).values.get("messages", [])
        out, tools = [], []
        for m in messages:
            if isinstance(m, HumanMessage):
                out.append(ThreadMessage(role="user", content=m.text))
                tools = []
            elif isinstance(m, AIMessage):
                tools += [c["name"] for c in m.tool_calls]
                if not m.tool_calls and m.text:
                    out.append(ThreadMessage(role="assistant", content=m.text, tools=tools))
                    tools = []
        return out

    # ---- ingestion
    @app.post("/ingest/file", response_model=IngestResponse, dependencies=[*auth, Depends(writable)])
    def ingest_file(file: Annotated[UploadFile, File(description="pdf, docx, md, txt, code or csv")]):
        name = safe_filename(file.filename)
        if Path(name).suffix.lower() not in UPLOAD_EXTS:
            raise HTTPException(415, f"unsupported file type; allowed: {', '.join(sorted(UPLOAD_EXTS))}")
        s.upload_dir.mkdir(parents=True, exist_ok=True)
        dest = s.upload_dir / name
        with write_lock:
            save_upload(file, dest, int(s.max_file_mb * 1e6))
            is_csv = dest.suffix.lower() == ".csv"
            docs, skipped = collect([] if is_csv else [dest], [], [dest] if is_csv else [], 0, s)
            if not docs:
                dest.unlink(missing_ok=True)
                raise HTTPException(422, f"nothing could be read from {name}: "
                                         + "; ".join(why for _, why in skipped))
            return ingest_response(ingest_documents(docs, kb, s), skipped)

    @app.post("/ingest/url", response_model=IngestResponse, dependencies=[*auth, Depends(writable)])
    def ingest_url(req: UrlRequest):
        if req.crawl_depth > s.max_crawl_depth:
            raise HTTPException(422, f"crawl_depth must be <= {s.max_crawl_depth}")
        with write_lock:
            docs, skipped = collect([], [req.url], [], req.crawl_depth, s, web=web_factory())
            if not docs:
                raise HTTPException(422, "nothing could be read: " + "; ".join(f"{w}: {y}" for w, y in skipped))
            return ingest_response(ingest_documents(docs, kb, s), skipped)

    # ---- sources
    @app.get("/sources", response_model=list[SourceInfo], dependencies=auth)
    def sources():
        return [SourceInfo(source=src, **info) for src, info in kb.sources().items()]

    @app.delete("/sources/{source:path}", response_model=DeleteResponse, dependencies=[*auth, Depends(writable)])
    def delete_source(source: str):
        """`source` exactly as /sources lists it; URL-encode it (a URL source contains "://")."""
        with write_lock:
            metas = kb.collection.get(where={"source": source}, include=["metadatas"])["metadatas"]
            if not metas:
                raise HTTPException(404, f"no source {source!r}")
            deleted = kb.delete_source(source)
            table = metas[0].get("table") if metas[0].get("type") == "table" else None
            if table:  # otherwise run_sql could still query a table the knowledge base no longer lists
                with sqlite3.connect(s.sqlite_path) as conn:
                    conn.execute(f'DROP TABLE IF EXISTS "{table}"')
            path = Path(source).resolve()
            uploaded = path.is_relative_to(s.upload_dir.resolve()) and path.is_file()
            if uploaded:  # only files we stored ourselves; never anything else on disk
                path.unlink()
            return DeleteResponse(source=source, chunks_deleted=deleted, file_deleted=uploaded, table_dropped=table)

    return app


def main() -> None:
    import uvicorn

    s = get_settings()
    uvicorn.run("jarvis_rag.api:create_app", factory=True, host=s.api_host, port=s.api_port)


if __name__ == "__main__":
    main()
