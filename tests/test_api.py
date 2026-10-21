"""The HTTP API with FastAPI's TestClient: a scripted model, fake embeddings, a
mock web server. No Ollama, no network, no GPU."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fakes import ScriptedModel, call
from jarvis_rag import api as api_module
from jarvis_rag.agent import build_agent
from jarvis_rag.api import create_app, safe_filename
from jarvis_rag.config import Settings
from jarvis_rag.ingest import ingest_documents
from jarvis_rag.loaders import WebLoader
from jarvis_rag.vectorstore import KnowledgeBase

KEY = {"X-API-Key": "s3cret"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None, storage_dir=tmp_path / "st", collection_name="apit", memory_backend="memory",
                 retrieval_mode="hybrid", api_key="s3cret", max_file_mb=0.01)
    kb = KnowledgeBase(s, embeddings=DeterministicFakeEmbedding(size=32))
    (tmp_path / "specs.md").write_text("# Specs\nBattery life: 9 hours.")
    ingest_documents([Document("# Specs\n## W1\nBattery life: 9 hours.",
                               metadata={"source": "specs.md", "type": "text", "title": "Specs", "path": "specs.md"})],
                     kb, s)
    return s, kb


def client_for(env, *script, web=None, **overrides):
    s, kb = env
    s = s.model_copy(update=overrides) if overrides else s
    model = ScriptedModel(script=list(script), seen=[])
    agent = build_agent(s, model=model, kb=kb, checkpointer=InMemorySaver())
    return TestClient(create_app(s, kb=kb, agent=agent, web_factory=web)), model


def sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines() if not line.startswith(":"))
        if "data" in lines:
            events.append((lines.get("event", "message"), json.loads(lines["data"])))
    return events


# --------------------------------------------------------------------------- startup, auth, health

def test_refuses_network_bind_without_key(env):
    s, kb = env
    with pytest.raises(RuntimeError, match="API_KEY"):
        create_app(s.model_copy(update={"api_key": None, "api_host": "0.0.0.0"}), kb=kb, agent=object())
    create_app(s.model_copy(update={"api_key": None, "api_host": "127.0.0.1"}), kb=kb, agent=object())


def test_key_required_everywhere_but_health(env, monkeypatch):
    monkeypatch.setattr(api_module.httpx, "get", lambda *a, **k: httpx.Response(200))
    c, _ = client_for(env)
    assert c.get("/health").json() == {"status": "ok", "ollama": True, "provider": "ollama",
                                       "model": "qwen3:4b-instruct"}
    for method, path in [("get", "/sources"), ("post", "/chat"), ("delete", "/sources/x")]:
        assert getattr(c, method)(path).status_code == 401
        assert getattr(c, method)(path, headers={"X-API-Key": "wrong"}).status_code == 401
    assert c.get("/sources", headers=KEY).status_code == 200


def test_health_degraded_when_ollama_down(env, monkeypatch):
    def down(*a, **k):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(api_module.httpx, "get", down)
    r = client_for(env)[0].get("/health")
    assert r.status_code == 503 and r.json()["status"] == "degraded"


# --------------------------------------------------------------------------- chat

def test_chat_returns_answer_tools_and_checked_citations(env):
    c, _ = client_for(env, AIMessage("9 hours [specs.md]. Also [manual.pdf, page 4]."))
    body = c.post("/chat", json={"message": "W1 battery?"}, headers=KEY).json()
    assert body["answer"].startswith("9 hours") and len(body["thread_id"]) == 32
    assert body["tool_calls"][0]["name"] == "search_knowledge_base"
    assert body["citations"] == [{"label": "specs.md", "verified": True},
                                 {"label": "manual.pdf, page 4", "verified": False}]


def test_chat_thread_memory(env):
    c, model = client_for(env, AIMessage("Noted."), AIMessage("Rusty."), auto_search=False)
    c.post("/chat", json={"message": "My robot is Rusty.", "thread_id": "t-1"}, headers=KEY)
    c.post("/chat", json={"message": "Its name?", "thread_id": "t-1"}, headers=KEY)
    assert any("Rusty" in m.text for m in model.seen[1])


@pytest.mark.parametrize("body", [{"message": ""}, {"message": "x" * 5000}, {"message": "hi", "thread_id": "../x"},
                                  {"message": "hi", "thread_id": "a" * 65}, {}])
def test_chat_validation(env, body):
    assert client_for(env)[0].post("/chat", json=body, headers=KEY).status_code == 422


def test_chat_stream_events(env):
    c, _ = client_for(env, call("calculator", expression="2+2"), AIMessage("4 [specs.md]"))
    r = c.post("/chat/stream", json={"message": "2+2?", "thread_id": "s1"}, headers=KEY)
    assert r.headers["content-type"].startswith("text/event-stream")
    events = sse(r.text)
    assert [e for e, _ in events] == ["start", "tool_call", "tool_result", "tool_call", "tool_result", "answer"]
    assert events[0][1] == {"thread_id": "s1"}
    assert events[4][1]["content"] == "2+2 = 4"
    assert events[-1][1]["citations"] == [{"label": "specs.md", "verified": True}]


def test_chat_stream_reports_errors_in_band(env):
    class Broken:
        def stream(self, *a, **k):
            raise RuntimeError("ollama exploded at /secret/path")

    s, kb = env
    c = TestClient(create_app(s, kb=kb, agent=Broken()))
    events = sse(c.post("/chat/stream", json={"message": "hi"}, headers=KEY).text)
    assert events[-1] == ("error", {"message": "internal error (RuntimeError)"})


# --------------------------------------------------------------------------- ingestion

def test_upload_markdown_then_list_and_delete(env, tmp_path):
    c, _ = client_for(env)
    r = c.post("/ingest/file", headers=KEY, files={"file": ("../../etc/parking.md", b"# Parking\nBasement rack B2.")})
    assert r.status_code == 200, r.text
    assert r.json()["sources"][0] == {"source": "data/uploads/parking.md", "chunks": 1, "added": 1, "removed": 0,
                                      "unchanged": 0}
    assert (tmp_path / "data/uploads/parking.md").exists() and not (tmp_path / "etc").exists()
    assert "data/uploads/parking.md" in [x["source"] for x in c.get("/sources", headers=KEY).json()]
    d = c.delete("/sources/data/uploads/parking.md", headers=KEY).json()
    assert d == {"source": "data/uploads/parking.md", "chunks_deleted": 1, "file_deleted": True, "table_dropped": None}
    assert not (tmp_path / "data/uploads/parking.md").exists()


def test_upload_csv_becomes_a_table_and_delete_drops_it(env):
    s, _ = env
    c, _ = client_for(env)
    r = c.post("/ingest/file", headers=KEY, files={"file": ("Staff Laptops.csv", b"name,laptop\nAna,X1\nBo,M2\n")})
    assert r.json()["sources"][0]["source"] == "data/uploads/Staff_Laptops.csv"
    from jarvis_rag.tools import table_names
    assert "staff_laptops" in table_names(s.sqlite_path)
    d = c.delete("/sources/data/uploads/Staff_Laptops.csv", headers=KEY).json()
    assert d["table_dropped"] == "staff_laptops" and "staff_laptops" not in table_names(s.sqlite_path)


@pytest.mark.parametrize("name,content,status", [
    ("evil.exe", b"MZ", 415), ("noext", b"x", 400), ("big.md", b"x" * 20_000, 413), ("empty.md", b"", 422),
    ("binary.py", b"\x00\x01\x02" * 10, 422),
])
def test_upload_rejections(env, tmp_path, name, content, status):
    r = client_for(env)[0].post("/ingest/file", headers=KEY, files={"file": (name, content)})
    assert r.status_code == status, r.text
    uploads = tmp_path / "data" / "uploads"
    assert not uploads.exists() or not any(uploads.iterdir())  # nothing left behind, not even a .part


def test_delete_never_touches_files_outside_uploads(env, tmp_path):
    c, _ = client_for(env)
    d = c.delete("/sources/specs.md", headers=KEY).json()
    assert d["chunks_deleted"] == 1 and d["file_deleted"] is False and (tmp_path / "specs.md").exists()
    assert c.delete("/sources/specs.md", headers=KEY).status_code == 404


def mock_site():
    page = ("<html><head><title>Docking</title></head><body><article><h1>Docking</h1><p>"
            + "Dock the robot nightly to keep firmware current. " * 6 + "</p></article></body></html>")

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.path == "/dock":
            return httpx.Response(200, text=page, headers={"content-type": "text/html"})
        return httpx.Response(404)

    return lambda: WebLoader("t", client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_ingest_url(env):
    c, _ = client_for(env, web=mock_site())
    r = c.post("/ingest/url", headers=KEY, json={"url": "https://ex.com/dock"})
    assert r.status_code == 200 and r.json()["sources"][0]["source"] == "https://ex.com/dock"
    assert c.post("/ingest/url", headers=KEY, json={"url": "https://ex.com/gone"}).status_code == 422
    assert c.post("/ingest/url", headers=KEY, json={"url": "ftp://ex.com/x"}).status_code == 422
    assert c.post("/ingest/url", headers=KEY, json={"url": "https://ex.com/dock", "crawl_depth": 9}).status_code == 422
    assert c.delete("/sources/https://ex.com/dock", headers=KEY).json()["chunks_deleted"] == 1


def test_ingest_url_blocks_private_addresses_by_default(env):
    c, _ = client_for(env)  # real guarded loader; 127.0.0.1 is refused before any connection
    r = c.post("/ingest/url", headers=KEY, json={"url": "http://127.0.0.1:9/admin"})
    assert r.status_code == 422 and "private or local" in r.json()["detail"]


@pytest.mark.parametrize("raw,clean", [("report.pdf", "report.pdf"), ("../../x.md", "x.md"),
                                       ("C:\\Users\\me\\notes v2.txt", "notes_v2.txt"), (".bashrc.md", "bashrc.md")])
def test_safe_filename(raw, clean):
    assert safe_filename(raw) == clean


def test_cors_preflight(env):
    c, _ = client_for(env)
    ok = c.options("/chat", headers={"Origin": "http://localhost:7860", "Access-Control-Request-Method": "POST",
                                     "Access-Control-Request-Headers": "x-api-key,content-type"})
    assert ok.headers["access-control-allow-origin"] == "http://localhost:7860"
    bad = c.options("/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers


def test_body_size_limit_rejects_before_parsing(env):
    c, _ = client_for(env)  # max_file_mb=0.01 -> limit 1.01 MB
    big = b"x" * 1_100_000
    r = c.post("/ingest/file", headers=KEY, files={"file": ("big.md", big)})
    assert r.status_code == 413 and "larger than" in r.json()["detail"]

    def chunked():  # no Content-Length: the byte counter has to catch it
        for _ in range(12):
            yield b"y" * 100_000

    r = c.post("/chat", headers=KEY | {"content-type": "application/json"}, content=chunked())
    assert r.status_code == 413
    assert c.post("/chat", headers=KEY, json={"message": "x" * 10}).status_code != 413


class AuthenticationError(Exception):  # named like the groq/openai SDK classes
    status_code = 401


class RateLimitError(Exception):
    status_code = 429


@pytest.mark.parametrize("exc,status,text", [
    (AuthenticationError("Invalid API Key gsk_secret"), 502, "rejected the credentials"),
    (RateLimitError("slow down"), 503, "rate-limiting"),
    (httpx.ConnectError("refused"), 503, "unreachable"),
    (KeyError("x"), 500, "internal error (KeyError)"),
])
def test_provider_failures_are_explained_not_leaked(env, exc, status, text):
    class Failing:
        def stream(self, *a, **k):
            raise exc

    s, kb = env
    c = TestClient(create_app(s, kb=kb, agent=Failing()), raise_server_exceptions=False)
    r = c.post("/chat", json={"message": "hi"}, headers=KEY)
    assert r.status_code == status and text in r.json()["detail"] and "gsk_secret" not in r.text
    events = sse(c.post("/chat/stream", json={"message": "hi"}, headers=KEY).text)
    assert events[-1][0] == "error" and text in events[-1][1]["message"]
