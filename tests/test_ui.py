"""The Gradio UI's handlers with a fake client, and JarvisClient against the real
API through TestClient. No browser, no Ollama."""

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from fakes import ScriptedModel, call
from jarvis_rag import ui
from jarvis_rag.agent import build_agent
from jarvis_rag.api import create_app
from jarvis_rag.client import ApiError, JarvisClient
from jarvis_rag.config import Settings
from jarvis_rag.ingest import ingest_documents
from jarvis_rag.vectorstore import KnowledgeBase


class FakeClient:
    def __init__(self, events=(), fail=None):
        self.events, self.fail, self.calls = list(events), fail, []

    def chat_stream(self, message, thread_id=None):
        self.calls.append((message, thread_id))
        yield from self.events
        if self.fail:
            raise self.fail

    def health(self):
        if self.fail:
            raise self.fail
        return {"status": "ok", "ollama": True, "model": "m"}

    def thread(self, thread_id):
        return [{"role": "user", "content": "q", "tools": []},
                {"role": "assistant", "content": "a [x.md]", "tools": ["search_knowledge_base", "run_sql"]}]

    def upload(self, path, filename=None):
        if filename == "bad.md":
            raise ApiError(415, "unsupported file type")
        return {"sources": [{"source": f"data/uploads/{filename}", "chunks": 2, "added": 2, "removed": 0,
                             "unchanged": 0}], "skipped": []}

    def delete_source(self, source):
        return {"source": source, "chunks_deleted": 1, "file_deleted": True, "table_dropped": "t"}


STREAM = [
    ("start", {"thread_id": "th1"}),
    ("tool_call", {"name": "search_knowledge_base", "args": {"query": "q"}, "id": "auto-1"}),
    ("tool_result", {"name": "search_knowledge_base", "content": "[x.md]\nfacts", "id": "auto-1", "error": False}),
    ("token", {"text": "Maybe 20"}),
    ("discard", {"reason": "checking with SQL (t)"}),
    ("tool_call", {"name": "run_sql", "args": {"query": "SELECT 1"}, "id": "c1"}),
    ("tool_result", {"name": "run_sql", "content": "Error: Only SELECT", "id": "c1", "error": True}),
    ("token", {"text": "15 "}), ("token", {"text": "[x.md]"}),
    ("answer", {"text": "15 [x.md]", "citations": [{"label": "x.md", "verified": True},
                                                   {"label": "y.pdf", "verified": False}]}),
]


def test_chat_turn_renders_tools_discards_and_citations():
    updates = list(ui.chat_turn(FakeClient(STREAM), " how many? ", [], None))
    history, thread, box = updates[-1]
    assert thread == "th1" and box == ""
    assert history[0] == {"role": "user", "content": "how many?"}
    search, draft, sql, answer = history[1:]
    assert search["metadata"]["title"] == "🔎 search_knowledge_base" and search["metadata"]["status"] == "done"
    assert draft["content"] == "Maybe 20" and draft["metadata"]["title"].startswith("✗ draft discarded")
    assert sql["metadata"]["title"] == "🗄️ run_sql (error)"
    assert answer["content"].startswith("15 [x.md]") and "✅ x.md" in answer["content"]
    assert "⚠️ ~~y.pdf~~" in answer["content"]
    assert len(updates) == len(STREAM) + 1  # one redraw per event, plus the user's message


def test_chat_turn_continues_thread_and_ignores_blank_input():
    client = FakeClient([("answer", {"text": "hi", "citations": []})])
    list(ui.chat_turn(client, "hello", [], "th9"))
    assert client.calls == [("hello", "th9")]
    assert list(ui.chat_turn(client, "   ", [{"role": "user", "content": "x"}], "th9")) == [
        ([{"role": "user", "content": "x"}], "th9", "")]


@pytest.mark.parametrize("exc,text", [(httpx.ConnectError("refused"), "API unreachable"),
                                      (ApiError(401, "missing or wrong X-API-Key"), "wrong X-API-Key")])
def test_chat_turn_shows_failures_in_the_chat(exc, text):
    history, *_ = list(ui.chat_turn(FakeClient(fail=exc), "q", [], None))[-1]
    assert history[-1]["role"] == "assistant" and text in history[-1]["content"]


def test_error_event_and_restore_thread():
    history, *_ = list(ui.chat_turn(FakeClient([("error", {"message": "internal error (X)"})]), "q", [], None))[-1]
    assert history[-1]["content"] == "❌ internal error (X)"
    restored = ui.restore_thread(FakeClient(), "th1")
    assert [m["role"] for m in restored] == ["user", "assistant", "assistant"]
    assert restored[1]["metadata"]["title"] == "🔧 search_knowledge_base → run_sql"
    assert ui.restore_thread(FakeClient(), None) == []


def test_kb_handlers():
    out = ui.upload_files(FakeClient(), ["/tmp/x/notes.md", "/tmp/x/bad.md"])
    assert "✅ data/uploads/notes.md: 2 chunks" in out and "❌ unsupported file type (bad.md)" in out
    assert ui.upload_files(FakeClient(), None).startswith("Choose")
    assert ui.delete_source(FakeClient(), "data/uploads/a.csv") == \
        "🗑️ deleted 1 chunks of data/uploads/a.csv and its file and SQL table t"
    assert ui.delete_source(FakeClient(), None).startswith("Pick")
    assert ui.ingest_url(FakeClient(), " ", 0).startswith("Enter")
    assert ui.status_line(FakeClient(fail=httpx.ConnectError("x")), None, "http://h:1") == \
        "🔴 API unreachable at http://h:1"


def test_ui_auth_policy():
    s = Settings(_env_file=None)
    assert ui.ui_auth(s, "127.0.0.1") is None
    with pytest.raises(RuntimeError, match="UI_PASSWORD"):
        ui.ui_auth(s, "0.0.0.0")
    assert ui.ui_auth(Settings(_env_file=None, ui_password="pw"), "0.0.0.0") == ("jarvis", "pw")


def test_build_ui_exposes_only_named_endpoints():
    blocks = ui.build_ui(FakeClient(), Settings(_env_file=None))
    public = sorted(f.api_name for f in blocks.fns.values() if f.api_visibility == "public")
    assert public == ["chat", "delete", "ingest_url", "sources", "upload"]


# --------------------------------------------------------------------------- JarvisClient against the real API

@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None, storage_dir=tmp_path / "st", collection_name="uic", memory_backend="memory",
                 retrieval_mode="hybrid", api_key="k")
    kb = KnowledgeBase(s, embeddings=DeterministicFakeEmbedding(size=32))
    ingest_documents([Document("# A\nAlpha facts.", metadata={"source": "https://ex.com/a?x=1#top", "type": "web",
                                                               "title": "A", "url": "https://ex.com/a"})], kb, s)
    model = ScriptedModel(script=[call("calculator", expression="6*7"), AIMessage("42 [https://ex.com/a?x=1#top]")],
                          seen=[])
    agent = build_agent(s, model=model, kb=kb, checkpointer=InMemorySaver())
    return TestClient(create_app(s, kb=kb, agent=agent))


def test_client_streams_and_reads_back_the_thread(api_client):
    jc = JarvisClient(http=api_client, api_key="k")
    events = list(jc.chat_stream("6 times 7?", "t-1"))
    assert [e for e, _ in events][0] == "start" and events[-1][0] == "answer"
    assert events[-1][1]["citations"] == [{"label": "https://ex.com/a?x=1#top", "verified": True}]
    assert jc.thread("t-1") == [{"role": "user", "content": "6 times 7?", "tools": []},
                                {"role": "assistant", "content": "42 [https://ex.com/a?x=1#top]",
                                 "tools": ["search_knowledge_base", "calculator"]}]
    assert jc.thread("never-used") == []


def test_client_deletes_sources_with_url_characters(api_client):
    jc = JarvisClient(http=api_client, api_key="k")
    assert jc.delete_source("https://ex.com/a?x=1#top")["chunks_deleted"] == 1  # ? and # survive the trip


def test_client_raises_api_errors(api_client):
    with pytest.raises(ApiError) as e:
        JarvisClient(http=api_client, api_key="wrong").sources()
    assert e.value.status == 401
    with pytest.raises(ApiError):
        list(JarvisClient(http=api_client, api_key="wrong").chat_stream("hi"))
