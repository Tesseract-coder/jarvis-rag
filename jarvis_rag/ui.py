"""Gradio web UI: a client of the HTTP API, not a second copy of the agent.

    make api & make ui          # two processes: API on :8000, UI on :7860 (talks to API_URL)
    make app                    # one process: API on :8000 with the UI mounted at /ui

    browser ──▶ Gradio server (holds API_KEY) ──HTTP/SSE──▶ JARVIS API ──▶ agent, Chroma, SQLite, Ollama

The browser never sees the API key, and the UI gets nothing the API doesn't
offer, so anything the UI can do, a script can do too.

The event handlers (`chat_turn`, `upload_files`, ...) are plain functions of a
JarvisClient, so tests drive them with a fake client and no browser.
"""

import argparse
import json
import os
import time
from collections.abc import Iterator

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # before importing gradio: no usage telemetry

import gradio as gr  # noqa: E402
import httpx  # noqa: E402

from jarvis_rag.api import UPLOAD_EXTS, is_loopback  # noqa: E402
from jarvis_rag.client import ApiError, JarvisClient  # noqa: E402
from jarvis_rag.config import Settings, get_settings  # noqa: E402

EXAMPLES = [
    "How many safety incidents were recorded in 2026, and what was the main cause?",
    "What was the total revenue per region?",
    "What does error E-217 mean, and how many support tickets have it?",
    "Where is the chunk id hash computed?",
    "What is 15% of the APAC revenue?",
]
TOOL_ICONS = {"search_knowledge_base": "🔎", "run_sql": "🗄️", "describe_table": "🗄️", "list_tables": "🗄️",
              "fetch_url": "🌐", "calculator": "🧮", "ingest_text": "📝"}


def failure(e: Exception) -> str:
    if isinstance(e, ApiError):
        return f"❌ {e.detail}"
    if isinstance(e, httpx.HTTPError):
        return f"❌ API unreachable ({type(e).__name__}). Is `make api` running?"
    return f"❌ {type(e).__name__}: {e}"


def citations_footer(citations: list[dict]) -> str:
    if not citations:
        return ""
    parts = [f"✅ {c['label']}" if c["verified"] else f"⚠️ ~~{c['label']}~~ (no tool returned this)"
             for c in citations]
    return "\n\n<sub>Sources: " + " · ".join(parts) + "</sub>"


# --------------------------------------------------------------------------- chat

def chat_turn(client: JarvisClient, message: str, history: list, thread_id: str | None
              ) -> Iterator[tuple[list, str | None, str]]:
    """Stream one turn into the chat history. Yields (history, thread_id, textbox).
    Tool calls become collapsible "🔎 search_knowledge_base" steps (Gradio's message
    metadata); answer tokens are appended as they arrive."""
    message = (message or "").strip()
    if not message:
        yield history, thread_id, ""
        return
    history = [*history, {"role": "user", "content": message}]
    yield history, thread_id, ""
    tools: dict[str, tuple[int, float]] = {}  # tool_call_id -> (index in history, start time)
    answer_at: int | None = None
    try:
        for event, data in client.chat_stream(message, thread_id):
            if event == "start":
                thread_id = data["thread_id"]
            elif event == "tool_call":
                answer_at = None  # text after a tool call belongs to a new bubble
                args = json.dumps(data["args"], ensure_ascii=False)
                history.append({"role": "assistant", "content": "",
                                "metadata": {"title": f"{TOOL_ICONS.get(data['name'], '🔧')} {data['name']}",
                                             "log": args[:300], "status": "pending", "id": data["id"]}})
                tools[data["id"]] = (len(history) - 1, time.perf_counter())
            elif event == "tool_result" and data.get("id") in tools:
                i, t0 = tools[data["id"]]
                meta = history[i]["metadata"] | {"status": "done"}
                if (took := time.perf_counter() - t0) >= 0.1:  # auto-search already ran before its call event
                    meta["duration"] = round(took, 1)
                if data.get("error"):
                    meta["title"] += " (error)"
                history[i] = {**history[i], "metadata": meta, "content": f"```\n{data['content']}\n```"}
            elif event == "discard" and answer_at is not None:
                # A guard rejected the streamed draft: fold it into a collapsed note
                # rather than silently deleting what the user just watched appear.
                draft = history[answer_at]["content"]
                history[answer_at] = {"role": "assistant", "content": draft,
                                      "metadata": {"title": f"✗ draft discarded: {data['reason']}", "status": "done"}}
                answer_at = None
            elif event == "token":
                if answer_at is None:
                    history.append({"role": "assistant", "content": ""})
                    answer_at = len(history) - 1
                history[answer_at] = {**history[answer_at], "content": history[answer_at]["content"] + data["text"]}
            elif event == "answer":
                text = data["text"] + citations_footer(data.get("citations", []))
                if answer_at is None:
                    history.append({"role": "assistant", "content": text})
                else:
                    history[answer_at] = {**history[answer_at], "content": text}
            elif event == "error":
                history.append({"role": "assistant", "content": f"❌ {data['message']}"})
            yield history, thread_id, ""
    except Exception as e:  # API down, 401, ...: show it in the chat instead of a red toast
        history.append({"role": "assistant", "content": failure(e)})
        yield history, thread_id, ""


def restore_thread(client: JarvisClient, thread_id: str | None) -> list:
    """Redraw a saved conversation after a page reload (tool steps summarised)."""
    if not thread_id:
        return []
    try:
        msgs = client.thread(thread_id)
    except Exception:
        return []
    out = []
    for m in msgs:
        if m["role"] == "assistant" and m["tools"]:
            out.append({"role": "assistant", "content": "",
                        "metadata": {"title": "🔧 " + " → ".join(m["tools"]), "status": "done"}})
        out.append({"role": m["role"], "content": m["content"]})
    return out


# --------------------------------------------------------------------------- knowledge base

def sources_table(client: JarvisClient) -> tuple[list[list], list[str], str]:
    try:
        srcs = client.sources()
    except Exception as e:
        return [], [], failure(e)
    rows = [[s["source"], s["type"], s["chunks"], s["ingested_at"].replace("T", " ")[:16]] for s in srcs]
    chunks = sum(s["chunks"] for s in srcs)
    return rows, [s["source"] for s in srcs], f"{len(srcs)} sources, {chunks} chunks"


def describe_ingest(result: dict) -> list[str]:
    lines = [f"✅ {s['source']}: {s['chunks']} chunks ({s['added']} new, {s['removed']} removed)"
             for s in result["sources"]]
    return lines + [f"⚠️ skipped {s['what']}: {s['why']}" for s in result["skipped"]]


def upload_files(client: JarvisClient, files: list | None) -> str:
    if not files:
        return "Choose one or more files first."
    lines = []
    for f in files:
        path = getattr(f, "name", f)  # gr.File gives paths (or tempfile wrappers in older versions)
        name = os.path.basename(path)
        try:
            lines += describe_ingest(client.upload(path, filename=name))
        except Exception as e:
            lines.append(f"{failure(e)} ({name})")
    return "\n\n".join(lines)


def ingest_url(client: JarvisClient, url: str, depth: float) -> str:
    if not (url or "").strip():
        return "Enter a URL first."
    try:
        return "\n\n".join(describe_ingest(client.ingest_url(url.strip(), int(depth or 0))))
    except Exception as e:
        return failure(e)


def delete_source(client: JarvisClient, source: str | None) -> str:
    if not source:
        return "Pick a source to delete."
    try:
        r = client.delete_source(source)
    except Exception as e:
        return failure(e)
    extra = (" and its file" if r["file_deleted"] else "") + (f" and SQL table {r['table_dropped']}"
                                                                if r["table_dropped"] else "")
    return f"🗑️ deleted {r['chunks_deleted']} chunks of {r['source']}{extra}"


def status_line(client: JarvisClient, s: Settings, api_url: str) -> str:
    try:
        h = client.health()
    except Exception:
        return f"🔴 API unreachable at {api_url}"
    ollama = {True: "Ollama 🟢", False: "Ollama 🔴", None: "cloud LLM"}[h.get("ollama")]
    return f"{'🟢' if h.get('status') == 'ok' else '🟠'} API {h.get('status')} · {ollama} · model `{h.get('model')}`"


# --------------------------------------------------------------------------- layout

def build_ui(client: JarvisClient, s: Settings | None = None, api_url: str | None = None) -> gr.Blocks:
    s = s or get_settings()
    api_url = api_url or s.api_url
    with gr.Blocks(title=s.agent_name, analytics_enabled=False, fill_height=True) as ui:
        thread = gr.BrowserState(None, storage_key="jarvis_thread_id")  # survives page reloads
        with gr.Row():
            gr.Markdown(f"## {s.agent_name}")
            status = gr.Markdown()
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(height=560, show_label=False, render_markdown=True,
                                 placeholder=f"Ask {s.agent_name} about your documents, code and tables.")
            with gr.Row():
                box = gr.Textbox(placeholder="Ask a question…", show_label=False, scale=8, autofocus=True)
                send = gr.Button("Send", variant="primary", scale=1)
                new = gr.Button("New chat", scale=1)
            gr.Examples(EXAMPLES, inputs=box)
            thread_info = gr.Markdown()
        with gr.Tab("Knowledge base"):
            with gr.Row():
                with gr.Column():
                    files = gr.File(label="Upload files", file_count="multiple",
                                    file_types=sorted(UPLOAD_EXTS))
                    upload_btn = gr.Button("Ingest files", variant="primary")
                with gr.Column():
                    url = gr.Textbox(label="URL", placeholder="https://…")
                    depth = gr.Slider(0, s.max_crawl_depth, value=0, step=1, label="Crawl depth (same site)")
                    url_btn = gr.Button("Ingest URL", variant="primary")
            result = gr.Markdown()
            with gr.Row():
                summary = gr.Markdown()
                refresh = gr.Button("Refresh", scale=0)
            table = gr.Dataframe(headers=["source", "type", "chunks", "ingested"], interactive=False, wrap=True)
            with gr.Row():
                pick = gr.Dropdown(label="Source", choices=[], scale=4)
                delete = gr.Button("Delete source", variant="stop", scale=1)

        def show_thread(tid):
            return f"<sub>thread `{tid}` (saved on the server; reload keeps it, *New chat* starts another)</sub>" \
                if tid else ""

        def on_chat(msg, history, tid):
            for h, tid2, text in chat_turn(client, msg, history, tid):
                yield h, tid2, text, show_thread(tid2)

        def refresh_kb():
            rows, choices, text = sources_table(client)
            return rows, gr.update(choices=choices, value=None), text

        # Every Gradio event is also an HTTP endpoint (see "Use via API" in the footer).
        # Name the useful ones and hide duplicates and page-load helpers.
        hidden = dict(api_visibility="private")
        chat_io = dict(inputs=[box, chatbot, thread], outputs=[chatbot, thread, box, thread_info])
        box.submit(on_chat, **chat_io, api_name="chat")
        send.click(on_chat, **chat_io, **hidden)
        new.click(lambda: ([], None, ""), outputs=[chatbot, thread, thread_info], **hidden)

        kb_out = [table, pick, summary]
        upload_btn.click(lambda f: upload_files(client, f), files, result, api_name="upload").then(
            refresh_kb, outputs=kb_out, **hidden)
        url_btn.click(lambda u, d: ingest_url(client, u, d), [url, depth], result, api_name="ingest_url").then(
            refresh_kb, outputs=kb_out, **hidden)
        delete.click(lambda p: delete_source(client, p), pick, result, api_name="delete").then(
            refresh_kb, outputs=kb_out, **hidden)
        refresh.click(refresh_kb, outputs=kb_out, api_name="sources")

        ui.load(lambda: status_line(client, s, api_url), outputs=status, **hidden)
        ui.load(lambda tid: (restore_thread(client, tid), show_thread(tid)), thread, [chatbot, thread_info], **hidden)
        ui.load(refresh_kb, outputs=kb_out, **hidden)
    return ui


def ui_auth(s: Settings, host: str) -> tuple[str, str] | None:
    """Like the API: on a network address the UI needs a password, because the
    Gradio server holds the API key and would otherwise hand its power to anyone."""
    if s.ui_password:
        return s.ui_username, s.ui_password.get_secret_value()
    if not is_loopback(host):
        raise RuntimeError(f"UI on {host} is reachable from the network: set UI_PASSWORD in .env first.")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="JARVIS web UI.")
    ap.add_argument("--with-api", action="store_true",
                    help="one process: serve the API on API_HOST:API_PORT and the UI at /ui")
    args = ap.parse_args()
    s = get_settings()
    key = s.api_key.get_secret_value() if s.api_key else None
    max_size = f"{int(s.max_file_mb)}mb"  # Gradio rejects bigger uploads before they reach disk

    if args.with_api:
        import uvicorn

        from jarvis_rag.api import create_app

        api_url = f"http://127.0.0.1:{s.api_port}"
        app = gr.mount_gradio_app(create_app(s), build_ui(JarvisClient(api_url, key), s, api_url), path="/ui",
                                  auth=ui_auth(s, s.api_host), max_file_size=max_size, footer_links=[],
                                  mcp_server=False)
        print(f"API on http://{s.api_host}:{s.api_port}  UI on http://{s.api_host}:{s.api_port}/ui")
        uvicorn.run(app, host=s.api_host, port=s.api_port)
        return

    build_ui(JarvisClient(s.api_url, key), s).queue(default_concurrency_limit=4).launch(
        server_name=s.ui_host, server_port=s.ui_port, auth=ui_auth(s, s.ui_host), share=False,
        max_file_size=max_size, footer_links=[], mcp_server=False, show_error=False)


if __name__ == "__main__":
    main()
