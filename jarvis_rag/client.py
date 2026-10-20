"""Typed Python client for the JARVIS HTTP API, used by the Gradio UI.

    from jarvis_rag.client import JarvisClient
    jc = JarvisClient("http://127.0.0.1:8000", api_key="...")
    for event, data in jc.chat_stream("How many tickets are open?"):
        ...

`examples/client.py` is the dependency-free teaching version of the same thing.
"""

import json
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import quote

import httpx


class ApiError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class JarvisClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", api_key: str | None = None,
                 http: httpx.Client | None = None, timeout: float = 300):
        """`http` lets tests pass FastAPI's TestClient (an httpx.Client) instead of a network client."""
        self.http = http or httpx.Client(base_url=base_url, timeout=httpx.Timeout(10, read=timeout))
        if api_key:
            self.http.headers["X-API-Key"] = api_key

    @staticmethod
    def _check(r: httpx.Response) -> httpx.Response:
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except (ValueError, AttributeError):
                detail = r.text
            raise ApiError(r.status_code, str(detail))
        return r

    def health(self) -> dict:
        r = self.http.get("/health")
        return r.json() if r.headers.get("content-type", "").startswith("application/json") else {"status": "down"}

    def chat(self, message: str, thread_id: str | None = None) -> dict:
        return self._check(self.http.post("/chat", json={"message": message, "thread_id": thread_id})).json()

    def chat_stream(self, message: str, thread_id: str | None = None) -> Iterator[tuple[str, dict]]:
        """(event, data) pairs from /chat/stream: start, tool_call, tool_result, token, answer, error."""
        with self.http.stream("POST", "/chat/stream", json={"message": message, "thread_id": thread_id}) as r:
            if r.status_code >= 400:
                r.read()
                self._check(r)
            event, data = "message", []
            for line in r.iter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data.append(line[5:].strip())
                elif not line and data:
                    yield event, json.loads("\n".join(data))
                    event, data = "message", []

    def thread(self, thread_id: str) -> list[dict]:
        return self._check(self.http.get(f"/threads/{quote(thread_id, safe='')}")).json()

    def upload(self, path: str | Path, filename: str | None = None) -> dict:
        path = Path(path)
        with path.open("rb") as f:
            return self._check(self.http.post("/ingest/file", files={"file": (filename or path.name, f)})).json()

    def ingest_url(self, url: str, crawl_depth: int = 0) -> dict:
        return self._check(self.http.post("/ingest/url", json={"url": url, "crawl_depth": crawl_depth})).json()

    def sources(self) -> list[dict]:
        return self._check(self.http.get("/sources")).json()

    def delete_source(self, source: str) -> dict:
        # Percent-encode everything but "/", so "https://x.com/a?b=1" arrives as one path
        # parameter instead of being cut at "?" (a query string) or "#" (a fragment).
        return self._check(self.http.delete(f"/sources/{quote(source, safe='/')}")).json()
