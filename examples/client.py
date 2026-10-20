"""Minimal Python client for the JARVIS API (only needs httpx).

    JARVIS_URL=http://127.0.0.1:8000 JARVIS_API_KEY=... python examples/client.py "How many tickets are open?"
    python examples/client.py --upload notes.pdf
    python examples/client.py --sources
"""

import argparse
import json
import os
import sys

import httpx

URL = os.environ.get("JARVIS_URL", "http://127.0.0.1:8000")
HEADERS = {"X-API-Key": os.environ.get("JARVIS_API_KEY", "")}


def chat(client: httpx.Client, message: str, thread_id: str | None = None) -> dict:
    r = client.post("/chat", json={"message": message, "thread_id": thread_id})
    r.raise_for_status()
    return r.json()


def stream(client: httpx.Client, message: str, thread_id: str | None = None):
    """Yield (event, data) pairs from the SSE stream. SSE is plain text:
    `event: name` and `data: json` lines, with a blank line ending each event."""
    event, data = "message", []
    with client.stream("POST", "/chat/stream", json={"message": message, "thread_id": thread_id}) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].strip())
            elif line == "" and data:
                yield event, json.loads("\n".join(data))
                event, data = "message", []
            # lines starting with ":" are keep-alive comments: ignore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?")
    ap.add_argument("--thread", help="continue a conversation")
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--upload", help="ingest a file")
    ap.add_argument("--sources", action="store_true")
    args = ap.parse_args()

    with httpx.Client(base_url=URL, headers=HEADERS, timeout=httpx.Timeout(10, read=300)) as client:
        if args.upload:
            with open(args.upload, "rb") as f:
                r = client.post("/ingest/file", files={"file": (os.path.basename(args.upload), f)})
            print(r.status_code, r.json())
        if args.sources:
            for src in client.get("/sources").raise_for_status().json():
                print(f"{src['chunks']:>5}  {src['type']:<6} {src['source']}")
        if not args.message:
            return
        if args.no_stream:
            reply = chat(client, args.message, args.thread)
            print(reply["answer"])
            print(f"\nthread: {reply['thread_id']}  citations: {reply['citations']}")
            return
        for event, data in stream(client, args.message, args.thread):
            if event == "tool_call":
                print(f"  → {data['name']}({data['args']})", file=sys.stderr)
            elif event == "token":
                print(data["text"], end="", flush=True)
            elif event == "answer":
                unverified = [c["label"] for c in data["citations"] if not c["verified"]]
                print(f"\n\nthread: {data['thread_id']}" + (f"  UNVERIFIED citations: {unverified}" if unverified else ""))
            elif event == "error":
                print(f"\n[error] {data['message']}", file=sys.stderr)


if __name__ == "__main__":
    main()
