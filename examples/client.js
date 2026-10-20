// Minimal JavaScript client for the JARVIS API. No dependencies: Node 18+ or any browser.
//
//   JARVIS_API_KEY=... node examples/client.js "Which robot has the longer battery life?"
//
// Browsers: EventSource only supports GET, and /chat/stream is a POST (the question is in
// the body), so read the stream with fetch() and parse SSE yourself, as below. The page's
// origin must be listed in CORS_ORIGINS. Never ship your API key in public front-end code:
// put a small backend in front of the API instead.

const BASE = (typeof process !== "undefined" && process.env.JARVIS_URL) || "http://127.0.0.1:8000";
const KEY = (typeof process !== "undefined" && process.env.JARVIS_API_KEY) || "";
const headers = { "Content-Type": "application/json", "X-API-Key": KEY };

export async function chat(message, threadId = null) {
  const r = await fetch(`${BASE}/chat`, {
    method: "POST", headers, body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.json(); // { answer, thread_id, tool_calls, citations }
}

// Async generator of {event, data}. An SSE event is "event: x" + "data: {...}" lines,
// ended by a blank line. Network chunks can split an event anywhere, so buffer.
export async function* stream(message, threadId = null) {
  const r = await fetch(`${BASE}/chat/stream`, {
    method: "POST", headers, body: JSON.stringify({ message, thread_id: threadId }),
  });
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  const reader = r.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replaceAll("\r\n", "\n");
    let end;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const raw = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      let event = "message";
      const data = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trim());
      }
      if (data.length) yield { event, data: JSON.parse(data.join("\n")) };
    }
  }
}

async function main() {
  const question = process.argv[2] ?? "What does error E-217 mean?";
  for await (const { event, data } of stream(question)) {
    if (event === "tool_call") process.stderr.write(`  → ${data.name}(${JSON.stringify(data.args)})\n`);
    else if (event === "token") process.stdout.write(data.text);
    else if (event === "answer") {
      const bad = data.citations.filter((c) => !c.verified).map((c) => c.label);
      console.log(`\n\nthread: ${data.thread_id}${bad.length ? `  UNVERIFIED: ${bad.join(", ")}` : ""}`);
    } else if (event === "error") console.error(`\n[error] ${data.message}`);
  }
}

if (typeof process !== "undefined" && import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => { console.error(e.message); process.exit(1); });
}
