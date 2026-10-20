"""Grade the whole agent, not just retrieval: did it pick the right tools, get
the numbers right, cite, and say "I don't know" when it should?

    python eval/run_agent_eval.py                  # all of eval/agent_set.jsonl (~4 min on a GTX 1650)
    python eval/run_agent_eval.py --only a01 a13   # a few items
    python eval/run_agent_eval.py --set auto_search=false --json eval/results/agent_noauto.json

Checks are deterministic (no LLM judge): `tools` must all have been called,
every `numbers` value must appear in the answer (± tolerance, thousands
separators ignored), every `contains` string must appear, `cite` must appear
(it is a substring of the expected citation label), `abstain` items must
abstain and `no_abstain` items must not.
"""

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402

from jarvis_rag.agent import build_agent, stream_events  # noqa: E402
from jarvis_rag.config import get_settings  # noqa: E402

SET_PATH = Path(__file__).with_name("agent_set.jsonl")
ABSTAIN = re.compile(r"don.t know|do not know|not (?:mentioned|available|included|found|present|specified|provided|"
                     r"contain)|no (?:information|data|record|mention)|isn.t (?:mentioned|available)|"
                     r"(?:does|do|did) not (?:include|contain|cover|mention)|don.t mention", re.I)


def load_set(path: Path = SET_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def numbers_in(text: str) -> list[float]:
    text = re.sub(r"(?<=\d)[,   ](?=\d{3}\b)", "", text)  # 13,434,000 -> 13434000
    return [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text)]


def grade(item: dict, answer: str, called: list[str]) -> dict:
    found = numbers_in(answer)
    low = answer.lower()
    abstained = bool(ABSTAIN.search(answer))
    answer_ok = (all(any(abs(n - v) <= tol for n in found) for v, tol in item.get("numbers", []))
                 and all(c.lower() in low for c in item.get("contains", [])))
    if item.get("abstain"):
        answer_ok = abstained
    if item.get("no_abstain"):
        answer_ok = answer_ok and not abstained
    tools_ok = set(item.get("tools", [])) <= set(called)
    cite_ok = item["cite"].lower() in low if "cite" in item else None
    return {"answer_ok": answer_ok, "tools_ok": tools_ok, "cite_ok": cite_ok, "abstained": abstained,
            "pass": answer_ok and tools_ok and cite_ok is not False}


def run_item(agent, item: dict) -> dict:
    thread = f"eval-{item['id']}-{time.time_ns()}"
    t0 = time.perf_counter()
    for turn in item["turns"][:-1]:  # earlier turns only build up memory
        for _ in stream_events(agent, turn, thread):
            pass
    calls, answer = [], ""
    for ev in stream_events(agent, item["turns"][-1], thread):
        if ev["type"] == "tool_call":
            calls.append({"name": ev["name"], "args": ev["args"], "auto": str(ev["id"]).startswith("auto-")})
        elif ev["type"] == "tool_result" and ev["error"]:
            calls[-1]["error"] = ev["content"][:200]
        elif ev["type"] == "answer":
            answer = ev["text"]
    row = {"id": item["id"], "kind": item["kind"], "answer": answer, "calls": calls,
           "s": time.perf_counter() - t0, **grade(item, answer, [c["name"] for c in calls])}
    return row


def print_row(r: dict) -> None:
    mark = "✓" if r["pass"] else "✗"
    flags = "".join(f" {name}" for name, ok in (("ANSWER", r["answer_ok"]), ("TOOLS", r["tools_ok"]),
                                                 ("CITE", r["cite_ok"] is not False)) if not ok)
    tools = " → ".join(c["name"] + ("!" if "error" in c else "") for c in r["calls"] if not c["auto"]) or "-"
    print(f"{mark} {r['id']} {r['kind']:<12} {r['s']:5.1f}s  model tools: {tools}{'   FAILED:' + flags if flags else ''}")
    print(f"      {r['answer'][:200].replace(chr(10), ' ')}")


def summarize(rows: list[dict]) -> dict:
    cited = [r for r in rows if r["cite_ok"] is not None]
    return {"n": len(rows), "pass": sum(r["pass"] for r in rows), "answer_ok": sum(r["answer_ok"] for r in rows),
            "tools_ok": sum(r["tools_ok"] for r in rows), "cite_ok": f"{sum(r['cite_ok'] for r in cited)}/{len(cited)}",
            "model_tool_calls": statistics.mean(sum(not c["auto"] for c in r["calls"]) for r in rows),
            "tool_errors": sum("error" in c for r in rows for c in r["calls"]),
            "p50_s": statistics.median(r["s"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="+", help="item ids to run")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a setting")
    ap.add_argument("--json", type=Path, help="write results here")
    args = ap.parse_args()

    s = get_settings()
    if args.set:
        cast = lambda k, v: v.lower() in ("1", "true", "yes") if isinstance(getattr(s, k), bool) else type(getattr(s, k))(v)
        s = s.model_copy(update={k: cast(k, v) for k, v in (kv.split("=", 1) for kv in args.set)})
    items = [i for i in load_set() if not args.only or i["id"] in args.only]
    agent = build_agent(s, checkpointer=InMemorySaver())
    print(f"{len(items)} items, model {s.chat_model}, auto_search={s.auto_search}"
          + (f", overrides: {', '.join(args.set)}" if args.set else ""))
    rows = []
    for item in items:
        rows.append(run_item(agent, item))
        print_row(rows[-1])
    summ = summarize(rows)
    print(f"\npassed {summ['pass']}/{summ['n']}   answer {summ['answer_ok']}/{summ['n']}   "
          f"tools {summ['tools_ok']}/{summ['n']}   citations {summ['cite_ok']}   "
          f"model tool calls/q {summ['model_tool_calls']:.1f}   tool errors {summ['tool_errors']}   "
          f"p50 {summ['p50_s']:.1f}s")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"settings": args.set, "summary": summ, "rows": rows}, indent=2,
                                        ensure_ascii=False))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
