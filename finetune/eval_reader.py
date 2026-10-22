"""Compare readers on the Halcyon eval set: same retrieved passages for every model.

    .venv/bin/python finetune/eval_reader.py --contexts          # retrieve once, cache (needs the app index)
    GROQ_API_KEY=... .venv/bin/python finetune/eval_reader.py --models qwen3:1.7b jarvis-reader qwen3:4b-instruct

Cases built from eval/qa_set.jsonl (tables excluded: they need SQL, not reading):
  answerable   top-5 reranked passages                         -> correct? cites the right source?
  removed      same question, every chunk of the right source
               removed (topped up from the next results)       -> abstain (or answer correctly from
                                                                  another doc that has the same fact)
  unanswerable the 3 questions nothing answers                 -> must abstain

Deterministic: abstention, citation validity (every cited label is in the context),
cites-expected-source. Correctness of answerable cases: LLM judge (gpt-oss-120b on Groq).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from common import ABSTAIN_RE, READER_PROMPT, citations, user_message  # noqa: E402

OUT = ROOT / "finetune" / "out"
CONTEXTS = OUT / "eval_contexts.json"

JUDGE = """Grade an answer against a reference. Be strict about facts, lenient about wording.
Question: {question}
Reference answer: {reference}
Answer to grade: {answer}
correct = the answer states the key facts of the reference and nothing contradicting them.
An answer that says it doesn't know is NOT correct."""


def build_contexts() -> None:
    from eval.run_eval import load_qa
    from jarvis_rag.retriever import Retriever
    from jarvis_rag.tools import cite_label
    from jarvis_rag.vectorstore import KnowledgeBase

    r = Retriever(KnowledgeBase())
    cases = []
    for q in load_qa():
        if q["kind"] == "table":
            continue
        hits = r.retrieve(q["question"], 10, mode="rerank")
        passages = [(cite_label(d), d.page_content) for d, _ in hits]
        if q["kind"] == "unanswerable":
            cases.append({"id": q["id"], "case": "unanswerable", "question": q["question"],
                          "passages": passages[:5], "reference": None, "expected": None})
            continue
        cases.append({"id": q["id"], "case": "answerable", "question": q["question"], "passages": passages[:5],
                      "reference": q["expected_answer"], "expected": q["expected_source"]})
        removed = [p for (d, _), p in zip(hits, passages) if d.metadata["source"] != q["expected_source"]][:5]
        # Some facts live in two documents, so "removed" may still be answerable from
        # the other one; the reference lets the judge recognise that (Halcyon is
        # fictional: a correct answer can only have come from the passages).
        cases.append({"id": q["id"], "case": "removed", "question": q["question"], "passages": removed,
                      "reference": q["expected_answer"], "expected": None})
    OUT.mkdir(parents=True, exist_ok=True)
    CONTEXTS.write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    print(f"{len(cases)} cases -> {CONTEXTS}")


def generate(model: str, cases: list[dict]) -> list[dict]:
    from langchain_ollama import ChatOllama

    from jarvis_rag.config import get_settings

    s = get_settings()
    llm = ChatOllama(model=model, base_url=s.ollama_base_url, temperature=0, num_ctx=4096, num_gpu=s.num_gpu,
                     reasoning=False, keep_alive=600)
    out = []
    for i, c in enumerate(cases, 1):
        t0 = time.perf_counter()
        msg = llm.invoke([("system", READER_PROMPT), ("user", user_message([tuple(p) for p in c["passages"]],
                                                                             c["question"]))])
        meta = msg.response_metadata
        out.append({**c, "answer": msg.text.strip(), "s": time.perf_counter() - t0,
                    "tok_s": meta.get("eval_count", 0) / max(meta.get("eval_duration", 1) / 1e9, 1e-9)})
        if i % 20 == 0:
            print(f"  {model}: {i}/{len(cases)}")
    return out


def judge(rows: list[dict]) -> None:
    from langchain_groq import ChatGroq
    from pydantic import BaseModel, Field

    from jarvis_rag.config import get_settings

    class Verdict(BaseModel):
        correct: bool = Field(description="states the reference's key facts")

    s = get_settings()
    # json_schema: Groq constrains the reply to the schema. The default (a forced tool
    # call) failed with "Tool choice is required, but model did not call a tool".
    j = ChatGroq(model="openai/gpt-oss-120b", api_key=s.groq_api_key.get_secret_value(), temperature=0,
                 max_retries=8, reasoning_effort="low").with_structured_output(Verdict, method="json_schema")
    for r in rows:
        if r.get("correct") is not None and "judge_error" not in r:
            continue  # already judged (--rejudge only fills the gaps)
        r.pop("judge_error", None)
        r["correct"] = False
        if not r["reference"] or r["abstained"]:
            continue
        time.sleep(4)  # free tier: 8,000 tokens/minute; ~400 tokens per verdict
        for attempt in range(3):
            try:
                r["correct"] = j.invoke(JUDGE.format(question=r["question"], reference=r["reference"],
                                                     answer=r["answer"])).correct
                break
            except Exception as e:  # one bad judge reply must not throw away 100 generations
                r["judge_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        if "judge_error" in r and attempt == 2:
            r["correct"] = None


def score(rows: list[dict]) -> None:
    for r in rows:
        labels = {p[0] for p in r["passages"]}
        cited = citations(r["answer"])
        r["abstained"] = bool(ABSTAIN_RE.search(r["answer"]))
        r["invalid_citations"] = [c for c in cited if c not in labels]
        r["cites_expected"] = bool(r["expected"]) and any(c.split(",")[0] == r["expected"] for c in cited)


def summarize(model: str, rows: list[dict]) -> dict:
    ans = [r for r in rows if r["case"] == "answerable"]
    neg = [r for r in rows if r["case"] in ("removed", "unanswerable")]
    pct = lambda xs: 100 * sum(bool(x) for x in xs) / max(len(xs), 1)
    return {"model": model, "n": len(rows),
            "correct": pct([r["correct"] for r in ans]),
            "cites_source": pct([r["cites_expected"] for r in ans]),
            "wrong_abstain": pct([r["abstained"] for r in ans]),
            "unjudged": sum(r["correct"] is None for r in rows),
            "safe_when_absent": pct([r["abstained"] or r["correct"] for r in neg]),
            "hallucinated": sum(not r["abstained"] and not r["correct"] for r in neg),
            "invalid_cite_rows": sum(bool(r["invalid_citations"]) for r in rows),
            "words": statistics.median(len(r["answer"].split()) for r in ans),
            "p50_s": statistics.median(r["s"] for r in rows),
            "tok_s": statistics.median(r["tok_s"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--contexts", action="store_true", help="(re)build the cached retrieval contexts")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--rejudge", action="store_true", help="reuse saved generations; judge only what's missing")
    args = ap.parse_args()
    if args.contexts:
        build_contexts()
    cases = json.loads(CONTEXTS.read_text())[: args.limit]
    summaries = []
    for model in args.models:
        path = OUT / f"eval_{model.replace(':', '_').replace('/', '_')}.json"
        if args.rejudge and path.exists():
            rows = json.loads(path.read_text())
        else:
            rows = generate(model, cases)
            path.write_text(json.dumps(rows, indent=1, ensure_ascii=False))  # save before judging
        score(rows)
        judge(rows)
        path.write_text(json.dumps(rows, indent=1, ensure_ascii=False))
        summaries.append(summarize(model, rows))
    if summaries:
        n_neg = sum(c["case"] != "answerable" for c in cases)
        print(f"\n{'model':<22} {'correct':>8} {'cites src':>9} {'wrong abst':>10} {'safe|absent':>11} "
              f"{'halluc':>6} {'bad cites':>9} {'words':>6} {'p50 s':>6} {'tok/s':>6}")
        for s in summaries:
            print(f"{s['model']:<22} {s['correct']:>7.0f}% {s['cites_source']:>8.0f}% {s['wrong_abstain']:>9.0f}% "
                  f"{s['safe_when_absent']:>10.0f}% {s['hallucinated']:>3}/{n_neg:<2} {s['invalid_cite_rows']:>9} "
                  f"{s['words']:>6.0f} {s['p50_s']:>6.1f} {s['tok_s']:>6.1f}")


if __name__ == "__main__":
    main()
