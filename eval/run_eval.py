"""Measure retrieval (and optionally answers) instead of guessing.

    python eval/run_eval.py                          # all modes, hit@k + MRR, per-kind table
    python eval/run_eval.py --modes hybrid rerank --misses
    python eval/run_eval.py --set bm25_weight=0.3 --set fetch_k=30
    python eval/run_eval.py --answers --answer-mode rerank   # + LLM-as-judge (slow: ~1 min per 5 questions)

Requires the demo corpus: `make demo-data ingest`.

A retrieved chunk is RELEVANT to a question when it comes from `expected_source`
and contains at least one `must_contain` phrase (case/whitespace-insensitive).
The source alone is not enough: product_specs.md holds both robots, and only the
chunk with the right robot answers the question.

  hit@k       share of questions with a relevant chunk in the top k
  MRR         mean of 1/rank of the first relevant chunk (0 if none in top k)
  complete@k  comparison questions: share where EVERY must_contain fact is in the top k
"""

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pydantic import BaseModel, Field  # noqa: E402

from jarvis_rag.config import get_settings  # noqa: E402
from jarvis_rag.retriever import Retriever  # noqa: E402
from jarvis_rag.vectorstore import KnowledgeBase  # noqa: E402

MODES = ["similarity", "mmr", "hybrid", "rerank"]
QA_PATH = Path(__file__).with_name("qa_set.jsonl")


def load_qa(path: Path = QA_PATH) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def is_relevant(doc, item: dict) -> bool:
    if doc.metadata.get("source") != item["expected_source"]:
        return False
    text = norm(doc.page_content)
    return not item["must_contain"] or any(norm(p) in text for p in item["must_contain"])


def score_ranking(docs: list, item: dict, k: int = 5) -> dict:
    top = docs[:k]
    rank = next((n for n, d in enumerate(top, 1) if is_relevant(d, item)), None)
    found = {p for d in top if d.metadata.get("source") == item["expected_source"]
             for p in item["must_contain"] if norm(p) in norm(d.page_content)}
    return {"rank": rank, "complete": len(found) == len(item["must_contain"])}


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {}
    ranks = [r["rank"] for r in rows]
    comps = [r["complete"] for r in rows if r["kind"] == "comparison"]
    return {
        "n": n,
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "complete@5": sum(comps) / len(comps) if comps else None,
        "p50_ms": statistics.median(r["ms"] for r in rows),
    }


def unreachable(kb: KnowledgeBase, qa: list[dict]) -> list[str]:
    """Gold labels that no stored chunk can satisfy: a typo, a stale corpus, or a
    phrase cut in half by the chunker. Better to know than to blame the retriever."""
    bad = []
    for item in qa:
        if not item["expected_source"]:
            continue
        got = kb.collection.get(where={"source": item["expected_source"]}, include=["documents"])["documents"]
        for p in item["must_contain"]:
            if not any(norm(p) in norm(t) for t in got):
                bad.append(f"{item['id']}: {p!r} not found intact in {item['expected_source']} ({len(got)} chunks)")
    return bad


def run_retrieval(retriever: Retriever, qa: list[dict], modes: list[str], k: int) -> dict[str, list[dict]]:
    results = {}
    for mode in modes:
        retriever.retrieve("warm-up", k, mode=mode)  # load models / build BM25 outside the timings
        rows = []
        for item in qa:
            if not item["expected_source"]:
                continue
            t0 = time.perf_counter()
            hits = retriever.retrieve(item["question"], k, mode=mode)
            ms = (time.perf_counter() - t0) * 1000
            docs = [d for d, _ in hits]
            rows.append({"id": item["id"], "kind": item["kind"], "ms": ms,
                         "top": [d.page_content.splitlines()[0][:70] for d in docs[:3]],
                         **score_ranking(docs, item, k)})
        results[mode] = rows
    return results


def print_report(results: dict[str, list[dict]], qa: list[dict], show_misses: bool) -> None:
    print(f"\n{'mode':<11} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6} {'complete@5':>11} {'p50 ms':>7}")
    for mode, rows in results.items():
        s = summarize(rows)
        comp = f"{s['complete@5']:.2f}" if s["complete@5"] is not None else "-"
        print(f"{mode:<11} {s['hit@1']:>6.2f} {s['hit@3']:>6.2f} {s['hit@5']:>6.2f} {s['mrr']:>6.3f} {comp:>11} {s['p50_ms']:>7.0f}")

    kinds = sorted({r["kind"] for rows in results.values() for r in rows})
    print(f"\nMRR by question kind (n = questions of that kind)\n{'kind':<12} {'n':>3} " +
          " ".join(f"{m:>10}" for m in results))
    for kind in kinds:
        cells = []
        for rows in results.values():
            cells.append(f"{summarize([r for r in rows if r['kind'] == kind])['mrr']:>10.3f}")
        n = sum(1 for r in next(iter(results.values())) if r["kind"] == kind)
        print(f"{kind:<12} {n:>3} " + " ".join(cells))

    if show_misses:
        questions = {q["id"]: q for q in qa}
        for mode, rows in results.items():
            misses = [r for r in rows if r["rank"] != 1]
            print(f"\n{mode}: {len(misses)} questions not ranked first")
            for r in misses:
                print(f"  {r['id']} rank={r['rank'] or '-':<2} {questions[r['id']]['question']}")
                print(f"       got: {r['top'][0]}")


# --------------------------------------------------------------------------- answers + judge

ANSWER_PROMPT = """You answer questions using ONLY the numbered context passages.
- Cite the passage(s) you used after each claim, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Never use outside knowledge."""

JUDGE_PROMPT = """You grade an AI assistant's answer. Be strict and literal.

Question: {question}
Reference answer: {reference}

Context the assistant was given:
{context}

Assistant's answer:
{answer}

correct  = the answer agrees with the reference answer on the key facts (wording may differ).
faithful = every factual claim in the answer is supported by the context above
           (an answer that says it doesn't know is faithful)."""


class Judgement(BaseModel):
    correct: bool = Field(description="answer agrees with the reference on the key facts")
    faithful: bool = Field(description="every claim is supported by the context")
    reason: str = Field(description="one short sentence")


def run_answers(retriever: Retriever, qa: list[dict], mode: str, k: int, limit: int | None) -> list[dict]:
    from jarvis_rag.llm import get_chat_model

    llm = get_chat_model().model_copy(update={"temperature": 0})  # grading should be repeatable
    judge = llm.with_structured_output(Judgement)
    items = [q for q in qa if q["kind"] not in ("table",)]  # tables need SQL: the agent's job (Phase 4)
    items = items[:limit] if limit else items
    out = []
    for n, item in enumerate(items, 1):
        hits = retriever.retrieve(item["question"], k, mode=mode)
        context = "\n\n".join(f"[{i}] {d.page_content}" for i, (d, _) in enumerate(hits, 1))
        t0 = time.perf_counter()
        answer = llm.invoke([("system", ANSWER_PROMPT),
                             ("user", f"Context:\n{context}\n\nQuestion: {item['question']}")]).content.strip()
        gen_s = time.perf_counter() - t0
        cited = {int(c) for c in re.findall(r"\[(\d+)\]", answer)}
        abstained = "don't know" in answer.lower() or "do not know" in answer.lower()
        if item["kind"] == "unanswerable":
            j = Judgement(correct=abstained, faithful=abstained or bool(cited), reason="abstention check")
        else:
            j = judge.invoke(JUDGE_PROMPT.format(question=item["question"], reference=item["expected_answer"],
                                                 context=context, answer=answer))
        row = {"id": item["id"], "kind": item["kind"], "answer": answer, "correct": j.correct,
               "faithful": j.faithful, "reason": j.reason, "abstained": abstained,
               "bad_citations": sorted(c for c in cited if not 1 <= c <= len(hits)), "gen_s": gen_s}
        out.append(row)
        mark = "✓" if j.correct else "✗"
        print(f"  {mark} {item['id']} {'F' if j.faithful else 'UNFAITHFUL'} {gen_s:4.1f}s  {item['question'][:60]}"
              f"\n      {answer[:160].replace(chr(10), ' ')}")
    n = len(out)
    answerable = [r for r in out if r["kind"] != "unanswerable"]
    unans = [r for r in out if r["kind"] == "unanswerable"]
    print(f"\nanswers ({mode}, k={k}): correct {sum(r['correct'] for r in answerable)}/{len(answerable)}, "
          f"faithful {sum(r['faithful'] for r in out)}/{n}, "
          f"abstained correctly {sum(r['abstained'] for r in unans)}/{len(unans)}, "
          f"wrongly abstained {sum(r['abstained'] for r in answerable)}, "
          f"invented citation numbers {sum(bool(r['bad_citations']) for r in out)}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a setting for this run, e.g. bm25_weight=0.3")
    ap.add_argument("--misses", action="store_true", help="list questions not ranked first")
    ap.add_argument("--answers", action="store_true", help="also generate answers and judge them (slow)")
    ap.add_argument("--answer-mode", choices=MODES, default=None)
    ap.add_argument("--limit", type=int, help="answer only the first N questions")
    ap.add_argument("--json", type=Path, help="write all results here")
    args = ap.parse_args()

    s = get_settings()
    if args.set:
        s = s.model_copy(update={k: type(getattr(s, k))(v) for k, v in (kv.split("=", 1) for kv in args.set)})
    kb = KnowledgeBase(s)
    qa = load_qa()
    print(f"{len(qa)} questions, {kb.count()} chunks in collection {s.collection_name!r}"
          + (f", overrides: {', '.join(args.set)}" if args.set else ""))
    if bad := unreachable(kb, qa):
        print("WARNING: gold labels no chunk can satisfy (run `make demo-data ingest`?):")
        print("\n".join(f"  {b}" for b in bad))

    retriever = Retriever(kb, s)
    results = run_retrieval(retriever, qa, args.modes, args.k)
    print_report(results, qa, args.misses)
    out = {"settings": {k: str(v) for k, v in s.model_dump().items()}, "retrieval": results}
    if args.answers:
        mode = args.answer_mode or s.retrieval_mode
        print(f"\nGenerating + judging answers with {s.chat_model} (mode={mode}) ...")
        out["answers"] = run_answers(retriever, qa, mode, args.k, args.limit)
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
