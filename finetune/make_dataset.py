"""Build the reader fine-tuning set: (passages, question) -> cited answer or abstention.

    GROQ_API_KEY=... .venv/bin/python finetune/make_dataset.py --n 450

The corpus is this project's own docs/ and learn/ (RAG and ML topics), NOT the Halcyon
documents the eval uses, and any chunk that mentions Halcyon terms is dropped. The
model should learn the *skill* (cite, abstain, be brief), not memorise eval answers.

Two kinds of example:
  answerable    target chunk + 4 distractors from other files, shuffled
                -> teacher's short answer + [target label]
  unanswerable  5 distractors only, question written from the (absent) target chunk
                -> the exact abstention sentence, kept only if the teacher confirms
                   the distractors really don't answer it
"""

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "finetune"))

from pydantic import BaseModel, Field  # noqa: E402

from common import ABSTAIN, READER_PROMPT, user_message  # noqa: E402
from jarvis_rag.config import get_settings  # noqa: E402
from jarvis_rag.ingest import Chunker, clean  # noqa: E402
from jarvis_rag.loaders import load_code, load_text  # noqa: E402
from jarvis_rag.tools import cite_label  # noqa: E402

OUT = ROOT / "finetune" / "out"
# Anything from the eval's fictional company: product names, error codes, part numbers,
# table names, places and facts that the teaching docs quote as examples.
CONTAMINATION = re.compile(r"halcyon|kestrel|wren|\bK2\b|\bW1\b|E-\d{3}|HK-\d{3,}|robot_sales|support_tickets|"
                           r"lisbon|lidar|safety.report|maintenance manual|battery", re.I)
VAGUE = re.compile(r"\b(this|the) (passage|text|document|section|snippet|context|chunk)\b", re.I)


class QA(BaseModel):
    question: str = Field(description="a natural question a developer might ask, answerable from the passage alone")
    answer: str = Field(description="1-3 sentence answer using only facts in the passage, no citation")


class Check(BaseModel):
    answerable: bool = Field(description="true if the context contains the answer to the question")


QA_PROMPT = """Write one question that a developer reading project documentation might ask, and its answer.
Rules: the question must be answerable from the passage alone; don't refer to "the passage" or "the text";
ask about a specific fact, number, name, reason or how-to; the answer is 1-3 sentences using only the passage.

Passage:
{text}"""

CHECK_PROMPT = """Does the context below contain enough information to answer the question? Be strict: partial or
related information that does not actually answer it means false.

Question: {question}

Context:
{context}"""


def corpus_chunks() -> list[tuple[str, str]]:
    s = get_settings()
    chunker, docs = Chunker(s), []
    for p in sorted((ROOT / "docs").glob("*.md")) + [ROOT / "README.md"]:
        docs += load_text(p, 10**7)
    for p in sorted((ROOT / "learn").glob("*.py")):
        docs += load_code(p, 10**7)
    chunks = []
    for d in docs:
        d.page_content = clean(d.page_content)
        for c in chunker.split(d):
            body = c.page_content
            if len(body) >= 300 and not CONTAMINATION.search(body):
                chunks.append((cite_label(c), body))
    return chunks


def make_teacher(model: str):
    from langchain_groq import ChatGroq

    s = get_settings()
    if not s.groq_api_key:
        sys.exit("set GROQ_API_KEY")
    llm = ChatGroq(model=model, api_key=s.groq_api_key.get_secret_value(), temperature=0.7, max_retries=8,
                   reasoning_effort="low")
    # json_schema, not the default forced tool call: the first run lost 345 of 450 calls to
    # "Tool choice is required, but model did not call a tool".
    return (llm.with_structured_output(QA, method="json_schema"),
            llm.model_copy(update={"temperature": 0}).with_structured_output(Check, method="json_schema"))


def ask(runnable, prompt: str, tries: int = 3):
    for attempt in range(tries):
        try:
            return runnable.invoke(prompt)
        except Exception:
            if attempt == tries - 1:
                raise


def build_one(i: int, chunks, by_source, teacher, checker, seed: int) -> dict | None:
    rng = random.Random(seed * 100_003 + i)
    target = rng.choice(chunks)
    source = target[0].split(",")[0]
    others = [c for src, cs in by_source.items() if src != source for c in cs]
    try:
        qa = ask(teacher, QA_PROMPT.format(text=target[1]))
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
    if not qa or VAGUE.search(qa.question) or len(qa.answer) > 600 or not qa.question.strip().endswith("?"):
        return None
    kind = "answerable" if rng.random() < 0.6 else "unanswerable"
    if kind == "answerable":
        passages = rng.sample(others, 4) + [target]
        rng.shuffle(passages)
        answer = f"{qa.answer.strip()} [{target[0]}]"
    else:
        passages = rng.sample(others, 5)
        try:  # the docs overlap a lot: make sure the distractors really don't answer it
            if ask(checker, CHECK_PROMPT.format(question=qa.question, context=user_message(passages, ""))).answerable:
                return None
        except Exception as e:
            return {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        answer = ABSTAIN
    return {"kind": kind, "source": source, "messages": [
        {"role": "system", "content": READER_PROMPT},
        {"role": "user", "content": user_message(passages, qa.question)},
        {"role": "assistant", "content": answer}]}


def parse_context(user: str) -> tuple[list[tuple[str, str]], str]:
    ctx, question = user.removeprefix("Context:\n").rsplit("\n\nQuestion: ", 1)
    blocks = [b.split("\n", 1) for b in ctx.split("\n\n---\n\n")]
    return [(label.strip("[]"), text) for label, text in blocks], question


def augment(row: dict, by_source: dict, rng: random.Random) -> dict:
    """A second context for the same Q/A, no teacher call needed.
    answerable:   keep the answering passage, draw 4 new distractors, new position
    unanswerable: the teacher-verified distractors, reshuffled (new ones could answer it)"""
    passages, question = parse_context(row["messages"][1]["content"])
    if row["kind"] == "answerable":
        cited = row["messages"][2]["content"].rsplit("[", 1)[1].rstrip("]")
        target = next(p for p in passages if p[0] == cited)
        others = [c for src, cs in by_source.items() if src != row["source"] for c in cs]
        passages = rng.sample(others, 4) + [target]
    rng.shuffle(passages)
    return {**row, "augmented": True, "messages": [row["messages"][0], {"role": "user",
            "content": user_message(passages, question)}, row["messages"][2]]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, default=450, help="attempts (some are filtered out)")
    ap.add_argument("--teacher", default="openai/gpt-oss-20b",
                    help="Groq free tier: 1,000 requests/day and 8,000 tokens/minute per model")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-sources", type=int, default=3, help="files held out entirely for validation")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--augment", type=int, default=1, help="extra contexts per TRAIN example (val stays clean)")
    args = ap.parse_args()

    chunks = corpus_chunks()
    by_source: dict[str, list] = {}
    for c in chunks:
        by_source.setdefault(c[0].split(",")[0], []).append(c)
    print(f"{len(chunks)} clean chunks from {len(by_source)} files (Halcyon-contaminated chunks dropped)")
    teacher, checker = make_teacher(args.teacher) if args.n else (None, None)
    # Rows are appended to raw.jsonl as they arrive, keyed by attempt index, so a quota
    # cut-off loses nothing and a re-run resumes where it stopped.
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "raw.jsonl"
    done = {json.loads(line)["i"] for line in raw.open()} if raw.exists() else set()
    todo = [i for i in range(args.n) if i not in done]
    print(f"{len(done)} attempts already done, {len(todo)} to go (teacher {args.teacher})")
    if not args.n:
        todo = []  # --n 0: rebuild the split/augmentation from raw.jsonl without calling the teacher
    t0, errors = time.time(), []
    with ThreadPoolExecutor(args.workers) as ex, raw.open("a") as f:
        for n, (i, r) in enumerate(zip(todo, ex.map(
                lambda i: build_one(i, chunks, by_source, teacher, checker, args.seed), todo)), 1):
            if r and "error" in r:
                errors.append(r["error"])  # not recorded as done: retried on the next run
            else:
                f.write(json.dumps({"i": i, "row": r}, ensure_ascii=False) + "\n")
                f.flush()
            if n % 25 == 0:
                print(f"  {n}/{len(todo)} attempts, {len(errors)} errors, {time.time() - t0:.0f}s", flush=True)
    rows = [rec["row"] for rec in map(json.loads, raw.open()) if rec["row"]]

    # Validation = whole files the model never trains on, so val loss measures transfer.
    rng = random.Random(args.seed)
    val_sources = set(rng.sample(sorted(by_source), args.val_sources))
    split = {"train": [r for r in rows if r["source"] not in val_sources],
             "val": [r for r in rows if r["source"] in val_sources]}
    aug_rng = random.Random(args.seed + 1)
    split["train"] += [augment(r, by_source, aug_rng) for r in split["train"] for _ in range(args.augment)]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, part in split.items():
        with (OUT / f"{name}.jsonl").open("w") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        kinds = {k: sum(r["kind"] == k for r in part) for k in ("answerable", "unanswerable")}
        print(f"{name}: {len(part)} examples {kinds} -> {OUT / f'{name}.jsonl'}")
    print(f"held-out files: {sorted(val_sources)}; {len(errors)} teacher errors; {time.time() - t0:.0f}s")
    if errors:
        print("  first error:", errors[0])


if __name__ == "__main__":
    main()
