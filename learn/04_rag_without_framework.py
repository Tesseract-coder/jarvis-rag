"""04: Retrieval-Augmented Generation (RAG) by hand.

Run: python learn/04_rag_without_framework.py

    docs ──chunk──▶ chunks ──embed──▶ TinyVectorDB          (ingest, once)
    question ──embed──▶ top-k chunks ──▶ prompt ──▶ LLM ──▶ answer + [citations]   (every query)

Uses the fictional Halcyon Robotics docs in learn/sample_docs/. The model has
never seen them, so any correct answer must come from retrieval.
"""

import re
import time

from ollama_api import SAMPLE_DOCS, banner, chat, embed, load_script

TinyVectorDB = load_script("03_vector_db_from_scratch").TinyVectorDB

SEPARATORS = ["\n\n", "\n", ". ", " "]


def split_text(text: str, chunk_size: int = 800, overlap: int = 120, separators=SEPARATORS) -> list[str]:
    """Recursive character splitter, the same idea as LangChain's RecursiveCharacterTextSplitter:
    split on the coarsest separator that works (paragraphs, then lines, then sentences,
    then words), then greedily pack pieces back together up to chunk_size, carrying
    up to `overlap` chars of trailing pieces into the next chunk."""
    return _merge(_split(text, chunk_size, separators), chunk_size, overlap)


def _split(text: str, size: int, separators: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not separators:  # no natural boundary left: hard cut
        return [text[i:i + size] for i in range(0, len(text), size)]
    sep, finer = separators[0], separators[1:]
    parts = text.split(sep)
    pieces = []
    for n, part in enumerate(parts):
        if n < len(parts) - 1:
            part += sep  # keep the separator so packing reproduces the original text
        if part:
            pieces += [part] if len(part) <= size else _split(part, size, finer)
    return pieces


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    chunks, current = [], []
    for piece in pieces:
        if current and sum(map(len, current)) + len(piece) > size:
            chunks.append("".join(current).strip())
            carry = []  # trailing pieces that fit in the overlap budget start the next chunk
            while current and sum(map(len, carry)) + len(current[-1]) <= overlap:
                carry.insert(0, current.pop())
            while carry and sum(map(len, carry)) + len(piece) > size:
                carry.pop(0)
            current = carry
        current.append(piece)
    if current:
        chunks.append("".join(current).strip())
    return [c for c in chunks if c]


def load_docs() -> list[tuple[str, str]]:
    return [(p.name, p.read_text()) for p in sorted(SAMPLE_DOCS.glob("*.md"))]


def build_index(chunk_size: int = 800, overlap: int = 120) -> TinyVectorDB:
    ids, texts, metas = [], [], []
    for source, text in load_docs():
        for n, chunk in enumerate(split_text(text, chunk_size, overlap)):
            ids.append(f"{source}#{n}")
            texts.append(chunk)
            metas.append({"source": source, "chunk": n})
    db = TinyVectorDB(dim=768)
    db.add(ids, embed(["search_document: " + t for t in texts]), texts, metas)
    return db


SYSTEM_PROMPT = """You answer questions using ONLY the numbered context passages provided.
Rules:
- After each claim, cite the passage(s) you used, like [1] or [2][3].
- If the context does not contain the answer, reply exactly: "I don't know based on the provided documents."
- Never use outside knowledge, and never invent citations."""


def build_messages(question: str, hits: list[dict]) -> list[dict]:
    context = "\n\n".join(f"[{n}] (source: {h['id']})\n{h['doc']}" for n, h in enumerate(hits, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}]


def cited_sources(answer: str, hits: list[dict]) -> list[str]:
    """Map [n] markers in the answer back to chunk ids, ignoring invented numbers."""
    nums = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})
    return [hits[n - 1]["id"] for n in nums if 1 <= n <= len(hits)]


def rag_answer(db: TinyVectorDB, question: str, k: int = 4) -> dict:
    t0 = time.perf_counter()
    hits = db.query(embed(["search_query: " + question])[0], k=k)
    t_retrieve = time.perf_counter() - t0
    resp = chat(build_messages(question, hits), temperature=0)
    answer = resp["message"]["content"].strip()
    return {"answer": answer, "hits": hits, "sources": cited_sources(answer, hits),
            "prompt_tokens": resp["prompt_eval_count"], "retrieve_s": t_retrieve,
            "generate_s": resp["total_duration"] / 1e9}


def main() -> None:
    banner("1. Ingest: chunk + embed the sample docs")
    t0 = time.perf_counter()
    db = build_index()
    lengths = [len(d) for d in db.docs]
    print(f"  {len(load_docs())} docs -> {len(db)} chunks (800 chars / 120 overlap),"
          f" avg {sum(lengths) / len(lengths):.0f} chars, embedded in {time.perf_counter() - t0:.1f}s")
    print(f"\n  example chunk {db.ids[1]!r}:\n  " + db.docs[1][:300].replace("\n", "\n  ") + " ...")

    banner("2. Without RAG: the model has never seen these documents")
    q = "How many days of paid annual leave do Halcyon Robotics employees get?"
    print(f"  Q: {q}\n  A: {chat([{'role': 'user', 'content': q}], temperature=0)['message']['content'].strip()}")

    banner("3. With RAG")
    questions = [
        q,
        "What should I do if error E-217 keeps coming back?",
        "Can I approve my own pull request?",
        "Which robot has the longer battery life, and by how many hours?",
        "What is the name of Halcyon's CEO?",  # not in the docs
    ]
    for question in questions:
        r = rag_answer(db, question)
        print(f"\n  Q: {question}")
        print("  retrieved: " + ", ".join(f"{h['id']} ({h['score']:.2f})" for h in r["hits"]))
        print(f"  A: {r['answer']}")
        print(f"  cited: {r['sources'] or '-'}   | prompt {r['prompt_tokens']} tokens,"
              f" retrieve {r['retrieve_s'] * 1000:.0f} ms, generate {r['generate_s']:.1f}s")


if __name__ == "__main__":
    main()
