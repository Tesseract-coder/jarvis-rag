"""06: Chunk size and overlap experiments, measured instead of guessed.

Run: python learn/06_chunking_experiments.py

For each (chunk_size, overlap) we re-chunk and re-embed the sample docs, ask
14 questions whose answers we know, and score retrieval:

  hit@k  = share of questions where a chunk containing the answer is in the top k
  MRR    = mean of 1/rank of the first correct chunk (1.0 = always ranked first)
  ctx    = characters the LLM must read if we pass the top 3 chunks (cost + noise)

The answer phrase must appear *intact* in one chunk, so a fact split across a
chunk boundary counts as a miss, which is exactly what overlap is meant to prevent.
"""

import time

from ollama_api import banner, embed, load_script

rag = load_script("04_rag_without_framework")

EVAL_SET = [
    ("How many days of paid annual leave do employees get?", "25 days of paid annual leave"),
    ("When do carried-over leave days expire?", "March 31"),
    ("How many days a week can I work from home?", "up to 3 days per week"),
    ("How long is parental leave for the primary caregiver?", "20 weeks"),
    ("What is the yearly learning budget?", "1,500"),
    ("What is the minimum password length?", "at least 14 characters"),
    ("How fast must I report a stolen laptop?", "within 1 hour"),
    ("When is code freeze before a release?", "Friday at 17:00"),
    ("How many approvals does a pull request need?", "two approvals"),
    ("How quickly must on-call respond to a SEV1?", "Response within 15 minutes"),
    ("What is the Kestrel K2 payload?", "Payload: 45 kg"),
    ("What does error E-217 mean?", "wheel motor overcurrent"),
    ("How far ahead must visitors be registered?", "24 hours in advance"),
    ("How much does a replacement badge cost?", "€20"),
]

CONFIGS = [(size, ov) for size in (200, 400, 800, 1600) for ov in (0, 0.15)]


def norm(s: str) -> str:
    return " ".join(s.lower().split())


def evaluate(chunk_size: int, overlap: int, query_vecs, k_max: int = 5) -> dict:
    t0 = time.perf_counter()
    db = rag.build_index(chunk_size, overlap)
    build_s = time.perf_counter() - t0
    ranks, ctx = [], []
    for qv, (_, expected) in zip(query_vecs, EVAL_SET):
        hits = db.query(qv, k=k_max)
        rank = next((n for n, h in enumerate(hits, 1) if norm(expected) in norm(h["doc"])), None)
        ranks.append(rank)
        ctx.append(sum(len(h["doc"]) for h in hits[:3]))
    n = len(EVAL_SET)
    return {
        "chunks": len(db), "build_s": build_s,
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@5": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "ctx": sum(ctx) / n,
        "misses": [q for (q, _), r in zip(EVAL_SET, ranks) if r is None],
    }


def main() -> None:
    banner(f"Chunking grid: {len(CONFIGS)} configs × {len(EVAL_SET)} questions")
    query_vecs = embed(["search_query: " + q for q, _ in EVAL_SET])  # queries don't change, embed once
    print(f"  {'size':>5} {'overlap':>8} {'chunks':>7} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'MRR':>6}"
          f" {'ctx@3 chars':>12} {'build':>6}")
    results = {}
    for size, ov in CONFIGS:
        r = results[(size, ov)] = evaluate(size, int(size * ov), query_vecs)
        print(f"  {size:>5} {int(size * ov):>8} {r['chunks']:>7} {r['hit@1']:>6.2f} {r['hit@3']:>6.2f}"
              f" {r['hit@5']:>6.2f} {r['mrr']:>6.3f} {r['ctx']:>12.0f} {r['build_s']:>5.1f}s")

    banner("Misses at hit@5 (answer phrase not intact in any of the top 5 chunks)")
    for (size, ov), r in results.items():
        if r["misses"]:
            print(f"  size={size} overlap={int(size * ov)}: {r['misses']}")

    best = max(results, key=lambda c: results[c]["mrr"])
    worst = min(results, key=lambda c: results[c]["mrr"])
    print(f"\n  Best MRR:  size={best[0]} overlap={int(best[0] * best[1])} ({results[best]['mrr']:.3f})")
    print(f"  Worst MRR: size={worst[0]} overlap={int(worst[0] * worst[1])} ({results[worst]['mrr']:.3f})")

    banner("What a small chunk loses: the context around the fact")
    for size in (200, 800):
        chunk = next(c for _, text in rag.load_docs() for c in rag.split_text(text, size, 0)
                     if "Battery life: 9 hours" in c)
        print(f"  size={size}, the chunk holding 'Battery life: 9 hours':\n    " + chunk.replace("\n", "\n    ") + "\n")

    print("""  How to read this:
  - Retrieval favours chunks about ONE thing. A huge chunk is the average of many topics,
    so its vector is close to nothing in particular ("embedding dilution").
  - But retrieval score isn't answer quality. Neither chunk above says WHICH robot lasts
    9 hours: the "## Wren W1" heading landed in the previous chunk, even at 800 chars.
    Chunk size can't fix that. Prepending the doc title + section heading to every
    chunk can ("contextual chunk headers"); Phase 2's ingestion does this.
  - Overlap only matters when a fact can be cut in half. A structure-aware splitter on
    well-formatted docs rarely does that; long unbroken prose and hard cuts do.
  - 14 questions = 0.07 per question. Treat small differences as noise, and build a
    bigger eval set (Phase 3) before tuning on your real documents.""")

if __name__ == "__main__":
    main()
