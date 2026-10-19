"""03: A vector database from scratch, then why real ones use ANN indexes.

Run: python learn/03_vector_db_from_scratch.py

Part 1: TinyVectorDB, a working vector store in ~60 lines (add/upsert, delete,
        top-k query, metadata filter, save/load).
Part 2: Brute force is O(N × dim) per query. We measure that, then build a
        tiny IVF index (cluster first, search only nearby clusters) to see the
        recall vs latency trade-off every real vector DB makes.
"""

import json
import time
from pathlib import Path

import numpy as np


class TinyVectorDB:
    """Everything a vector DB fundamentally does, minus the index."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype=np.float32)
        self.ids: list[str] = []
        self.docs: list[str] = []
        self.metas: list[dict] = []

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray, docs: list[str], metas: list[dict] | None = None) -> None:
        """Upsert: an existing id is replaced, never duplicated."""
        self.delete([i for i in ids if i in self.ids])
        vectors = np.asarray(vectors, dtype=np.float32)
        assert vectors.shape[1] == self.dim, f"expected dim {self.dim}, got {vectors.shape[1]}"
        # Store unit vectors, so cosine similarity becomes a plain dot product at query time.
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        self.vectors = np.vstack([self.vectors, unit])
        self.ids += ids
        self.docs += docs
        self.metas += metas or [{} for _ in ids]

    def delete(self, ids: list[str]) -> None:
        keep = [n for n, i in enumerate(self.ids) if i not in set(ids)]
        self.vectors = self.vectors[keep]
        self.ids, self.docs, self.metas = ([lst[n] for n in keep] for lst in (self.ids, self.docs, self.metas))

    def query(self, vector: np.ndarray, k: int = 5, where: dict | None = None) -> list[dict]:
        candidates = np.arange(len(self))
        if where:  # metadata pre-filter: only score rows whose metadata matches
            candidates = np.array([n for n in candidates
                                   if all(self.metas[n].get(f) == v for f, v in where.items())], dtype=int)
        if candidates.size == 0:
            return []
        q = np.asarray(vector, dtype=np.float32)
        scores = self.vectors[candidates] @ (q / np.linalg.norm(q))  # cosine for every row: O(N × dim)
        k = min(k, candidates.size)
        top = np.argpartition(-scores, k - 1)[:k]                   # O(N) partial sort...
        top = top[np.argsort(-scores[top])]                         # ...then sort only the k winners
        return [{"id": self.ids[candidates[t]], "score": float(scores[t]), "doc": self.docs[candidates[t]],
                 "meta": self.metas[candidates[t]]} for t in top]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "records.json").write_text(json.dumps({"ids": self.ids, "docs": self.docs, "metas": self.metas}))

    @classmethod
    def load(cls, path: Path) -> "TinyVectorDB":
        vectors = np.load(path / "vectors.npy")
        rec = json.loads((path / "records.json").read_text())
        db = cls(vectors.shape[1])
        db.vectors, db.ids, db.docs, db.metas = vectors, rec["ids"], rec["docs"], rec["metas"]
        return db


# --------------------------------------------------------------------------- part 1

FACTS = [
    ("The Kestrel K2 robot carries up to 45 kg.", "product"),
    ("The Wren W1 robot carries up to 12 kg.", "product"),
    ("Error E-217 means wheel motor overcurrent.", "product"),
    ("Employees get 25 days of paid annual leave.", "hr"),
    ("Remote work is allowed up to 3 days per week.", "hr"),
    ("Passwords must be at least 14 characters.", "security"),
    ("Lost laptops must be reported within 1 hour.", "security"),
    ("Code freeze is on Friday at 17:00 before a release.", "engineering"),
]


def demo_tiny_db() -> None:
    from ollama_api import banner, embed

    banner("Part 1: TinyVectorDB with real embeddings")
    texts = [t for t, _ in FACTS]
    db = TinyVectorDB(dim=768)
    db.add([f"fact-{n}" for n in range(len(FACTS))], embed(["search_document: " + t for t in texts]),
           texts, [{"topic": topic} for _, topic in FACTS])
    print(f"  stored {len(db)} vectors, matrix shape {db.vectors.shape}, {db.vectors.nbytes / 1024:.0f} KiB")

    for question, where in [("How heavy a load can the big robot lift?", None),
                            ("What should I do if my computer is stolen?", None),
                            ("How much time off do I get?", {"topic": "hr"}),
                            ("How much time off do I get?", {"topic": "product"})]:
        hits = db.query(embed(["search_query: " + question])[0], k=2, where=where)
        print(f"\n  Q: {question}" + (f"   where={where}" if where else ""))
        for h in hits:
            print(f"     {h['score']:.3f}  [{h['meta']['topic']}] {h['doc']}")
    print("\n  Q1 is a miss: 'big robot' scored the 12 kg Wren 0.003 above the 45 kg Kestrel. The")
    print("  embedding doesn't know Kestrel is the heavy-duty model. Phase 3 (hybrid + rerank) targets this.")
    print("  The last query shows a metadata filter's sharp edge: restricting to 'product' still")
    print("  returns the *closest* product facts, however irrelevant. Top-k always returns k things.")

    db.add(["fact-0"], embed(["search_document: The Kestrel K2 robot carries up to 50 kg."]),
           ["The Kestrel K2 robot carries up to 50 kg."], [{"topic": "product"}])
    print(f"\n  upsert fact-0 -> still {len(db)} records; fact-0 is now: {db.docs[db.ids.index('fact-0')]!r}")


# --------------------------------------------------------------------------- part 2

def clustered_data(n: int, dim: int, n_topics: int, rng: np.random.Generator) -> np.ndarray:
    """Synthetic 'embeddings': real text embeddings cluster by topic, so random noise
    around topic centres is a far better stand-in than uniform random vectors."""
    centres = rng.standard_normal((n_topics, dim)).astype(np.float32)
    x = centres[rng.integers(0, n_topics, n)] + 0.6 * rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def kmeans(x: np.ndarray, k: int, iters: int, rng: np.random.Generator) -> np.ndarray:
    centroids = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        assign = np.argmax(x @ centroids.T, axis=1)          # nearest centroid (cosine)
        for c in range(k):
            members = x[assign == c]
            if len(members):
                centroids[c] = members.mean(0) / np.linalg.norm(members.mean(0))
    return centroids


class IVFIndex:
    """Inverted file index: the simplest approximate-nearest-neighbour (ANN) structure.
    Build: cluster vectors into `nlist` buckets. Query: only scan the `nprobe` nearest buckets."""

    def __init__(self, x: np.ndarray, nlist: int, rng: np.random.Generator):
        self.x = x
        self.centroids = kmeans(x, nlist, iters=8, rng=rng)
        assign = np.argmax(x @ self.centroids.T, axis=1)
        self.buckets = [np.where(assign == c)[0] for c in range(nlist)]

    def search(self, q: np.ndarray, k: int, nprobe: int) -> np.ndarray:
        nearest = np.argsort(-(self.centroids @ q))[:nprobe]
        cand = np.concatenate([self.buckets[c] for c in nearest])
        scores = self.x[cand] @ q
        return cand[np.argsort(-scores)[:k]]


def brute_force(x: np.ndarray, q: np.ndarray, k: int) -> np.ndarray:
    scores = x @ q
    top = np.argpartition(-scores, k)[:k]
    return top[np.argsort(-scores[top])]


def timed(fn, queries) -> tuple[list, float]:
    start = time.perf_counter()
    out = [fn(q) for q in queries]
    return out, (time.perf_counter() - start) / len(queries) * 1000


def demo_scaling() -> None:
    from ollama_api import banner

    banner("Part 2a: brute force cost grows linearly with N")
    rng = np.random.default_rng(0)
    dim, k = 768, 10
    for n in (1_000, 10_000, 100_000):
        x = clustered_data(n, dim, n_topics=5000, rng=rng)
        queries = x[rng.choice(n, 200, replace=False)] + 0.05 * rng.standard_normal((200, dim)).astype(np.float32)
        timed(lambda q: brute_force(x, q, k), queries[:20])  # warm-up: CPU caches, numpy/BLAS init
        _, ms = timed(lambda q: brute_force(x, q, k), queries)
        print(f"  N={n:>7,}  matrix {x.nbytes / 1e6:6.1f} MB  {ms:7.2f} ms/query")
    print("  10× the data = 10× the time. At 10M chunks that's seconds per query, and the")
    print("  vectors alone would need 30 GB of RAM.")

    banner("Part 2b: IVF index on N=100,000: trade recall for speed")
    t0 = time.perf_counter()
    ivf = IVFIndex(x, nlist=256, rng=rng)
    print(f"  built 256 clusters in {time.perf_counter() - t0:.1f}s (one-off cost at ingest time)\n")
    truth, brute_ms = timed(lambda q: brute_force(x, q, k), queries)
    print(f"  {'method':<22} {'ms/query':>9} {'recall@10':>10} {'scanned':>9}")
    print(f"  {'brute force':<22} {brute_ms:>9.2f} {1.0:>10.3f} {'100%':>9}")
    for nprobe in (1, 4, 16, 64):
        found, ms = timed(lambda q: ivf.search(q, k, nprobe), queries)
        recall = np.mean([len(set(f) & set(t)) / k for f, t in zip(found, truth)])
        print(f"  {'IVF nprobe=' + str(nprobe):<22} {ms:>9.2f} {recall:>10.3f} {nprobe / 256:>8.1%}")
    print("""
  recall@10 = fraction of the TRUE top-10 the index actually returned.
  Scanning fewer clusters is faster but can miss neighbours that sit just across a
  cluster boundary. Every ANN index exposes a knob like nprobe for this trade-off.
  (At high nprobe our Python IVF is slower than brute force: gathering scattered rows
  costs more than one contiguous matmul. Real indexes are C++/SIMD and avoid this.)

  Chroma (Phase 2) uses HNSW instead of IVF: a multi-layer graph where each vector
  links to its nearest neighbours. A query starts at the sparse top layer, greedily
  hops toward the target, then descends to denser layers. Cost ~O(log N) hops.
  Its knobs: M (links per node: memory vs recall) and ef_search (candidates kept
  while searching: latency vs recall).""")


if __name__ == "__main__":
    demo_tiny_db()
    demo_scaling()
