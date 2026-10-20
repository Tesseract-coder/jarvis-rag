"""Four ways to pick the chunks an answer is built from.

    similarity  vector top-k (cosine)                          what Phase 2 did
    mmr         vector top-fetch_k, then pick k that are relevant AND unlike each other
    hybrid      BM25 top-fetch_k + vector top-fetch_k, merged by reciprocal rank fusion
    rerank      hybrid candidates re-scored by a cross-encoder that reads query+chunk together

    python -m jarvis_rag.retriever "what does E-217 mean?" --mode rerank

LangChain v1 dropped `langchain.retrievers`: EnsembleRetriever survives only in
the legacy langchain-classic package and BM25Retriever in the sunset
langchain-community. Both algorithms are short, so they live here in plain
sight (EnsembleRetriever is exactly the weighted RRF in `rrf_fuse`).
"""

import argparse
import hashlib
import logging
import math
import re
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence

from langchain_core.documents import Document
from py_rust_stemmers import SnowballStemmer

from jarvis_rag.config import Settings, get_settings
from jarvis_rag.vectorstore import KnowledgeBase

Hits = list[tuple[Document, float]]
log = logging.getLogger(__name__)

STOPWORDS = set("""a an and are as at be by can do does for from has have how i if in is it its
of on or our should that the their them they this to was we what when where which who why will
with you your""".split())
_stem = SnowballStemmer("english").stem_word


def tokenize(text: str) -> list[str]:
    """Lowercase words, stemmed; compound tokens like `e-217` or `get_settings` are
    kept whole AND split, so "E-217" matches exactly while "217" still matches."""
    out = []
    for tok in re.findall(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*", text.lower()):
        parts = re.split(r"[-_.]", tok)
        if len(parts) > 1:
            out.append(tok)
        out += [_stem(p) for p in parts if p not in STOPWORDS]
    return out


class BM25:
    """Okapi BM25: the classic keyword score.

        score(q, d) = Σ_t∈q  idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
        idf(t)      = ln(1 + (N − df + 0.5) / (df + 0.5))

    Rare terms (high idf) dominate; tf saturates (k1) so repeating a word 50
    times doesn't win; long chunks are normalised (b)."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.lengths = []
        for i, text in enumerate(texts):
            tokens = tokenize(text)
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term][i] = tf
        n = len(self.lengths)
        self.avgdl = sum(self.lengths) / n if n else 0.0
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()}

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, {}).items():
                norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avgdl)
                out[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out


def matches(meta: dict, where: dict | None) -> bool:
    """The subset of Chroma's filter language we use, for filtering BM25 results
    the same way Chroma filters vector results."""
    if not where:
        return True
    if "$and" in where:
        return all(matches(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(matches(meta, w) for w in where["$or"])
    for field, cond in where.items():
        value = meta.get(field)
        if isinstance(cond, dict):
            if "$in" in cond and value not in cond["$in"]:
                return False
            if "$eq" in cond and value != cond["$eq"]:
                return False
            if "$ne" in cond and value == cond["$ne"]:
                return False
        elif value != cond:
            return False
    return True


def rrf_fuse(rankings: list[list[str]], weights: list[float], c: int = 60) -> dict[str, float]:
    """Weighted reciprocal rank fusion: score(d) = Σ w_i / (c + rank_i(d)).
    Uses only ranks, so BM25's unbounded scores and cosine's 0..1 never need
    to be put on the same scale. c=60 damps the gap between rank 1 and rank 2."""
    fused: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights):
        for rank, key in enumerate(ranking, 1):
            fused[key] += w / (c + rank)
    return dict(sorted(fused.items(), key=lambda kv: -kv[1]))


class CrossEncoderReranker:
    """FastEmbed ONNX cross-encoder. A bi-encoder (our embedder) compresses query and
    chunk separately into one vector each; a cross-encoder reads them *together*,
    so it can see that "longer battery" is answered by two numbers. ~100x slower
    per pair, hence only used on the top fetch_k candidates."""

    def __init__(self, model_name: str, cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder  # lazy: ~1 s import

        self.model = TextCrossEncoder(model_name, cache_dir=cache_dir)

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        # batch_size=1: a batch is padded to its longest pair, so one 1,100-char code
        # chunk made 19 short ones pay for 512 tokens each. Measured on 20 candidates
        # (Ryzen 5600H): batch 64 = 2,342 ms, batch 4 = 965 ms, batch 1 = 281 ms.
        return [float(s) for s in self.model.rerank(query, texts, batch_size=1)]


class Retriever:
    def __init__(self, kb: KnowledgeBase, settings: Settings | None = None, mode: str | None = None,
                 reranker: Callable[[str, list[str]], list[float]] | None = None):
        self.kb, self.s = kb, settings or get_settings()
        self.mode = mode or self.s.retrieval_mode
        self._reranker = reranker
        self._bm25: BM25 | None = None
        self._reranker_failed = False
        self._signature = None

    # ---- the lexical side: rebuilt whenever the collection's ids change
    def _corpus(self) -> tuple[BM25, list[str], list[Document]]:
        ids = self.kb.collection.get(include=[])["ids"]
        sig = hashlib.sha256("".join(sorted(ids)).encode()).hexdigest()
        if sig != self._signature:
            got = self.kb.collection.get(include=["documents", "metadatas"])
            self._ids = got["ids"]
            self._docs = [Document(t, metadata=m, id=i) for i, t, m in zip(got["ids"], got["documents"], got["metadatas"])]
            self._bm25 = BM25(got["documents"])
            self._signature = sig
        return self._bm25, self._ids, self._docs

    def bm25(self, query: str, k: int, where: dict | None = None) -> Hits:
        index, _, docs = self._corpus()
        ranked = sorted(index.scores(query).items(), key=lambda kv: -kv[1])
        return [(docs[i], s) for i, s in ranked if matches(docs[i].metadata, where)][:k]

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoderReranker(self.s.reranker_model, str(self.s.models_dir))
        return self._reranker

    # ---- the four modes
    def similarity(self, query: str, k: int, where: dict | None = None) -> Hits:
        return self.kb.search(query, k, where)

    def mmr(self, query: str, k: int, where: dict | None = None) -> Hits:
        docs = self.kb.store.max_marginal_relevance_search(query, k=k, fetch_k=self.s.fetch_k,
                                                           lambda_mult=self.s.mmr_lambda, filter=where)
        return [(d, 1 - n / max(len(docs), 1)) for n, d in enumerate(docs)]  # MMR gives order, not scores

    def hybrid(self, query: str, k: int, where: dict | None = None) -> Hits:
        vec = self.kb.search(query, self.s.fetch_k, where)
        lex = self.bm25(query, self.s.fetch_k, where)
        by_id = {d.metadata["chunk_id"]: d for d, _ in vec + lex}
        fused = rrf_fuse([[d.metadata["chunk_id"] for d, _ in vec], [d.metadata["chunk_id"] for d, _ in lex]],
                         [1 - self.s.bm25_weight, self.s.bm25_weight], self.s.rrf_c)
        return [(by_id[cid], score) for cid, score in list(fused.items())[:k]]

    def rerank(self, query: str, k: int, where: dict | None = None) -> Hits:
        candidates = [d for d, _ in self.hybrid(query, self.s.fetch_k, where)]
        if not candidates:
            return []
        if self._reranker_failed:
            return self.hybrid(query, k, where)
        try:
            reranker = self.reranker
        except Exception as e:  # model not downloadable (offline, HF down): degrade, don't fail
            log.warning("reranker %s unavailable (%s); falling back to hybrid", self.s.reranker_model, e)
            self._reranker_failed = True
            return self.hybrid(query, k, where)
        scores = reranker(query, [d.page_content for d in candidates])
        return sorted(zip(candidates, scores), key=lambda ds: -ds[1])[:k]

    def retrieve(self, query: str, k: int | None = None, where: dict | None = None, mode: str | None = None) -> Hits:
        mode = mode or self.mode
        if mode not in ("similarity", "mmr", "hybrid", "rerank"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        return getattr(self, mode)(query, k or self.s.retrieval_k, where)


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieve chunks with a chosen strategy.")
    ap.add_argument("query")
    ap.add_argument("--mode", choices=["similarity", "mmr", "hybrid", "rerank", "all"], default="all")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--type", help="only chunks of this type")
    args = ap.parse_args()
    r = Retriever(KnowledgeBase())
    where = {"type": args.type} if args.type else None
    for mode in ["similarity", "mmr", "hybrid", "rerank"] if args.mode == "all" else [args.mode]:
        t0 = time.perf_counter()
        hits = r.retrieve(args.query, args.k, where, mode)
        print(f"\n{mode} ({(time.perf_counter() - t0) * 1000:.0f} ms)")
        for n, (doc, score) in enumerate(hits, 1):
            print(f"  [{n}] {score:8.4f}  {doc.page_content.splitlines()[0][:100]}")


if __name__ == "__main__":
    main()
