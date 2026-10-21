"""Chroma wrapper: one persistent collection, cosine distance, and per-source
bookkeeping so re-ingesting is cheap and never duplicates.

    chunk id = sha256(source + content)
    same file again   -> every id already stored  -> nothing embedded (metadata refreshed)
    file edited       -> only new ids embedded, ids that vanished are deleted
    file removed      -> delete_source(source)

Search from the shell:
    python -m jarvis_rag.vectorstore "how do I reset E-217?" -k 3 --type pdf
"""

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from jarvis_rag.config import Settings, get_settings
from jarvis_rag.llm import get_embeddings


FINGERPRINT_KEY = "embed_fingerprint"
PROBE = "JARVIS embedding fingerprint: the Kestrel K2 battery lasts 6.5 hours."
SAME_SPACE = 0.99  # Ollama vs FastEmbed nomic-v1.5: 1.0000; vs its quantized -Q variant: ~0.95


class EmbeddingMismatch(RuntimeError):
    """The collection's vectors came from a different embedding model than the one configured."""


def chunk_id(source: str, content: str) -> str:
    return hashlib.sha256(f"{source}\n{content}".encode()).hexdigest()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) or 1.0)


@dataclass
class UpsertResult:
    source: str
    chunks: int
    added: int
    removed: int

    @property
    def unchanged(self) -> int:
        return self.chunks - self.added


class KnowledgeBase:
    def __init__(self, settings: Settings | None = None, embeddings: Embeddings | None = None,
                 persist_dir: Path | None = None):
        s = settings or get_settings()
        persist_dir = Path(persist_dir or s.chroma_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(persist_dir),
                                           settings=chromadb.Settings(anonymized_telemetry=False))
        self.store = Chroma(
            client=client,
            collection_name=s.collection_name,
            embedding_function=embeddings or get_embeddings(s),
            # Chroma defaults to squared L2. Cosine makes scores comparable across
            # models whether or not they normalise their vectors.
            collection_configuration={"hnsw": {"space": "cosine"}},
        )
        self.collection = self.store._collection  # raw chromadb collection for metadata-only calls
        self._fingerprint_ok = False

    def check_embeddings(self) -> None:
        """Refuse to mix vector spaces. Vectors from two embedding models are not
        comparable, and nothing else would notice: same dimension, valid-looking
        scores, silently wrong results. So each collection stores the vector of a
        fixed probe sentence, and every embedder must reproduce it before use.
        Runs lazily (first search/ingest), so opening a collection never needs a model."""
        if self._fingerprint_ok:
            return
        probe = self.store.embeddings.embed_documents([PROBE])[0]
        meta = dict(self.collection.metadata or {})
        if FINGERPRINT_KEY not in meta:  # new collection (or one from before Phase 7): adopt this embedder
            self.collection.modify(metadata=meta | {FINGERPRINT_KEY: json.dumps([round(x, 6) for x in probe])})
        else:
            ref = json.loads(meta[FINGERPRINT_KEY])
            sim = cosine(ref, probe) if len(ref) == len(probe) else 0.0
            if sim < SAME_SPACE:
                raise EmbeddingMismatch(
                    f"collection {self.collection.name!r} was built with a different embedding model "
                    f"(dim {len(ref)} vs {len(probe)}, probe similarity {sim:.3f}). Switch EMBED_PROVIDER/"
                    f"EMBED_MODEL/FASTEMBED_MODEL back, or set a new COLLECTION_NAME and re-ingest.")
        self._fingerprint_ok = True

    def count(self) -> int:
        return self.collection.count()

    def ids_for(self, source: str) -> set[str]:
        return set(self.collection.get(where={"source": source}, include=[])["ids"])

    def upsert_source(self, source: str, chunks: list[Document]) -> UpsertResult:
        """Make the stored chunks for `source` exactly `chunks`, embedding only what's new."""
        new: dict[str, Document] = {}
        for c in chunks:
            cid = c.metadata["chunk_id"]
            new.setdefault(cid, c)  # identical chunks within one source collapse to one
        old = self.ids_for(source)
        stale = list(old - new.keys())
        fresh = [cid for cid in new if cid not in old]
        kept = [cid for cid in new if cid in old]
        if stale:
            self.collection.delete(ids=stale)
        if kept:  # same text, so no re-embedding; refresh metadata (line numbers, ingested_at)
            self.collection.update(ids=kept, metadatas=[new[cid].metadata for cid in kept])
        if fresh:
            self.check_embeddings()
            self.store.add_documents([new[cid] for cid in fresh], ids=fresh)
        return UpsertResult(source, len(new), len(fresh), len(stale))

    def delete_source(self, source: str) -> int:
        ids = list(self.ids_for(source))
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)

    def sources(self) -> dict[str, dict]:
        """source -> {type, chunks, ingested_at} for everything in the collection."""
        metas = self.collection.get(include=["metadatas"])["metadatas"]
        counts = Counter(m["source"] for m in metas)
        out = {}
        for m in metas:
            info = out.setdefault(m["source"], {"type": m.get("type"), "chunks": counts[m["source"]],
                                                "ingested_at": m.get("ingested_at", "")})
            info["ingested_at"] = max(info["ingested_at"], m.get("ingested_at", ""))
        return dict(sorted(out.items()))

    def search(self, query: str, k: int = 5, where: dict | None = None) -> list[tuple[Document, float]]:
        """Top-k chunks with cosine *similarity* (1 = identical direction, 0 = unrelated)."""
        self.check_embeddings()
        hits = self.store.similarity_search_with_score(query, k=k, filter=where)
        return [(doc, 1 - distance) for doc, distance in hits]  # Chroma returns cosine distance


def main() -> None:
    ap = argparse.ArgumentParser(description="Search the knowledge base.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--type", help="only chunks of this type: pdf, docx, text, code, web, table")
    ap.add_argument("--source", help="only chunks from this source")
    args = ap.parse_args()
    filters = [{f: v} for f, v in (("type", args.type), ("source", args.source)) if v]
    where = filters[0] if len(filters) == 1 else ({"$and": filters} if filters else None)

    kb = KnowledgeBase()
    for n, (doc, score) in enumerate(kb.search(args.query, args.k, where), 1):
        m = doc.metadata
        loc = f"p.{m['page']}" if "page" in m else m.get("symbols", "")
        print(f"\n[{n}] {score:.3f}  {m['source']} {loc}  ({m['type']})")
        print("    " + doc.page_content[:400].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
