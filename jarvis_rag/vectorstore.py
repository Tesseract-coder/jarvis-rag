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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from jarvis_rag.config import Settings, get_settings
from jarvis_rag.llm import get_embeddings


def chunk_id(source: str, content: str) -> str:
    return hashlib.sha256(f"{source}\n{content}".encode()).hexdigest()


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
