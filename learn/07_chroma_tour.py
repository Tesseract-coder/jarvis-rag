"""07: What Chroma is doing for us: collections, distance metrics, persistence, filters.

Run after `make demo-data ingest`:  python learn/07_chroma_tour.py

Compare with learn/03: TinyVectorDB was a numpy matrix + a list of dicts.
Chroma is the same idea plus an HNSW index, a SQLite file for documents and
metadata, and a filter language. This script pokes at each piece.
"""

import sqlite3
import sys
import tempfile
import time

import chromadb
import numpy as np

from ollama_api import banner, embed  # also puts the repo root on sys.path

from jarvis_rag.config import get_settings
from jarvis_rag.vectorstore import KnowledgeBase

S = get_settings()
NO_TELEMETRY = chromadb.Settings(anonymized_telemetry=False)


def collections_and_metrics() -> None:
    banner("1. One client, many collections; each collection fixes its distance metric")
    sentences = ["The Wren W1 battery lasts 9 hours.", "Kestrel K2 carries 45 kg.",
                 "Annual leave is 25 days.", "Passwords need at least 14 characters."]
    vecs = embed(["search_document: " + t for t in sentences])
    q = embed(["search_query: How long does the small robot run on a charge?"])[0]
    print(f"  nomic vectors are unit length: norms = {np.round(np.linalg.norm(vecs, axis=1), 4)}")
    client = chromadb.EphemeralClient(settings=NO_TELEMETRY)  # in-memory, gone at exit
    for space in ("cosine", "l2", "ip"):
        col = client.create_collection(f"demo_{space}", configuration={"hnsw": {"space": space}})
        col.add(ids=[str(i) for i in range(len(sentences))], embeddings=vecs, documents=sentences)
        r = col.query(query_embeddings=[q], n_results=4)
        ranks = ", ".join(f"{d[:18]}… {dist:.3f}" for d, dist in zip(r["documents"][0], r["distances"][0]))
        print(f"  {space:>6}: {ranks}")
    print("""
  Same order three times. For unit vectors: cosine distance = 1 - cos,
  ip distance = 1 - dot = 1 - cos, and squared l2 = 2 - 2·cos. They only differ
  when vectors are NOT normalised; then l2 also punishes length, which is
  rarely what you want for text. We use cosine so swapping in a model that
  doesn't normalise (Phase 7) can't silently change rankings.""")


def persistence() -> None:
    banner("2. Persistence: it's just files")
    root = S.chroma_dir
    for p in sorted(root.rglob("*")):
        if p.is_file():
            print(f"  {p}  {p.stat().st_size / 1024:,.0f} KiB")
    with sqlite3.connect(root / "chroma.sqlite3") as db:
        tables = [r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        print(f"\n  chroma.sqlite3: {len(tables)} tables, including a SQLite FTS5 full-text index"
              f" ({'embedding_fulltext_search' in tables}) that powers where_document filters")
        n = db.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        ops = db.execute("SELECT COUNT(*) FROM embeddings_queue").fetchone()[0]
        print(f"  `embeddings` (ids, documents; metadata in `embedding_metadata`): {n} rows")
        print(f"  `embeddings_queue` (write-ahead log of every add/update/delete, vectors included): {ops} ops")
    print("""  Writes land in the SQLite log first (durable immediately); the HNSW .bin files
  are the search index built from it, flushed every `sync_threshold` writes. A
  crash loses nothing: the index is rebuilt from the log on the next open.""")

    t0 = time.perf_counter()
    kb = KnowledgeBase(S)  # a fresh client re-opens the same directory
    print(f"\n  reopened in {(time.perf_counter() - t0) * 1000:.0f} ms -> {kb.count()} chunks, nothing re-embedded")
    print(f"  collection config: {kb.collection.configuration['hnsw']}")
    print("""  ef_search / max_neighbors are the HNSW knobs from learn/03: higher = better
  recall, slower queries. At a few thousand chunks the defaults are near-exact.""")


def filters(kb: KnowledgeBase) -> None:
    banner("3. Metadata filters: narrow the search BEFORE ranking")
    q = "how do I fix error E-217?"
    cases = [
        ("no filter", None),
        ('{"type": "docx"}', {"type": "docx"}),
        ('{"type": {"$in": ["pdf", "docx"]}}', {"type": {"$in": ["pdf", "docx"]}}),
        ('{"$and": [{"type": "code"}, {"language": "python"}]}', {"$and": [{"type": "code"}, {"language": "python"}]}),
    ]
    for label, where in cases:
        hits = kb.search(q, k=3, where=where)
        print(f"\n  where={label}")
        for doc, score in hits:
            print(f"    {score:.3f}  {doc.page_content.splitlines()[0][:90]}")
    r = kb.collection.get(where_document={"$contains": "E-217"}, include=["metadatas"])
    print(f"\n  where_document={{'$contains': 'E-217'}} (exact text match, no vectors): "
          f"{sorted({m['source'] for m in r['metadatas']})}")
    print("""
  Filters run inside the index search, so you still get k results (when k
  exist) rather than "top-k, then throw most away". The agent's
  search_knowledge_base(source_type=...) in Phase 4 is exactly this.""")


def isolation_demo() -> None:
    banner("4. Collections are isolated namespaces")
    with tempfile.TemporaryDirectory() as tmp:
        client = chromadb.PersistentClient(path=tmp, settings=NO_TELEMETRY)
        a = client.create_collection("team_a")
        b = client.create_collection("team_b")
        a.add(ids=["1"], embeddings=[[1.0, 0.0]], documents=["only in A"])
        print(f"  team_a: {a.count()} doc, team_b: {b.count()} docs -> {[c.name for c in client.list_collections()]}")
        print("  One collection per embedding model (or tenant) keeps vector spaces from mixing.")
        print(f"  Ours is COLLECTION_NAME={S.collection_name!r}; changing EMBED_MODEL means a new collection + re-ingest.")


def main() -> None:
    kb = KnowledgeBase(S)
    if kb.count() == 0:
        sys.exit("Knowledge base is empty: run `make demo-data ingest` first.")
    collections_and_metrics()
    persistence()
    filters(kb)
    isolation_demo()


if __name__ == "__main__":
    main()
