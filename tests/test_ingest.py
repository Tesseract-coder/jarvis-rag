"""Chunking, chunk ids, and the Chroma knowledge base, offline with fake embeddings."""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from jarvis_rag import demo_data
from jarvis_rag.config import Settings
from jarvis_rag.ingest import Chunker, collect, ingest_documents, prune
from jarvis_rag.vectorstore import KnowledgeBase, chunk_id


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # sources are relative to cwd
    return Settings(_env_file=None, storage_dir=tmp_path / "storage", data_dir=tmp_path / "data",
                    collection_name="test")


@pytest.fixture
def kb(settings):
    return KnowledgeBase(settings, embeddings=DeterministicFakeEmbedding(size=32))


def md_doc(text, source="specs.md", title="Specs"):
    return Document(text, metadata={"source": source, "type": "text", "title": title, "path": source})


SPECS = """# Specs
Two robots.

## Kestrel K2
- Payload: 45 kg

## Wren W1
### Battery
- Battery life: 9 hours
"""


def test_markdown_chunks_carry_their_section(settings):
    chunks = Chunker(settings).split(md_doc(SPECS))
    battery = next(c for c in chunks if "9 hours" in c.page_content)
    assert battery.page_content.startswith("[Specs > Wren W1 > Battery]\n")  # title not repeated
    assert battery.metadata["section"] == "Wren W1 > Battery"


def test_chunk_size_and_ids(settings):
    s = settings.model_copy(update={"chunk_size": 200, "chunk_overlap": 30})
    text = "# T\n\n" + " ".join(f"Sentence {n} about robots." for n in range(100))
    chunks = Chunker(s).split(md_doc(text, title="T"))
    assert len(chunks) > 5
    for c in chunks:
        header, body = c.page_content.split("\n", 1)
        assert header == "[T]" and len(body) <= 200
        assert c.metadata["chunk_id"] == chunk_id("specs.md", c.page_content)
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_chunk_id_depends_on_source_and_content():
    assert chunk_id("a.md", "x") == chunk_id("a.md", "x")
    assert chunk_id("a.md", "x") != chunk_id("b.md", "x") != chunk_id("a.md", "y")
    assert len(chunk_id("a", "b")) == 64  # sha256 hex


def test_code_chunks_name_their_symbols_and_line(settings):
    s = settings.model_copy(update={"code_chunk_size": 300, "code_chunk_overlap": 0})
    funcs = "\n\n".join(f"def func_{n}(x):\n    '''Doc {n}.'''\n    return x * {n} + {n} * {n}\n" for n in range(12))
    doc = Document(funcs, metadata={"source": "m.py", "type": "code", "title": "m.py", "path": "m.py",
                                    "language": "python"})
    chunks = Chunker(s).split(doc)
    assert len(chunks) > 2
    for c in chunks:
        assert c.page_content.startswith("[m.py (python): func_")
        first = c.metadata["symbols"].split(", ")[0]
        assert funcs.splitlines()[c.metadata["line"] - 1].startswith(f"def {first}(")
        assert len(c.page_content.split("\n", 1)[1]) <= 300


def test_pdf_chunks_keep_page(settings):
    doc = Document("Some text on page two.", metadata={"source": "r.pdf", "type": "pdf", "title": "Report",
                                                         "path": "r.pdf", "page": 2})
    (c,) = Chunker(settings).split(doc)
    assert c.page_content == "[Report, page 2]\nSome text on page two." and c.metadata["page"] == 2


# --------------------------------------------------------------------------- knowledge base

def test_upsert_is_idempotent_and_incremental(kb, settings):
    v1 = SPECS
    r = ingest_documents([md_doc(v1)], kb, settings)[0]
    assert (r.chunks, r.added, r.removed) == (3, 3, 0)
    r = ingest_documents([md_doc(v1)], kb, settings)[0]
    assert (r.added, r.removed, r.unchanged) == (0, 0, 3)  # nothing re-embedded

    v2 = v1.replace("9 hours", "10 hours")  # edit one section
    r = ingest_documents([md_doc(v2)], kb, settings)[0]
    assert (r.added, r.removed, r.unchanged) == (1, 1, 2)
    assert kb.count() == 3
    texts = kb.collection.get(include=["documents"])["documents"]
    assert any("10 hours" in t for t in texts) and not any("9 hours" in t for t in texts)


def test_metadata_is_complete_on_every_chunk(kb, settings):
    ingest_documents([md_doc(SPECS)], kb, settings)
    for m in kb.collection.get(include=["metadatas"])["metadatas"]:
        assert {"source", "type", "path", "ingested_at", "chunk_id", "title"} <= m.keys()


def test_duplicate_chunks_within_a_source_collapse(kb):
    c = Document("[x]\nsame", metadata={"source": "d.md", "type": "text", "chunk_id": chunk_id("d.md", "[x]\nsame")})
    assert kb.upsert_source("d.md", [c, c]).chunks == 1 and kb.count() == 1


def test_delete_sources_and_filters(kb, settings):
    ingest_documents([md_doc(SPECS), md_doc("# FAQ\nParking is free.", "faq.md", "FAQ")], kb, settings)
    assert set(kb.sources()) == {"specs.md", "faq.md"}
    hits = kb.search("parking", k=10, where={"source": "faq.md"})
    assert hits and all(d.metadata["source"] == "faq.md" for d, _ in hits)
    assert all(-1.01 <= score <= 1.01 for _, score in kb.search("parking", k=10))
    assert kb.delete_source("specs.md") == 3
    assert set(kb.sources()) == {"faq.md"}


def test_collection_uses_cosine(kb):
    assert kb.collection.configuration["hnsw"]["space"] == "cosine"


def test_persistence_across_instances(kb, settings):
    ingest_documents([md_doc(SPECS)], kb, settings)
    again = KnowledgeBase(settings, embeddings=DeterministicFakeEmbedding(size=32))
    assert again.count() == 3


def test_end_to_end_demo_corpus_and_prune(kb, settings):
    files = demo_data.write_demo_data(settings.data_dir)
    assert files
    docs, skipped = collect([settings.data_dir / "docs", settings.data_dir / "code"], [],
                            [settings.data_dir / "csv", settings.data_dir / "nope"], 0, settings)
    assert skipped == [(str(settings.data_dir / "nope"), "does not exist")]
    ingest_documents(docs, kb, settings)
    types = {info["type"] for info in kb.sources().values()}
    assert types == {"text", "pdf", "docx", "code", "table"}
    assert settings.sqlite_path.exists()

    (settings.data_dir / "docs" / "office_faq.md").unlink()
    gone = prune(kb, [settings.data_dir / "docs"])
    assert gone == ["data/docs/office_faq.md"]
    assert "data/docs/office_faq.md" not in kb.sources()
