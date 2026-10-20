"""BM25, rank fusion, filters and the four retrieval modes, offline."""

import math

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from jarvis_rag import retriever as R
from jarvis_rag.config import Settings
from jarvis_rag.ingest import ingest_documents
from jarvis_rag.retriever import BM25, Retriever, matches, rrf_fuse, tokenize
from jarvis_rag.vectorstore import KnowledgeBase


def test_tokenize_keeps_codes_whole_and_split():
    toks = tokenize("Error E-217 on get_settings()")
    assert {"e-217", "217", "get_settings", "get", "set"} <= set(toks)  # "settings" is stemmed to "set"
    assert "on" not in toks  # stopword


def test_tokenize_stems():
    assert tokenize("batteries") == tokenize("battery")


def test_bm25_rare_terms_win_and_tf_saturates():
    docs = ["robot robot robot robot robot robot", "robot battery", "office lunch menu"]
    bm = BM25(docs)
    s = bm.scores("robot battery")
    assert max(s, key=s.get) == 1  # 'battery' is rare, so doc 1 wins over 6x 'robot'
    assert 2 not in s
    one, many = BM25(["robot x x x x", "robot robot robot robot robot"]).scores("robot").values()
    assert many < one * 2.5  # saturation: 5x the term is nowhere near 5x the score


def test_bm25_idf_formula():
    bm = BM25(["cat dog", "cat fish", "bird owl"])
    assert bm.idf["cat"] == pytest.approx(math.log(1 + (3 - 2 + 0.5) / (2 + 0.5)))
    assert bm.scores("") == {}


@pytest.mark.parametrize("where,expected", [
    (None, True), ({"type": "pdf"}, True), ({"type": "code"}, False),
    ({"type": {"$in": ["code", "pdf"]}}, True), ({"type": {"$ne": "pdf"}}, False),
    ({"$and": [{"type": "pdf"}, {"page": 2}]}, True), ({"$and": [{"type": "pdf"}, {"page": 3}]}, False),
    ({"$or": [{"type": "code"}, {"page": 2}]}, True),
])
def test_filter_matcher(where, expected):
    assert matches({"type": "pdf", "page": 2}, where) is expected


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "d", "a"]], [0.5, 0.5], c=60)
    assert list(fused)[:2] == ["b", "a"]  # in both lists beats first-in-one
    assert fused["b"] == pytest.approx(0.5 / 62 + 0.5 / 61)
    only_vec = rrf_fuse([["a"], ["b"]], [1.0, 0.0])
    assert list(only_vec) == ["a", "b"] and only_vec["b"] == 0


# --------------------------------------------------------------------------- modes

CORPUS = {
    "specs.md": "# Specs\nRobots.\n\n## Kestrel K2\n- Battery life: 6.5 hours\n\n## Wren W1\n- Battery life: 9 hours\n",
    "errors.md": "# Errors\n## Codes\nE-217: wheel motor overcurrent. Power-cycle the robot.\n",
    "faq.md": "# FAQ\n## Lunch\nLunch costs 4.50 per meal in the cafeteria.\n",
}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None, storage_dir=tmp_path / "st", collection_name="retr", fetch_k=10)
    kb = KnowledgeBase(s, embeddings=DeterministicFakeEmbedding(size=32))
    docs = [Document(t, metadata={"source": n, "type": "text", "title": n, "path": n}) for n, t in CORPUS.items()]
    ingest_documents(docs, kb, s)
    return kb, s


def test_bm25_mode_finds_exact_code_that_random_vectors_cannot(setup):
    kb, s = setup
    r = Retriever(kb, s)
    (doc, _), *_ = r.bm25("what is E-217?", k=3)
    assert doc.metadata["source"] == "errors.md"
    hits = r.retrieve("what is E-217?", k=1, mode="hybrid")
    assert hits[0][0].metadata["source"] == "errors.md"  # fake vectors are noise; BM25 carries it


def test_bm25_respects_filters(setup):
    kb, s = setup
    hits = Retriever(kb, s).bm25("battery lunch", k=5, where={"source": "faq.md"})
    assert hits and {d.metadata["source"] for d, _ in hits} == {"faq.md"}


def test_hybrid_dedupes_and_limits(setup):
    kb, s = setup
    hits = Retriever(kb, s).retrieve("battery", k=3, mode="hybrid")
    ids = [d.metadata["chunk_id"] for d, _ in hits]
    assert len(ids) == len(set(ids)) == 3
    assert [sc for _, sc in hits] == sorted([sc for _, sc in hits], reverse=True)


def test_rerank_uses_cross_encoder_order(setup):
    kb, s = setup
    calls = []

    def fake_reranker(query, texts):
        calls.append(len(texts))
        return [1.0 if "9 hours" in t else 0.0 for t in texts]

    hits = Retriever(kb, s, reranker=fake_reranker).retrieve("anything", k=2, mode="rerank")
    assert "9 hours" in hits[0][0].page_content and hits[0][1] == 1.0
    assert calls == [kb.count()]  # every candidate scored once (fetch_k >= corpus here)


def test_rerank_falls_back_to_hybrid(setup, monkeypatch):
    kb, s = setup

    class Broken:
        def __init__(self, *a, **kw):
            raise OSError("offline")

    monkeypatch.setattr(R, "CrossEncoderReranker", Broken)
    r = Retriever(kb, s)
    assert r.retrieve("E-217", k=2, mode="rerank") == r.retrieve("E-217", k=2, mode="hybrid")
    assert r._reranker_failed


def test_mmr_and_similarity_return_k(setup):
    kb, s = setup
    r = Retriever(kb, s)
    assert len(r.retrieve("battery", k=2, mode="mmr")) == 2
    assert len(r.retrieve("battery", k=2, mode="similarity")) == 2
    with pytest.raises(ValueError):
        r.retrieve("x", mode="magic")


def test_bm25_index_follows_new_ingests(setup):
    kb, s = setup
    r = Retriever(kb, s)
    assert not r.bm25("parking basement", k=3)
    doc = Document("# Parking\nBike parking is in the basement.", metadata={"source": "p.md", "type": "text",
                                                                            "title": "Parking", "path": "p.md"})
    ingest_documents([doc], kb, s)
    assert r.bm25("parking basement", k=1)[0][0].metadata["source"] == "p.md"
