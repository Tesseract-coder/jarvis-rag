"""Offline tests for the pure parts of the learn/ scripts (no Ollama calls)."""

import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "learn"))

emb = importlib.import_module("02_embeddings_cosine")
vdb = importlib.import_module("03_vector_db_from_scratch")
rag = importlib.import_module("04_rag_without_framework")
agent = importlib.import_module("05_agent_loop_from_scratch")
TinyVectorDB = vdb.TinyVectorDB

PROSE = " ".join(f"Sentence number {n} talks about robots and warehouses." for n in range(60))


# ---------------------------------------------------------------- 02 cosine

def test_cosine_by_hand_matches_numpy():
    a, b = [0.3, -1.2, 2.0], [1.0, 0.5, -0.4]
    m = emb.cosine_matrix(np.array([a, b]))
    assert emb.cosine_by_hand(a, b) == pytest.approx(m[0, 1])
    assert np.allclose(np.diag(m), 1.0)


# ---------------------------------------------------------------- 03 vector db

def make_db() -> TinyVectorDB:
    db = TinyVectorDB(dim=3)
    db.add(["x", "y", "z"], np.array([[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]]), ["doc x", "doc y", "doc z"],
           [{"t": "a"}, {"t": "b"}, {"t": "a"}])
    return db


def test_query_ranks_by_cosine():
    hits = make_db().query(np.array([1, 0, 0]), k=2)
    assert [h["id"] for h in hits] == ["x", "z"]
    assert hits[0]["score"] == pytest.approx(1.0)


def test_vectors_are_stored_normalised():
    db = TinyVectorDB(dim=2)
    db.add(["a"], np.array([[3.0, 4.0]]), ["a"])
    assert np.linalg.norm(db.vectors[0]) == pytest.approx(1.0)


def test_upsert_replaces_instead_of_duplicating():
    db = make_db()
    db.add(["x"], np.array([[0, 0, 1]]), ["new x"])
    assert len(db) == 3
    assert db.query(np.array([0, 0, 1]), k=1)[0]["doc"] == "new x"


def test_delete_and_metadata_filter():
    db = make_db()
    assert [h["id"] for h in db.query(np.array([0, 1, 0]), k=3, where={"t": "a"})] == ["z", "x"]
    assert db.query(np.array([1, 0, 0]), where={"t": "missing"}) == []
    db.delete(["z"])
    assert db.ids == ["x", "y"] and len(db.vectors) == 2


def test_k_larger_than_db():
    assert len(make_db().query(np.array([1, 0, 0]), k=50)) == 3


def test_dim_mismatch_rejected():
    with pytest.raises(AssertionError):
        TinyVectorDB(dim=4).add(["a"], np.ones((1, 3)), ["a"])


def test_save_load_roundtrip(tmp_path):
    db = make_db()
    db.save(tmp_path / "db")
    loaded = TinyVectorDB.load(tmp_path / "db")
    assert loaded.ids == db.ids and loaded.metas == db.metas
    assert np.allclose(loaded.vectors, db.vectors)


def test_ivf_with_all_clusters_equals_brute_force():
    rng = np.random.default_rng(1)
    x = vdb.clustered_data(2000, 32, n_topics=20, rng=rng)
    ivf = vdb.IVFIndex(x, nlist=16, rng=rng)
    q = x[7]
    assert set(ivf.search(q, 10, nprobe=16)) == set(vdb.brute_force(x, q, 10))
    assert vdb.brute_force(x, q, 1)[0] == 7


# ---------------------------------------------------------------- 04 chunking + prompt

def test_short_text_is_one_chunk():
    assert rag.split_text("Just a short note.", 800, 120) == ["Just a short note."]


@pytest.mark.parametrize("size,overlap", [(100, 0), (200, 30), (400, 60)])
def test_chunks_respect_size_and_cover_text(size, overlap):
    chunks = rag.split_text(PROSE, size, overlap)
    assert all(len(c) <= size for c in chunks)
    assert " ".join(chunks).split()[-1] == PROSE.split()[-1]
    for n in range(60):  # every sentence survives intact in some chunk
        assert any(f"number {n} talks" in c for c in chunks)


def sentence_numbers(chunk: str) -> set[int]:
    return {int(n) for n in re.findall(r"number (\d+) talks", chunk)}


def test_overlap_repeats_text_between_neighbours():
    with_overlap = rag.split_text(PROSE, 200, 60)
    assert all(sentence_numbers(a) & sentence_numbers(b) for a, b in zip(with_overlap, with_overlap[1:]))
    without = rag.split_text(PROSE, 200, 0)
    assert not any(sentence_numbers(a) & sentence_numbers(b) for a, b in zip(without, without[1:]))


def test_paragraph_boundaries_preferred():
    text = "Alpha paragraph here.\n\nBeta paragraph here.\n\nGamma paragraph here."
    assert rag.split_text(text, 30, 0) == ["Alpha paragraph here.", "Beta paragraph here.", "Gamma paragraph here."]


def test_unbreakable_text_is_hard_cut():
    assert rag.split_text("x" * 250, 100, 0) == ["x" * 100, "x" * 100, "x" * 50]


def test_build_messages_numbers_passages():
    hits = [{"id": "a.md#0", "doc": "First."}, {"id": "b.md#3", "doc": "Second."}]
    system, user = rag.build_messages("Q?", hits)
    assert system["role"] == "system" and "I don't know" in system["content"]
    assert "[1] (source: a.md#0)\nFirst." in user["content"]
    assert "[2] (source: b.md#3)\nSecond." in user["content"]
    assert user["content"].endswith("Question: Q?")


def test_cited_sources_ignores_invented_numbers():
    hits = [{"id": "a#0"}, {"id": "b#1"}]
    assert rag.cited_sources("Yes [2][1], also [2] and [7].", hits) == ["a#0", "b#1"]
    assert rag.cited_sources("No citations.", hits) == []


# ---------------------------------------------------------------- 05 agent

@pytest.mark.parametrize("expr,expected", [("17 * 23", "391"), ("(25 - 7) / 2", "9.0"), ("-2 ** 3", "-8"),
                                           ("10 // 3 + 10 % 3", "4"), ("0.1 + 0.2", "0.3")])
def test_calculator(expr, expected):
    assert agent.calculator(expr) == expected


@pytest.mark.parametrize("evil", ["__import__('os').system('ls')", "open('x')", "a + 1", "(1).__class__", "9 ** 999999"])
def test_calculator_rejects_non_arithmetic(evil):
    with pytest.raises(ValueError):
        agent.calculator(evil)


def test_run_tool_turns_errors_into_results():
    assert agent.run_tool("nope", {}).startswith("Error: unknown tool")
    assert agent.run_tool("calculator", {"expression": "1/0"}).startswith("Error: ZeroDivisionError")
    assert agent.run_tool("calculator", {"wrong_arg": "1"}).startswith("Error: TypeError")


def tool_call(name, **args):
    return {"message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}}


def final(text):
    return {"message": {"role": "assistant", "content": text}}


def test_agent_loop_executes_tool_then_answers():
    script = iter([tool_call("calculator", expression="17*23"), final("17 × 23 = 391.")])
    seen = []

    def fake_chat(messages, tools, temperature):
        seen.append([m["role"] for m in messages])
        assert {t["function"]["name"] for t in tools} == set(agent.TOOLS)
        return next(script)

    answer, messages = agent.run_agent("What is 17*23?", chat_fn=fake_chat, verbose=False)
    assert answer == "17 × 23 = 391."
    assert seen == [["system", "user"], ["system", "user", "assistant", "tool"]]
    assert messages[3] == {"role": "tool", "tool_name": "calculator", "content": "391"}


def test_agent_loop_stops_at_step_limit():
    def always_tools(messages, tools, temperature):
        return tool_call("get_current_date")

    answer, messages = agent.run_agent("loop forever", chat_fn=always_tools, max_steps=3, verbose=False)
    assert answer.startswith("Stopped")
    assert sum(m["role"] == "tool" for m in messages) == 3


def test_tool_schemas_match_python_functions():
    import inspect
    for schema in agent.TOOL_SCHEMAS:
        fn = agent.TOOLS[schema["function"]["name"]]
        assert set(schema["function"]["parameters"]["properties"]) == set(inspect.signature(fn).parameters)
