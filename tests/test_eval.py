"""The eval harness itself: metrics, relevance rule, and the QA set's shape."""

import importlib.util
from pathlib import Path

import pytest
from langchain_core.documents import Document

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

KINDS = {"fact", "paraphrase", "exact", "comparison", "code", "table", "pdf", "docx", "unanswerable"}


def doc(source, text):
    return Document(text, metadata={"source": source})


ITEM = {"expected_source": "specs.md", "must_contain": ["Battery life: 6.5 hours", "Battery life: 9 hours"],
        "kind": "comparison"}


def test_relevance_needs_source_and_phrase():
    assert ev.is_relevant(doc("specs.md", "[Wren]\n- battery   LIFE: 9 hours"), ITEM)  # case/space-insensitive
    assert not ev.is_relevant(doc("specs.md", "Warranty: 3 years"), ITEM)
    assert not ev.is_relevant(doc("other.md", "Battery life: 9 hours"), ITEM)


def test_rank_and_completeness():
    docs = [doc("specs.md", "Warranty"), doc("specs.md", "Battery life: 9 hours"), doc("x", "y")]
    assert ev.score_ranking(docs, ITEM) == {"rank": 2, "complete": False}
    docs.append(doc("specs.md", "Battery life: 6.5 hours"))
    assert ev.score_ranking(docs, ITEM)["complete"] is True
    assert ev.score_ranking(docs, ITEM, k=3)["complete"] is False  # 4th doc is outside top 3


def test_summarize_metrics():
    rows = [{"rank": 1, "complete": True, "kind": "fact", "ms": 10},
            {"rank": 2, "complete": True, "kind": "fact", "ms": 30},
            {"rank": None, "complete": False, "kind": "comparison", "ms": 20},
            {"rank": 4, "complete": True, "kind": "comparison", "ms": 40}]
    s = ev.summarize(rows)
    assert s["hit@1"] == 0.25 and s["hit@3"] == 0.5 and s["hit@5"] == 0.75
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0 + 0.25) / 4)
    assert s["complete@5"] == 0.5 and s["p50_ms"] == 25


def test_qa_set_is_well_formed():
    qa = ev.load_qa()
    assert len(qa) >= 50
    assert len({q["id"] for q in qa}) == len(qa)
    for q in qa:
        assert q["kind"] in KINDS and q["question"].endswith("?") and q["expected_answer"]
        if q["kind"] == "unanswerable":
            assert q["expected_source"] is None and not q["must_contain"]
        else:
            assert q["expected_source"].startswith("data/") and q["must_contain"]
        if q["kind"] == "comparison":
            assert len(q["must_contain"]) >= 2


def test_unreachable_labels_are_reported():
    class FakeCollection:
        def get(self, where, include):
            return {"documents": ["Battery life: 9 hours"] if where["source"] == "specs.md" else []}

    class FakeKB:
        collection = FakeCollection()

    bad = ev.unreachable(FakeKB(), [{"id": "q1", **ITEM}])
    assert len(bad) == 1 and "6.5 hours" in bad[0]
