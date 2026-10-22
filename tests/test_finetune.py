"""Phase 10 data plumbing, offline: the prompt format, augmentation and eval scoring.
(Training itself needs the GPU and .venv-train: `make train`.)"""

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune"))

import common  # noqa: E402
import eval_reader  # noqa: E402
import make_dataset  # noqa: E402

P = [("docs/a.md", "Alpha is 3."), ("learn/b.py, lines 1-9", "def beta(): ...")]


def test_context_round_trips_through_the_user_message():
    msg = common.user_message(P, "What is alpha?")
    assert msg.startswith("Context:\n[docs/a.md]\nAlpha is 3.\n\n---\n\n[learn/b.py, lines 1-9]")
    assert make_dataset.parse_context(msg) == (P, "What is alpha?")


def test_prompt_has_no_example_path_to_copy():
    # "[docs/report.md]" in an earlier prompt was copied verbatim by the base 1.7B
    assert "report.md" not in common.READER_PROMPT and common.ABSTAIN in common.READER_PROMPT


def test_contamination_filter_catches_eval_facts():
    for s in ["Halcyon Robotics", "the Kestrel K2", "error E-217", "part HK-2231", "robot_sales", "battery life"]:
        assert make_dataset.CONTAMINATION.search(s), s
    assert not make_dataset.CONTAMINATION.search("BM25 uses idf to weight rare terms")


def test_augment_keeps_the_answering_passage():
    by_source = {f"docs/s{i}.md": [(f"docs/s{i}.md", f"text {i}")] for i in range(8)}
    row = {"kind": "answerable", "source": "docs/a.md", "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": common.user_message(P, "What is alpha?")},
        {"role": "assistant", "content": "Alpha is 3. [docs/a.md]"}]}
    for seed in range(5):
        aug = make_dataset.augment(row, by_source, random.Random(seed))
        passages, q = make_dataset.parse_context(aug["messages"][1]["content"])
        assert q == "What is alpha?" and ("docs/a.md", "Alpha is 3.") in passages and len(passages) == 5
        assert aug["messages"][2] == row["messages"][2] and aug["augmented"]


def test_eval_scoring():
    rows = [
        {"case": "answerable", "passages": P, "expected": "docs/a.md", "answer": "It is 3 [docs/a.md].",
         "correct": True, "s": 1.0, "tok_s": 50},
        {"case": "answerable", "passages": P, "expected": "docs/a.md", "answer": "It is 3 [docs/x.md].",
         "correct": True, "s": 2.0, "tok_s": 50},
        {"case": "removed", "passages": P[1:], "expected": None, "answer": common.ABSTAIN, "correct": False,
         "s": 1.0, "tok_s": 50},
        {"case": "removed", "passages": P[1:], "expected": None, "answer": "It is 7.", "correct": False,
         "s": 1.0, "tok_s": 50},
    ]
    eval_reader.score(rows)
    assert [r["cites_expected"] for r in rows[:2]] == [True, False]
    assert rows[1]["invalid_citations"] == ["docs/x.md"] and rows[2]["abstained"]
    s = eval_reader.summarize("m", rows)
    assert (s["correct"], s["cites_source"], s["safe_when_absent"], s["hallucinated"]) == (100, 50, 50, 1)
