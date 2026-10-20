"""The agent loop with a scripted chat model: no Ollama, no GPU.

ScriptedModel (tests/fakes.py) replays AIMessages and records what it was shown, so the
tests can check what actually reached the model (system prompt, auto-search,
trimmed history, nudges) as well as what came out.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from fakes import ScriptedModel, call

from jarvis_rag.agent import ask, build_agent, catalogue, fit_history, needs_sql_nudge, stream_events, system_prompt
from jarvis_rag.config import Settings
from jarvis_rag.ingest import ingest_documents
from jarvis_rag.loaders import load_csv
from jarvis_rag.vectorstore import KnowledgeBase


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    s = Settings(_env_file=None, storage_dir=tmp_path / "st", collection_name="agnt", memory_backend="memory",
                 retrieval_mode="hybrid", agent_name="FRIDAY", address_user_as="boss")
    s.storage_dir.mkdir()
    kb = KnowledgeBase(s, embeddings=DeterministicFakeEmbedding(size=32))
    (tmp_path / "sales.csv").write_text("Region,Units\nEMEA,10\nAPAC,5\n")
    docs = load_csv(tmp_path / "sales.csv", s.sqlite_path)
    docs.append(Document("# Specs\n## K2\nBattery life: 6.5 hours.",
                         metadata={"source": "specs.md", "type": "text", "title": "Product Specs", "path": "specs.md"}))
    ingest_documents(docs, kb, s)
    return s, kb


def make(env, *script, **overrides):
    s, kb = env
    model = ScriptedModel(script=list(script), seen=[])
    s = s.model_copy(update=overrides) if overrides else s
    return build_agent(s, model=model, kb=kb, checkpointer=InMemorySaver()), model


def test_prompt_has_persona_catalogue_and_rules(env):
    s, kb = env
    p = system_prompt(s, catalogue(kb))
    assert 'You are FRIDAY' in p and 'Address the user as "boss"' in p
    assert "- Product Specs (specs.md)" in p and "- SQL tables: sales" in p
    assert "never answer from memory" in p
    assert "Do not call the user by any name" in system_prompt(s.model_copy(update={"address_user_as": ""}))


def test_auto_search_runs_before_the_model(env):
    agent, model = make(env, AIMessage("6.5 hours [specs.md]"))
    events = list(stream_events(agent, "battery life of the K2?", "t1"))
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_call", "tool_result", "answer"]
    assert events[0]["id"].startswith("auto-") and events[0]["args"] == {"query": "battery life of the K2?"}
    first = model.seen[0]
    assert isinstance(first[0], SystemMessage) and "FRIDAY" in first[0].content
    assert isinstance(first[-1], ToolMessage) and "[specs.md]" in first[-1].content
    assert events[-1]["text"] == "6.5 hours [specs.md]"


def test_auto_search_can_be_disabled(env):
    agent, model = make(env, AIMessage("Hello."), auto_search=False)
    assert [e["type"] for e in stream_events(agent, "hi", "t")] == ["answer"]
    assert isinstance(model.seen[0][-1], HumanMessage)


def test_tool_loop_runs_sql_and_feeds_result_back(env):
    agent, model = make(env, call("run_sql", query="SELECT SUM(units) FROM sales"), AIMessage("15 units [table sales]"))
    events = list(stream_events(agent, "total units?", "t"))
    results = [e for e in events if e["type"] == "tool_result"]
    assert results[-1]["name"] == "run_sql" and "15" in results[-1]["content"] and not results[-1]["error"]
    assert events[-1] == {"type": "answer", "text": "15 units [table sales]"}
    assert len(model.seen) == 2


def test_tool_errors_go_back_to_the_model_instead_of_crashing(env):
    agent, model = make(env, call("run_sql", query="DELETE FROM sales"), AIMessage("I can only read."))
    events = list(stream_events(agent, "delete everything", "t"))
    err = [e for e in events if e["type"] == "tool_result" and e["error"]]
    assert err and "Only read-only SELECT" in err[0]["content"]
    assert events[-1]["text"] == "I can only read."


def test_unknown_internal_errors_are_not_leaked(env, monkeypatch):
    from jarvis_rag import tools

    def boom(*a, **kw):
        raise RuntimeError("secret path /home/x/.env")

    monkeypatch.setattr(tools, "safe_eval", boom)
    agent, _ = make(env, call("calculator", expression="1+1"), AIMessage("ok"), auto_search=False)
    err = [e for e in stream_events(agent, "1+1?", "t") if e["type"] == "tool_result"][0]
    assert err["error"] and "RuntimeError" in err["content"] and "secret" not in err["content"]


def test_memory_is_per_thread(env):
    agent, model = make(env, AIMessage("Noted."), AIMessage("You asked about the K2."), AIMessage("No idea."),
                        auto_search=False)
    ask(agent, "Remember: I care about the K2.", "alice")
    ask(agent, "What do I care about?", "alice")
    ask(agent, "What do I care about?", "bob")
    alice_2nd, bob = model.seen[1], model.seen[2]
    assert any("K2" in m.text for m in alice_2nd if isinstance(m, HumanMessage))
    assert not any("K2" in m.text for m in bob if isinstance(m, HumanMessage))


def test_sqlite_memory_survives_a_restart(env):
    s, kb = env
    s = s.model_copy(update={"memory_backend": "sqlite", "auto_search": False})
    first = build_agent(s, model=ScriptedModel(script=[AIMessage("Noted.")], seen=[]), kb=kb)
    ask(first, "My robot is called Rusty.", "t")
    model = ScriptedModel(script=[AIMessage("Rusty.")], seen=[])
    second = build_agent(s, model=model, kb=kb)  # new process, same storage/memory.db
    ask(second, "What is my robot called?", "t")
    assert any("Rusty" in m.text for m in model.seen[0])
    assert s.memory_path.exists()


def test_model_call_limit_stops_a_looping_model(env):
    loop = [call("calculator", expression=f"{i}+1") for i in range(20)]
    agent, model = make(env, *loop, agent_max_model_calls=3, auto_search=False)
    list(stream_events(agent, "loop forever", "t"))
    assert len(model.seen) == 3


def test_history_is_trimmed_for_the_model_but_kept_in_state(env):
    agent, model = make(env, *[AIMessage("x" * 400) for _ in range(6)], history_max_tokens=300, auto_search=False)
    for i in range(6):
        ask(agent, f"question {i} " + "y" * 400, "t")
    last = model.seen[-1]
    assert isinstance(last[1], HumanMessage) and "question 5" in last[-1].text
    assert not any("question 0" in m.text for m in last)
    assert len(agent.get_state({"configurable": {"thread_id": "t"}}).values["messages"]) == 12


def test_fit_history_keeps_an_oversized_current_turn():
    msgs = [HumanMessage("old"), AIMessage("old answer"), HumanMessage("new " + "z" * 5000),
            AIMessage("", tool_calls=[{"name": "t", "args": {}, "id": "1"}]), ToolMessage("r", tool_call_id="1")]
    assert fit_history(msgs, 100) == msgs[2:]


def test_sql_nudge_detection():
    q = HumanMessage("How many units in total?")
    table_hit = ToolMessage("[table sales]\nSQL table `sales`...", tool_call_id="a", name="search_knowledge_base")
    assert needs_sql_nudge([q, table_hit], AIMessage("There are several rows.")) == "sales"
    assert needs_sql_nudge([HumanMessage("What does E-217 mean?"), table_hit], AIMessage("Overcurrent.")) is None
    assert needs_sql_nudge([HumanMessage("hi")], AIMessage("You could run SELECT COUNT(*) ...")) == "the right table"
    ran = ToolMessage("1 row(s):\n15", tool_call_id="b", name="run_sql")
    assert needs_sql_nudge([q, table_hit, ran], AIMessage("15 units.")) is None


def test_sql_follow_through_asks_again(env):
    agent, model = make(env, AIMessage("You can find this by running a SELECT on sales."),
                        call("run_sql", query="SELECT SUM(units) FROM sales"), AIMessage("15 [table sales]"))
    assert ask(agent, "How many units in total?", "t") == "15 [table sales]"
    assert "call run_sql on sales now" in model.seen[1][-1].text
    stored = agent.get_state({"configurable": {"thread_id": "t"}}).values["messages"]
    assert not any("automatic check" in m.text for m in stored)  # the nudge is not saved to memory


def test_sql_follow_through_tells_streaming_clients_to_discard_the_draft(env):
    agent, _ = make(env, AIMessage("Probably about 20."), call("run_sql", query="SELECT SUM(units) FROM sales"),
                    AIMessage("15 [table sales]"))
    kinds = [e["type"] for e in stream_events(agent, "How many units in total?", "t")]
    assert kinds.index("discard") < kinds.index("tool_call", 1)  # discard comes before the SQL call
    assert kinds[-1] == "answer"


def test_catalogue_refreshes_after_ingest(env):
    s, kb = env
    agent, model = make(env, AIMessage("a"), AIMessage("b"), auto_search=False)
    ask(agent, "q1", "t")
    ingest_documents([Document("# Parking\nBasement.", metadata={"source": "parking.md", "type": "text",
                                                                 "title": "Parking Guide", "path": "parking.md"})], kb, s)
    ask(agent, "q2", "t")
    assert "Parking Guide" not in model.seen[0][0].content and "Parking Guide" in model.seen[1][0].content
