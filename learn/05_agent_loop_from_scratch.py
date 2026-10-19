"""05: An agent loop from scratch. This is what LangChain's create_agent does internally.

Run: python learn/05_agent_loop_from_scratch.py

An "agent" is just a loop:

    messages = [system, user]
    loop:
        reply = LLM(messages, tool_schemas)
        if reply has no tool calls: return reply.text          # final answer
        for each tool call: result = run_tool(name, args)
                            messages += [reply, tool result]    # model sees what happened

Part 1 uses Ollama's native `tools` field. Part 2 drops one level lower and shows
that a tool call is just text the model was trained to emit.
"""

import ast
import json
import operator
from datetime import date

from ollama_api import CHAT_MODEL, CHAT_OPTIONS, S, banner, chat, client, load_script

rag = load_script("04_rag_without_framework")

# --------------------------------------------------------------------------- tools

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def calculator(expression: str) -> str:
    """Safe arithmetic: parse to an AST and evaluate only numbers and math operators.
    Never eval() model output: `__import__('os').system(...)` is one bad generation away."""
    def ev(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")
    result = ev(ast.parse(expression, mode="eval").body)
    return str(round(result, 10) if isinstance(result, float) else result)


_db = None


def search_handbook(query: str) -> str:
    """Vector search over the Halcyon docs (script 04's index), returned as cited text."""
    global _db
    _db = _db or rag.build_index()
    hits = _db.query(rag.embed(["search_query: " + query])[0], k=3)
    return "\n\n".join(f"[source: {h['id']}]\n{h['doc']}" for h in hits)


def get_current_date() -> str:
    return date.today().isoformat()


TOOLS = {"calculator": calculator, "search_handbook": search_handbook, "get_current_date": get_current_date}

# The JSON Schema the model sees. create_agent generates this from a function's
# signature + docstring; here we write it by hand. Descriptions ARE the prompt:
# they are how the model decides which tool fits.
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "calculator",
        "description": "Evaluate an arithmetic expression exactly. Use for ANY math, e.g. '17*23' or '(25-7)/2'.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "Python-style arithmetic, numbers and + - * / // % ** only"}},
            "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "search_handbook",
        "description": "Search Halcyon Robotics internal documents: HR policies, IT security, "
                       "engineering process, robot product specs, office FAQ.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, as a natural-language question"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_current_date",
        "description": "Return today's date in ISO format (YYYY-MM-DD).",
        "parameters": {"type": "object", "properties": {}}}},
]

# The first version of this prompt (kept for the demo below) let the model decide
# when to search, and it often answered "I don't have that information" without
# trying. Telling it what it does NOT know is what makes it reach for the tool.
WEAK_SYSTEM_PROMPT = f"""You are {S.agent_name}, a concise, precise assistant for Halcyon Robotics employees.
- For questions about Halcyon, call search_handbook and answer only from its results, citing the [source: ...].
- For any arithmetic, call calculator. Never do math in your head.
- If the tools don't give you the answer, say you don't know."""

SYSTEM_PROMPT = f"""You are {S.agent_name}, a concise, precise assistant for Halcyon Robotics employees.
You know NOTHING about Halcyon (its robots, products, policies, processes or office) except what search_handbook returns.
- Before answering ANY question that mentions Halcyon or its robots, policies, processes or office, call search_handbook first.
- For any arithmetic, call calculator. Never do math in your head.
- Answer only from tool results and cite the [source: ...]. If the tools don't give the answer, say you don't know."""

# --------------------------------------------------------------------------- the loop


def run_tool(name: str, args: dict) -> str:
    """Errors become tool results, not crashes, so the model can read them and retry."""
    if name not in TOOLS:
        return f"Error: unknown tool {name!r}. Available: {', '.join(TOOLS)}"
    try:
        return TOOLS[name](**args)
    except Exception as e:  # noqa: BLE001 (the model should see any failure)
        return f"Error: {type(e).__name__}: {e}"


def run_agent(question: str, chat_fn=chat, max_steps: int = 6, verbose: bool = True,
              system_prompt: str = SYSTEM_PROMPT) -> tuple[str, list[dict]]:
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
    for step in range(1, max_steps + 1):
        reply = chat_fn(messages, tools=TOOL_SCHEMAS, temperature=0)["message"]
        messages.append(reply)
        calls = reply.get("tool_calls") or []
        if not calls:
            return reply["content"].strip(), messages
        for call in calls:
            name, args = call["function"]["name"], call["function"]["arguments"]
            result = run_tool(name, args)
            if verbose:
                preview = result.replace("\n", " ")[:110]
                print(f"    step {step}: {name}({json.dumps(args)}) -> {preview}{'...' if len(result) > 110 else ''}")
            messages.append({"role": "tool", "tool_name": name, "content": result})
    return "Stopped: step limit reached without a final answer.", messages  # guard against infinite loops


# --------------------------------------------------------------------------- part 2

QWEN_RAW_PROMPT = """<|im_start|>system
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call><|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""


def show_raw_tool_call() -> None:
    banner("Part 2: under the hood, a tool call is just text")
    prompt = QWEN_RAW_PROMPT.format(tools="\n".join(json.dumps(t) for t in TOOL_SCHEMAS[:1]),
                                    question="What is 1234 * 5678?")
    print("  Qwen's chat template renders tools=[...] into the system prompt like this (we build the")
    print("  same text by hand here and send it with raw=True, bypassing Ollama's template):\n")
    print("    " + prompt[prompt.index("<tools>\n"):prompt.index("\n</tools>") + 9].replace("\n", "\n    "))
    r = client.post("/api/generate", json={"model": CHAT_MODEL, "prompt": prompt, "raw": True, "stream": False,
                                           "options": CHAT_OPTIONS | {"temperature": 0}}).json()
    print("\n  ...and the model's raw output (raw=True, no parsing) is:\n")
    print("    " + r["response"].strip().replace("\n", "\n    "))
    print("\n  Ollama parses <tool_call>...</tool_call> into message.tool_calls. If a small model")
    print("  emits malformed JSON or forgets the tags, the 'tool call' is just garbled text in")
    print("  the answer. That's the failure mode to watch for with 4B models.")


if __name__ == "__main__":
    banner(f"Part 1: the agent loop ({CHAT_MODEL})")
    for q in [
        "What is 17 * 23?",
        "How many paid annual leave days do I get, and how many are left if I take 7 days in March and 4 in August?",
        "Which Halcyon robot carries more weight, and what is the payload difference in kg?",
        "Hi, who are you?",
    ]:
        print(f"\n  Q: {q}")
        answer, msgs = run_agent(q)
        used = [c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []]
        print(f"  A: {answer}\n     tools used: {used or 'none'}, {len(msgs)} messages in the conversation")

    banner("Part 1b: same question, the weaker system prompt")
    q = "Which Halcyon robot carries more weight, and what is the payload difference in kg?"
    for label, prompt in [("weak prompt  ", WEAK_SYSTEM_PROMPT), ("strong prompt", SYSTEM_PROMPT)]:
        answer, msgs = run_agent(q, verbose=False, system_prompt=prompt)
        used = [c["function"]["name"] for m in msgs for c in m.get("tool_calls") or []]
        print(f"  {label} tools={used or 'none'}: {answer.replace(chr(10), ' ')[:100]}")
    print("  Same model, same tools, same temperature. Only the instructions changed.")

    if CHAT_MODEL.startswith("qwen"):
        show_raw_tool_call()
