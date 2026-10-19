"""01: Talk to the LLM with nothing but HTTP.

Run: python learn/01_tokens_and_llm_call.py

1. One raw /api/generate call: what goes in, what comes out, where the time goes.
2. Streaming: tokens arrive one at a time; time-to-first-token vs total time.
3. Tokens: how text becomes tokens, and why code/non-English text costs more.
4. Temperature: same prompt, deterministic vs creative sampling.
5. Context window: what happens when the prompt doesn't fit.
"""

import json
import time

from ollama_api import CHAT_MODEL, CHAT_OPTIONS, S, banner, client, generate, tok_per_s


def raw_call() -> None:
    banner("1. One raw call to POST /api/generate")
    body = {"model": CHAT_MODEL, "prompt": "Explain what an LLM is in two sentences.",
            "stream": False, "think": S.reasoning, "options": CHAT_OPTIONS | {"temperature": 0}}
    print("request:", json.dumps(body))
    r = client.post("/api/generate", json=body).json()
    print("\nresponse:", r["response"].strip())
    print("\nwhere the time went (Ollama reports nanoseconds):")
    print(f"  load_duration         {r['load_duration'] / 1e9:6.2f}s   weights -> VRAM (≈0 if already loaded)")
    print(f"  prompt_eval_duration  {r['prompt_eval_duration'] / 1e9:6.2f}s   {r['prompt_eval_count']:4d} input tokens"
          f"  = {tok_per_s(r, 'prompt_eval'):6.1f} tok/s  (parallel: all tokens at once)")
    print(f"  eval_duration         {r['eval_duration'] / 1e9:6.2f}s   {r['eval_count']:4d} output tokens"
          f" = {tok_per_s(r):6.1f} tok/s  (sequential: one token per forward pass)")
    print(f"  total_duration        {r['total_duration'] / 1e9:6.2f}s")


def streaming() -> None:
    banner("2. Streaming: the same API with stream=true returns one JSON line per token")
    body = {"model": CHAT_MODEL, "prompt": "Name three planets, comma separated.", "stream": True,
            "think": S.reasoning, "options": CHAT_OPTIONS | {"temperature": 0}}
    start, first, pieces = time.perf_counter(), None, []
    with client.stream("POST", "/api/generate", json=body) as resp:
        for line in resp.iter_lines():
            chunk = json.loads(line)
            if chunk["response"]:
                first = first or time.perf_counter() - start
                pieces.append(chunk["response"])
                print(f"  chunk {len(pieces):2d}: {chunk['response']!r}")
    total = time.perf_counter() - start
    print(f"\ntime to first token {first:.2f}s, total {total:.2f}s. Streaming makes UIs feel fast:"
          f" the user sees text after {first:.2f}s, not {total:.2f}s.")


def count_tokens(text: str) -> int:
    # Ollama has no /tokenize endpoint, but raw=True skips the chat template, so
    # prompt_eval_count is exactly the number of tokens in `text`.
    return generate(text, raw=True, num_predict=1)["prompt_eval_count"]


def tokens() -> None:
    banner("3. Tokens: the unit models read, write, bill and limit by")
    samples = [
        "hello world",
        "The quick brown fox jumps over the lazy dog.",
        "Supercalifragilisticexpialidocious",
        "def add(a, b):\n    return a + b",
        '{"user_id": 4821, "active": true}',
        "नमस्ते दुनिया",           # Hindi: "hello world"
        "こんにちは世界",              # Japanese: "hello world"
        "2026-10-19T21:38:43+05:30",
    ]
    print(f"  {'text':<48} {'chars':>5} {'tokens':>6} {'chars/tok':>9}")
    for s in samples:
        n = count_tokens(s)
        print(f"  {s!r:<48} {len(s):>5} {n:>6} {len(s) / n:>9.1f}")
    print("\n  English prose ≈ 4-5 chars/token. Rare words, code, JSON and numbers split into")
    print("  more pieces (Qwen gives every digit its own token), so they eat context faster.")
    print("  Cost per language depends on the tokenizer's training data: Qwen (a Chinese lab)")
    print("  packs Japanese/Chinese densely but spends one token per Hindi character.")


def temperature() -> None:
    banner("4. Temperature: how the next token is picked")
    prompt = "Invent a name for a friendly warehouse robot. Reply with only the name."
    for temp in (0.0, 1.5):
        names = [generate(prompt, temperature=temp, num_predict=10)["response"].strip() for _ in range(4)]
        print(f"  temperature={temp}: {names}")
    names = [generate(prompt, temperature=1.5, seed=42, num_predict=10)["response"].strip() for _ in range(2)]
    print(f"  temperature=1.5, seed=42: {names}  <- a fixed seed makes sampling repeatable")
    print("\n  The model outputs a probability for every token in its vocabulary.")
    print("  temperature=0 always takes the most likely one (deterministic); higher values")
    print("  flatten the distribution so unlikely tokens get picked. Agents and RAG use 0-0.3.")


def context_window() -> None:
    banner("5. Context window: what happens when the prompt doesn't fit")
    filler = "The warehouse was quiet and the robots were charging. " * 300  # ~3,300 tokens
    prompt = f"The secret code is TANGERINE-42.\n\n{filler}\n\nWhat is the secret code? Reply with only the code."
    for num_ctx in (8192, 1024):
        r = generate(prompt, num_ctx=num_ctx, temperature=0, num_predict=20)
        print(f"  num_ctx={num_ctx:<5} prompt tokens the model saw: {r['prompt_eval_count']:>5}"
              f"  answer: {r['response'].strip()!r}")
    print("\n  When the prompt is longer than num_ctx, Ollama silently cuts it to fit, and")
    print("  the start of the prompt (our secret) is gone. You get no error, just a")
    print("  wrong answer. That's why RAG must budget tokens: system prompt + chunks + history + answer.")


if __name__ == "__main__":
    print(f"model: {CHAT_MODEL}   options: {CHAT_OPTIONS}")
    raw_call()
    streaming()
    tokens()
    temperature()
    context_window()
