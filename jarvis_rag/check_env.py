"""Environment doctor: `python -m jarvis_rag.check_env` (or `make check`).

Checks Python, GPU, Ollama and the configured models, predicts VRAM use
from the model's architecture, then runs one chat call and one embedding
call through the same factories the app uses and reports real numbers.
"""

import os
import re
import shutil
import subprocess
import sys
import time

import httpx
import numpy as np

from jarvis_rag.config import Settings, get_settings
from jarvis_rag.llm import get_chat_model, get_embeddings, model_name

MIB = 1024**2

# Effective bits per weight of common GGUF quantizations (includes the
# per-block scales, which is why Q4 is not exactly 4.0).
BITS_PER_WEIGHT = {
    "Q4_0": 4.5,
    "Q4_K_M": 4.85,
    "Q5_K_M": 5.7,
    "Q6_K": 6.56,
    "Q8_0": 8.5,
    "F16": 16.0,
}

_failures = 0


def report(ok: bool | None, label: str, detail: str = "") -> None:
    """ok=True pass, False fail, None warning."""
    global _failures
    mark = {True: "✓", False: "✗", None: "!"}[ok]
    if ok is False:
        _failures += 1
    print(f"  {mark} {label:<22} {detail}")


def parse_nvidia_smi(text: str) -> list[dict]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version
    --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        name, total, used, driver = (p.strip() for p in line.split(","))
        gpus.append({"name": name, "total_mib": int(total), "used_mib": int(used), "driver": driver})
    return gpus


def estimate_vram_mib(
    n_params: int,
    bits_per_weight: float,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    n_ctx: int,
    kv_bytes: int = 2,
) -> dict:
    """weights = params x bits / 8
    KV cache = 2 (K and V) x layers x kv_heads x head_dim x bytes x tokens"""
    weights = n_params * bits_per_weight / 8
    kv = 2 * n_layers * n_kv_heads * head_dim * kv_bytes * n_ctx
    return {"weights": weights / MIB, "kv_cache": kv / MIB, "total": (weights + kv) / MIB}


def model_arch(show: dict) -> dict:
    """Pull the numbers the VRAM formula needs out of Ollama's /api/show."""
    info = show["model_info"]
    arch = info["general.architecture"]
    return {
        "n_params": info["general.parameter_count"],
        "quant": show["details"]["quantization_level"],
        "n_layers": info[f"{arch}.block_count"],
        "n_kv_heads": info[f"{arch}.attention.head_count_kv"],
        "head_dim": info.get(f"{arch}.attention.key_length")
        or info[f"{arch}.embedding_length"] // info[f"{arch}.attention.head_count"],
    }


def _full_name(model: str) -> str:
    return model if ":" in model else f"{model}:latest"


def cosine(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def check_python() -> None:
    v = sys.version_info
    report(v >= (3, 10), "python", f"{v.major}.{v.minor}.{v.micro} ({sys.executable})")


def check_gpu() -> None:
    if not shutil.which("nvidia-smi"):
        report(None, "gpu", "nvidia-smi not found; Ollama will run on CPU")
        return
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout
    for g in parse_nvidia_smi(out):
        report(True, "gpu", f"{g['name']}, {g['total_mib']} MiB total, {g['used_mib']} MiB used, driver {g['driver']}")


def check_ollama(s: Settings, client: httpx.Client) -> bool:
    try:
        version = client.get("/api/version").json()["version"]
    except httpx.HTTPError as e:
        report(False, "ollama", f"not reachable at {s.ollama_base_url} ({e.__class__.__name__})")
        return False
    report(True, "ollama", f"v{version} at {s.ollama_base_url}")

    pulled = {m["name"] for m in client.get("/api/tags").json()["models"]}
    needed = []
    if "ollama" in (s.llm_provider, s.llm_fallback):
        needed += [("chat model", s.chat_model), ("fallback model", s.fallback_chat_model)]
    if s.embed_provider == "ollama":
        needed.append(("embed model", s.embed_model))
    for role, name in needed:
        ok = _full_name(name) in pulled
        report(ok if role != "fallback model" else (ok or None), role,
               name if ok else f"{name} missing: run `ollama pull {name}`")
    return _full_name(s.chat_model) in pulled


def check_vram_estimate(s: Settings, client: httpx.Client) -> None:
    arch = model_arch(client.post("/api/show", json={"model": s.chat_model}).json())
    bpw = BITS_PER_WEIGHT.get(arch["quant"], 4.85)
    est = estimate_vram_mib(arch["n_params"], bpw, arch["n_layers"], arch["n_kv_heads"],
                            arch["head_dim"], s.num_ctx)
    report(True, "vram estimate",
           f"{arch['n_params'] / 1e9:.1f}B @ {arch['quant']} (~{bpw} bpw): weights {est['weights']:.0f} MiB"
           f" + KV@{s.num_ctx} {est['kv_cache']:.0f} MiB = {est['total']:.0f} MiB (+ compute buffers)")


def smoke_chat(s: Settings) -> bool:
    start = time.perf_counter()
    try:
        msg = get_chat_model(s).invoke("In one short sentence, who are you?")
    except Exception as e:  # missing key, 401, 429, no network: say which, don't crash the doctor
        report(False, "chat smoke test", f"{s.llm_provider}/{model_name(s)}: {type(e).__name__}: {str(e)[:160]}")
        return False
    wall = time.perf_counter() - start
    meta = msg.response_metadata
    if "eval_count" in meta:  # Ollama reports its own timings
        detail = (f"{meta['eval_count'] / (meta['eval_duration'] / 1e9):.1f} tok/s, "
                  f"load {meta['load_duration'] / 1e9:.1f}s, wall {wall:.1f}s")
    else:
        out = (msg.usage_metadata or {}).get("output_tokens", 0)
        detail = f"{s.llm_provider}/{model_name(s)}: {out} tokens in {wall:.1f}s (includes network)"
    report(True, "chat smoke test", detail)
    print(f"      → {msg.text.strip()[:120]!r}")
    return True


def smoke_tools(s: Settings) -> None:
    """The agent is useless if the model can't emit structured tool calls."""
    from langchain_core.tools import tool

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression exactly."""
        return expression

    try:
        msg = get_chat_model(s).bind_tools([calculator]).invoke("Use the calculator tool to compute 37 * 43.")
    except Exception as e:
        report(False, "tool-calling", f"{type(e).__name__}: {str(e)[:160]}")
        return
    calls = msg.tool_calls
    report(bool(calls), "tool-calling", f"called {calls[0]['name']}({calls[0]['args']})" if calls
           else f"model answered in text instead: {msg.text[:80]!r}")


def smoke_embed(s: Settings) -> None:
    start = time.perf_counter()
    try:
        vecs = get_embeddings(s).embed_documents(["car", "automobile", "banana"])
    except Exception as e:
        report(False, "embed smoke test", f"{s.embed_provider}: {type(e).__name__}: {str(e)[:160]}")
        return
    model = s.fastembed_model if s.embed_provider == "fastembed" else s.embed_model
    report(True, "embed smoke test",
           f"{s.embed_provider}/{model} dim={len(vecs[0])}  cos(car,automobile)={cosine(vecs[0], vecs[1]):.3f}"
           f"  cos(car,banana)={cosine(vecs[0], vecs[2]):.3f}  ({(time.perf_counter() - start) * 1000:.0f} ms)")


LLAMA_CACHE_RAM_DEFAULT_MIB = 8192


def parse_cache_ram(systemd_env: str) -> int | None:
    """LLAMA_ARG_CACHE_RAM from `systemctl show ollama -p Environment` output."""
    m = re.search(r"LLAMA_ARG_CACHE_RAM=(-?\d+)", systemd_env)
    return int(m.group(1)) if m else None


def check_prompt_cache_ram() -> None:
    """Ollama's llama-server keeps every recent prompt's KV state in *system* RAM
    (~140 MiB each for qwen3:4b at 8k ctx), up to LLAMA_ARG_CACHE_RAM (default
    8192 MiB). On a small-RAM machine that grows until the OOM killer stops Ollama.
    Found the hard way in Phase 3; see docs/phase-3.md."""
    try:
        total_mib = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // MIB
        env = subprocess.run(["systemctl", "show", "ollama", "-p", "Environment"],
                             capture_output=True, text=True, timeout=5).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return  # not Linux/systemd: nothing to check
    limit = parse_cache_ram(env)
    effective = LLAMA_CACHE_RAM_DEFAULT_MIB if limit is None else limit
    detail = f"limit {effective} MiB{' (default)' if limit is None else ''}, system RAM {total_mib} MiB"
    if effective < 0 or effective > total_mib // 4:
        report(None, "prompt cache RAM", detail + " -> may OOM-kill Ollama; set LLAMA_ARG_CACHE_RAM=1024 "
               "in a systemd drop-in (docs/phase-3.md)")
    else:
        report(True, "prompt cache RAM", detail)


def check_placement(client: httpx.Client) -> None:
    for m in client.get("/api/ps").json()["models"]:
        gpu_pct = 100 * m["size_vram"] / m["size"] if m["size"] else 0
        placement = "100% GPU" if gpu_pct >= 99.5 else ("100% CPU" if gpu_pct < 0.5 else f"{gpu_pct:.0f}% GPU")
        report(True, "loaded", f"{m['name']:<28} {m['size'] / MIB:>6.0f} MiB  {placement}")


def main() -> int:
    s = get_settings()
    print(f"{s.agent_name} environment check\n")
    check_python()
    local_llm = s.llm_provider == "ollama"
    if s.llm_provider != "ollama":
        report(None, "hosted LLM", f"{s.llm_provider}/{model_name(s)}: prompts, including retrieved document "
                                   f"chunks, are sent to {s.llm_provider}")
    if local_llm:
        check_gpu()
    uses_ollama = "ollama" in (s.llm_provider, s.embed_provider, s.llm_fallback)
    ollama_ok = False
    if uses_ollama:
        with httpx.Client(base_url=s.ollama_base_url, timeout=300) as client:
            ollama_ok = check_ollama(s, client)
            if ollama_ok and local_llm:
                check_vram_estimate(s, client)
    if not local_llm or ollama_ok:
        if smoke_chat(s):
            smoke_tools(s)
    if s.embed_provider == "fastembed" or ollama_ok:
        smoke_embed(s)
    if local_llm and ollama_ok:
        with httpx.Client(base_url=s.ollama_base_url, timeout=30) as client:
            check_placement(client)
        check_prompt_cache_ram()
    print("\nAll checks passed." if _failures == 0 else f"\n{_failures} check(s) failed.")
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
