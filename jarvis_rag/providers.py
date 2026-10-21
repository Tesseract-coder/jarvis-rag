"""Which models can I use right now? Free-tier catalogues change monthly, so ask
the providers instead of trusting a default written months ago.

    python -m jarvis_rag.providers                  # every provider you have a key for, + OpenRouter's free list
    python -m jarvis_rag.providers openrouter       # free OpenRouter models that support tool calling (no key needed)
    python -m jarvis_rag.providers groq --ping      # list, then one chat + one tool call with the configured model
"""

import argparse
import sys
import time

import httpx

from jarvis_rag.config import Settings, get_settings
from jarvis_rag.llm import OPENROUTER_BASE_URL, MissingAPIKey, api_key, model_name


def list_groq(s: Settings) -> list[str]:
    r = httpx.get("https://api.groq.com/openai/v1/models", timeout=20,
                  headers={"Authorization": f"Bearer {api_key(s, 'groq')}"})
    r.raise_for_status()
    return sorted(m["id"] for m in r.json()["data"] if m.get("active", True))


def list_gemini(s: Settings) -> list[str]:
    r = httpx.get("https://generativelanguage.googleapis.com/v1beta/models", timeout=20, params={"pageSize": 200},
                  headers={"x-goog-api-key": api_key(s, "gemini")})  # header, not ?key=, keeps it out of logs
    r.raise_for_status()
    return sorted(m["name"].removeprefix("models/") for m in r.json()["models"]
                  if "generateContent" in m.get("supportedGenerationMethods", []))


def list_openrouter_free(s: Settings) -> list[str]:
    """Public endpoint: no key needed. Only ':free' models that accept `tools`."""
    r = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=20)
    r.raise_for_status()
    free = [m for m in r.json()["data"]
            if m["id"].endswith(":free") and "tools" in (m.get("supported_parameters") or [])]
    return [f"{m['id']}  (context {m.get('context_length', '?'):,})"
            for m in sorted(free, key=lambda m: -(m.get("context_length") or 0))]


LISTERS = {"groq": list_groq, "gemini": list_gemini, "openrouter": list_openrouter_free}


def ping(s: Settings, provider: str) -> None:
    from langchain_core.tools import tool

    from jarvis_rag.llm import get_chat_model

    @tool
    def calculator(expression: str) -> str:
        """Evaluate an arithmetic expression exactly."""
        return expression

    s = s.model_copy(update={"llm_provider": provider})
    m = get_chat_model(s)
    t0 = time.perf_counter()
    text = m.invoke("Reply with one word: ready").text.strip()
    t1 = time.perf_counter()
    calls = m.bind_tools([calculator]).invoke("Use the calculator tool to compute 37 * 43.").tool_calls
    t2 = time.perf_counter()
    print(f"  ping {model_name(s, provider)}: {text[:40]!r} in {t1 - t0:.1f}s; "
          f"tool call: {calls[0]['args'] if calls else 'NONE (answered in text)'} in {t2 - t1:.1f}s")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("provider", nargs="?", choices=list(LISTERS))
    ap.add_argument("--ping", action="store_true", help="also send one chat and one tool call")
    args = ap.parse_args()
    s = get_settings()
    failed = False
    for provider in [args.provider] if args.provider else list(LISTERS):
        configured = model_name(s, provider)
        print(f"\n{provider} (configured: {configured})")
        try:
            models = LISTERS[provider](s)
        except MissingAPIKey as e:
            print(f"  skipped: {e}")
            continue
        except httpx.HTTPStatusError as e:
            print(f"  error: HTTP {e.response.status_code} from {e.request.url.host}")
            failed = True
            continue
        for m in models[:40]:
            print(f"  {'*' if m.split()[0] == configured else ' '} {m}")
        if len(models) > 40:
            print(f"    ... and {len(models) - 40} more")
        if not any(m.split()[0] == configured for m in models):
            print(f"  ! configured model {configured!r} is not in this list")
        if args.ping:
            try:
                ping(s, provider)
            except Exception as e:
                print(f"  ping failed: {type(e).__name__}: {str(e)[:200]}")
                failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
