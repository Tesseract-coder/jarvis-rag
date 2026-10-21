# Phase 7: Provider switch

**Goal:** the same agent, API and UI on a different model with one `.env` change.
Either a hosted free tier (Groq, Gemini, OpenRouter) for machines without a GPU, or for
a bigger model than 4 GB of VRAM can hold. Or CPU-only embeddings (FastEmbed), so
nothing needs an Ollama server. All of it without silently breaking the existing index.

```bash
make providers                          # which models your keys can use right now (+ OpenRouter's free list)
make providers PING=1                   # ... and one chat + one tool call per provider
LLM_PROVIDER=groq make check            # doctor adapts: no GPU checks, a privacy warning, a tool-calling test
EMBED_PROVIDER=fastembed make chat      # embeddings in-process, same vectors as Ollama's
```

## What we built

| File | Job |
|---|---|
| `jarvis_rag/llm.py` | `get_chat_model(provider=…)` for ollama / groq / gemini / openrouter. `MissingAPIKey` names the variable and where to get a free key. `get_fallback_model`. `FastEmbedEmbeddings`. |
| `jarvis_rag/vectorstore.py` | **Embedding fingerprint**: `check_embeddings()`, `EmbeddingMismatch` |
| `jarvis_rag/agent.py` | `ModelFallbackMiddleware` when `LLM_FALLBACK` is set; warns when prompts leave the machine; clean one-line startup errors |
| `jarvis_rag/providers.py` | Lists each provider's current models (OpenRouter's needs no key) and checks that the configured name exists; `--ping` |
| `jarvis_rag/check_env.py` | Provider-aware: skips GPU/VRAM for hosted LLMs, adds a **tool-calling** smoke test, embeds with whichever provider is set |
| `tests/test_providers.py` | 13 tests, all offline: key errors, each client's settings, `bind_tools` for every provider, fallback, FastEmbed wrapper, fingerprint |

New settings: `GROQ_API_KEY/GROQ_MODEL`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)/`GEMINI_MODEL`,
`OPENROUTER_API_KEY/OPENROUTER_MODEL`, `LLM_FALLBACK`, `LLM_MAX_RETRIES=3`, `LLM_TIMEOUT=120`,
`REASONING_EFFORT`, `FASTEMBED_MODEL=nomic-ai/nomic-embed-text-v1.5`, `FASTEMBED_BATCH_SIZE=1`.

## Choosing the defaults (checked, not remembered)

Free-tier catalogues change every month, so the defaults come from the providers' own
documentation and APIs at the time of writing, and `make providers` re-checks them.

| Provider | Default | Why |
|---|---|---|
| Groq | `openai/gpt-oss-20b` | A production model; Groq docs say all hosted models support tool use. GPT-OSS doesn't do *parallel* tool calls, which our agent never needs. The Llama 3.x models were being deprecated for free/developer tiers (Aug 2026 notice). |
| Gemini | `gemini-3.5-flash-lite` | The stable Flash-Lite model. Forum reports (unofficial) put free Flash at about 20 requests/day against about 500 for Flash-Lite. An agent makes 2–4 model calls per question, so 20/day is roughly 7 questions. |
| OpenRouter | `google/gemma-4-31b-it:free` | From the public `/models` list, filtered to `:free` **and** `tools` in `supported_parameters`: 16 models at the time of writing. A 31B instruct model with 262k context. |

Sources:
- [Groq models](https://console.groq.com/docs/models)
- [Groq tool use](https://console.groq.com/docs/tool-use/overview)
- [Groq deprecations](https://console.groq.com/docs/deprecations)
- [Gemini rate limits](https://ai.google.dev/gemini-api/docs/rate-limits)
- [Gemini 3.5 Flash-Lite](https://ai.google.dev/gemini-api/docs/models/gemini-3.5-flash-lite)
- [Gemini free-tier forum report](https://discuss.ai.google.dev/t/gemini-3-8-flash-free-tier-20-rpd-is-too-limited-for-practical-evaluation/180609)
- `https://openrouter.ai/api/v1/models`

**Not tested live:** no provider keys were configured during this phase, so the hosted
paths are covered by offline tests only. These check that each client is built with the
right model, key, base URL and retries, and that our tool schemas convert. A real call
needs a key: `make providers PING=1` sends one chat and one tool call. Then run
`make eval-agent` with `LLM_PROVIDER=groq` to see how a 20B model does on Phase 4's
19 scenarios, and try `--set auto_search=false` to see whether a bigger model still
needs that crutch.

## Dependencies: one pin we refused

| Package | Version | Note |
|---|---|---|
| `langchain-groq` | 1.1.3 | |
| `langchain-google-genai` | 4.4.0 | 4.4.1/4.4.2 were under two weeks old |
| `langchain-openai` | 1.6.7 | used for OpenRouter |
| transitive | `google-genai<2.29`, `google-auth<2.61`, `pycparser<3.1`, `openai<=3.24.0` | the resolver picked releases 1–8 days old; held back to ≥ two weeks |

The first choice for OpenRouter was the official `langchain-openrouter` integration. The
lock diff showed it **downgrading pydantic for the whole application** (2.14.1 → 2.12.5),
because its `openrouter` SDK requires `pydantic<2.13`. Changing a core dependency of
every module because of one optional provider is the wrong trade. OpenRouter speaks the
OpenAI protocol, so `ChatOpenAI(base_url="https://openrouter.ai/api/v1")` does the same
job with no pin. Lesson: **read the lock diff, not just the requirement you added.**

A global `uv pip compile --exclude-newer <two weeks ago>` was tried too. It fails here:
`pydantic-settings>=2.16` itself is younger than two weeks. Writing the lock to a new
file also lost uv's preference for the versions already pinned, and bumped unrelated
packages (one published that day). Re-locking *in place* (`make lock`) keeps existing
pins unless something forces a change.

## Hosted LLMs: what changes, and what doesn't

```
LLM_PROVIDER=groq ──▶ get_chat_model() ──▶ ChatGroq(model, api_key, temperature, max_retries, timeout)
                                  └────────▶ same BaseChatModel interface: bind_tools, invoke, stream
create_agent(model, tools, middleware=[…, ModelFallbackMiddleware(ChatOllama) if LLM_FALLBACK=ollama, …])
```

Nothing above `llm.py` knows which provider is in use. The agent, the eval scripts, the
API and the UI all call `get_chat_model()`.

- **Privacy.** A hosted model receives the system prompt (including the catalogue of
  your file names), your question, and every tool result: the retrieved chunks and SQL
  rows. `build_agent` logs a warning and `make check` shows a `!` line. Keep confidential
  corpora on Ollama.
- **Rate limits.** Free tiers return HTTP 429. Each SDK retries with exponential backoff
  and honours `Retry-After`, up to `LLM_MAX_RETRIES`. One agent question is 2–4 model
  calls (more with the SQL follow-through), so requests/minute limits bite sooner
  than you'd expect.
- **Fallback.** With `LLM_FALLBACK=ollama`, a model call that still fails after its
  retries (429, 5xx, timeout, no network) is retried on the local model.
  `ModelFallbackMiddleware` wraps every model call, so the switch happens per step, even
  in the middle of a tool loop (`test_fallback_model_takes_over_when_the_primary_fails`).
  A *missing key* is a configuration error, not a runtime failure: startup stops with
  one line, not a silent fallback.
- **Small-model scaffolding stays on.** Auto-search, the catalogue, the point-of-use
  footer and the SQL follow-through cost a bigger model little. Whether they still
  *help* is an empirical question for `make eval-agent` once you have a key.

## FastEmbed: the same vectors without Ollama

FastEmbed runs ONNX models in-process on CPU: no server, a ~520 MB download once into
`storage/models`. The question was whether its `nomic-ai/nomic-embed-text-v1.5` is
*compatible* with the index Ollama built using `nomic-embed-text`:

| Pair (4 prefixed texts) | cosine of the two vectors |
|---|---|
| Ollama `nomic-embed-text` vs FastEmbed `nomic-embed-text-v1.5` | **1.0000, 1.0000, 1.0000, 1.0000** |
| Ollama vs FastEmbed `nomic-embed-text-v1.5-Q` (quantized) | 0.959, 0.952, 0.947, 0.957 |

The same weights give the same space, whatever runs them. Quantization changes the
space enough to matter. Retrieval eval on the Ollama-built `jarvis` collection with
FastEmbed embedding the queries:

| query embedder | hit@1 | hit@5 | MRR (similarity) | MRR (rerank) | query p50 |
|---|---|---|---|---|---|
| Ollama | 0.85 | 0.96 | 0.903 | 0.956 | 25 ms |
| FastEmbed | 0.85 | 0.96 | 0.903 | 0.956 | 86 ms |

Identical quality. Ollama is faster per query (llama.cpp's CPU kernels beat ONNX
Runtime here), but FastEmbed needs no server, which is what a free CPU host needs
(Phase 9).

### The padding trap, again, and probably the cause of a crash

The first FastEmbed ingest took **67 s against Ollama's 20 s**, and the session running
the benchmark died partway through. FastEmbed's default batch is 256 (we had set 32), and a batch is padded to
its longest member. nomic accepts 8,192 tokens, and code chunks are long. Measured on 60
real chunks:

| batch size | time per chunk | peak RSS |
|---|---|---|
| 1 | **133 ms** | **1.1 GB** |
| 4 | 280 ms | 1.4 GB |
| 32 | 488 ms | 2.8 GB |

2.8 GB, on a 5.2 GB machine already holding Ollama, is a likely way to lose a process.
With `FASTEMBED_BATCH_SIZE=1` the full ingest takes **26.6 s** (113 chunks; Ollama:
20.1 s). Phase 3 found the same effect in the cross-encoder (2,342 → 281 ms). On CPU,
with variable-length inputs, small batches beat big ones.

### A smaller model: bge-small-en-v1.5

`BAAI/bge-small-en-v1.5`: 384 dimensions, 67 MB (nomic: 768-d, 520 MB). It uses a query
instruction and no document prefix, so it needs its own collection. Same 113-chunk corpus:

| | ingest | query p50 | MRR (similarity) | hit@1 | MRR (rerank) |
|---|---|---|---|---|---|
| nomic-v1.5 (FastEmbed) | 26.6 s | 39 ms | 0.903 | 0.85 | 0.956 |
| **bge-small-en-v1.5** | **8.1 s** | **10 ms** | **0.938** | **0.91** | 0.956 |

On this corpus the 7× smaller model is *better* at first-stage retrieval, and with the
cross-encoder both reach the same ceiling. With 53 questions that's a hint, not a law:
nomic's 8k context and wider training may win on long or unusual documents. Still, for
a CPU-only deployment bge-small + rerank is the obvious candidate. The default stays
nomic, because it's compatible with the existing index and with Ollama.

## The embedding fingerprint

Changing the embedding model behind an existing collection is the classic silent RAG
failure. Chroma can't tell, because a 768-d vector is a 768-d vector. Every search
returns *something*, scores look normal, and answers just get worse. So each collection
now records the vector of a fixed probe sentence in its metadata. Before the first
search or ingest, the current embedder must reproduce it (cosine ≥ 0.99):

```
$ EMBED_PROVIDER=fastembed FASTEMBED_MODEL=nomic-ai/nomic-embed-text-v1.5-Q python -m jarvis_rag.vectorstore "battery life"
EmbeddingMismatch: collection 'jarvis' was built with a different embedding model (dim 768 vs 768,
probe similarity 0.956). Switch EMBED_PROVIDER/EMBED_MODEL/FASTEMBED_MODEL back, or set a new
COLLECTION_NAME and re-ingest.
```

- **Checked lazily.** Opening a `KnowledgeBase` needs no model, so the API still starts
  (and reports `degraded`) when Ollama is down.
- **Existing collections adopt it.** A collection from before Phase 7 has no fingerprint;
  the first embedder to use it is recorded. That is correct as long as it's the one
  that built it, as here.
- **It compares behaviour, not names.** "nomic-embed-text" (Ollama) and
  "nomic-ai/nomic-embed-text-v1.5" (FastEmbed) are different names for the same space,
  and are accepted. "-Q" is almost the same name for a different space, and is refused.
  Comparing configuration strings would get both cases wrong.
- **It also guards ingest.** Writing vectors from the wrong space into a collection
  would corrupt it for everyone (`test_fingerprint_guards_ingest_too`).

## Tests (14 new, 210 total, all offline)

- `tests/test_providers.py` (13):
  - missing-key messages for all three providers;
  - each client's model, key, retries and reasoning effort; `GOOGLE_API_KEY` accepted;
  - OpenRouter via the OpenAI protocol (base URL, attribution header);
  - `bind_tools` for every provider;
  - fallback selection;
  - FastEmbed wrapped with prefixes, batch size 1 (fake ONNX model);
  - fingerprint adopted, then enforced for a different dimension and for a same-dimension
    "flipped" space, and guarding ingest.
- `tests/test_agent.py` (+1): a primary model that always fails with a 429-style error
  → the fallback model answers.

## Try it

```bash
make providers                                     # OpenRouter's free tool-capable models, no key needed
# get a free Groq key, then:
echo "GROQ_API_KEY=gsk_..." >> .env
make providers PING=1
LLM_PROVIDER=groq LLM_FALLBACK=ollama make chat
LLM_PROVIDER=groq .venv/bin/python eval/run_agent_eval.py --json eval/results/agent_groq.json
LLM_PROVIDER=groq .venv/bin/python eval/run_agent_eval.py --set auto_search=false

# a CPU-only setup, no Ollama at all for retrieval:
COLLECTION_NAME=jarvis-bge EMBED_PROVIDER=fastembed FASTEMBED_MODEL=BAAI/bge-small-en-v1.5 \
EMBED_QUERY_PREFIX="Represent this sentence for searching relevant passages: " EMBED_DOCUMENT_PREFIX= \
  make ingest eval
```

## Quiz

1. Ollama's `nomic-embed-text` and FastEmbed's `nomic-embed-text-v1.5` are interchangeable,
   but `nomic-embed-text-v1.5-Q` is not, although it is "the same model, quantized".
   Why, and why does a fingerprint catch this where comparing model names can't?
2. `LLM_FALLBACK=ollama` is set, `GROQ_API_KEY` is missing, and the agent refuses to
   start instead of falling back. Bug or feature?
3. Batch size 32 was 3.7× *slower* than batch size 1 for FastEmbed on CPU. Batching is
   supposed to make things faster. What is going on?

<details><summary>Answers</summary>

1. Interchangeable means vectors for the same text point the same way. The fp32 ONNX
   export and Ollama's GGUF are the same weights, so cosine is 1.0000. Quantizing to
   8-bit (or fewer) changes every weight a little, and the errors add up through 12
   layers into vectors about 0.95 similar. That's enough to reorder near-ties in
   retrieval, and it is a different space. Names are arbitrary labels: two different
   names here describe the same space, and two nearly identical names describe
   different ones. A fingerprint tests what actually matters, whether the embedder
   reproduces a known vector.
2. A feature. Fallback exists for *transient runtime* failures: rate limits, outages, a
   dropped connection. A missing key is a *configuration* error that will never fix
   itself. Silently running everything on the fallback would hide it, and would quietly
   switch a user who chose a 20B hosted model to a 4B local one. Fail fast, with the fix
   in the message.
3. Padding. A batch is a rectangle: every sequence is padded to the longest one. Our
   chunks range from a few hundred to over a thousand tokens, so a batch of 32 pays the
   longest chunk's cost 32 times, and transformer attention grows quadratically with
   length. On a GPU the wasted work runs in parallel and batching still wins. On a
   6-core CPU there's little parallelism to gain and the waste is real time (and RAM).
   Sorting by length before batching is the other fix; batch size 1 is the simplest.

</details>
