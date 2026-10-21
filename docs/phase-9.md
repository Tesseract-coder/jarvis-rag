# Phase 9: Hugging Face Spaces

**Goal:** a shareable web demo on Hugging Face's free CPU tier: the same code, a
read-only knowledge base baked into the image, FastEmbed for embeddings and a hosted
free-tier LLM. Publishing to an external service is outward-facing, so everything is
built and tested locally first, and the push happens only after the owner decides on
account, keys and visibility.

> **Status:** bundle built and tested locally, exactly as HF runs it (uid 1000, port 7860,
> secrets as env vars, even with no network). **Not deployed: Hugging Face now requires a
> paid PRO subscription to host Docker (and Gradio) Spaces**, even on the basic CPU tier.
> This project uses no paid services, so the Space stays a ready-to-push bundle. See
> "What happened when we pushed" below.

```bash
make space          # assemble build/hf-space: exactly what would be published
make space-test     # docker build it: bakes the index, gates on the retrieval eval
cp deploy/hf-space/secrets.example .space-secrets && $EDITOR .space-secrets
hf auth login                                   # in your own terminal (interactive)
make space-push SPACE=<your-user>/jarvis-rag    # private by default; PUBLIC=1 to make it public
```

## What we built

| File | Job |
|---|---|
| `deploy/hf-space/Dockerfile` | The Space image: uid 1000, port 7860, FastEmbed bge-small, Groq, `KB_READ_ONLY`, demo index and models **baked in**, build **gated on the retrieval eval** |
| `deploy/hf-space/README.md` | The Space card. Its YAML front matter is the Space config (`sdk: docker`, `app_port: 7860`) |
| `deploy/build_space.sh` | Assembles `build/hf-space`: code, lock file, demo sources, eval. No `.env`, `storage/`, `data/` or git history |
| `deploy/hf-space/secrets.example` | `GROQ_API_KEY`, `API_KEY`, `UI_PASSWORD`, sent as Space secrets, never committed |
| `jarvis_rag/api.py`, `ui.py`, `config.py` | `KB_READ_ONLY` (403 on ingest/delete, write controls hidden), `/` → `/ui/` redirect, `read_only` in `/health`, provider-aware "unreachable" message |
| `eval/run_eval.py` | `--min-hit5`: exits 1 below the threshold or if any gold label is unreachable. This is the build gate. |
| `jarvis_rag/demo_data.py` | Raises if the Markdown sample docs are missing (it used to glob nothing, silently) |

## How Spaces shape the design

Verified against the current docs:
[Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker),
[config reference](https://huggingface.co/docs/hub/en/spaces-config-reference),
[CLI](https://huggingface.co/docs/huggingface_hub/main/guides/cli).

| Space fact | Consequence here |
|---|---|
| The app must listen on `app_port` (default 7860) | `API_PORT=7860`; one process serves the API and the UI (`--with-api`), and `/` redirects to `/ui/` |
| The container runs as **uid 1000** | `useradd --uid 1000`; `/app` owned by it before anything is written |
| **The disk is wiped on restart**; persistent storage is no longer offered for new Spaces | The demo corpus, the Chroma index *and* every model download are built **into the image**. Cold start: no network, no re-ingest. Conversations and anything written at runtime reset on restart, so the KB is read-only anyway. |
| No GPU, no Ollama on the free tier (2 vCPU, 16 GB) | `EMBED_PROVIDER=fastembed` with **bge-small** (Phase 7: 3× faster ingest, better first-stage MRR), plus the cross-encoder; `LLM_PROVIDER=groq` |
| Secrets come in as environment variables | `GROQ_API_KEY`, `API_KEY`, `UI_PASSWORD` via `hf repos create --secrets-file`, never in the repo |
| Spaces are public by default and run on your API quota | Private by default (`make space-push`); `UI_PASSWORD` required either way; `KB_READ_ONLY=true`, so visitors can't upload files or crawl URLs from your Space |

Why a password even on a *private* Space? The app listens on `0.0.0.0` inside the
container and can't know HF's access control sits in front of it. Phase 8's rule
applies: if the app might be exposed, it demands credentials. Inside the
huggingface.co page the Space runs in an iframe, where browsers may block Gradio's
login cookie. The direct `https://<user>-<space>.hf.space/ui/` URL always works (noted on
the Space card).

## The bug the image almost shipped with

The first Space image built fine, started fine, passed health and auth checks, and
answered "Which robot has the longer battery life?" from the **maintenance manual** with
a reranker score of −4.63. The knowledge base had **18 sources instead of 23**. All five
Markdown documents (handbook, product specs, FAQ, IT policy, release process) were
missing.

Cause: `demo_data` copies them from `learn/sample_docs/`, which the bundle didn't
include. `Path.glob()` on a missing directory returns nothing, so there was no error and
no warning, just a smaller corpus and worse answers. Every check that ran (health, auth,
read-only, "retrieval returns something") passed, because none of them measured
*quality*.

Two fixes:
1. **Fail loudly at the source.** `write_demo_data` raises `FileNotFoundError` when it
   finds no sample docs. An empty glob is not an error to Python, but it is a broken
   corpus.
2. **Gate the build on the eval.** The Space Dockerfile now ends with
   `python eval/run_eval.py --modes rerank --min-hit5 0.95`. A missing document, a
   broken embedder or a bad prefix now fails the `docker build`, instead of shipping a
   demo that quietly answers worse. Phase 3's eval became a deployment test.

The gate also rejected the next build, for a different reason. **Hybrid** scored hit@5
0.943 (< 0.95), because the Space indexes *today's* code snapshot (23 sources, 235
chunks: `api.py`, `ui.py`, `agent.py`…), so more code chunks compete with the
documents than in the local corpus. The served mode, **rerank**, scored 1.00. The gate
now tests what is served (`--modes rerank`); lowering the threshold until hybrid passed
would have defeated the point.

### Eval contamination, spotted on the way

"Where is the chunk id hash computed?" ranks `agent.py` first in the Space. Its
docstring quotes that exact question, as the Phase 4 example of a model answering from
imagination. The correct answer (`vectorstore.py`) is still in the top 5, so the gate
passes, but rerank MRR falls from 0.956 to 0.926. When the corpus contains the code
that discusses the test set, the test set leaks into the corpus. A good reason to keep
eval questions out of the documents being searched.

## Local test, the way HF runs it

`docker run --network none --user 1000 -e API_KEY=… -e UI_PASSWORD=… -e GROQ_API_KEY=<fake>`:

| Check | Result |
|---|---|
| image build (cold / code-only change) | ~2 min; 1.61 GB on disk, 426 MB compressed (models + index included) |
| ingest during the build | 23 sources, 235 chunks in 13.9 s (bge-small, CPU) |
| eval gate (rerank) | hit@1 0.89, hit@5 **1.00**, MRR 0.926, all gold labels reachable |
| `/health` with **no network** | 200 `{"ollama": null, "provider": "groq", "read_only": true}`; no download attempted |
| `/` | redirects to `/ui/` |
| `/ui/config` without login | 401 |
| `DELETE /sources/…`, `POST /ingest/url` | 403 "the knowledge base is read-only on this deployment" |
| offline retrieval | battery → `product_specs.md` · E-217 → maintenance manual · dog policy → `office_faq.md` |
| `/chat` with no network | 503 "a model service is unreachable (APIConnectionError): can this server reach groq?" |

Not yet tested: a real answer from Groq on HF's own infrastructure. That needs the key and
the push.

## Deploying (needs a PRO account; see the next section)

1. **Account and login.** Create a free account at huggingface.co, then in *your own
   terminal* (it's interactive): `hf auth login` with a token that has **write**
   access. The token is stored in `~/.cache/huggingface/`, never in this repo.
2. **Secrets.** `cp deploy/hf-space/secrets.example .space-secrets`, then fill in
   `GROQ_API_KEY` (free: console.groq.com/keys), a random `API_KEY` and a `UI_PASSWORD`.
   `.space-secrets` is gitignored.
3. **Visibility.** `make space-push SPACE=<user>/jarvis-rag` creates a **private**
   Space. `PUBLIC=1` makes it public: the URL is then shareable, but the UI still needs
   the password, and every question uses *your* Groq quota.
4. Watch the build log on the Space page (~5 min on HF's builders). The eval gate runs
   there too.
5. Updating: `make space-push SPACE=…` again. `--exist-ok` keeps the Space and replaces
   the files; each push is a commit in the Space repo, tagged with the local git hash.

## What happened when we pushed

Everything up to the push worked:
- `hf auth login` (token with write permission).
- `.space-secrets` filled in.
- The Groq key checked locally first: `make providers` → `groq --ping` → "ready" in 0.6 s,
  and a correct tool call in 0.2 s.

Then `hf repos create … --type space --sdk docker --private` returned:

```
Error: Client error '402 Payment Required' for url 'https://huggingface.co/api/repos/create'
Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on free cpu-basic
requires a PRO subscription.
```

Nothing was created or uploaded. The repo creation failed before any file or secret was
sent (checked afterwards: `repo_exists` → False, no Spaces on the account). The
documentation pages read earlier the same day still described `cpu-basic` as the free
default. Pricing pages and API behaviour can change before docs do, and the API is
the authority.

What remains true and useful:
- The bundle is a complete Docker Space. With PRO (or on an organisation that has it),
  `make space-push SPACE=<user>/jarvis-rag` deploys it unchanged, and the eval gate runs
  on HF's builder too.
- The same image runs on any Docker host: `make space-test`, then `docker run` with the
  three secrets.
- A *static* Space (free) can't run this app: retrieval, the reranker and the agent need a
  Python server.

Free alternatives considered, not done:
- **A free tunnel to the local Docker app** (`cloudflare/cloudflared` as a container,
  with `API_KEY`, `UI_PASSWORD` and `KB_READ_ONLY`). It works only while the PC is on,
  and it exposes the machine.
- **Other free container hosts.** Most now require a payment card or offer about 512 MB
  of RAM, against the ~1.5 GB this app uses.

If you're done with it, delete `.space-secrets`: it holds your Groq key (gitignored,
mode 600).

## Tests (+4, 223 total)

- `tests/test_api.py` (+2): read-only deployment blocks all three writes but allows
  chat and listing; "unreachable" names the right service.
- `tests/test_ui.py` (+1): read-only UI hides the upload/URL and delete rows.
- `tests/test_docker.py` (+1): the bundle contains exactly the publishable files and no
  `.env`/`storage`/`data`/`.git`/secrets; the Space card declares `sdk: docker`,
  `app_port: 7860`; the Space Dockerfile keeps the eval gate, read-only and
  `--require-hashes`.
- `tests/test_eval.py`, `tests/test_ingest.py`: unchanged and passing.

## Quiz

1. The broken image passed health, auth and "retrieval returns results" checks. What
   kind of check would have caught it, and why did none of the existing ones?
2. Why bake the index and models into the image instead of ingesting at startup? Name
   one downside.
3. The Space is private on Hugging Face. Why does the app still require `UI_PASSWORD`
   and `API_KEY`?

<details><summary>Answers</summary>

1. A *quality* check against known answers: the retrieval eval with a threshold. Health
   checks prove the process is up, auth checks prove the doors are locked, "returns
   results" proves the index isn't empty. A knowledge base missing a fifth of its
   documents passes all three, because it still returns plausible-looking results, just
   wrong ones. Only comparing results with expected sources (hit@5, gold labels
   reachable) notices that the right document is *absent*.
2. Spaces wipe the disk on every restart and sleep when idle. Ingesting at startup would
   mean a slow cold start (download ~150 MB of models, embed 235 chunks) on every wake-up,
   a runtime dependency on the Hugging Face model hub, and a deployment that could start
   broken without anyone seeing a build failure. Baking makes startup instant, offline and
   identical to what the gate tested. Downsides: a bigger image (1.61 GB), and any corpus
   change needs a rebuild and push. Fine for a demo, wrong for a knowledge base users edit.
3. Defence in depth, and the rule from Phase 8: the app decides from what it can see
   (binding `0.0.0.0`), not from access control configured elsewhere. HF's private
   setting can be flipped to public in one click (or by `PUBLIC=1`), and then the only
   protection left would be the app's own. Also, private Spaces can be shared with an
   organisation, and the API key protects programmatic access that the Gradio login
   doesn't cover.

</details>
