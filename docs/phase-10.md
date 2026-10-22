# Phase 10: Fine-tuning a reader (QLoRA → GGUF → Ollama)

**Goal:** teach a small local model one narrow skill, reading retrieved passages and
answering with a citation or an exact abstention. Then measure whether it beats
the base model, and the bigger model it would replace, on the *same* contexts.
Everything runs on the 4 GB GTX 1650, with only free services.

```bash
make train-setup     # separate .venv-train: Unsloth + PyTorch CUDA wheels (~6 GB), hash-locked
make train-data      # Groq teacher writes Q/A pairs from docs/ + learn/ (resumable)
make train           # QLoRA, 1 epoch, ~4 h on a GTX 1650 (stops Ollama's models first)
make export-reader   # merge -> f16 GGUF -> Q4_K_M -> `ollama create jarvis-reader:1.7b`
make eval-reader     # qwen3:1.7b vs jarvis-reader:1.7b vs qwen3:4b-instruct, judged by Groq
```

## Result

103 cases, identical cached contexts for every model, temperature 0, same system
prompt. Groq `gpt-oss-120b` judged correctness (0 unjudged rows).

| model | correct | cites source | wrong abstain | safe when absent | hallucinated | bad-cite rows | words | p50 s | tok/s |
|---|---|---|---|---|---|---|---|---|---|
| qwen3:1.7b (base) | **90%** | 18% | 2% | 74% | 14/53 | 17 | 28 | 2.3 | 56 |
| **jarvis-reader:1.7b** | 88% | **84%** | 2% | 83% | 9/53 | **13** | **22** | **2.1** | 59 |
| qwen3:4b-instruct | 88% | 46% | 2% | **94%** | **3/53** | 26 | 22 | 4.7 | 28 |

- **cites source** (answerable cases): the answer's citation matches the passage that
  actually holds the answer.
- **safe when absent** (removed + unanswerable cases): the model abstained, or was
  still correct because a second document held the fact.
- **bad-cite rows**: answers that cite a label that doesn't appear in the context.

What the fine-tune bought, and what it didn't:

- **Citations: 18% → 84%.** This is the skill the training data was mostly about, and
  it's the clearest win. The base model knows the facts but cites the wrong thing, or
  writes "this is stated in the … passage" instead of a label.
- **Abstention: 74% → 83%, with the exact sentence.** The base model's hedges
  ("not explicitly stated…") are harder to detect. Two of its 14 "hallucinations" are
  really hedges that the abstention regex misses, so the true base figure is ~78%. The
  fine-tune says `I don't know based on the available sources.` verbatim, so the UI and
  the eval can detect it reliably.
- **Correctness unchanged** (88% vs 90%: one question out of 50). It didn't learn
  facts, as intended. The training corpus contains no Halcyon text.
- **Still behind the 4B at refusing.** Nine of its 53 "answer is absent" cases got a
  confident answer stitched together from *nearby* passages (core hours from building
  access hours, "car at the office" from the remote-work policy). The 4B does this
  three times. Twice as many parameters buys judgement that 312 examples don't.
- **2.2× faster than the 4B** (2.1 s vs 4.7 s median, 59 vs 28 tok/s) and 1.1 GB of
  VRAM instead of 2.5 GB.

So it is a better 1.7B reader, not a replacement for the 4B agent. A sensible use is
the CPU/low-VRAM fallback, or a reader stage behind a separate tool-calling model. It
was never trained on tool calls, so `CHAT_MODEL=jarvis-reader:1.7b` would break the
agent loop.

### One finding that applies to every model: two things look like a label

Markdown chunks start with a breadcrumb in brackets, so a passage reads:

```
[data/docs/employee_handbook.md]
[Halcyon Robotics — Employee Handbook > Annual leave]
Every full-time employee receives 25 days …
```

All three models sometimes cite the *second* bracket. That is most of the "bad cite"
column: 13, 17 and 26 rows, and it's part of why the 4B scores only 46% on citing
the source. The fine-tune reduced this but never fully got rid of it, because the
training contexts have the same layout. The fix belongs in the format, not the model:
render the breadcrumb as `Section: …`, without brackets. That changes what the agent sees too,
so it's a Phase 11 change, followed by retraining and re-running this eval.

## Pipeline

```
docs/*.md, README, learn/*.py ──chunk──► 213 clean chunks (19 files)
        │  drop any chunk mentioning eval terms (Halcyon, Kestrel, E-217, Lisbon, …)
        ▼
Groq teacher (gpt-oss-20b, json_schema) ──► question + 1-3 sentence answer per chunk
        │  60% answerable: target + 4 distractors       → answer + [target label]
        │  40% unanswerable: 5 distractors, checker-verified → exact abstention
        │  + 1 teacher-free augmentation per train row (new distractors / order)
        ▼
train 312 (194 answerable / 118 unanswerable)   val 38 (3 whole files held out)
        ▼
QLoRA on unsloth/Qwen3-1.7B-unsloth-bnb-4bit  (r=16, all 7 projections, 1 epoch)
        ▼
merge (fp16) → convert_hf_to_gguf (f16, 3.2 GB) → llama-quantize Q4_K_M (1.03 GB)
        ▼
ollama create jarvis-reader:1.7b  (qwen3:1.7b's chat template + stop tokens)
```

| File | Job |
|---|---|
| `finetune/common.py` | `READER_PROMPT`, the abstention sentence, the context format. Shared by data, training and eval, so the model sees at inference the same prompt it was trained on. |
| `finetune/make_dataset.py` | Corpus, contamination filter, teacher and checker, resumable `raw.jsonl`, held-out split, augmentation |
| `finetune/train.py` | Unsloth QLoRA, loss on the answer only, checkpoints every 3 steps, `train_report.json` |
| `finetune/export.py` | Merge, GGUF, quantize, Modelfile, `ollama create`; clones/builds llama.cpp's quantizer if missing |
| `finetune/eval_reader.py` | Cached contexts, generation via Ollama, scoring, Groq judge, summary table |
| `requirements-train.{in,txt}` | 104 hash-locked packages, nothing newer than 2026-10-07 |

## Concepts

**LoRA.** Freeze the weight matrix W (d×d) and learn a low-rank update B·A, with
B d×r and A r×d, r = 16. Only A and B get gradients. Here that is 17.4 M trainable
parameters out of 1.74 B (1.0%), applied to q, k, v, o, gate, up and down in all 28
layers. After training, `W + B·A` is merged back, so inference costs nothing extra.

**QLoRA.** The frozen W is stored in 4-bit NF4 (bitsandbytes) and dequantized
on the fly in each forward pass. The model loads in 1.4 GB, not ~3.4 GB in fp16.
Add 8-bit AdamW state and gradient checkpointing (activations recomputed in the
backward pass, and with Unsloth's variant also offloaded to CPU RAM), and peak VRAM was
3.2 GB. That's how a 1.7B model trains on a 4 GB card.

**Loss only on the response.** `train_on_responses_only` masks everything before
`<|im_start|>assistant`. Otherwise ~95% of every example's tokens would be the system
prompt and passages, and the model would mostly learn to predict documentation text.

**Turing constraints.** The GTX 1650 has no bfloat16 and no tensor cores, so training
runs in fp16 with loss scaling. Unsloth picks this automatically with `dtype=None`.

**Why GGUF, and why quantize twice?** The QLoRA 4-bit format (NF4) is a training
format that Ollama can't load. Export merges into fp16, converts to GGUF (llama.cpp's
single-file format), then re-quantizes to Q4_K_M: llama.cpp's 4-bit scheme with
per-block scales, the same one the stock `qwen3:1.7b` uses. With the same
quantization on both sides, the comparison measures the fine-tune.

## Training run

| | |
|---|---|
| data | 312 train / 38 val; 1,264 tokens per example median, 1,715 max (5 passages) |
| steps | 39 (batch 1 × gradient accumulation 8), lr 2e-4 cosine, warmup 5% |
| time | **243.6 min, 46.9 s per example**; GPU-bound (~100% util), 6 min per step |
| VRAM | 1.4 GB after load, **3.2 GB peak** of 4 GB |
| val loss | **2.29 → 0.27** (held-out files the model never saw) |
| train loss | 2.93 (steps 1-5) → 0.67 → 0.80 → 1.28 → 0.37 → 0.40 |

The jump at step 20 (1.28) was a window of harder examples, not divergence. It was
still the first epoch, so every batch was new data, and the next window fell to 0.37.
Answerable answers are paraphrases with many valid wordings. Abstentions are one
fixed sentence the model soon predicts almost perfectly, so the loss of a 5-step window
depends on its mix of the two.

## What went wrong, and the fixes

| Problem | Cause | Fix |
|---|---|---|
| 345 of 450 teacher calls failed: "Tool choice is required, but model did not call a tool" | `with_structured_output` defaults to a forced tool call, which gpt-oss on Groq sometimes ignores | `method="json_schema"`: Groq constrains the reply to the schema |
| 429s, then a whole day's quota gone | Groq free tier per model: 1,000 req/day, 8,000 tokens/min, **200,000 tokens/day**. 120b and then 20b both hit the daily cap. | 20b teacher, 1-2 workers, `raw.jsonl` written per row so reruns resume; augmentation doubles the data without the teacher |
| Models cited `[docs/report.md]`, a file that doesn't exist | The first prompt gave an example label, and small models copy examples literally | Describe the label ("copied exactly as it appears above the passage"), never show one. The first baseline numbers were invalid and were re-run. |
| OOM in the *eval* step of the smoke test (2.94 GB asked for) | `SFTTrainer` evaluates with batch 8 and keeps full logits: 8 × 1,700 × 152k vocab | `per_device_eval_batch_size=1`, `prediction_loss_only=True` |
| The judge crashed mid-run and 100 generations were lost | Generation and judging in one unsaved step | Save generations first; the judge retries, paces itself (4 s per call), and `--rejudge` fills only the gaps |
| The first 4-hour run was gone after 5 minutes, with no checkpoint | **The laptop suspended.** The CUDA context doesn't survive (`NVRM: … NV_ERR_NO_MEMORY` in nvidia-suspend). | Run under `systemd-inhibit --what=sleep:idle:handle-lid-switch`: blocks suspend only while the process lives. Checkpoints every 3 steps (~18 min) for anything else. |
| `ollama create -q q4_K_M` refused the GGUF | Ollama 0.40 quantizes only safetensors imports, and only to its own types (int4, int8, mxfp4, nvfp4, mxfp8) | Build only `llama-quantize` from the pinned llama.cpp (CPU, cmake via `uvx`, no system packages), quantize, then `ollama create` without `-q` |
| `pytest` from the repo root collected llama.cpp's own tests | `finetune/llama.cpp` is a full checkout | `make test` runs `pytest tests` |

## Contamination: the rule that shaped the dataset

The eval asks about Halcyon Robotics, a fictional company. Training on Halcyon text would
let the model memorize answers and look better than it is. So:

1. The training corpus is this project's *own* docs and teaching scripts (RAG, embeddings,
   Docker…), a different domain from the eval.
2. Any chunk matching `halcyon|kestrel|wren|E-\d{3}|lisbon|battery|…` is dropped. The
   phase docs quote eval questions as examples (see Phase 9's "eval contamination"),
   so this filter caught real leaks.
3. Validation holds out 3 whole files, so val loss measures transfer to unseen
   documents, not unseen paragraphs of familiar ones.

The result shows it worked: correctness on Halcyon stayed where the base model had it,
and only the *behaviour* changed (citing, abstaining, length).

## Things to try

- Change the breadcrumb format (`Section: …`, no brackets), regenerate the data with
  `make_dataset.py --n 0` (no teacher calls needed), retrain, and watch "bad cites" fall.
- Train 2 epochs from the existing checkpoint (`train.py --epochs 2 --resume`) and check
  whether "safe when absent" improves or the model starts over-abstaining ("wrong abstain").
- More unanswerable examples that are *near misses* (a distractor from the same file as
  the target). This is the exact failure behind most of the 9 hallucinations.
- Rank 8 vs 32: does a smaller adapter lose anything on 312 examples?
- `qwen3:4b` as the fine-tuning base: it won't fit 4 GB for training at this context
  length. Try a free Colab/Kaggle T4 with the same scripts.

## Tests (+5, 228 total)

`tests/test_finetune.py`: the context format round-trips (`user_message` →
`parse_context`); the prompt contains no example path; the contamination filter drops
Halcyon chunks; augmentation keeps the answering passage; eval scoring (abstention,
invalid citations, expected source). None of them need a GPU, Ollama or a network.

## Quiz

1. Validation loss fell from 2.29 to 0.27, but correctness didn't change. Why is that
   not a contradiction, and which columns show what the loss drop *did* buy?
2. Why mask the loss to the assistant's answer only? What would the model learn otherwise?
3. The fine-tuned 1.7B hallucinated on 9 of 53 "answer absent" cases, the 4B on 3.
   Which training-data change would most directly target that gap, and why?

<details><summary>Answers</summary>

1. Loss measures how well the model predicts *this dataset's* answers: their format,
   their citation, the exact abstention sentence. Much of the drop is the model learning
   that format, which "cites source" (18% → 84%), "words" (28 → 22) and the exact
   abstentions show. Correctness on Halcyon depends on reading facts from passages, which
   the base model could already do, and the training set deliberately contained no
   Halcyon facts to memorize.
2. ~95% of each example is system prompt and passages. Without the mask, most of the
   gradient would teach the model to predict documentation text (language modelling
   on the docs), and only a sliver would teach it how to answer. The mask spends the
   whole update on the behaviour.
3. Harder unanswerable examples: distractors from the *same* document or topic as the
   missing answer (near misses), not random other files. The 9 failures all answered from
   a neighbouring passage that was related but wrong. Random distractors are usually
   obviously off-topic, so the model never practised refusing when something *looks*
   relevant.

</details>
