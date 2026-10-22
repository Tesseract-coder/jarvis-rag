"""QLoRA fine-tune of Qwen3-1.7B as the JARVIS reader, sized for a 4 GB GTX 1650.

    .venv-train/bin/python finetune/train.py                 # ~train.jsonl/val.jsonl from make_dataset.py
    .venv-train/bin/python finetune/train.py --max-steps 5   # smoke test: does it fit in VRAM?

Stop Ollama's resident models first (`make train` does): the 4B model alone holds
3.6 GB of the card's 4 GB.

QLoRA = the base weights frozen in 4-bit (bitsandbytes NF4, ~1.2 GB for 1.7B params)
+ small trainable low-rank matrices (LoRA) on every attention and MLP projection.
Only the LoRA matrices (~1% of parameters) get gradients and optimizer state.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from unsloth import FastLanguageModel  # import first: it patches transformers/trl for speed and memory
from unsloth.chat_templates import train_on_responses_only

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune" / "out"
sys.path.insert(0, str(ROOT / "finetune"))
from common import READER_PROMPT  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", default="unsloth/Qwen3-1.7B-unsloth-bnb-4bit")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=-1, help="override epochs (smoke tests)")
    ap.add_argument("--checkpointing", choices=["unsloth", "plain"], default="unsloth",
                    help="unsloth = offload checkpointed activations to CPU RAM; plain = keep them on the GPU")
    ap.add_argument("--no-eval", action="store_true", help="skip validation loss (profiling)")
    ap.add_argument("--resume", action="store_true", help="continue from the last checkpoint in out/checkpoints")
    ap.add_argument("--save-steps", type=int, default=3, help="checkpoint every N optimizer steps (~18 min here)")
    args = ap.parse_args()

    torch.cuda.reset_peak_memory_stats()
    model, tok = FastLanguageModel.from_pretrained(
        args.base, max_seq_length=args.max_len, load_in_4bit=True,
        dtype=None,  # auto: float16 here; Turing (GTX 16xx) has no bfloat16
    )
    loaded = torch.cuda.max_memory_allocated() / 2**20
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # recompute activations in backward: trades ~20% time for VRAM ("unsloth" also offloads them to CPU RAM)
        use_gradient_checkpointing="unsloth" if args.checkpointing == "unsloth" else True,
        random_state=3407,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    def render(ex):  # the exact chat template Ollama will use, thinking disabled
        # Always the current reader prompt, so a prompt fix doesn't require regenerating data.
        messages = [{"role": "system", "content": READER_PROMPT}, *ex["messages"][1:]]
        return {"text": tok.apply_chat_template(messages, tokenize=False, enable_thinking=False)}

    ds = load_dataset("json", data_files={"train": str(OUT / "train.jsonl"), "val": str(OUT / "val.jsonl")})
    ds = ds.map(render, remove_columns=ds["train"].column_names)
    lengths = [len(tok(t)["input_ids"]) for t in ds["train"]["text"]]
    print(f"{len(ds['train'])} train / {len(ds['val'])} val examples; tokens per example: "
          f"median {sorted(lengths)[len(lengths) // 2]}, max {max(lengths)} (limit {args.max_len})")

    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=ds["train"], eval_dataset=ds["val"],
        args=SFTConfig(
            dataset_text_field="text", max_length=args.max_len, packing=False,
            per_device_train_batch_size=1, gradient_accumulation_steps=args.grad_accum,  # effective batch 8
            num_train_epochs=args.epochs, max_steps=args.max_steps,
            learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.05, weight_decay=0.01,
            fp16=True, bf16=False, optim="adamw_8bit",  # 8-bit optimizer state: another VRAM saving
            logging_steps=1 if args.max_steps > 0 else 5,
            eval_strategy="no" if args.no_eval else ("steps" if args.max_steps > 0 else "epoch"),
            eval_steps=args.max_steps if args.max_steps > 0 else None,
            # Eval defaults to batch 8 and keeps full logits (8 x 1,700 tokens x 152k vocab):
            # that asked for 2.94 GB on a 3.6 GB card. Batch 1, loss only.
            per_device_eval_batch_size=1, prediction_loss_only=True,
            # ~45 s per example on a GTX 1650: a 4-hour run must survive a crash or a reboot.
            save_strategy="no" if args.max_steps > 0 else "steps", save_steps=args.save_steps, save_total_limit=2,
            output_dir=str(OUT / "checkpoints"), report_to="none", seed=3407,
        ),
    )
    # Loss only on the assistant's answer, not on the system prompt and passages the
    # model is merely conditioned on: otherwise most of the gradient teaches it to
    # predict documentation text instead of how to answer.
    trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n",
                                      response_part="<|im_start|>assistant\n")

    before = None
    if not args.no_eval and not args.resume:  # the untrained model's val loss: the baseline to beat
        before = round(trainer.evaluate()["eval_loss"], 4)
        print(f"val loss before training: {before}", flush=True)
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=True if args.resume else None)
    minutes = (time.time() - t0) / 60
    examples = trainer.state.global_step * args.grad_accum
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    report = {
        "base": args.base, "rank": args.rank, "epochs": args.epochs, "steps": trainer.state.global_step,
        "trainable_params": trainable, "total_params": total, "minutes": round(minutes, 1),
        "seconds_per_example": round(minutes * 60 / max(examples, 1), 1),
        "train_loss": round(stats.training_loss, 4), "eval_loss_before": before,
        "eval_loss": [round(e["eval_loss"], 4) for e in evals],
        "vram_after_load_mib": round(loaded), "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
        "loss_curve": [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h],
    }
    print(json.dumps({k: v for k, v in report.items() if k != "loss_curve"}, indent=1))
    if args.max_steps < 0:
        model.save_pretrained(str(OUT / "lora"))  # adapter only: ~70 MB
        tok.save_pretrained(str(OUT / "lora"))
        (OUT / "train_report.json").write_text(json.dumps(report, indent=1))
        print(f"adapter -> {OUT / 'lora'}")


if __name__ == "__main__":
    main()
