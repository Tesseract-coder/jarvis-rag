"""LoRA adapter -> merged fp16 model -> GGUF -> Q4_K_M -> Ollama model `jarvis-reader:1.7b`.

    .venv-train/bin/python finetune/export.py

1. merge     Unsloth folds the LoRA matrices into the base weights (W + B·A) and writes
             a plain fp16 Hugging Face checkpoint (~3.4 GB). After this, no PEFT needed.
             (First run downloads the 16-bit base, unsloth/Qwen3-1.7B, 3.2 GB.)
2. GGUF      llama.cpp's convert_hf_to_gguf.py (pinned release b11457) rewrites it as
             one f16 GGUF file: the format Ollama/llama.cpp load.
3. quantize  llama.cpp's llama-quantize -> Q4_K_M (~1.1 GB), the same scheme as the stock
             qwen3:1.7b, so the comparison is about the fine-tune, not the quantization.
             Ollama 0.40 can't do this itself: `ollama create -q` refuses GGUF input, and
             for safetensors it only offers its own int4/int8/mxfp4/... types.
4. Ollama    `ollama create` registers it with the base model's chat template and stop
             tokens, so it is prompted exactly the way it was trained.

llama.cpp is cloned into finetune/llama.cpp (gitignored) and only llama-quantize is
built, CPU-only, with a pinned cmake run through uvx: no system packages.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "finetune" / "out"
LLAMA_CPP = ROOT / "finetune" / "llama.cpp"
LLAMA_TAG = "b11457"  # 2026-10-06
CMAKE = ["uvx", "--from", "cmake==3.31.6", "cmake"]


def run(cmd: list[str], **kw) -> str:
    print("$", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, capture_output=True, **kw)
    if p.returncode:  # surface the tool's own message, not just "exit status 1"
        sys.exit(f"{cmd[0]} failed ({p.returncode}):\n{(p.stdout + p.stderr)[-2000:]}")
    return p.stdout


def llama_quantize() -> Path:
    if not (LLAMA_CPP / "convert_hf_to_gguf.py").exists():
        run(["git", "clone", "--depth", "1", "--branch", LLAMA_TAG,
             "https://github.com/ggml-org/llama.cpp", str(LLAMA_CPP)])
    binary = LLAMA_CPP / "build" / "bin" / "llama-quantize"
    if not binary.exists():
        run([*CMAKE, "-S", str(LLAMA_CPP), "-B", str(LLAMA_CPP / "build"), "-DCMAKE_BUILD_TYPE=Release",
             "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_SERVER=OFF"])
        run([*CMAKE, "--build", str(LLAMA_CPP / "build"), "--target", "llama-quantize", "-j", str(os.cpu_count())])
    return binary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--name", default="jarvis-reader:1.7b")
    ap.add_argument("--base-ollama", default="qwen3:1.7b", help="whose template/stop tokens to reuse")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--skip-merge", action="store_true")
    args = ap.parse_args()
    merged, f16 = OUT / "merged", OUT / "jarvis-reader-f16.gguf"
    quantized = OUT / f"jarvis-reader-{args.quant.lower()}.gguf"
    quantize = llama_quantize()  # fail fast, before the slow merge

    if not args.skip_merge:
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(str(OUT / "lora"), max_seq_length=2048, load_in_4bit=True)
        model.save_pretrained_merged(str(merged), tok, save_method="merged_16bit")
        del model

    env = os.environ | {"PYTHONPATH": str(LLAMA_CPP / "gguf-py")}  # the converter's own gguf, version-matched
    run([sys.executable, str(LLAMA_CPP / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(f16),
         "--outtype", "f16"], env=env)
    run([str(quantize), str(f16), str(quantized), args.quant])
    print(f"GGUF: {f16.name} {f16.stat().st_size / 2**30:.2f} GiB -> {quantized.name} "
          f"{quantized.stat().st_size / 2**30:.2f} GiB")

    template = run(["ollama", "show", args.base_ollama, "--template"])
    params = [line for line in run(["ollama", "show", args.base_ollama, "--modelfile"]).splitlines()
              if line.startswith("PARAMETER stop")]
    modelfile = OUT / "Modelfile"
    modelfile.write_text("\n".join([
        f"FROM {quantized}",
        f'TEMPLATE """{template}"""',
        *params,
        "PARAMETER temperature 0.2",  # a reader should be near-deterministic
        "PARAMETER num_ctx 4096",
    ]) + "\n")
    run(["ollama", "create", args.name, "-f", str(modelfile)])
    print(run(["ollama", "list"]))


if __name__ == "__main__":
    main()
