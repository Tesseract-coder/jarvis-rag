#!/usr/bin/env bash
# Assemble the Hugging Face Space repository in build/hf-space: exactly the files that
# get pushed, nothing else (no .env, no storage/, no data/, no git history).
set -euo pipefail
cd "$(dirname "$0")/.."
out=build/hf-space
rm -rf "$out"
mkdir -p "$out"
cp deploy/hf-space/Dockerfile deploy/hf-space/README.md requirements.txt "$out"/
cp -r jarvis_rag "$out"/
# demo corpus sources + the retrieval eval, which gates the image build
mkdir -p "$out/learn" "$out/eval"
cp -r learn/sample_docs "$out/learn/"
cp eval/run_eval.py eval/qa_set.jsonl "$out/eval/"
find "$out" -name __pycache__ -type d -prune -exec rm -rf {} +
echo "Space bundle in $out:"
(cd "$out" && find . -type f | sort | sed 's/^/  /')
du -sh "$out" | awk '{print "  total " $1}'
