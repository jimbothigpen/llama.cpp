#!/bin/bash
set -euo pipefail

cd /mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/llama.cpp/src/worktrees/feat-kv-mma-ph1a-turbotier-2026-07-02
BUILD_DIR="build"
CLI="${BUILD_DIR}/bin/llama-cli"
MODEL="/home/builduser/local-models-ai02/Qwen3.5-0.8B-IQ4_KS.gguf"

if [ ! -f "$CLI" ]; then
    echo "llama-cli not found in $CLI"
    exit 1
fi

echo "Running with KV cache = TurboQ4_0"
$CLI -m "$MODEL" -ngl 99 \
    -n 128 \
    -c 2048 \
    -ctk tq4_0 -ctv tq4_0 \
    -p "The quick brown fox jumps over the lazy dog" \
    --temp 0.0

echo "Running with KV cache = TurboQ3_0"
$CLI -m "$MODEL" -ngl 99 \
    -n 128 \
    -c 2048 \
    -ctk tq3_0 -ctv tq3_0 \
    -p "The quick brown fox jumps over the lazy dog" \
    --temp 0.0

echo "Running with KV cache = TurboQ2_0"
$CLI -m "$MODEL" -ngl 99 \
    -n 128 \
    -c 2048 \
    -ctk tq2_0 -ctv tq2_0 \
    -p "The quick brown fox jumps over the lazy dog" \
    --temp 0.0
