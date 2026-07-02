#!/bin/bash
set -e

BIN_DIR="/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/llama.cpp/src/worktrees/feat-kv-mma-ph1a-turbotier-2026-07-02/build/bin"
MODEL="/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/llama.cpp/kernel-work/worker-scratch/adopt1-complete-nbeforeuser-port-2026-07-01/Qwen3-0.6B-Q4_K_M.gguf"

# First, test generation to verify output is coherent and valid
echo ">>> Running generation test with turboq4_0 KV cache..."
$BIN_DIR/llama-cli -m "$MODEL" -p "The capital of France is" -n 20 -ctk turboq4 -ctv turboq4 -c 128 -ngl 99 -st -no-cnv -fa on --no-mmap

# Second, run with debug logging to confirm flash attention is used
echo -e "\n>>> Verifying flash attention is hit..."
GGML_CUDA_DEBUG=1 $BIN_DIR/llama-cli -m "$MODEL" -p "Test." -n 1 -ctk turboq4 -ctv turboq4 -c 128 -ngl 99 -st -no-cnv -fa on --no-mmap > validate_debug.log 2>&1

echo -e "\n>>> Debug log excerpts (flash attention):"
grep -i "flash" validate_debug.log || echo "No flash attention output found in debug log"

echo "Done."
