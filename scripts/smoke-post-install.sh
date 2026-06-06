#!/usr/bin/env bash
# smoke-post-install.sh — /opt post-install smoke matrix for llama-yggdrasil cells.
# Runs the base generation smoke + MTP spec-decode smoke (TODO 90) for one cell prefix.
# Called by manual-reship.sh smoke phase; also runnable standalone.
#
# USAGE:
#   PREFIX=/opt/llama-yggdrasil-rocm MODEL=<gguf> ./smoke-post-install.sh
# or positional:
#   ./smoke-post-install.sh <prefix> <model.gguf>
set -uo pipefail

# Accept positional args or env vars
PREFIX="${1:-${PREFIX:-}}"
MODEL="${2:-${MODEL:-}}"

: "${PREFIX:?ERROR: set PREFIX=/opt/llama-yggdrasil-{rocm,vulkan} or pass as \$1}"
: "${MODEL:?ERROR: set MODEL=<path-to-gguf> or pass as \$2}"

CLI="$PREFIX/bin/llama-cli"
LIB_PATH="$PREFIX/lib:/usr/lib/llvm-19/lib"

[ -x "$CLI" ] || { echo "ERROR: llama-cli not found: $CLI" >&2; exit 3; }

pass=0; fail=0
smoke() {
    local desc="$1"; shift
    echo "  SMOKE: $desc"
    if LD_LIBRARY_PATH="$LIB_PATH" "$@"; then
        echo "    -> PASS"
        pass=$((pass+1))
    else
        echo "    -> FAIL (rc=$?)"
        fail=$((fail+1))
    fi
}

# S1: base generation (-st: single-turn, exits after one response)
smoke "base generation" \
    "$CLI" --no-mmap -fa on -ngl 99 -st -n 24 -c 512 \
    -p "The capital of France is" -m "$MODEL"

# S2: MTP spec-decode (TODO 90) — timeout guards REPL hang; -st ensures single-turn exit
smoke "MTP spec-decode (--spec-type draft-mtp)" \
    timeout 60s "$CLI" --no-mmap -fa on -ngl 99 -st -n 8 -c 512 \
    --spec-type draft-mtp -p "Hello" -m "$MODEL"

echo ""
echo "post-install smoke: ${pass} PASS / ${fail} FAIL"
[ "$fail" -eq 0 ]
