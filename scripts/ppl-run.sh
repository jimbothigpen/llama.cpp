#!/bin/bash
# ppl-run.sh — run llama-perplexity with the canonical anchor-matching flags + corpus.
# Wrong flags or wrong corpus silently change the PPL value; this wrapper makes them non-optional so
# every measurement is comparable. Corpus comes from $PPL_CORPUS (set it in matrix-env.sh, or export it).
# Usage: ppl-run.sh <logfile> <binary> <model.gguf> <chunks> [extra llama-perplexity args...]
set -uo pipefail
CORPUS="${PPL_CORPUS:?PPL_CORPUS not set — point it at your wikitext-2-raw/wiki.test.raw (see matrix-env.sh)}"

LOG="${1:?usage: ppl-run.sh <logfile> <binary> <model.gguf> <chunks> [extra...]}"
BIN="${2:?need binary}"; MODEL="${3:?need model.gguf}"; CHUNKS="${4:?need chunks}"; shift 4 || true
[ -f "$CORPUS" ] || { echo "ABORT: corpus missing: $CORPUS"; exit 1; }
[ -x "$BIN" ] || { echo "ABORT: binary not executable: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "ABORT: model not found: $MODEL"; exit 1; }

SHA=$("$BIN" --version 2>&1 | grep -oE 'build: [0-9]+ \([0-9a-f]+\)' | head -1 || echo "build:?")
echo "PPL-RUN $(date -u +%H:%M:%SZ) | bin=$BIN ($SHA) | model=$(basename "$MODEL") | chunks=$CHUNKS" | tee "$LOG"
# byte-stream tee (never tee|tail — SIGPIPE kills the producer); capture real RC via PIPESTATUS
"$BIN" --no-mmap -fa on -ngl 999 -c 4096 -ub 512 -b 512 \
  -m "$MODEL" -f "$CORPUS" --chunks "$CHUNKS" "$@" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "---- RC=$RC ($([ "$RC" -ge 128 ] && echo "SIGNAL $((RC-128))" || echo ok)) ----" | tee -a "$LOG"
grep -E 'Final estimate|PPL =|^\[[0-9]+\]' "$LOG" | tail -3
exit "$RC"
