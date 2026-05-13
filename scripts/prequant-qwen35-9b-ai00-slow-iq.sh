#!/usr/bin/env bash
# Pre-quantize Qwen3.5-9B BF16 → the slow IQ family on ai00 (gfx1150 host).
# Parallel partner to scripts/prequant-qwen35-9b-with-imatrix.sh running on
# ai01. ai00 takes the 11 slowest IQ types; ai01 takes IQ4_XS + the 16
# fast/medium types.
#
# Coordination: the coordinator script touches an empty sentinel file for
# each type in this list at <model>/<type>.gguf so the ai01 script's
# "skip if exists" check fires and ai01 does not redo our work.

set -euo pipefail

MODEL_DIR=/mnt/cephfs/0/Container/systems/ai00/users/builduser/models/Qwen3.5-9B
BASE="$MODEL_DIR/Qwen3.5-9B-BF16.gguf"
IMATRIX="$MODEL_DIR/Qwen3.5-9B-imatrix.gguf"
QUANTIZE=/opt/llama-yggdrasil-rocm/bin/llama-quantize

# No HSA_OVERRIDE on ai00 — gfx1150 binary is native.

if [[ ! -f "$BASE" ]]; then
  echo "ERROR: base GGUF not found: $BASE" >&2
  exit 1
fi
if [[ ! -f "$IMATRIX" ]]; then
  echo "ERROR: imatrix not found: $IMATRIX" >&2
  exit 1
fi

echo "Source:  $BASE ($(du -h "$BASE" | cut -f1))"
echo "Imatrix: $IMATRIX ($(du -h "$IMATRIX" | cut -f1))"
echo "Available disk: $(df -h "$MODEL_DIR" | tail -1 | awk '{print $4}')"
echo

TYPES=(
  IQ4_NL
  IQ3_M IQ3_S IQ3_XS IQ3_XXS
  IQ2_M IQ2_S IQ2_XS IQ2_XXS
  IQ1_M IQ1_S
)

for TYPE in "${TYPES[@]}"; do
  OUT="$MODEL_DIR/Qwen3.5-9B-$TYPE.gguf"
  # Real file = size > 1 MB. Sentinel = size 0. Skip only if real.
  if [[ -f "$OUT" && $(stat -c %s "$OUT") -gt 1048576 ]]; then
    echo "SKIP (real file exists): $OUT"
    continue
  fi
  rm -f "$OUT"
  echo "=== $TYPE ==="
  date
  "$QUANTIZE" --imatrix "$IMATRIX" "$BASE" "$OUT" "$TYPE"
  echo "DONE: $(du -h "$OUT" | cut -f1)  $OUT"
  echo
done

echo "=== AI00 IQ-FAMILY QUANTIZE COMPLETE ==="
