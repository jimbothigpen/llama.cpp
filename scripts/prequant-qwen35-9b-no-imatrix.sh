#!/usr/bin/env bash
# Pre-quantize Qwen3.5-9B BF16 → the 7 weight quants that do not use imatrix.
# Run on ai01 (idle host) while ai00 generates the imatrix.
#
# Output: /mnt/cephfs/0/Container/systems/ai00/users/builduser/models/Qwen3.5-9B/Qwen3.5-9B-<TYPE>.gguf

set -euo pipefail

MODEL_DIR=/mnt/cephfs/0/Container/systems/ai00/users/builduser/models/Qwen3.5-9B
BASE="$MODEL_DIR/Qwen3.5-9B-BF16.gguf"
QUANTIZE=/opt/llama-yggdrasil-rocm/bin/llama-quantize

# Always set HSA_OVERRIDE on ai01 ROCm even for non-kernel tools — the binary
# initializes the ROCm context at startup.
export HSA_OVERRIDE_GFX_VERSION=11.0.2

if [[ ! -f "$BASE" ]]; then
  echo "ERROR: base GGUF not found: $BASE" >&2
  exit 1
fi

echo "Source: $BASE ($(du -h "$BASE" | cut -f1))"
echo "Available disk: $(df -h "$MODEL_DIR" | tail -1 | awk '{print $4}')"
echo

for TYPE in F16 Q8_0 Q5_1 Q5_0 Q4_1 Q4_0 Q1_0; do
  OUT="$MODEL_DIR/Qwen3.5-9B-$TYPE.gguf"
  if [[ -f "$OUT" ]]; then
    echo "SKIP (exists): $OUT"
    continue
  fi
  echo "=== $TYPE ==="
  date
  "$QUANTIZE" "$BASE" "$OUT" "$TYPE"
  echo "DONE: $(du -h "$OUT" | cut -f1)  $OUT"
  echo
done

echo "=== ALL QUANTIZE STEPS COMPLETE ==="
ls -lh "$MODEL_DIR"/Qwen3.5-9B-*.gguf
