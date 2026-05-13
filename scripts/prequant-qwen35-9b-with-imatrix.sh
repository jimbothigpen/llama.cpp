#!/usr/bin/env bash
# Pre-quantize Qwen3.5-9B BF16 → the imatrix-using weight quants.
# Run on ai01 (idle host).
#
# Order:
#   1. IQ4_XS first (priority — sanity-check vs pre-existing IQ4_XS GGUF)
#   2. Recovered legacy types Q5_1, Q5_0, Q4_1, Q4_0 (mainline DOES use imatrix
#      on these via quantize_row_q*_impl + make_qkx3_quants; the earlier
#      no-imatrix run for these was incorrect tier)
#   3. K-quants high→low quality
#   4. IQ family high→low quality
#   5. Q1_0 last (Q1_0 ignores the --imatrix arg; harmless to pass it,
#      keeps the script structure uniform)
#
# F16 and Q8_0 are not included — those are correctly no-imatrix and already
# done by the prior sweep.

set -euo pipefail

MODEL_DIR=/mnt/cephfs/0/Container/systems/ai00/users/builduser/models/Qwen3.5-9B
BASE="$MODEL_DIR/Qwen3.5-9B-BF16.gguf"
IMATRIX="$MODEL_DIR/Qwen3.5-9B-imatrix.gguf"
QUANTIZE=/opt/llama-yggdrasil-rocm/bin/llama-quantize

export HSA_OVERRIDE_GFX_VERSION=11.0.2

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
  IQ4_XS
  Q5_1 Q5_0 Q4_1 Q4_0
  Q6_K
  Q5_K_M Q5_K_S
  Q4_K_M Q4_K_S
  Q3_K_L Q3_K_M Q3_K_S
  Q2_K Q2_K_S
  IQ4_NL
  IQ3_M IQ3_S IQ3_XS IQ3_XXS
  IQ2_M IQ2_S IQ2_XS IQ2_XXS
  IQ1_M IQ1_S
  Q1_0
)

for TYPE in "${TYPES[@]}"; do
  OUT="$MODEL_DIR/Qwen3.5-9B-$TYPE.gguf"
  if [[ -f "$OUT" ]]; then
    echo "SKIP (exists): $OUT"
    continue
  fi
  echo "=== $TYPE ==="
  date
  "$QUANTIZE" --imatrix "$IMATRIX" "$BASE" "$OUT" "$TYPE"
  echo "DONE: $(du -h "$OUT" | cut -f1)  $OUT"
  echo
done

echo "=== ALL IMATRIX QUANTIZE STEPS COMPLETE ==="
ls -lh "$MODEL_DIR"/Qwen3.5-9B-*.gguf
