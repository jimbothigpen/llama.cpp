#!/usr/bin/env bash
# weight-matrix-driver.sh <ai00|ai01> — TODO 202 STAGE 3 weight-quant matrix, ONE host.
# Runs both models (9B first = validation default + fast) × both backends (rocm first = clean),
# SERIALLY on this host (gpu-exclusive enforces one GPU run at a time; PPL is unflocked so we must
# never overlap invocations on a host). ai00 ‖ ai01 run in parallel via two detached driver procs.
#
# Per cell: kernel-work/run-weight-quant-matrix.sh measures every base GGUF under the model dir.
# Idempotent (skips CSV rows already present) → safe to stop/resume.
#
# MEASUREMENT METHODOLOGY (LOCKED 2026-06-04: "PPL-reference + bench-only"):
#  - PPL is expensive and host-invariant, so ONE reference leg (ai00 + rocm) measures full 50-chunk
#    PPL for every quant (PPL_MODE=full). Every other cell is bench-only + a short 5-chunk PPL
#    sanity sample (PPL_MODE=sanity) — a cheap drift check vs the reference, recorded as PPLSANITY rows.
#  - ai01 (16 GiB VRAM) per-quant GTT: the run script toggles GTT_VAR by GGUF size
#    (≤ GTT_MIN_GB native VRAM, ≤ GTT_MAX_GB GTT, > GTT_MAX_GB SKIP-OOM). GTT spills into the SAME
#    unified system RAM → PP/TG ~identical to native; mode is operational, NOT tagged in the CSV.
#  - ai00 (96 GiB unified): no GTT gating (GTT_VAR left unset) — everything fits natively.
#
# HARD RULES baked in:
#  - NEVER export HSA_OVERRIDE_GFX_VERSION on ai00 (wedges the gfx1150 ASIC; reboot-only recovery).
#  - ai01 ROCm REQUIRES HSA_OVERRIDE_GFX_VERSION=11.0.2.
#  - 35B fits ai00 natively (96GiB unified); ai01 (16GiB) needs GTT oversubscribe env (per-quant).
set -uo pipefail
PROFILE="${1:?usage: weight-matrix-driver.sh <ai00|ai01>}"
case "$PROFILE" in ai00|ai01) ;; *) echo "bad profile: $PROFILE" >&2; exit 2;; esac

REPOROOT=/mnt/cephfs/0/Container/systems/ai00/users/builduser/projects/llama.cpp
RUN="$REPOROOT/kernel-work/run-weight-quant-matrix.sh"
LOGDIR="$REPOROOT/kernel-work/worker-scratch/weight-matrix-2026-06-04"
mkdir -p "$LOGDIR"

# Defense in depth: scrub HSA from the inherited env up front.
unset HSA_OVERRIDE_GFX_VERSION 2>/dev/null || true

echo "=== [$(date -Is)] DRIVER START profile=$PROFILE host=$(hostname) ===" | tee -a "$LOGDIR/driver-$PROFILE.log"

for MODEL in Qwen3.5-9B Qwen3.6-35B-A3B; do
  for BK in rocm vulkan; do
    # Reset per-cell env so nothing leaks between backends/hosts.
    unset HSA_OVERRIDE_GFX_VERSION GGML_CUDA_ENABLE_UNIFIED_MEMORY GGML_VK_PREFER_HOST_MEMORY HIP_VISIBLE_DEVICES 2>/dev/null || true
    unset GTT_VAR GTT_MIN_GB GTT_MAX_GB 2>/dev/null || true

    # ai00+rocm is the single full-PPL reference leg; every other cell is bench-only + 5-chunk sanity.
    if [ "$PROFILE" = ai00 ] && [ "$BK" = rocm ]; then
      export PPL_MODE=full
    else
      export PPL_MODE=sanity
    fi

    if [ "$BK" = rocm ]; then
      BIN=/opt/llama-yggdrasil-rocm/bin
      export HIP_VISIBLE_DEVICES=0
      if [ "$PROFILE" = ai01 ]; then
        export HSA_OVERRIDE_GFX_VERSION=11.0.2                    # REQUIRED on ai01/gfx1103
        # Per-quant GTT: run script toggles this var by GGUF size (see header thresholds).
        export GTT_VAR=GGML_CUDA_ENABLE_UNIFIED_MEMORY GTT_MIN_GB=14 GTT_MAX_GB=36
      fi
    else
      BIN=/opt/llama-yggdrasil-vulkan/bin
      if [ "$PROFILE" = ai01 ]; then
        export GTT_VAR=GGML_VK_PREFER_HOST_MEMORY GTT_MIN_GB=14 GTT_MAX_GB=36
      fi
    fi
    # ai00: GTT_VAR stays unset → run script does no GTT management and no size SKIP-OOM.

    # HARD GUARD: abort the whole driver if HSA ever ends up set on ai00.
    if [ "$PROFILE" = ai00 ] && [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
      echo "FATAL: HSA_OVERRIDE set on ai00 — aborting to protect the ASIC" | tee -a "$LOGDIR/driver-$PROFILE.log" >&2
      exit 99
    fi
    LOG="$LOGDIR/cell-$PROFILE-$MODEL-$BK.log"
    echo "=== [$(date -Is)] CELL $PROFILE $MODEL $BK BIN=$BIN PPL_MODE=$PPL_MODE HSA=${HSA_OVERRIDE_GFX_VERSION:-unset} GTT_VAR=${GTT_VAR:-none} GTT_MIN=${GTT_MIN_GB:-na} GTT_MAX=${GTT_MAX_GB:-na} ===" | tee -a "$LOGDIR/driver-$PROFILE.log"
    bash "$RUN" --repo "Qwen/$MODEL" --bin-dir "$BIN" >> "$LOG" 2>&1
    rc=$?
    echo "=== [$(date -Is)] CELL DONE $PROFILE $MODEL $BK rc=$rc ===" | tee -a "$LOGDIR/driver-$PROFILE.log"
  done
done
echo "=== [$(date -Is)] DRIVER COMPLETE profile=$PROFILE ===" | tee -a "$LOGDIR/driver-$PROFILE.log"
touch "$LOGDIR/driver-$PROFILE-COMPLETE.flag"
