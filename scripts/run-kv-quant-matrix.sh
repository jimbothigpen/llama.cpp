#!/usr/bin/env bash
# run-kv-quant-matrix.sh — KV cache quantization quality + speed matrix (model-parameterized).
# Measures PPL and generation TPS for symmetric and asymmetric KV type pairs
# on the Q8_0 and Q4_K_M base models. Run generate-quants.sh --repo X FIRST.
#
# USAGE:
#   ./run-kv-quant-matrix.sh --repo <org/model> [--bin-dir <path>] [--label <tag>] [--force]
#   e.g. ./run-kv-quant-matrix.sh --repo Qwen/Qwen3.5-9B
#        ./run-kv-quant-matrix.sh --repo Qwen/Qwen3.5-9B --bin-dir /path/to/build/bin --label rocm-b912
#
# RUN ORDER: generate-quants.sh --repo X  →  run-weight-quant-matrix.sh  →  THIS SCRIPT
#
# Local staging: copies each base GGUF (Q8_0, Q4_K_M) OUTPUT→STAGING NVME once, measures all
# KV-pairs, then removes the staging copy. Do NOT run this on both hosts concurrently.
#
# TPS cells use gpu-exclusive-run.sh (hard host-level GPU mutex — never co-run two GPU procs).
# PPL cells use ppl-run.sh wrapper (canonical flags + corpus; PPL is stack-2, no exclusive lock).
#
# OUTPUT: $OUTPUT_DIR/matrices/kv-quant-<model>-<label>.csv (appends; idempotent)
# LOGS:   $OUTPUT_DIR/matrices/logs/kvq-<model>-<kt>-<kv>-{ppl,tps}-<label>.log

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse layout + naming + repo parsing (OUTPUT_DIR, STAGING_DIR, MATRICES_DIR, gq_* helpers).
source "$HERE/generate-quants.sh"

# ── ARG PARSING ─────────────────────────────────────────────────────────────
REPO="" LABEL="" FORCE_ARG=0
while [ $# -gt 0 ]; do case "$1" in
  --repo)        REPO="$2";    shift 2;;
  --bin-dir)     BIN_DIR="$2"; shift 2;;
  --label)       LABEL="$2";   shift 2;;
  --force|--yes) FORCE_ARG=1;  shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
[ -n "$REPO" ] || { echo "ERROR: --repo <org/model> is required" >&2; exit 2; }
[ "$FORCE_ARG" = 1 ] && FORCE=1

# Resolve measurement binaries from BIN_DIR (set in matrix-env.sh, env, or --bin-dir above)
: "${BENCH_BIN:=${BIN_DIR:+$BIN_DIR/llama-bench}}"
: "${BENCH_BIN:=$(command -v llama-bench 2>/dev/null || true)}"
: "${PPL_BIN:=${BIN_DIR:+$BIN_DIR/llama-perplexity}}"
: "${PPL_BIN:=$(command -v llama-perplexity 2>/dev/null || true)}"
[ -n "$BENCH_BIN" ] && [ -x "$BENCH_BIN" ] || { echo "ERROR: llama-bench not found (set BIN_DIR or BENCH_BIN)" >&2; exit 3; }
[ -n "$PPL_BIN"   ] && [ -x "$PPL_BIN"   ] || { echo "ERROR: llama-perplexity not found (set BIN_DIR or PPL_BIN)" >&2; exit 3; }

# Default label = smart basename of BIN_DIR ('bin' → parent dir name)
if [ -z "$LABEL" ] && [ -n "${BIN_DIR:-}" ]; then
  _lbase="$(basename "$BIN_DIR")"
  if [ "$_lbase" = "bin" ]; then _lbase="$(basename "$(dirname "$BIN_DIR")")"; fi
  LABEL="$_lbase"
fi
: "${LABEL:=default}"

PPL_WRAP="${PPL_WRAP:-$HERE/ppl-run.sh}"   # co-located canonical PPL wrapper (ships beside this script)
GPU_EXCL="${GPU_EXCL:-$HERE/scripts/gpu-exclusive-run.sh}"

# Result CSVs + logs live under OUTPUT_DIR/matrices (MATRICES_DIR from generate-quants.sh).
RESULTS_DIR="${RESULTS_DIR:-$MATRICES_DIR}"
LOG_DIR="${LOG_DIR:-$MATRICES_DIR/logs}"

: "${PPL_CHUNKS:=50}"    # user standard 2026-06-01: 50 chunks for PPL across all matrices
: "${TPS_CTX:=8192}"     # long enough to show VRAM benefit; short enough to complete quickly

gq_parse_repo "$REPO" || exit 2
OUT_MD="$(gq_model_dir "$OUTPUT_DIR"  "$REPO_ORG" "$REPO_MODEL")"
STG_MD="$(gq_model_dir "$STAGING_DIR" "$REPO_ORG" "$REPO_MODEL")"
[ -d "$OUT_MD" ] || { echo "ERROR: model dir not found: $OUT_MD (run generate-quants.sh --repo $REPO first)" >&2; exit 4; }

# Staging policy (unified with generate-quants.sh): copy OUTPUT→STAGING when paths differ.
if gq_same_path "$STG_MD" "$OUT_MD"; then STAGE_LOCAL=0; else STAGE_LOCAL=1; mkdir -p "$STG_MD"; fi

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

CSV="$RESULTS_DIR/kv-quant-${REPO_MODEL}-${LABEL}.csv"
if [ ! -f "$CSV" ]; then
  echo "kv_k,kv_v,model,metric,ctx,label,binary_sha,value,notes" > "$CSV"
else
  if [ "${FORCE:-0}" != "1" ]; then
    echo "WARN: CSV exists: $CSV" >&2
    echo "      Rows already recorded will be skipped (idempotent)." >&2
    echo "      For a fresh run, use a different --label or OUTPUT_DIR." >&2
  fi
fi

SHA=$("$PPL_BIN" --version 2>&1 | grep -oE 'build: [0-9]+ \([0-9a-f]+\)' | head -1 || echo unknown)

echo "=== KV quant matrix: $REPO  label=$LABEL ==="
echo "    bench=$BENCH_BIN  ppl=$PPL_BIN"
echo "    model dir=$OUT_MD   csv=$CSV"

PROMPT="Write a detailed technical essay about the history of artificial intelligence from 1950 to 2025."

have()     { [ -f "$1" ]; }
done_row() { grep -q "^$1,$2,$3,$4,$5,${LABEL}," "$CSV" 2>/dev/null; }

# ── KV TYPE PAIRS ───────────────────────────────────────────────────────────
# Format: "k_type:v_type"
# Symmetric (K=V): fp baseline, standard quant, TurboQuant tiers, RQ, Oscar.
# Asymmetric (K≠V): K at higher quality (keys are harder to compress).
# USER RULE 2026-06-01: only measure cells where bpw(K) >= bpw(V).
KV_PAIRS=(
  "f16:f16"
  "q8_0:q8_0"
  "q4_0:q4_0"
  "iq4_nl:iq4_nl"
  "turboq4:turboq4"
  "turboq3:turboq3"
  "turboq2:turboq2"
  "turboq3_tcq:turboq3_tcq"
  "turboq2_tcq:turboq2_tcq"
  "planar4:planar4"
  "planar3:planar3"
  "iso4:iso4"
  "iso3:iso3"
  "kv_oscar_int2:kv_oscar_int2"
  "turboq4:turboq2"
  "turboq4:turboq3"
  "turboq3:turboq2"
  "q8_0:q4_0"
  "q8_0:turboq2"
  "turboq3_tcq:turboq2_tcq"
)

kv_bpw() {
  case "$1" in
    f16) echo 16 ;;
    q8_0) echo 9 ;;
    q5_0|q5_1) echo 6 ;;
    q4_0|q4_1|iq4_nl) echo 5 ;;
    turboq4|planar4|iso4) echo 4 ;;
    turboq3|turboq3_tcq|planar3|iso3) echo 3 ;;
    turboq2|turboq2_tcq|kv_oscar_int2) echo 2 ;;
    *) echo 8 ;;
  esac
}

ppl_cell() {
  local kk="$1" kv="$2" mlabel="$3" local_gguf="$4"
  done_row "$kk" "$kv" "$mlabel" "ppl" "4096" && { echo "  skip(done): ppl $kk/$kv $mlabel"; return; }
  local log="$LOG_DIR/kvq-${mlabel}-${kk//\//_}-${kv//\//_}-ppl-${LABEL}.log"
  echo ">>> PPL  kv=$kk/$kv  model=$mlabel  label=$LABEL"
  timeout -k 30s 3600s bash "$PPL_WRAP" "$log" "$PPL_BIN" "$local_gguf" "$PPL_CHUNKS" \
    --cache-type-k "$kk" --cache-type-v "$kv"
  local rc=$?
  local ppl
  ppl=$(grep -oE 'Final estimate[^0-9]*[0-9]+\.[0-9]+' "$log" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
  [ -z "$ppl" ] && ppl=$(grep -oE 'PPL = [0-9]+\.[0-9]+' "$log" | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
  local note="rc=$rc"
  [ "$rc" -ne 0 ] && note="FAIL rc=$rc $(grep -iE 'error|abort|assert' "$log" | tail -1 | cut -c1-60)"
  echo "$kk,$kv,$mlabel,ppl,4096,${LABEL},$SHA,${ppl:-},$note" >> "$CSV"
  echo "    → ppl=${ppl:-?} ($note)"
}

tps_cell() {
  local kk="$1" kv="$2" mlabel="$3" local_gguf="$4"
  done_row "$kk" "$kv" "$mlabel" "tg128_tps" "$TPS_CTX" && { echo "  skip(done): tps $kk/$kv $mlabel ctx=$TPS_CTX"; return; }
  local log="$LOG_DIR/kvq-${mlabel}-${kk//\//_}-${kv//\//_}-tps-c${TPS_CTX}-${LABEL}.log"
  echo ">>> TPS(llama-bench)  kv=$kk/$kv  model=$mlabel  ctx=$TPS_CTX  label=$LABEL"
  # CANONICAL TPS = llama-bench (pp512+tg128, -r5); -d sets KV depth so KV-quant impact shows at ctx.
  # gpu-exclusive-run.sh: hard host-level mutex (never co-run two GPU procs).
  timeout -k 30s 3600s bash "$GPU_EXCL" "$log" "$BENCH_BIN" \
    -m "$local_gguf" --mmap 0 -fa on -ngl 99 \
    -ctk "$kk" -ctv "$kv" -p 512 -n 128 -d "$TPS_CTX" -r 5
  local rc=$?
  local pp tg
  pp=$(awk '/pp512/ { for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/){print $i; exit} }' "$log")
  tg=$(awk '/tg128/ { for(i=1;i<=NF;i++) if($i~/^[0-9]+\.[0-9]+$/){print $i; exit} }' "$log")
  local note="rc=$rc"
  [ "$rc" -ne 0 ] && note="FAIL rc=$rc $(grep -iE 'error|abort|assert' "$log" | tail -1 | cut -c1-60)"
  echo "$kk,$kv,$mlabel,pp512_tps,$TPS_CTX,${LABEL},$SHA,${pp:-},$note" >> "$CSV"
  echo "$kk,$kv,$mlabel,tg128_tps,$TPS_CTX,${LABEL},$SHA,${tg:-},$note" >> "$CSV"
  echo "    → pp512=${pp:-?} tg128=${tg:-?} ($note)"
}

run_model() {
  local mlabel="$1" local_gguf="$2"
  echo ""
  echo "=== MODEL: $mlabel ==="
  local pair kk kv rest
  for pair in "${KV_PAIRS[@]}"; do
    kk="${pair%%:*}"
    rest="${pair#*:}"
    kv="${rest%%:*}"
    # USER RULE 2026-06-01: bpw(K) >= bpw(V) — never quantize K more aggressively than V
    if [ "$(kv_bpw "$kk")" -lt "$(kv_bpw "$kv")" ]; then
      echo "  skip $kk/$kv (bpw K<V — violates K>=V rule)"
      continue
    fi
    ppl_cell "$kk" "$kv" "$mlabel" "$local_gguf"
    tps_cell "$kk" "$kv" "$mlabel" "$local_gguf"
  done
}

# ── MAIN: copy each base GGUF OUTPUT→STAGING, measure all KV-pairs, rm staging copy ──
BASE_QUANTS=(Q8_0 Q4_K_M)
for bq in "${BASE_QUANTS[@]}"; do
  gguf_name="$(gq_quant_name "$REPO_MODEL" "$bq")"
  src_gguf="$OUT_MD/$gguf_name"
  if ! have "$src_gguf"; then
    echo "=== SKIP $gguf_name: not found — run generate-quants.sh --repo $REPO first ==="
    continue
  fi
  run_gguf="$src_gguf"
  if [ "$STAGE_LOCAL" = 1 ]; then
    run_gguf="$STG_MD/$gguf_name"
    echo "=== Copying $gguf_name OUTPUT→STAGING NVME ==="
    df -h "$STG_MD"
    cp "$src_gguf" "$run_gguf"
  fi
  run_model "${REPO_MODEL}-${bq}" "$run_gguf"
  if [ "$STAGE_LOCAL" = 1 ]; then
    echo "  Removing staging copy"
    rm -f "$run_gguf"
  fi
done

echo ""
echo "DONE → $CSV"
