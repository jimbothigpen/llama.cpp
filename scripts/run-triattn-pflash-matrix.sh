#!/usr/bin/env bash
# run-triattn-pflash-matrix.sh — TriAttention + PFlash feature TPS matrix (model-parameterized).
# Measures PP+TG speed across 6 conditions (off / tria-25/50/75 / pflash / tria50+pflash)
# at two long contexts (8192 and 24576) where eviction / prompt-compression matters.
#
# USAGE:
#   ./run-triattn-pflash-matrix.sh --repo <org/model> [--bin-dir <path>] [--label <tag>] [--force]
#   e.g. ./run-triattn-pflash-matrix.sh --repo Qwen/Qwen3.5-9B
#        ./run-triattn-pflash-matrix.sh --repo Qwen/Qwen3.5-9B --bin-dir /path/to/build/bin --label rocm-b912
#
# RUN ORDER: generate-quants.sh --repo X  →  run-weight-quant-matrix.sh  →  run-kv-quant  →  THIS SCRIPT
#
# PREREQUISITES:
#   1. TriAttention .tria calibration file at <kernel-work>/<model>.tria (auto-detected by REPO_MODEL).
#      Generate with: llama-tria-gen -m <iq4_xs.gguf> -f <corpus.txt> -o <out.tria>
#   2. PFlash scorer GGUF (see PFLASH_SCORER below). Cells skipped if scorer is missing.
#   3. $OUTPUT_DIR/<org>/<model>/<model>-<QUANT_TYPE>.gguf from generate-quants.sh.
#
# USES llama-bench --triattention / --pflash-scorer flags (wired in
# feature/triattn-pflash-bench-ppl-flags-2026-06-02).  llama-bench produces
# structured CSV directly; no manual TPS parsing from log files needed.
#
# OUTPUT: $OUTPUT_DIR/matrices/triattn-pflash-<model>-<label>.csv (appends; idempotent)
# LOGS:   $OUTPUT_DIR/matrices/logs/bench-<cond>-<model>-c<ctx>-<label>.{log,csv}

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

# Resolve measurement binary (llama-bench) from BIN_DIR
: "${BENCH_BIN:=${BIN_DIR:+$BIN_DIR/llama-bench}}"
: "${BENCH_BIN:=$(command -v llama-bench 2>/dev/null || true)}"
[ -n "$BENCH_BIN" ] && [ -x "$BENCH_BIN" ] || { echo "ERROR: llama-bench not found (set BIN_DIR or BENCH_BIN)" >&2; exit 3; }

# Default label = auto-detected <hw>-<backend> (gfx1150-rocm, gfx1103-hsa1102-rocm, T4-cuda, …):
# hardware-honest + collision-free across hosts. Override with --label. (gq_detect_label from generate-quants.sh)
[ -z "$LABEL" ] && LABEL="$(gq_detect_label "${BIN_DIR:-}")"
: "${LABEL:=default}"

# Result CSVs + logs live under OUTPUT_DIR/matrices (MATRICES_DIR from generate-quants.sh).
RESULTS_DIR="${RESULTS_DIR:-$MATRICES_DIR}"
LOG_DIR="${LOG_DIR:-$MATRICES_DIR/logs}"

gq_parse_repo "$REPO" || exit 2
OUT_MD="$(gq_model_dir "$OUTPUT_DIR"  "$REPO_ORG" "$REPO_MODEL")"
STG_MD="$(gq_model_dir "$STAGING_DIR" "$REPO_ORG" "$REPO_MODEL")"
[ -d "$OUT_MD" ] || { echo "ERROR: model dir not found: $OUT_MD (run generate-quants.sh --repo $REPO first)" >&2; exit 4; }

# Staging policy (unified with generate-quants.sh): copy OUTPUT→STAGING when paths differ.
if gq_same_path "$STG_MD" "$OUT_MD"; then STAGE_LOCAL=0; else STAGE_LOCAL=1; mkdir -p "$STG_MD"; fi

# Quant type used for TriAttention/PFlash TPS measurement (good speed/quality balance)
: "${QUANT_TYPE:=IQ4_XS}"

# TriAttention calibration file — auto-resolved as <kernel-work>/<model>.tria
: "${TRIA_FILE:=$HERE/${REPO_MODEL}.tria}"

# PFlash scorer: small Qwen3 model, model-independent scorer.
# Search common local-model directories; override via PFLASH_SCORER env var.
if [ -z "${PFLASH_SCORER:-}" ]; then
  for _cand in \
      "${HOME}/local-models/Qwen3-0.6B-Q4_K_M.gguf" \
      "${HOME}/local-models/Qwen3-1.7B-Q4_K_M.gguf" \
      "${PFLASH_SCORER_DIR:+$PFLASH_SCORER_DIR/Qwen3-0.6B-Q4_K_M.gguf}" \
      "${PFLASH_SCORER_DIR:+$PFLASH_SCORER_DIR/Qwen3-1.7B-Q4_K_M.gguf}"; do
    [ -n "$_cand" ] && [ -f "$_cand" ] && { PFLASH_SCORER="$_cand"; break; }
  done
fi
: "${PFLASH_SCORER:=}"
: "${PFLASH_KEEP_RATIO:=0.05}"  # retain 5% of prompt tokens
: "${PFLASH_ALPHA:=0.12}"
: "${BENCH_REPS:=3}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

CSV="$RESULTS_DIR/triattn-pflash-${REPO_MODEL}-${LABEL}.csv"
if [ ! -f "$CSV" ]; then
  echo "condition,tria_budget_pct,pflash_keep_ratio,model,ctx,label,binary_sha,n_prompt_eff,pp_tps,tg_tps,notes" > "$CSV"
else
  if [ "${FORCE:-0}" != "1" ]; then
    echo "WARN: CSV exists: $CSV" >&2
    echo "      Rows already recorded will be skipped (idempotent)." >&2
    echo "      For a fresh run, use a different --label or OUTPUT_DIR." >&2
  fi
fi

SHA=$("$BENCH_BIN" --version 2>&1 | grep -oE 'build: [0-9]+ \([0-9a-f]+\)' | head -1 || echo unknown)

echo "=== triattn-pflash matrix: $REPO  label=$LABEL ==="
echo "    bench=$BENCH_BIN   model dir=$OUT_MD   quant=$QUANT_TYPE"
echo "    tria_file=$TRIA_FILE   pflash_scorer=${PFLASH_SCORER:-NONE}"
echo "    csv=$CSV"

have()     { [ -f "$1" ]; }
done_row() { grep -q "^$1,$2,$3,$4,$5,${LABEL}," "$CSV" 2>/dev/null; }

# parse_bench_csv <csvfile>
# Prints: <n_prompt_eff> <pp_tps> <tg_tps>
parse_bench_csv() {
  local f="$1"
  local header
  header=$(grep -m1 "^model," "$f" 2>/dev/null) || { echo "0 0 0"; return; }
  IFS=',' read -ra hdrs <<< "$header"
  local i=0
  local n_prompt_col=14 n_gen_col=15 tts_col=19   # sensible defaults
  for col in "${hdrs[@]}"; do
    case "$col" in
      n_prompt) n_prompt_col=$i ;;
      n_gen)    n_gen_col=$i    ;;
      t_ts)     tts_col=$i      ;;
    esac
    (( i++ )) || true   # (( x++ )) is falsy when x==0; || true keeps pipefail happy
  done
  local pp_tps=0 tg_tps=0 n_prompt_eff=0
  while IFS=',' read -ra row; do
    local np ng tts
    np="${row[$n_prompt_col]:-0}"
    ng="${row[$n_gen_col]:-0}"
    tts="${row[$tts_col]:-0}"
    if [[ "$np" -gt 0 && "$ng" -eq 0 ]]; then
      pp_tps="$tts"; n_prompt_eff="$np"
    elif [[ "$np" -eq 0 && "$ng" -gt 0 ]]; then
      tg_tps="$tts"
    fi
  done < <(grep -v "^model," "$f" | grep -v '^$' | grep -v '^#')
  echo "$n_prompt_eff $pp_tps $tg_tps"
}

run_bench() {
  local cond="$1" bpct="$2" krat="$3" mlabel="$4" local_gguf="$5" ctx="$6"
  shift 6  # remaining args: extra llama-bench flags for this condition

  done_row "$cond" "$bpct" "$krat" "$mlabel" "$ctx" && \
    { echo "  skip(done): $cond bpct=$bpct $mlabel ctx=$ctx"; return; }

  if [[ "$cond" == *tria* ]] && ! have "$TRIA_FILE"; then
    echo "$cond,$bpct,$krat,$mlabel,$ctx,$LABEL,$SHA,,,,BLOCKED-NO-TRIA-FILE:$(basename "$TRIA_FILE")" >> "$CSV"
    echo "  BLOCKED-NO-TRIA-FILE: $cond (generate with llama-tria-gen first)"
    return
  fi

  if [[ "$cond" == *pflash* ]] && ! have "$PFLASH_SCORER"; then
    echo "$cond,$bpct,$krat,$mlabel,$ctx,$LABEL,$SHA,,,,SKIP-SCORER-MISSING" >> "$CSV"
    echo "  SKIP-SCORER-MISSING: $cond"
    return
  fi

  local log="$LOG_DIR/bench-${cond}-${mlabel}-c${ctx}-${LABEL}.log"
  local benchcsv="$LOG_DIR/bench-${cond}-${mlabel}-c${ctx}-${LABEL}-raw.csv"

  echo ">>> BENCH  cond=$cond  model=$mlabel  ctx=$ctx  label=$LABEL"
  # llama-bench writes CSV to stdout; stderr goes to log.
  # Inline flock on /var/tmp/llama-gpu.lock (same lock as gpu-exclusive-run.sh —
  # prevents concurrent GPU usage with other matrix scripts on this host).
  {
    flock -x 200
    timeout -k 30s 3600s "$BENCH_BIN" \
        -m "$local_gguf" --mmap 0 -fa on -ngl 99 \
        -p "$ctx" -n 400 \
        -r "$BENCH_REPS" -o csv \
        "$@" \
      >"$benchcsv" 2>"$log"
  } 200>/var/tmp/llama-gpu.lock
  local rc=$?

  read -r n_prompt_eff pp_tps tg_tps <<< "$(parse_bench_csv "$benchcsv")"
  local note="rc=$rc"
  [ "$rc" -ge 128 ] && note="CRASH_SIGNAL=$((rc-128))"
  [ "$rc" -ne 0 ] && [ "$rc" -lt 128 ] && \
    note="FAIL rc=$rc $(grep -iE 'error|abort|assert' "$log" 2>/dev/null | tail -1 | cut -c1-60)"
  echo "$cond,$bpct,$krat,$mlabel,$ctx,$LABEL,$SHA,$n_prompt_eff,$pp_tps,$tg_tps,$note" >> "$CSV"
  echo "    → pp=${pp_tps:-?} tg=${tg_tps:-?} n_prompt_eff=${n_prompt_eff:-?} ($note)"
}

run_model() {
  local mlabel="$1" local_gguf="$2"
  echo ""
  echo "=== MODEL: $mlabel ==="

  local ctx bpct krat_fmt
  krat_fmt="$(printf '%.2f' "$PFLASH_KEEP_RATIO")"
  for ctx in 8192 24576; do
    echo ""
    echo "--- ctx=$ctx ---"

    # baseline: no TriAttention, no PFlash
    run_bench "off" "0" "0.00" "$mlabel" "$local_gguf" "$ctx"

    # TriAttention only: budget 25/50/75%
    # --cache-type-k q8_0 required for GPU scoring path (Vulkan; ROCm accepts any)
    for bpct in 25 50 75; do
      run_bench "tria" "$bpct" "0.00" "$mlabel" "$local_gguf" "$ctx" \
        --triattention "$TRIA_FILE" --tri-budget "$bpct" \
        --cache-type-k q8_0
    done

    # PFlash only: bench compresses the dummy prompt with scorer; reports effective n_prompt
    run_bench "pflash" "0" "$krat_fmt" "$mlabel" "$local_gguf" "$ctx" \
      --pflash-scorer "$PFLASH_SCORER" \
      --pflash-keep-ratio "$PFLASH_KEEP_RATIO" \
      --pflash-alpha "$PFLASH_ALPHA"

    # TriAttention 50% + PFlash combined
    run_bench "tria50+pflash" "50" "$krat_fmt" "$mlabel" "$local_gguf" "$ctx" \
      --triattention "$TRIA_FILE" --tri-budget 50 \
      --cache-type-k q8_0 \
      --pflash-scorer "$PFLASH_SCORER" \
      --pflash-keep-ratio "$PFLASH_KEEP_RATIO" \
      --pflash-alpha "$PFLASH_ALPHA"
  done
}

# ── MAIN: copy model GGUF OUTPUT→STAGING NVME, run all conditions, rm staging copy ──
gguf_name="$(gq_quant_name "$REPO_MODEL" "$QUANT_TYPE")"
src_gguf="$OUT_MD/$gguf_name"
if ! have "$src_gguf"; then
  echo "ERROR: $src_gguf not found — run generate-quants.sh --repo $REPO first" >&2
  exit 1
fi

if [ "$STAGE_LOCAL" = 1 ]; then
  run_gguf="$STG_MD/$gguf_name"
  echo "=== Copying $gguf_name OUTPUT→STAGING NVME ==="
  df -h "$STG_MD"
  cp "$src_gguf" "$run_gguf"
else
  run_gguf="$src_gguf"
fi

MLABEL="${REPO_MODEL}-${QUANT_TYPE}"
run_model "$MLABEL" "$run_gguf"

if [ "$STAGE_LOCAL" = 1 ]; then
  echo "  Removing staging copy"
  rm -f "$run_gguf"
fi

echo ""
echo "DONE → $CSV"
