#!/usr/bin/env bash
# run-weight-quant-matrix.sh — weight-quantization quality + speed matrix (model-parameterized).
# Measures PPL + llama-bench (PP512+TG128) for every quant type of a model.
# GGUFs must already exist under $OUTPUT_DIR/<org>/<model>/ — produce them with generate-quants.sh.
#
# USAGE:
#   ./run-weight-quant-matrix.sh --repo <org/model> [--bin-dir <path>] [--label <tag>] [--force]
#   e.g. ./run-weight-quant-matrix.sh --repo Qwen/Qwen3.5-9B
#        ./run-weight-quant-matrix.sh --repo Qwen/Qwen3.5-9B --bin-dir /path/to/build/bin --label rocm-b912
# To produce GGUFs first: ./generate-quants.sh --repo <org/model>
#
# --repo selects the collection: measures every GGUF under $OUTPUT_DIR/<org>/<model>/
#   matching <model>[-MTP]-<type>.gguf (set INCLUDE_MTP=1 to measure MTP-tagged variants).
# --label tags CSV rows and the CSV filename (default = basename of BIN_DIR; 'bin' → parent name).
# One GPU run at a time per host is enforced by gpu-exclusive-run.sh.
#
# IMATRIX: ADR-016 — IQ-K/KT families require imatrix (hard gate in generate-quants.sh).
#   Detected by presence of $OUTPUT_DIR/<org>/<model>/<model>-imatrix.gguf.
#   PPL corpus: wikitext-2-raw/wiki.test.raw (canonical; wrong file shifts all numbers ~+1.1 PPL).
#
# OUTPUT: $OUTPUT_DIR/matrices/weight-quant-<model>-<label>.csv (appends; idempotent)
# LOGS:   $OUTPUT_DIR/matrices/logs/wq-{ppl,bench}-<model>-<type>-<label>.log

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the suite's layout + naming + repo parsing (OUTPUT_DIR, STAGING_DIR, MATRICES_DIR, gq_* helpers).
source "$HERE/generate-quants.sh"

# ── ARG PARSING ─────────────────────────────────────────────────────────────
REPO="" LABEL="" FORCE_ARG=0
while [ $# -gt 0 ]; do case "$1" in
  --repo)        REPO="$2";   shift 2;;
  --bin-dir)     BIN_DIR="$2"; shift 2;;
  --label)       LABEL="$2";  shift 2;;
  --force|--yes) FORCE_ARG=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
[ -n "$REPO" ] || { echo "ERROR: --repo <org/model> is required — e.g. --repo Qwen/Qwen3.5-9B" >&2; exit 2; }
[ "$FORCE_ARG" = 1 ] && FORCE=1

# Resolve measurement binaries from BIN_DIR (set in matrix-env.sh, env, or --bin-dir above)
: "${BENCH_BIN:=${BIN_DIR:+$BIN_DIR/llama-bench}}"
: "${BENCH_BIN:=$(command -v llama-bench 2>/dev/null || true)}"
: "${PPL_BIN:=${BIN_DIR:+$BIN_DIR/llama-perplexity}}"
: "${PPL_BIN:=$(command -v llama-perplexity 2>/dev/null || true)}"
[ -n "$BENCH_BIN" ] && [ -x "$BENCH_BIN" ] || { echo "ERROR: llama-bench not found (set BIN_DIR or BENCH_BIN)" >&2; exit 3; }
[ -n "$PPL_BIN"   ] && [ -x "$PPL_BIN"   ] || { echo "ERROR: llama-perplexity not found (set BIN_DIR or PPL_BIN)" >&2; exit 3; }

# Default label = auto-detected <hw>-<backend> (gfx1150-rocm, gfx1103-hsa1102-rocm, T4-cuda, …):
# hardware-honest + collision-free across hosts. Override with --label. (gq_detect_label from generate-quants.sh)
[ -z "$LABEL" ] && LABEL="$(gq_detect_label "${BIN_DIR:-}")"
: "${LABEL:=default}"

PPL_WRAP="${PPL_WRAP:-$HERE/ppl-run.sh}"   # co-located canonical PPL wrapper (ships beside this script)
GPU_EXCL="${GPU_EXCL:-$HERE/scripts/gpu-exclusive-run.sh}"

# Result CSVs + logs live under OUTPUT_DIR/matrices (MATRICES_DIR from generate-quants.sh).
RESULTS_DIR="${RESULTS_DIR:-$MATRICES_DIR}"
LOG_DIR="${LOG_DIR:-$MATRICES_DIR/logs}"

# Measurement parameters (overridable)
: "${PPL_CHUNKS:=50}"       # user standard 2026-06-01: 50 chunks for PPL across all matrices
: "${BENCH_REPS:=5}"
: "${HARD_TIMEOUT:=3600}"   # PPL hard cap (s) — 50 chunks can take up to an hour
: "${BENCH_TIMEOUT:=1800}"  # llama-bench hard cap (s)
: "${STALL_SECS:=120}"      # PPL no-output watchdog (chunks write every ~30-60s; 120s catches stalls)

# --- Measurement methodology (LOCKED 2026-06-04: "PPL-reference + bench-only") ---
# PPL is expensive and quality is host-invariant, so a SINGLE reference leg (ai00 + rocm) measures
# full PPL_CHUNKS-chunk PPL for every quant; every other leg is bench-only + a short PPL sanity
# sample. The driver sets PPL_MODE per cell.
#   full   → full PPL_CHUNKS-chunk PPL for every quant + bench  (the ai00-rocm reference leg).
#   sanity → bench-only for every quant, PLUS a PPL_SANITY_CHUNKS-chunk PPL on the PPL_SANITY_TYPES
#            sample, emitted as separate status=PPLSANITY rows (a cheap drift check vs the reference).
#   none   → bench-only for every quant, no PPL at all.
: "${PPL_MODE:=full}"
: "${PPL_SANITY_CHUNKS:=5}"
: "${PPL_SANITY_TYPES:=BF16 Q8_0 Q4_K_M IQ2_K IQ1_S}"   # ~5 quants spanning the bpw ladder

# --- Per-quant GTT (unified-memory) gating ---
# For APU hosts whose dedicated VRAM is smaller than a given GGUF (ai01: 16 GB VRAM), the runtime
# spills into GTT — the SAME unified system RAM — so PP/TG is ~identical to native VRAM; GTT is a
# purely operational mode, NOT a separate throughput class, and is deliberately NOT tagged in the CSV.
# The driver sets these per cell (left empty on ai00, which has 96 GB unified and needs no gating):
#   GTT_VAR    = name of the env var that enables unified memory for this backend
#                (GGML_CUDA_ENABLE_UNIFIED_MEMORY for rocm/cuda, GGML_VK_PREFER_HOST_MEMORY for vulkan).
#                EMPTY (default) = no GTT management and no size-based SKIP-OOM.
#   GTT_MIN_GB = GGUFs at/below this run on native VRAM (GTT_VAR left unset).   [default 14 = 16-2]
#   GTT_MAX_GB = GGUFs above this won't fit even via GTT → SKIP-OOM.  EMPTY = no ceiling.
: "${GTT_VAR:=}"
: "${GTT_MIN_GB:=14}"
: "${GTT_MAX_GB:=}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
gq_parse_repo "$REPO" || exit 2

OUT_MD="$(gq_model_dir "$OUTPUT_DIR"  "$REPO_ORG" "$REPO_MODEL")"
STG_MD="$(gq_model_dir "$STAGING_DIR" "$REPO_ORG" "$REPO_MODEL")"
[ -d "$OUT_MD" ] || { echo "ERROR: model dir not found: $OUT_MD (run generate-quants.sh --repo $REPO first)" >&2; exit 4; }

# Staging policy (unified with generate-quants.sh): if STAGING_DIR differs from OUTPUT_DIR, copy each
# GGUF OUTPUT→STAGING (local NVME), measure, delete — network→local is significantly faster + consistent load.
# If they're the same path (default), measure in place.
if gq_same_path "$STG_MD" "$OUT_MD"; then STAGE_LOCAL=0; else STAGE_LOCAL=1; mkdir -p "$STG_MD"; fi

# Imatrix presence flag (informational — recorded in the CSV has_imatrix column)
HAS_IMATRIX=0
[ -f "$OUT_MD/$(gq_imatrix_name "$REPO_MODEL")" ] && HAS_IMATRIX=1

CSV="$RESULTS_DIR/weight-quant-${REPO_MODEL}-${LABEL}.csv"
if [ ! -f "$CSV" ]; then
  echo "model,quant,label,binary_sha,ppl,pp_tps,tg_tps,has_imatrix,status" > "$CSV"
else
  if [ "${FORCE:-0}" != "1" ]; then
    echo "WARN: CSV exists: $CSV" >&2
    echo "      Rows for (model,quant,label) already recorded will be skipped (idempotent)." >&2
    echo "      For a fresh run, use a different --label or OUTPUT_DIR." >&2
  fi
fi

# Binary SHA captured once
SHA="$("$PPL_BIN" --version 2>&1 | grep -oE '\([0-9a-f]{7,}\)' | tr -d '()' | head -1)"; SHA="${SHA:-NA}"

echo "=== weight quant matrix: $REPO  label=$LABEL ==="
echo "    bench=$BENCH_BIN  ppl=$PPL_BIN"
echo "    model dir=$OUT_MD   csv=$CSV"
echo "    PPL_CHUNKS=$PPL_CHUNKS  has_imatrix=$HAS_IMATRIX  binary_sha=$SHA"
echo "    PPL_MODE=$PPL_MODE  ppl_sanity_chunks=$PPL_SANITY_CHUNKS  ppl_sanity_types='$PPL_SANITY_TYPES'"
echo "    gtt_var='${GTT_VAR:-<none>}'  gtt_min_gb=$GTT_MIN_GB  gtt_max_gb='${GTT_MAX_GB:-<none>}'"

# Types: BF16 baseline + all quant types (ALL_QUANT_TYPES from generate-quants.sh)
MEASURE_TYPES=(BF16 "${ALL_QUANT_TYPES[@]}")

# --- CSV-row presence helpers (status-aware, so a bench/full "main" row and a PPLSANITY row for
#     the same (model,quant,label) are tracked independently for idempotent resume) ---
have_main()   { awk -F, -v m="$REPO_MODEL" -v q="$1" -v l="$LABEL" '$1==m&&$2==q&&$3==l&&$9!="PPLSANITY"{f=1} END{exit !f}' "$CSV" 2>/dev/null; }
have_sanity() { awk -F, -v m="$REPO_MODEL" -v q="$1" -v l="$LABEL" '$1==m&&$2==q&&$3==l&&$9=="PPLSANITY"{f=1} END{exit !f}' "$CSV" 2>/dev/null; }

# in_list <needle> <space-list> — exact whitespace-delimited membership test
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# run_ppl <chunks> <logpath> <gguf> — run llama-perplexity under the stall watchdog; prints "<ppl>|<status>".
# Pulled out of measure_one so the full (reference) and the short sanity sample share one impl.
run_ppl() {
  local chunks="$1" log_ppl="$2" gguf="$3"
  local ppl="" hang=0 runpid last=0 st=0 cur ppl_rc
  timeout -k 30s "${HARD_TIMEOUT}s" bash "$PPL_WRAP" "$log_ppl" "$PPL_BIN" "$gguf" "$chunks" &
  runpid=$!
  while kill -0 "$runpid" 2>/dev/null; do
    sleep 10
    cur=$(stat -c%s "$log_ppl" 2>/dev/null || echo 0)
    if [ "${cur:-0}" -gt "$last" ]; then last="$cur"; st=0; else st=$((st+10)); fi
    if [ "$st" -ge "$STALL_SECS" ]; then
      echo "    watchdog: no output for ${STALL_SECS}s → SIGKILL (HANG)" >&2
      kill -9 "$runpid" 2>/dev/null; pkill -9 -f "llama-perplexity" 2>/dev/null
      hang=1; break
    fi
  done
  wait "$runpid" 2>/dev/null; ppl_rc=$?
  ppl=$(grep -oE 'Final estimate[^0-9]*[0-9]+\.[0-9]+' "$log_ppl" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
  [ -z "$ppl" ] && ppl=$(grep -oE 'PPL = [0-9]+\.[0-9]+' "$log_ppl" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+$' | tail -1)
  local status="OK"
  if   [ "$hang"    = 1 ]; then status="HANG"
  elif [ "$ppl_rc" -ne 0 ]; then status="FAIL-rc${ppl_rc}"; fi
  printf '%s|%s' "${ppl:-NA}" "$status"
}

# measure_one: resolve GGUF via gq_* helpers, stage, apply per-quant GTT gating, run PPL (per PPL_MODE)
# + bench, optionally add a short PPL sanity row, rm stage copy, write CSV row(s).
# NOTE: script runs under `set -uo pipefail` (NOT -e) by design, so a no-match grep in the
# parse block is harmless (empty var → NA) and never aborts the loop.
measure_one() {
  local t="$1"
  local gguf_name
  if [ "$t" = "BF16" ]; then
    gguf_name="$(gq_bf16_name "$REPO_MODEL")"
  else
    gguf_name="$(gq_quant_name "$REPO_MODEL" "$t")"
  fi
  local src_gguf="$OUT_MD/$gguf_name"
  local run_gguf="$src_gguf"

  # What this cell's methodology asks for this quant:
  #   need_main   = the bench (and, in full mode, full-PPL) row
  #   need_sanity = a separate short-PPL row (sanity mode + sampled quant only)
  local is_sample=0
  [ "$PPL_MODE" = "sanity" ] && in_list "$t" "$PPL_SANITY_TYPES" && is_sample=1
  local need_main=1 need_sanity=0
  have_main "$t" && need_main=0
  if [ "$is_sample" = 1 ] && ! have_sanity "$t"; then need_sanity=1; fi

  if [ "$need_main" = 0 ] && [ "$need_sanity" = 0 ]; then
    echo "  [$t] already in CSV → skip"; return
  fi

  local row_has_im
  case "$t" in BF16) row_has_im=0 ;; *) row_has_im="$HAS_IMATRIX" ;; esac

  if ! [ -f "$src_gguf" ]; then
    [ "$need_main" = 1 ] && echo "${REPO_MODEL},${t},${LABEL},${SHA},,,,${row_has_im},SKIP-GGUF-MISSING" >> "$CSV"
    echo "  [$t] SKIP-GGUF-MISSING"
    return
  fi

  # --- Per-quant GTT gating (only when GTT_VAR is set; see header) ---
  # Decide native-VRAM vs GTT vs SKIP-OOM from the source GGUF size, then toggle GTT_VAR for the
  # bench/PPL child processes. GTT spills into the same unified RAM → NOT recorded as a distinct mode.
  local gtt_state="n/a"
  if [ -n "$GTT_VAR" ]; then
    local bytes sz_gib
    bytes="$(stat -c%s "$src_gguf" 2>/dev/null || echo 0)"
    sz_gib="$(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1073741824}')"
    if [ -n "$GTT_MAX_GB" ] && awk -v s="$sz_gib" -v m="$GTT_MAX_GB" 'BEGIN{exit !(s>m)}'; then
      [ "$need_main" = 1 ] && echo "${REPO_MODEL},${t},${LABEL},${SHA},,,,${row_has_im},SKIP-OOM" >> "$CSV"
      echo "  [$t] SKIP-OOM (${sz_gib} GiB > GTT_MAX_GB=${GTT_MAX_GB})"
      return
    fi
    if awk -v s="$sz_gib" -v m="$GTT_MIN_GB" 'BEGIN{exit !(s>m)}'; then
      export "${GTT_VAR}=1"; gtt_state="GTT (${sz_gib} GiB)"
    else
      unset "${GTT_VAR}";    gtt_state="native (${sz_gib} GiB)"
    fi
  fi

  if [ "$STAGE_LOCAL" = 1 ]; then
    run_gguf="$STG_MD/$gguf_name"
    if ! cp "$src_gguf" "$run_gguf"; then
      [ "$need_main" = 1 ] && echo "${REPO_MODEL},${t},${LABEL},${SHA},,,,${row_has_im},STAGE-COPY-FAIL" >> "$CSV"
      echo "  [$t] STAGE-COPY-FAIL (skipping)"
      [ -n "$GTT_VAR" ] && unset "${GTT_VAR}"
      return
    fi
  fi
  [ "$gtt_state" != "n/a" ] && echo "  [$t] mem: $gtt_state"

  # --- Main row: full PPL (reference leg only) + bench, or bench-only ---
  if [ "$need_main" = 1 ]; then
    local ppl="NA" ppl_status="OK"
    if [ "$PPL_MODE" = "full" ]; then
      local log_ppl="$LOG_DIR/wq-ppl-${REPO_MODEL}-${t}-${LABEL}.log"
      echo "  [$t] PPL(${PPL_CHUNKS}) → $(basename "$log_ppl")"
      local pr; pr="$(run_ppl "$PPL_CHUNKS" "$log_ppl" "$run_gguf")"
      ppl="${pr%%|*}"; ppl_status="${pr##*|}"
    fi

    # llama-bench PP512+TG128 — hard timeout ONLY; no watchdog (bench doesn't stream per-chunk progress)
    local pp_tps="" tg_tps="" bench_rc
    local log_bench="$LOG_DIR/wq-bench-${REPO_MODEL}-${t}-${LABEL}.log"
    echo "  [$t] BENCH → $(basename "$log_bench")"
    # NOTE: deployed /opt llama-bench has no bare `-fit` flag — fit is `-fitt/--fit-target <MiB>`,
    # default off. Passing `-fit off` made llama-bench print usage + exit 1 (lost all TG/PP). Omit it:
    # default fit-target=off already means "no auto-fit, full -ngl 99 offload" (GTT covers oversize).
    timeout -k 30s "${BENCH_TIMEOUT}s" bash "$GPU_EXCL" "$log_bench" "$BENCH_BIN" \
      -m "$run_gguf" --mmap 0 -ngl 99 -fa on -p 512 -n 128 -r "$BENCH_REPS"
    bench_rc=$?
    # llama-bench markdown row: | name | SIZE GiB | PARAMS B | backend | ... | pp512 | <t/s> ± <sd> |
    # The t/s is the first whitespace token of the LAST `|`-delimited data column. The old
    # field-scan grabbed the first decimal on the line = the model SIZE (e.g. 16.68 GiB), recording
    # garbage throughput. Parse by `|` and take col NF-1's leading number.
    pp_tps=$(awk -F'|' '/pp512/ { split($(NF-1), a, " "); print a[1]; exit }' "$log_bench")
    tg_tps=$(awk -F'|' '/tg128/ { split($(NF-1), a, " "); print a[1]; exit }' "$log_bench")
    local bench_status="OK"
    [ "$bench_rc" -ne 0 ] && bench_status="FAIL-rc${bench_rc}"

    # status: full mode keeps the historical "ppl;bench" form; bench-only modes carry the bench status.
    local status
    if [ "$PPL_MODE" = "full" ]; then
      status="$ppl_status"; [ "$bench_status" != "OK" ] && status="${status};${bench_status}"
    else
      status="$bench_status"
    fi

    echo "${REPO_MODEL},${t},${LABEL},${SHA},${ppl:-NA},${pp_tps:-NA},${tg_tps:-NA},$row_has_im,$status" >> "$CSV"
    echo "    → ppl=${ppl:-?}  pp=${pp_tps:-?} t/s  tg=${tg_tps:-?} t/s  ($status)"
  fi

  # --- Sanity row: short PPL on the sample, recorded separately as status=PPLSANITY ---
  if [ "$need_sanity" = 1 ]; then
    local log_s="$LOG_DIR/wq-pplsanity-${REPO_MODEL}-${t}-${LABEL}.log"
    echo "  [$t] PPL-SANITY(${PPL_SANITY_CHUNKS}) → $(basename "$log_s")"
    local sr; sr="$(run_ppl "$PPL_SANITY_CHUNKS" "$log_s" "$run_gguf")"
    local sppl="${sr%%|*}" sstat="${sr##*|}"
    local sanity_status="PPLSANITY"; [ "$sstat" != "OK" ] && sanity_status="PPLSANITY-${sstat}"
    echo "${REPO_MODEL},${t},${LABEL},${SHA},${sppl:-NA},NA,NA,$row_has_im,$sanity_status" >> "$CSV"
    echo "    → sanity ppl=${sppl:-?}  ($sanity_status)"
  fi

  [ -n "$GTT_VAR" ] && unset "${GTT_VAR}"
  [ "$STAGE_LOCAL" = 1 ] && rm -f "$run_gguf"
}

echo "=== MEASURE  label=$LABEL  repo=$REPO  has_imatrix=$HAS_IMATRIX ==="
for t in "${MEASURE_TYPES[@]}"; do
  measure_one "$t"
done
echo "=== MEASURE complete → $CSV ==="
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
