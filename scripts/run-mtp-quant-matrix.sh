#!/usr/bin/env bash
# run-mtp-quant-matrix.sh — MTP spec-decode sweep: accept-rate + decode t/s across all
# bundled-MTP quants of a model. Quality (PPL) is NOT measured here on purpose: spec-decode
# is verification-based, so output quality tracks the TARGET, not the draft/MTP head — the
# meaningful axes for an MTP quant are ACCEPTANCE RATE and net DECODE t/s.
#
# Each row is one `llama-speculative-simple -m <quant> --spec-type draft-mtp` run, parsed for
#   accept = <pct>%   and   decoded ... speed: <tps> t/s
#
# USAGE:
#   ./run-mtp-quant-matrix.sh --repo <org/model> [--bin-dir <path>] [--label <tag>] [--force]
#   e.g. ./run-mtp-quant-matrix.sh --repo Qwen/Qwen3.6-35B-A3B
#        ./run-mtp-quant-matrix.sh --repo Qwen/Qwen3.6-35B-A3B --bin-dir /path/to/build/bin --label rocm-b912
# --repo selects the collection: it measures every bundled-MTP quant under
#   $OUTPUT_DIR/<org>/<model>/  matching  <model>-MTP-<type>.gguf
# --label tags CSV rows and the CSV filename (default = basename of BIN_DIR; 'bin' → parent name).
# One GPU run at a time per host is enforced by gpu-exclusive-run.sh.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Reuse the suite's layout + naming + repo parsing (OUTPUT_DIR, gq_* helpers).
# INCLUDE_MTP=1 so the naming helpers resolve the '-MTP-' tagged quants.
INCLUDE_MTP=1
source "$HERE/generate-quants.sh"

# ── ARG PARSING ─────────────────────────────────────────────────────────────
REPO="" LABEL="" FORCE_ARG=0
while [ $# -gt 0 ]; do case "$1" in
  --repo)     REPO="$2";     shift 2;;
  --bin-dir)  BIN_DIR="$2";  shift 2;;
  --label)    LABEL="$2";    shift 2;;
  --force|--yes) FORCE_ARG=1; shift;;
  *) echo "unknown arg: $1" >&2; exit 2;;
esac; done
[ -n "$REPO" ] || { echo "ERROR: --repo <org/model> is required — it selects \$OUTPUT_DIR/<org>/<model>/ (e.g. --repo Qwen/Qwen3.6-35B-A3B)" >&2; exit 2; }
[ "$FORCE_ARG" = 1 ] && FORCE=1

# Resolve measurement binary from BIN_DIR (set in matrix-env.sh, env, or --bin-dir above)
: "${SPEC_BIN:=${BIN_DIR:+$BIN_DIR/llama-speculative-simple}}"
: "${SPEC_BIN:=$(command -v llama-speculative-simple 2>/dev/null || true)}"
[ -n "$SPEC_BIN" ] && [ -x "$SPEC_BIN" ] || { echo "ERROR: llama-speculative-simple not found (set BIN_DIR or SPEC_BIN)" >&2; exit 3; }

# Default label = smart basename of BIN_DIR (parent if basename is literally 'bin')
if [ -z "$LABEL" ] && [ -n "${BIN_DIR:-}" ]; then
  _lbase="$(basename "$BIN_DIR")"
  if [ "$_lbase" = "bin" ]; then _lbase="$(basename "$(dirname "$BIN_DIR")")"; fi
  LABEL="$_lbase"
fi
: "${LABEL:=default}"

GPU_EXCL="${GPU_EXCL:-$HERE/scripts/gpu-exclusive-run.sh}"
# Result CSVs + logs live under OUTPUT_DIR/matrices (MATRICES_DIR from generate-quants.sh).
RESULTS_DIR="${RESULTS_DIR:-$MATRICES_DIR}"
LOG_DIR="${LOG_DIR:-$MATRICES_DIR/logs}"

# Measurement parameters (overridable)
: "${N_DRAFT:=4}"        # speculative draft length (--spec-draft-n-max)
: "${CTX:=2048}"
: "${N_PREDICT:=200}"
: "${NGL:=99}"           # offload ALL layers
: "${FIT:=off}"          # -fit off: deterministic full offload (autofit would vary ngl per quant → invalid t/s comparison)
: "${HARD_TIMEOUT:=300}" # hard per-quant cap (s) — backstop for the watchdog (good quants finish in ~25-30s)
: "${STALL_SECS:=90}"    # no-output-progress watchdog: SIGKILL + mark HANG after this many idle seconds
                         # (catches IK base-K MUL_MAT CPU-fallback stalls under the MTP draft path)
: "${PROMPT:=Write a detailed technical explanation of how speculative decoding works in large language models, including the roles of the draft model and the target model, the verification step, and why the output distribution is preserved.}"
# Types to skip (references / non-quant artifacts). BF16/F16 are huge and won't fit small-VRAM hosts.
: "${SKIP_TYPES:=imatrix BF16 F16}"
# --- Tier 2 coherence (ADVISORY only — flags rows for review, never skips/rejects) ---
: "${SHORT_CHARS:=200}"  # generated chars below this (prompt asks long-form) → SHORT (output collapse)
: "${LOOP_RATIO:=0.35}"  # gzip(compressed/raw) of the completion below this → LOOP (repetitive)
: "${SLOW_TPS:=2.0}"     # decode t/s below this → +SLOW (CPU-fallback / pathological)
# =================================================================================

mkdir -p "$RESULTS_DIR" "$LOG_DIR"
gq_parse_repo "$REPO" || exit 2
OUT_MD="$(gq_model_dir "$OUTPUT_DIR"  "$REPO_ORG" "$REPO_MODEL")"
STG_MD="$(gq_model_dir "$STAGING_DIR" "$REPO_ORG" "$REPO_MODEL")"
[ -d "$OUT_MD" ] || { echo "ERROR: model dir not found: $OUT_MD" >&2; exit 4; }
# Staging policy (unified with generate-quants.sh): if STAGING_DIR differs from OUTPUT_DIR, copy each
# GGUF OUTPUT→STAGING (local), measure, delete, repeat. If same path, measure in place.
if gq_same_path "$STG_MD" "$OUT_MD"; then STAGE_LOCAL=0; else STAGE_LOCAL=1; mkdir -p "$STG_MD"; fi

CSV="$RESULTS_DIR/mtp-quant-${REPO_MODEL}-${LABEL}.csv"
CSV_HEADER="model,quant,label,n_draft,prompt_toks,n_predict,n_drafted,n_accept,accept_pct,decode_tps,sha,status,gen_chars,gzip_ratio,coherence"
if [ ! -f "$CSV" ]; then
  echo "$CSV_HEADER" > "$CSV"
else
  if [ "${FORCE:-0}" != "1" ]; then
    echo "WARN: CSV exists: $CSV" >&2
    echo "      Rows for (model,quant,label) already recorded will be skipped (idempotent)." >&2
    echo "      For a fresh run, use a different --label or OUTPUT_DIR." >&2
  fi
  if ! head -1 "$CSV" | grep -q "coherence"; then
    echo "WARN: $CSV predates the coherence columns — new rows will not align with its header." >&2
    echo "      Delete or rename it for a clean coherence-enabled run." >&2
  fi
fi

# Tier 2 coherence (advisory): sets GEN_CHARS / GZIP_RATIO / COHERENCE from the run log.
coherence_eval() {  # <log> <n_drafted> <decode_tps>
  local log="$1" ndr="$2" tps="$3" txt usz csz plen
  # free text = log minus START/END wrappers, timestamped log lines, and everything from the stats line on
  txt=$(awk '
    /^=== (START|END)/ {next}
    /encoded[[:space:]]+[0-9]+[[:space:]]+tokens/ {stats=1}
    stats {next}
    /^[0-9]+\.[0-9]+\.[0-9]+ [IWE] / {next}
    {print}' "$log" 2>/dev/null)
  usz=$(printf '%s' "$txt" | wc -c)
  csz=$(printf '%s' "$txt" | gzip -c 2>/dev/null | wc -c)
  plen=$(printf '%s' "$PROMPT" | wc -c)
  GEN_CHARS=$(( usz > plen ? usz - plen : usz ))          # completion chars ≈ free text minus echoed prompt
  if [ "${usz:-0}" -gt 0 ]; then GZIP_RATIO=$(awk "BEGIN{printf \"%.3f\", $csz/$usz}"); else GZIP_RATIO=NA; fi
  COHERENCE=OK
  [ "${GEN_CHARS:-0}" -lt "$SHORT_CHARS" ] 2>/dev/null && COHERENCE=SHORT
  [ "$GZIP_RATIO" != "NA" ] && awk "BEGIN{exit !($GZIP_RATIO < $LOOP_RATIO)}" && COHERENCE=LOOP
  case "$tps" in ''|NA) :;; *) awk "BEGIN{exit !($tps < $SLOW_TPS)}" && COHERENCE="${COHERENCE}+SLOW";; esac
}

echo "=== MTP quant matrix: $REPO  label=$LABEL ==="
echo "    bin=$SPEC_BIN"
echo "    model dir=$OUT_MD   csv=$CSV"
echo "    n_draft=$N_DRAFT ctx=$CTX n_predict=$N_PREDICT"
# binary build SHA, captured once (format: "version: <N> (<sha>)")
SHA="$("$SPEC_BIN" --version 2>&1 | grep -oE '\([0-9a-f]{7,}\)' | tr -d '()' | head -1)"; SHA="${SHA:-NA}"
echo "    binary sha=$SHA"

shopt -s nullglob
for gguf in "$OUT_MD/${REPO_MODEL}-MTP-"*.gguf; do
  base="$(basename "$gguf")"
  # type = strip '<model>-MTP-' prefix and '.gguf'
  t="${base#${REPO_MODEL}-MTP-}"; t="${t%.gguf}"
  gq_in_list "$t" $SKIP_TYPES && continue

  # idempotent: skip if this (model,quant,label) row already recorded
  if grep -q "^${REPO_MODEL},${t},${LABEL}," "$CSV"; then
    echo "  [$t] already in CSV → skip"; continue
  fi

  log="$LOG_DIR/mtp-${REPO_MODEL}-${t}-${LABEL}.log"
  echo "  [$t] spec-decode (MTP-ON) → $(basename "$log")"

  # stage to local only when STAGING_DIR differs from OUTPUT_DIR; else measure in place
  run_gguf="$gguf"
  if [ "$STAGE_LOCAL" = 1 ]; then
    run_gguf="$STG_MD/$base"
    if ! cp "$gguf" "$run_gguf"; then
      echo "${REPO_MODEL},${t},${LABEL},${N_DRAFT},NA,${N_PREDICT},NA,NA,NA,NA,NA,STAGE-COPY-FAIL,NA,NA,NA" >> "$CSV"
      echo "    → STAGE-COPY-FAIL (skipping)"; continue
    fi
  fi

  # NOTE: script runs under `set -uo pipefail` (NOT -e) by design, so a no-match grep in the
  # parse block below is harmless (empty var → NA) and never aborts the loop. Do not add `set -e`.
  # hard timeout backstop + no-progress watchdog (fast hang detection)
  timeout -k 30s "${HARD_TIMEOUT}s" bash "$GPU_EXCL" "$log" \
    "$SPEC_BIN" -m "$run_gguf" --spec-type draft-mtp --spec-draft-n-max "$N_DRAFT" \
    -c "$CTX" -n "$N_PREDICT" -ngl "$NGL" -fit "$FIT" -fa on --no-mmap -p "$PROMPT" &
  runpid=$!
  hang=0; last=0; st=0
  while kill -0 "$runpid" 2>/dev/null; do
    sleep 10
    cur=$(stat -c%s "$log" 2>/dev/null || echo 0)
    if [ "${cur:-0}" -gt "$last" ]; then last="$cur"; st=0; else st=$((st+10)); fi
    if [ "$st" -ge "$STALL_SECS" ]; then
      echo "    watchdog: no output for ${STALL_SECS}s → SIGKILL (HANG)"
      pkill -9 -f "llama-speculative-simple" 2>/dev/null
      hang=1; break
    fi
  done
  wait "$runpid" 2>/dev/null; rc=$?
  [ "$STAGE_LOCAL" = 1 ] && rm -f "$run_gguf"

  # ---- parse the log ----
  accept_pct=$(grep -oE 'accept[[:space:]]*=[[:space:]]*[0-9]+\.[0-9]+%' "$log" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  decode_tps=$(grep 'decoded' "$log" 2>/dev/null | grep -oE 'speed:[[:space:]]*[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+' | tail -1)
  n_drafted=$(grep -oE 'n_drafted[[:space:]]*=[[:space:]]*[0-9]+' "$log" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  n_accept=$(grep -oE 'n_accept[[:space:]]*=[[:space:]]*[0-9]+'  "$log" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  prompt_toks=$(grep -oE 'encoded[[:space:]]+[0-9]+[[:space:]]+tokens' "$log" 2>/dev/null | grep -oE '[0-9]+' | tail -1)

  status="OK"
  if [ "$hang" = 1 ]; then status="HANG"
  elif [ "$rc" -ne 0 ]; then status="FAIL-rc${rc}"; fi
  if [ -z "$decode_tps" ] || [ -z "$accept_pct" ]; then status="${status};PARSE-MISS"; fi

  coherence_eval "$log" "${n_drafted:-NA}" "${decode_tps:-NA}"
  echo "${REPO_MODEL},${t},${LABEL},${N_DRAFT},${prompt_toks:-NA},${N_PREDICT},${n_drafted:-NA},${n_accept:-NA},${accept_pct:-NA},${decode_tps:-NA},${SHA},${status},${GEN_CHARS},${GZIP_RATIO},${COHERENCE}" >> "$CSV"
  echo "    → accept=${accept_pct:-NA}%  decode=${decode_tps:-NA} t/s  (${status}; coherence=${COHERENCE})"
done

echo "=== done. results: $CSV ==="
column -t -s, "$CSV" 2>/dev/null || cat "$CSV"
