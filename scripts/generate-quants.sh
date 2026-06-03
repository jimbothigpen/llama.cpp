#!/usr/bin/env bash
# generate-quants.sh — download → convert → imatrix → quantize a HuggingFace model
# into a full set of GGUF weight-quants, with org/model namespacing and optional MTP.
#
# Two ways to use it:
#   • Run directly:   ./generate-quants.sh --repo Qwen/Qwen3.6-35B-A3B
#   • Source it as a library (the run-*-matrix.sh scripts do this) to reuse the exact
#     same config defaults, naming helpers, and quant type lists — one source of truth.
#     When sourced, NOTHING runs (the pipeline only executes when invoked directly).
#
# USAGE:
#   ./generate-quants.sh --repo <org/model> [--types T1,T2,...] [--bin-dir <path>] [--force]
#   MTP mode (pick at most one; default = plain, no MTP head):
#     --include-mtp                full model WITH the MTP head in the main GGUF  (tag: -MTP)
#     --assistant                  ONLY the MTP head, as a standalone draft GGUF  (tag: -assistant; converter --mtp)
#     (plain / --no-mtp)           full model WITHOUT the MTP head                (no tag)
#     --mtp-quant <type> [--mtp-pattern block|head]   (full+MTP only) pin MTP-block tensors to <type>
#   Imatrix: ONE shared, MTP-INCLUSIVE <model>-imatrix.gguf per model (auto, --imat-mtp when the model
#     has an MTP head — covers trunk+head, serves all three modes). Override with --imatrix-file <path>.
#   INCLUDE_MTP=1 OUTPUT_DIR=/mnt/bulk/models ./generate-quants.sh --repo Qwen/Qwen3.6-35B-A3B
#   BIN_DIR=/path/to/build/bin ./generate-quants.sh --repo ...
#
# Every variable below is overridable via the environment (sensible defaults otherwise).

# Auto-source site-local config (matrix-env.sh next to this script) if present — sets
# OUTPUT_DIR / STAGING_DIR / BIN_DIR / IMATRIX_CORPUS for this install so no per-run exports are
# needed. Runs BEFORE the defaults below so its exports win. Portable: absent in other projects
# → the $PATH/$HOME defaults apply.
_GQ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$_GQ_DIR/matrix-env.sh" ] && . "$_GQ_DIR/matrix-env.sh"

# ============================ EDIT THESE (or export) ============================
# ---- Binaries: set BIN_DIR to a build's bin/ directory; each binary defaults off it.
#      Override a specific binary with its env var (e.g. point QUANTIZE_BIN at a test build).
#      PATH fallback when BIN_DIR is unset and the binary is not found via its var.
# e.g.:  BIN_DIR=/path/to/llama-cpp-build/bin ./generate-quants.sh --repo X
: "${BIN_DIR:=}"
: "${IMATRIX_BIN:=${BIN_DIR:+$BIN_DIR/llama-imatrix}}"
: "${IMATRIX_BIN:=$(command -v llama-imatrix 2>/dev/null || true)}"
: "${QUANTIZE_BIN:=${BIN_DIR:+$BIN_DIR/llama-quantize}}"
: "${QUANTIZE_BIN:=$(command -v llama-quantize 2>/dev/null || true)}"
: "${CONVERT_BIN:=${BIN_DIR:+$BIN_DIR/convert_hf_to_gguf.py}}"
: "${CONVERT_BIN:=$(command -v convert_hf_to_gguf.py 2>/dev/null || true)}"

# ---- Storage layout ----
#   OUTPUT_DIR  : durable home for the src snapshot + BF16 + imatrix + quant GGUFs.
#   STAGING_DIR : fast scratch for in-flight quant work; defaults to OUTPUT_DIR.
#   Constrained local disk? Split them so OUTPUT_DIR lives on network/bulk storage and
#   STAGING_DIR on fast local NVMe — the script stages each artifact on STAGING_DIR,
#   publishes it to OUTPUT_DIR, and frees the staged copy. e.g.:
#     export STAGING_DIR="$HOME/.cache/llama-quant-stage"
#     export OUTPUT_DIR=/mnt/bulk/models
: "${OUTPUT_DIR:=$HOME/llama-quants}"
: "${STAGING_DIR:=$OUTPUT_DIR}"
: "${SRC_SUBDIR:=src}"            # subdir under <OUTPUT_DIR>/<org>/<model>/ for the HF snapshot
# All run-*-matrix.sh scripts write their result CSVs + logs here (one convention, suite-wide).
: "${MATRICES_DIR:=$OUTPUT_DIR/matrices}"

# ---- imatrix calibration (ADR-016) ----
#   IMATRIX_CORPUS must be a semantic calibration corpus DISJOINT from any PPL eval set.
#   Leave empty to skip imatrix entirely — but then imatrix-REQUIRED types error out.
: "${IMATRIX_CORPUS:=}"
# IMATRIX_CHUNKS: cap calibration chunks. EMPTY (default) = use the FULL corpus — the community
# standard is to run llama-imatrix over the entire calibration set (e.g. bartowski calibration_datav3).
# Set a positive integer only if you deliberately want to truncate (faster, lower-quality imatrix).
: "${IMATRIX_CHUNKS:=}"

# ---- MTP (multi-token-prediction / NextN draft head) ----
#   INCLUDE_MTP=1 keeps the MTP head in the GGUF, activates it during imatrix, and tags
#   every artifact '-MTP'. Default 0 (MTP-free; universal). With INCLUDE_MTP=1 on a model
#   that has NO MTP head, the run ERRORS OUT (no silent fallback).
: "${INCLUDE_MTP:=0}"
#   ASSISTANT=1 (--assistant): export ONLY the MTP head as a standalone speculative-draft GGUF
#   (gemma-4 style external assistant; converter --mtp). Tags artifacts '-assistant'. Requires an MTP
#   head (errors otherwise). Mutually exclusive with INCLUDE_MTP. Default 0.
: "${ASSISTANT:=0}"
#   IMATRIX_FILE: explicit path to a prebuilt imatrix to use for ALL quants (skips generation). Use to
#   share ONE imatrix across the no-MTP / full+MTP / assistant runs. Default unset = auto (see Step 3).
: "${IMATRIX_FILE:=}"
#   MTP_QUANT: when set + INCLUDE_MTP=1, overrides the quantization of MTP block tensors.
#   DEFAULT = unset = no override (MTP layers quantized at base type per imatrix — honest; IK MTP
#   bugs stay visible). Sweep usage: run with --mtp-quant q8_0 / f16 / q5_0 etc. to characterize
#   MTP-layer precision tradeoff. NEVER the silent default — see spec parity-goal note.
: "${MTP_QUANT:=}"
#   MTP_TENSOR_PATTERN: how to target MTP tensors. "block" (default) = pin the whole MTP block(s)
#   by layer index from config.json. "head" = pin only the nextn head via substring "nextn".
: "${MTP_TENSOR_PATTERN:=block}"

# ---- Retention ----
: "${KEEP_SAFETENSORS:=1}"        # 1=keep <model>/src/ snapshot; 0=delete after BF16 convert
: "${KEEP_BF16:=1}"               # 1=keep BF16 in OUTPUT; 0=delete after all quants are made

# ---- GPU offload for imatrix (CPU-only: NGL=0) ----
: "${NGL:=99}"

# ---- Collision / prompt behaviour ----
#   FORCE=1 (or --force / --yes): skip all collision prompts, default = reuse existing artifacts.
: "${FORCE:=0}"
# ===============================================================================

# ---- Quant type families (weight quants) — used for IMATRIX_REQUIRED_TYPES classification ----
# NOTE: ALL_QUANT_TYPES is no longer the default type list; gq_enumerate_types() derives it from
# the binary at runtime (Item 2b). These arrays are kept as the imatrix-required classification.
IQ_K_TYPES=(IQ2_K IQ2_KL IQ2_KT IQ3_K IQ3_KS IQ3_KT IQ4_K IQ4_KS IQ4_KSS IQ4_KT IQ5_K IQ6_K)
IQ_PLAIN=(IQ1_S IQ1_M IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XS IQ3_M IQ4_XS IQ4_NL)
K_QUANTS=(Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S Q5_K_M Q6_K)
STD_QUANTS=(Q8_0 Q5_0 Q5_1 Q4_0 Q4_1)
ALL_QUANT_TYPES=("${IQ_K_TYPES[@]}" "${IQ_PLAIN[@]}" "${K_QUANTS[@]}" "${STD_QUANTS[@]}")
# IQ-K/KT + low-bit IQ families REQUIRE an imatrix (ADR-016, hard gate):
IMATRIX_REQUIRED_TYPES=("${IQ_K_TYPES[@]}" "${IQ_PLAIN[@]}")

# Types to skip when enumerating all types from the binary (Item 2b — LOCKED skip set):
#   BF16/F16 = full-precision baselines (pipeline source, not compression targets)
#   COPY     = no-op pass-through, not a compression quant
#   TQ1_0/TQ2_0 = ternary quants (~1.5e6 PPL on non-ternary models — unusable for general matrices)
SKIP_QUANT_TYPES=(BF16 F16 COPY TQ1_0 TQ2_0)

# ================================ helpers ================================
# (these are what the run-*-matrix.sh scripts reuse when they `source` this file)
gq_in_list() { local x="$1"; shift; local e; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }
gq_mtp_tag() {
  if   [ "${ASSISTANT:-0}" = "1" ];   then printf -- "-assistant"
  elif [ "${INCLUDE_MTP:-0}" = "1" ]; then printf -- "-MTP"
  else printf ""; fi
}

# ---- Hardware-honest run label: <hw>-<backend> (used to tag matrix CSV rows) ----
# hw = physical accelerator (gfx####/T4/P100/…); on ROCm with HSA_OVERRIDE_GFX_VERSION active, a
# `-hsa<target>` qualifier is appended (kernels compiled for a different gfx than the silicon) — so the
# label is honest for BOTH throughput (physical device) AND PPL/reproducibility (override target flagged).
gq_hsa_to_gfx() { case "$1" in 11.0.2) echo gfx1102;; 11.0.0) echo gfx1100;; 10.3.0) echo gfx1030;; 9.0.*) echo gfx90a;; *) echo "gfx${1//./}";; esac; }
gq_detect_backend() {
  local b="${1:-${BIN_DIR:-}}"
  case "$b" in *vulkan*) echo vulkan; return;; *rocm*|*hip*) echo rocm; return;; *cuda*|*cublas*) echo cuda; return;; esac
  if command -v nvidia-smi >/dev/null 2>&1; then echo cuda
  elif command -v rocminfo >/dev/null 2>&1; then echo rocm
  else echo cpu; fi
}
gq_detect_hw() {
  local backend="$1"
  case "$backend" in
    cuda)
      local name n
      name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | grep -oiE 'T4|P100|V100|A100|H100|L40|L4|A10|RTX[0-9]+|[34]090' | head -1)
      n=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c . )
      [ -z "$name" ] && name=gpu
      [ "${n:-1}" -gt 1 ] && echo "${name}x${n}" || echo "$name" ;;
    rocm|vulkan)
      local phys
      phys=$(env -u HSA_OVERRIDE_GFX_VERSION rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
      [ -z "$phys" ] && phys=$(vulkaninfo --summary 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
      [ -z "$phys" ] && phys=gfxUNK
      if [ "$backend" = rocm ] && [ -n "${HSA_OVERRIDE_GFX_VERSION:-}" ]; then
        local ovr; ovr=$(gq_hsa_to_gfx "$HSA_OVERRIDE_GFX_VERSION")
        [ "$ovr" != "$phys" ] && echo "${phys}-hsa${ovr#gfx}" || echo "$phys"
      else echo "$phys"; fi ;;
    *) echo cpu ;;
  esac
}
# Compose <hw>-<backend>. Pass a backend hint (else inferred from BIN_DIR / available runtimes).
gq_detect_label() { local be; be=$(gq_detect_backend "${1:-}"); echo "$(gq_detect_hw "$be")-$be"; }
gq_model_dir()   { printf '%s/%s/%s' "$1" "$2" "$3"; }                  # <root> <org> <model>
gq_src_dir()     { printf '%s/%s' "$1" "${SRC_SUBDIR:-src}"; }          # <model_dir>
gq_bf16_name()   { printf '%s%s-BF16.gguf' "$1" "$(gq_mtp_tag)"; }      # <model>
gq_imatrix_name(){ printf '%s-imatrix.gguf' "$1"; }   # <model> — ONE shared, MTP-inclusive imatrix (no tag)
# <model> <type> — with MTP quant marker when MTP_QUANT is set, INCLUDE_MTP=1, and differs from base
gq_quant_name() {
  local model="$1" type="$2"
  local mtag; mtag="$(gq_mtp_tag)"
  # MTP-quant marker applies only to full+MTP (INCLUDE_MTP=1) — NOT assistant (all-MTP) or plain.
  if [ "${INCLUDE_MTP:-0}" = "1" ] && [ "${ASSISTANT:-0}" != "1" ] && [ -n "${MTP_QUANT:-}" ] && [ "${MTP_QUANT}" != "$type" ]; then
    printf '%s%s-%s-%s.gguf' "$model" "$mtag" "$MTP_QUANT" "$type"
  else
    printf '%s%s-%s.gguf' "$model" "$mtag" "$type"
  fi
}
# gq_ensure_mtp_bf16 <src_dir>: ensure the full+MTP BF16 exists (for the MTP-INCLUSIVE shared imatrix,
# regardless of the current quant scenario), echo its staged path on stdout. Cached + reused across passes.
gq_ensure_mtp_bf16() {  # <src_dir>
  local name out stg
  name="${REPO_MODEL}-MTP-BF16.gguf"
  out="$OUT_MD/$name"; stg="$STG_MD/$name"
  if [ -f "$stg" ]; then echo "$stg"; return 0; fi
  if [ -f "$out" ]; then gq_same_path "$stg" "$out" || cp -f "$out" "$stg"; echo "$stg"; return 0; fi
  echo "--- imatrix: building full+MTP BF16 for the shared imatrix → $name" >&2
  python3 "$CONVERT_BIN" "$1" --outfile "$stg" --outtype bf16 >&2 || return 1   # default conversion = MTP head included
  gq_publish "$stg" "$out"
  echo "$stg"
}
# parse "org/model" -> sets REPO_ORG / REPO_MODEL (returns nonzero on bad input)
gq_parse_repo() {
  local repo="$1"
  REPO_ORG="${repo%%/*}"; REPO_MODEL="${repo##*/}"
  if [ "$REPO_ORG" = "$repo" ] || [ -z "$REPO_ORG" ] || [ -z "$REPO_MODEL" ]; then
    echo "ERROR: --repo must be 'org/model' (got '$repo')" >&2; return 1
  fi
}
gq_same_path() { [ "$(readlink -m -- "$1")" = "$(readlink -m -- "$2")" ]; }
# publish a staged artifact to its output path (no-op if staging==output)
gq_publish() {  # <staged> <output>
  gq_same_path "$1" "$2" && return 0
  echo "    publish → $2"; df -h "$(dirname "$2")" | tail -1 | awk '{print "      output fs: "$4" free"}'
  cp -f "$1" "$2"
}
# true if the model snapshot declares an MTP / NextN head
gq_model_has_mtp() {  # <src_dir>
  local cfg="$1/config.json"
  [ -f "$cfg" ] || return 1
  python3 - "$cfg" <<'PY'
import json,sys
try: c=json.load(open(sys.argv[1]))
except Exception: sys.exit(1)
# Recognize all MTP-layer-count config keys: num_nextn_predict_layers / num_nextn_layers (NextN/MoE,
# e.g. 35B-A3B) AND mtp_num_hidden_layers (Qwen3.5/3.6 text variants, e.g. 9B) — the key the converter's
# _Qwen35MtpMixin actually reads. Missing the last one wrongly reports "no MTP" for those models.
n=c.get("num_nextn_predict_layers") or c.get("num_nextn_layers") or c.get("mtp_num_hidden_layers") or 0
sys.exit(0 if (isinstance(n,int) and n>0) else 1)
PY
}

# gq_enumerate_types: parse $QUANTIZE_BIN --help and return the type list minus SKIP_QUANT_TYPES.
# Output: space-separated type names on stdout (suitable for array capture via read -r -a).
gq_enumerate_types() {
  local bin="${1:-$QUANTIZE_BIN}"
  [ -n "$bin" ] && [ -x "$bin" ] || { echo "ERROR: gq_enumerate_types: QUANTIZE_BIN not found" >&2; return 1; }
  local skip=("${SKIP_QUANT_TYPES[@]}")
  "$bin" --help 2>&1 \
    | awk '/allowed quantization types/,0 {
        for (i=1; i<=NF; i++) {
          if ($i == "or" && i+1 <= NF && $(i-1) ~ /^[0-9]+$/) { print $(i+1); break }
        }
      }' \
    | while read -r t; do
        local s skip_it=0
        for s in "${skip[@]}"; do [ "$s" = "$t" ] && skip_it=1 && break; done
        [ "$skip_it" = 0 ] && echo "$t"
      done
}

# gq_mtp_block_indices: print one MTP block index per nextn layer (reads config.json).
gq_mtp_block_indices() {  # <src_dir>
  local cfg="$1/config.json"
  [ -f "$cfg" ] || return 0
  python3 - "$cfg" <<'PY'
import json, sys
try:
    c = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
n_layers = c.get("num_hidden_layers", 0)
n_nextn  = c.get("num_nextn_predict_layers") or c.get("num_nextn_layers") or 0
for i in range(n_nextn):
    print(n_layers + i)
PY
}

# gq_build_mtp_tensor_args: populate MTP_TENSOR_ARGS array for the quantize call.
# Uses MTP_QUANT, MTP_TENSOR_PATTERN, INCLUDE_MTP. Caller must declare MTP_TENSOR_ARGS as local.
gq_build_mtp_tensor_args() {  # <src_dir>
  MTP_TENSOR_ARGS=()
  [ -n "${MTP_QUANT:-}" ] && [ "${INCLUDE_MTP:-0}" = "1" ] || return 0
  case "${MTP_TENSOR_PATTERN:-block}" in
    head)
      MTP_TENSOR_ARGS=(--tensor-type "nextn=${MTP_QUANT}")
      ;;
    block|*)
      local idx
      while read -r idx; do
        MTP_TENSOR_ARGS+=(--tensor-type "blk.${idx}=${MTP_QUANT}")
      done < <(gq_mtp_block_indices "$1")
      ;;
  esac
}

# gq_ask_reuse: collision prompt for an existing artifact.
# Returns 0 = reuse (skip regen), 1 = regen.
# With FORCE=1, always returns 0 silently.
gq_ask_reuse() {  # <path> <description>
  local path="$1" desc="${2:-artifact}"
  echo "WARN: $desc already exists: $(basename "$path")" >&2
  if [ "${FORCE:-0}" = "1" ]; then return 0; fi
  echo "  Reuse existing $desc? [Y/n] (If this is a different build of the quantizer," >&2
  echo "  answer n and set a fresh OUTPUT_DIR instead)" >&2
  local ans
  read -r -t 60 ans 2>/dev/null || ans="y"
  case "${ans:-y}" in y|Y|"") return 0;; *) return 1;; esac
}

# ================================ pipeline ================================
gq_main() {
  set -uo pipefail
  local REPO="" TYPES_CSV="" BIN_DIR_ARG="" FORCE_ARG=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --repo)       REPO="$2"; shift 2;;
      --types)      TYPES_CSV="$2"; shift 2;;
      --include-mtp) INCLUDE_MTP=1; shift;;
      --no-mtp)     INCLUDE_MTP=0; shift;;
      --assistant)  ASSISTANT=1; shift;;
      --imatrix-file) IMATRIX_FILE="$2"; shift 2;;
      --bin-dir)    BIN_DIR_ARG="$2"; shift 2;;
      --mtp-quant)  MTP_QUANT="$2"; shift 2;;
      --mtp-pattern) MTP_TENSOR_PATTERN="$2"; shift 2;;
      --force|--yes) FORCE_ARG=1; shift;;
      -h|--help) sed -n '2,30p' "$0"; return 0;;
      *) echo "ERROR: unknown arg '$1'" >&2; return 2;;
    esac
  done
  [ -n "$REPO" ] || { echo "ERROR: --repo <org/model> is required" >&2; return 2; }
  if [ "${ASSISTANT:-0}" = 1 ] && [ "${INCLUDE_MTP:-0}" = 1 ]; then
    echo "ERROR: --assistant (MTP-only draft) and --include-mtp (full model+MTP) are mutually exclusive." >&2; return 2
  fi
  [ "$FORCE_ARG" = 1 ] && FORCE=1

  # --bin-dir: override BIN_DIR and re-derive generation binaries
  if [ -n "$BIN_DIR_ARG" ]; then
    BIN_DIR="$BIN_DIR_ARG"
    QUANTIZE_BIN="$BIN_DIR/llama-quantize"
    IMATRIX_BIN="$BIN_DIR/llama-imatrix"
    CONVERT_BIN="$BIN_DIR/convert_hf_to_gguf.py"
  fi

  # ---- resolve + validate binaries ----
  local b ok=1
  for b in CONVERT_BIN QUANTIZE_BIN; do
    [ -n "${!b}" ] || { echo "ERROR: $b not found in \$PATH (set BIN_DIR or $b explicitly)" >&2; ok=0; }
  done
  [ "$ok" = 1 ] || return 3

  # ---- selected types ----
  local SEL=()
  if [ -n "$TYPES_CSV" ]; then
    IFS=',' read -r -a SEL <<<"$TYPES_CSV"
  else
    # Dynamic enumeration from binary (Item 2b) — auto-adapts to fork IK/turbo/OScaR vs mainline
    local _enum_out
    _enum_out="$(gq_enumerate_types "$QUANTIZE_BIN")" || {
      echo "WARN: failed to enumerate types from $QUANTIZE_BIN --help; falling back to hardcoded list" >&2
      SEL=("${ALL_QUANT_TYPES[@]}")
    }
    if [ ${#SEL[@]} -eq 0 ] && [ -n "$_enum_out" ]; then
      IFS=$'\n' read -r -d '' -a SEL <<<"$_enum_out" || true
    fi
  fi

  # ---- layout ----
  gq_parse_repo "$REPO" || return 2
  local OUT_MD STG_MD SRC
  OUT_MD="$(gq_model_dir "$OUTPUT_DIR"  "$REPO_ORG" "$REPO_MODEL")"
  STG_MD="$(gq_model_dir "$STAGING_DIR" "$REPO_ORG" "$REPO_MODEL")"
  SRC="$(gq_src_dir "$OUT_MD")"          # snapshot always durable in OUTPUT
  mkdir -p "$OUT_MD" "$STG_MD" "$SRC"
  local TAG; TAG="$(gq_mtp_tag)"
  echo "=== generate-quants: $REPO  (INCLUDE_MTP=$INCLUDE_MTP tag='${TAG:-none}') ==="
  echo "    OUTPUT_DIR=$OUTPUT_DIR   STAGING_DIR=$STAGING_DIR$(gq_same_path "$OUT_MD" "$STG_MD" && echo '  [in-place: same path]')"
  echo "    model dir: $OUT_MD"
  [ -n "${MTP_QUANT:-}" ] && [ "$INCLUDE_MTP" = 1 ] && \
    echo "    MTP_QUANT=$MTP_QUANT  pattern=$MTP_TENSOR_PATTERN"

  # ---- imatrix availability gate (ADR-016) ----
  local need_imatrix=0 t
  for t in "${SEL[@]}"; do gq_in_list "$t" "${IMATRIX_REQUIRED_TYPES[@]}" && need_imatrix=1; done
  if [ "$need_imatrix" = 1 ] && [ -z "$IMATRIX_CORPUS" ]; then
    echo "ERROR: requested types include imatrix-REQUIRED quants but IMATRIX_CORPUS is unset." >&2
    echo "       Set IMATRIX_CORPUS=<semantic calibration corpus> or drop the IQ-K/KT/low-IQ types." >&2
    return 4
  fi
  if [ -n "$IMATRIX_CORPUS" ] && [ ! -f "$IMATRIX_CORPUS" ]; then
    echo "ERROR: IMATRIX_CORPUS '$IMATRIX_CORPUS' not found." >&2; return 4
  fi

  # ---- MTP tensor args (built once, used per-type in Step 4) ----
  local MTP_TENSOR_ARGS=()
  gq_build_mtp_tensor_args "$SRC"

  # ---- Step 1: download HF snapshot → OUTPUT/<org>/<model>/src/ ----
  if ls "$SRC"/*.safetensors >/dev/null 2>&1; then
    echo "--- Step 1: snapshot present → skip download ($SRC)"
  else
    echo "--- Step 1: hf download $REPO → $SRC"
    hf download "$REPO" --local-dir "$SRC" || { echo "ERROR: hf download failed" >&2; return 5; }
  fi

  # ---- MTP presence gate (error out clearly, per requirement) ----
  # Full+MTP (INCLUDE_MTP), assistant (MTP-only), AND the MTP-inclusive shared imatrix all need an MTP head.
  if { [ "$INCLUDE_MTP" = 1 ] || [ "${ASSISTANT:-0}" = 1 ]; } && ! gq_model_has_mtp "$SRC"; then
    local _why="INCLUDE_MTP=1"; [ "${ASSISTANT:-0}" = 1 ] && _why="--assistant"
    echo "ERROR: $_why but '$REPO' has no MTP/NextN head (config.json num_nextn_predict_layers absent/0)." >&2
    echo "       Use --no-mtp for this model (no MTP head to export)." >&2
    return 6
  fi

  # ---- Step 2: convert → BF16 (staged) ----
  local BF16_OUT BF16_STG
  BF16_OUT="$OUT_MD/$(gq_bf16_name "$REPO_MODEL")"
  BF16_STG="$STG_MD/$(gq_bf16_name "$REPO_MODEL")"
  # Converter MTP flag: assistant → --mtp (export ONLY the MTP head); full+MTP → none (head in main model); plain → --no-mtp.
  local convert_mtp=()
  if   [ "${ASSISTANT:-0}" = 1 ];   then convert_mtp=(--mtp)
  elif [ "${INCLUDE_MTP:-0}" = 1 ]; then convert_mtp=()
  else convert_mtp=(--no-mtp); fi
  if [ -f "$BF16_OUT" ]; then
    if gq_ask_reuse "$BF16_OUT" "BF16 GGUF"; then
      echo "--- Step 2: BF16 present → reuse ($(basename "$BF16_OUT"))"
      gq_same_path "$BF16_STG" "$BF16_OUT" || cp -f "$BF16_OUT" "$BF16_STG"
    else
      echo "--- Step 2: convert → $(basename "$BF16_STG") ${convert_mtp[*]:-} (regenerating)"
      python3 "$CONVERT_BIN" "$SRC" --outfile "$BF16_STG" --outtype bf16 "${convert_mtp[@]}" \
        || { echo "ERROR: convert failed" >&2; return 7; }
      gq_publish "$BF16_STG" "$BF16_OUT"
    fi
  else
    echo "--- Step 2: convert → $(basename "$BF16_STG") ${convert_mtp[*]:-}"
    python3 "$CONVERT_BIN" "$SRC" --outfile "$BF16_STG" --outtype bf16 "${convert_mtp[@]}" \
      || { echo "ERROR: convert failed" >&2; return 7; }
    gq_publish "$BF16_STG" "$BF16_OUT"
  fi

  # ---- Step 3: imatrix (staged) — ONE shared, MTP-INCLUSIVE imatrix per model ----
  local IMAT_OUT="" IMAT_STG="" IMAT_USE=""
  if [ -n "$IMATRIX_FILE" ]; then
    # Explicit prebuilt imatrix → use as-is for ALL quants (skip generation). Shares one imatrix across runs.
    [ -f "$IMATRIX_FILE" ] || { echo "ERROR: IMATRIX_FILE not found: $IMATRIX_FILE" >&2; return 8; }
    IMAT_USE="$IMATRIX_FILE"
    echo "--- Step 3: imatrix → using IMATRIX_FILE ($IMATRIX_FILE)"
  elif [ -n "$IMATRIX_CORPUS" ]; then
    IMAT_OUT="$OUT_MD/$(gq_imatrix_name "$REPO_MODEL")"
    IMAT_STG="$STG_MD/$(gq_imatrix_name "$REPO_MODEL")"
    if [ -f "$IMAT_OUT" ] && gq_ask_reuse "$IMAT_OUT" "imatrix"; then
      echo "--- Step 3: imatrix present → reuse ($(basename "$IMAT_OUT"))"
    else
      [ -n "$IMATRIX_BIN" ] || { echo "ERROR: IMATRIX_BIN not found in \$PATH" >&2; return 3; }
      # MTP-INCLUSIVE: if the model has an MTP head, compute from the full+MTP BF16 (built if needed) with
      # --imat-mtp, so the ONE imatrix covers trunk + MTP head + embd/output and serves the no-MTP, full+MTP,
      # AND assistant quant runs identically. No MTP head → compute from this run's BF16.
      local imat_src imat_mtp=()
      if gq_model_has_mtp "$SRC"; then
        imat_src="$(gq_ensure_mtp_bf16 "$SRC")" || { echo "ERROR: full+MTP BF16 build for imatrix failed" >&2; return 7; }
        imat_mtp=(--imat-mtp)
      else
        imat_src="$BF16_STG"
      fi
      echo "--- Step 3: imatrix → $(basename "$IMAT_STG") (chunks=${IMATRIX_CHUNKS:-full-corpus} ${imat_mtp[*]:-} src=$(basename "$imat_src"))"
      "$IMATRIX_BIN" -m "$imat_src" -f "$IMATRIX_CORPUS" -o "$IMAT_STG" \
        -ngl "$NGL" -fit off -fa on --no-mmap "${imat_mtp[@]}" -b 512 ${IMATRIX_CHUNKS:+--chunks $IMATRIX_CHUNKS} \
        || { echo "ERROR: imatrix failed" >&2; return 8; }
      gq_publish "$IMAT_STG" "$IMAT_OUT"
    fi
    # pick whichever imatrix file actually exists (staged preferred for fast reads)
    if   [ -f "$IMAT_STG" ]; then IMAT_USE="$IMAT_STG"
    elif [ -f "$IMAT_OUT" ]; then IMAT_USE="$IMAT_OUT"; fi
  else
    echo "--- Step 3: no IMATRIX_CORPUS → skipping imatrix (no imatrix-required types requested)"
  fi

  # ---- Step 4: quantize selected types (staged) ----
  echo "--- Step 4: quantize ${#SEL[@]} type(s)"
  local q_out q_stg imat_arg
  for t in "${SEL[@]}"; do
    q_out="$OUT_MD/$(gq_quant_name "$REPO_MODEL" "$t")"
    q_stg="$STG_MD/$(gq_quant_name "$REPO_MODEL" "$t")"
    if [ -f "$q_out" ]; then
      if gq_ask_reuse "$q_out" "quant GGUF [$t]"; then
        echo "  [$t] present → reuse"; continue
      fi
    fi
    imat_arg=()
    if gq_in_list "$t" "${IMATRIX_REQUIRED_TYPES[@]}"; then
      [ -n "$IMAT_USE" ] || { echo "  [$t] BLOCKED-NO-IMATRIX (imatrix-required) → skip"; continue; }
      imat_arg=(--imatrix "$IMAT_USE")
    elif [ -n "$IMAT_USE" ]; then
      imat_arg=(--imatrix "$IMAT_USE")                # ADR-016: use imatrix for all when available
    fi
    echo "  [$t] quantize → $(basename "$q_stg")${MTP_TENSOR_ARGS:+ (MTP-override=${MTP_QUANT})}"
    if "$QUANTIZE_BIN" "${imat_arg[@]}" "${MTP_TENSOR_ARGS[@]}" "$BF16_STG" "$q_stg" "$t"; then
      gq_publish "$q_stg" "$q_out"
      gq_same_path "$q_stg" "$q_out" || rm -f "$q_stg"
    else
      echo "  [$t] FAILED (continuing)"; rm -f "$q_stg"
    fi
  done

  # ---- Step 5: retention ----
  gq_same_path "$BF16_STG" "$BF16_OUT" || rm -f "$BF16_STG"   # always free staged BF16
  if [ "${KEEP_BF16:-1}" = 0 ]; then echo "--- retention: KEEP_BF16=0 → removing BF16"; rm -f "$BF16_OUT"; fi
  if [ -n "$IMAT_STG" ]; then gq_same_path "$IMAT_STG" "${IMAT_OUT:-}" || rm -f "$IMAT_STG"; fi
  if [ "${KEEP_SAFETENSORS:-1}" = 0 ]; then echo "--- retention: KEEP_SAFETENSORS=0 → removing $SRC"; rm -rf "$SRC"; fi

  echo "=== done: $OUT_MD ==="
  ls -la "$OUT_MD"/*.gguf 2>/dev/null | awk '{print "  "$5"  "$NF}'
}

# Run the pipeline ONLY when executed directly; do nothing when sourced as a library.
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  gq_main "$@"
fi
