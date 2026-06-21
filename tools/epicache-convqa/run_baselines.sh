#!/bin/bash
# Run the EpiCache conv-QA baseline arms end-to-end and print the comparison table.
#
# Required env (override as needed):
#   BIN     llama-server built with -DLLAMA_EPICACHE=ON   (REQUIRED)
#   MODEL   GGUF model path                                (REQUIRED)
#   TRIA    .tria TriAttention stats for MODEL             (REQUIRED for p1/plain-evict)
# Optional:
#   SUBSET  subset json (default: ./data/subset.json, built if missing)
#   BUDGET  EpiCache prefill budget M (default 1024)
#   NGL     GPU layers (default 0 = CPU)
#   THREADS CPU threads (default 16)
#   PORT    server port base (default 8190)
#   ARMS    space-separated arms (default "full p1 plain-evict")
#
# conv-QA accuracy is backend-independent, so CPU (NGL=0) gives valid accuracy
# numbers; set NGL=99 + a GPU build for the prod-representative (e.g. Qwen3.5-9B) run.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${BIN:?set BIN=<llama-server with LLAMA_EPICACHE=ON>}"
: "${MODEL:?set MODEL=<gguf>}"
TRIA="${TRIA:-}"
BUDGET="${BUDGET:-1024}"
NGL="${NGL:-0}"
THREADS="${THREADS:-16}"
PORT="${PORT:-8190}"
ARMS="${ARMS:-full p1 plain-evict}"
OUTDIR="${OUTDIR:-$HERE/results}"
SUBSET="${SUBSET:-$HERE/data/subset.json}"
mkdir -p "$OUTDIR"

if [ ! -f "$SUBSET" ]; then
    [ -f "$HERE/data/locomo10.json" ] || bash "$HERE/fetch_data.sh"
    python3 "$HERE/prepare_subset.py" --raw "$HERE/data/locomo10.json" \
        --out "$SUBSET" --convs "${CONVS:-2}" --qpc "${QPC:-10}"
fi

RES=()
i=0
for arm in $ARMS; do
    out="$OUTDIR/results_${arm}.json"
    echo "===== ARM: $arm ====="
    python3 "$HERE/run_eval.py" --bin "$BIN" --model "$MODEL" --subset "$SUBSET" \
        --arm "$arm" --tria "$TRIA" --budget "$BUDGET" \
        --ngl "$NGL" --threads "$THREADS" --port "$((PORT + i))" \
        --out "$out" ${ALLOW_UNIMPL:+--allow-unimplemented}
    RES+=("$out")
    i=$((i + 1))
done

echo "===== SUMMARY ====="
python3 "$HERE/summarize.py" "${RES[@]}" | tee "$OUTDIR/summary.md"
