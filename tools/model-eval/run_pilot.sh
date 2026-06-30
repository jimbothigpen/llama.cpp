#!/usr/bin/env bash
# Convenience wrapper: score one served model and write JSON + markdown scorecard.
#
# Usage:
#   ./run_pilot.sh                       # defaults to the 35B on :8080
#   BASE_URL=http://127.0.0.1:8080 MODEL=qwen36-35b ./run_pilot.sh
#   MODEL=qwopus35-9b BASE_URL=http://127.0.0.1:8080 ./run_pilot.sh
#
# Queries an ALREADY-RUNNING OpenAI-compat server; never launches/restarts one.
set -euo pipefail
cd "$(dirname "$0")"

BASE_URL="${BASE_URL:-http://127.0.0.1:8080}"
MODEL="${MODEL:-qwen36-35b}"
SUITE="${SUITE:-suites/default.json}"
RPS="${RPS:-0.7}"                 # modest load — do not hammer prod
OUT="results/${MODEL}.json"
MD="results/${MODEL}.md"

echo "scoring $MODEL @ $BASE_URL  (suite=$SUITE, rps=$RPS)"
python3 eval_harness.py \
  --base-url "$BASE_URL" --model "$MODEL" --suite "$SUITE" \
  --rps "$RPS" --req-timeout 180 \
  --out "$OUT" --md "$MD"
echo
echo "scorecard: $MD"
