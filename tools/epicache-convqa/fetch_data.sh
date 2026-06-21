#!/bin/bash
# Fetch the LoCoMo long-conversational-QA benchmark (10 multi-session conversations,
# ~200 QA pairs each). Single self-contained JSON, no auth, ~2.8 MB.
#
# Why LoCoMo (vs LongMemEval): both are named acceptable targets and both measure the
# same thing EpiCache headlines — LongConvQA *answer accuracy* over long multi-session
# dialogue. LoCoMo ships as ONE cleanly-fetchable JSON with gold short-form answers and
# per-question category labels (1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
# 5=adversarial), which is exactly the "small + reproducible" shape this gate needs.
# LongMemEval requires assembling a haystack corpus + per-question session selection;
# the harness loader is written so a LongMemEval adapter can be dropped in later
# (see prepare_subset.py: --format).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:-$HERE/data/locomo10.json}"
URL="https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
mkdir -p "$(dirname "$OUT")"
echo "fetching LoCoMo -> $OUT"
if curl -sSL --fail -o "$OUT" "$URL"; then
    sz=$(stat -c %s "$OUT" 2>/dev/null || echo 0)
    n=$(python3 -c "import json;print(len(json.load(open('$OUT'))))" 2>/dev/null || echo '?')
    echo "OK: $sz bytes, $n conversations"
else
    echo "FETCH FAILED ($URL). Network required (one-time). Aborting." >&2
    exit 1
fi
