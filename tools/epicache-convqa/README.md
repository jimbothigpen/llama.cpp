# EpiCache conv-QA accuracy harness

A small, reproducible **long-conversational-QA accuracy** gate for the EpiCache
work (arXiv 2509.17396). Our existing PPL/TPS anchors cannot measure EpiCache's
headline win — *answer accuracy over long multi-session dialogue* — so this
harness fills that gap and is the gate that future EpiCache phases (P2 layer
budgets, P3 episodic multi-cache routing) must pass.

## Why this exists

EpiCache's claimed wins are **conv-QA accuracy at a bounded KV budget** (near-full
accuracy at 4–6× compression; up to +40% over plain eviction). PPL on wikitext
does not reflect "can the model still answer questions about turn 3 after turn
200." This harness measures exactly that, and is built to compare the arms the
EpiCache roadmap produces.

## Arms

Each arm is **purely a `llama-server` launch configuration** (same binary,
different flags/env), so any accuracy difference isolates the KV policy.

| arm | what it is | server config |
|---|---|---|
| `full` | reference: full attention, no compression | `-fa on` |
| `p1` | **merged** EpiCache P1 block-wise prefill bounding (reuses TriAttention FA-safe proxy scorer) | `--triattention <stats> --tri-budget 50 --tri-window 256 --tri-sink 128 --tri-interval 1 -fa on` + `LLAMA_EPICACHE_PREFILL=1 LLAMA_EPICACHE_BUDGET=M` |
| `plain-evict` | TriAttention eviction *without* the EpiCache prefill budget — plain recent-window + top-K + sink | `--triattention <stats> ... -fa on` |
| `eq3` | **NOT YET IMPLEMENTED** — faithful EpiCache Eq.3 importance scorer (materialized `softmax(QK^T)`, non-FA). P3 work. | placeholder `-fa off` (see IMPL-PLAN) |

The `eq3` arm is scaffolded (refuses to run without `--allow-unimplemented`) so a
later impl worker only flips a flag once the faithful scorer lands. The harness,
subset format, and scorer already accommodate all three measurable arms plus
`eq3`.

> Eventual arm selection is by **conv-QA accuracy** (the user's prod metric), not
> PPL/TPS.

## Files

| file | role |
|---|---|
| `fetch_data.sh` | download LoCoMo (`data/locomo10.json`, ~2.8 MB, no auth) |
| `prepare_subset.py` | deterministic small subset → harness-internal JSON (LoCoMo + LongMemEval adapters) |
| `run_eval.py` | launch server per arm, ask questions, score, write results JSON |
| `score.py` | SQuAD-style EM / token-F1 / contains + adversarial abstention (`--selftest`) |
| `summarize.py` | combine per-arm results → comparison table + **P1-vs-full accuracy gap** |
| `run_baselines.sh` | one-shot: build subset, run all arms, print summary |

## Benchmark

**LoCoMo** (`snap-research/locomo`) — 10 multi-session conversations (~9–26 K
tokens each), ~200 QA pairs per conversation with gold short-form answers and
category labels: `1`=multi-hop, `2`=temporal, `3`=open-domain, `4`=single-hop,
`5`=adversarial (abstention). Chosen over LongMemEval because it ships as one
cleanly-fetchable JSON — ideal for a "small + reproducible" gate. A LongMemEval
adapter is included in `prepare_subset.py` (`--format longmemeval`) so the
benchmark can be swapped without touching the runner.

## Quick start

```bash
cd tools/epicache-convqa
./fetch_data.sh                                   # one-time, needs network
python3 score.py --selftest                       # sanity-check the scorer

# build a small deterministic subset (2 conversations × 10 questions)
python3 prepare_subset.py --raw data/locomo10.json --out data/subset.json \
    --convs 2 --qpc 10

# run all baseline arms (CPU; conv-QA accuracy is backend-independent)
BIN=/path/to/llama-server-LLAMA_EPICACHE-ON \
MODEL=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
TRIA=/path/to/Qwen3.5-0.8B.tria \
SUBSET=data/subset.json NGL=0 THREADS=16 \
./run_baselines.sh
```

`run_baselines.sh` writes `results/results_<arm>.json` + `results/summary.md`
(the comparison + P1-vs-full gap table).

### Single arm

```bash
python3 run_eval.py --bin $BIN --model $MODEL --subset data/subset.json \
    --arm p1 --tria $TRIA --budget 1024 --ngl 0 --threads 16 \
    --out results/results_p1.json
```

## Model choice

The committed baselines use **Qwen3.5-0.8B-Q4_K_M** (CPU): same family as the
user's prod (Qwen3.5/3.6), conv-capable, and fast enough for a reproducible CPU
gate (no GPU lease, no Vulkan device-lost risk). For the **prod-representative
headline**, point `MODEL=Qwen3.5-9B-Q4_K_M.gguf`, `TRIA=Qwen3.5-9B.tria`,
`NGL=99` at a GPU build — the harness is otherwise unchanged.

`-DLLAMA_EPICACHE=ON` is **required** for the `p1` arm (the env flags are inert
without it). The `full` arm runs on any build.

## Scoring

`score.py` reports SQuAD-style **EM**, token-level **F1** (the headline
"LongConvQA accuracy"), and a coarse **contains** signal. Category-5 questions
are scored by *abstention* (1.0 iff the model declines). The default is fully
offline (stdlib only). An LLM-judge can be layered on top of the results JSON if
desired; F1/EM is the reproducible default.

## Reproducibility

`prepare_subset.py` uses no RNG — same flags ⇒ byte-identical subset. Generation
is greedy (`temperature=0, top_k=1`) and `cache_prompt:false` (every question
re-prefills the full context, so the arm's prefill-time KV policy is exercised
identically per question). Stamp results with the binary commit SHA
(`llama-server --version`).
```
