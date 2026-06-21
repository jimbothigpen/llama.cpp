# Speculative-decode ensemble: n-gram + neural cascade

## Summary

The yggdrasil fork's `--spec-type` flag accepts a comma-separated list of speculative
strategies. When multiple types are listed, `common_speculative_draft()` applies them in
**priority order**: the first impl that proposes a non-empty draft wins for that step; later
impls are skipped. This is the **cascade ensemble** — it is already shipped and requires no
new code.

The recommended ensemble for most workloads is:

```
--spec-type ngram-simple,draft-mtp --spec-draft-n-max 1
```

For maximum throughput on repetitive/templated text, add `--spec-ngram-simple-size-n 4`
(shorter context window → higher n-gram fire-rate at the cost of lower accept%).

## How it works

- **n-gram arm** (ngram-simple, size_n=12 default): matches the last `size_n` tokens against
  the prompt/context window. On a hit it proposes up to `size_m=48` tokens — free CPU lookup.
- **MTP arm** (draft-mtp, n_max=1): runs the attached MTP head for 1-token neural speculation.
  On iGPU (Strix/Hawk Point), n_max=1 is the net-positive sweet-spot (see TODO 233).
- **Cascade rule** (`speculative.cpp:2374`): once an impl fills a non-empty draft for a sequence,
  `dp.drafting` is set false; subsequent impls skip that sequence for this step.
- **accept() is broadcast** to all impls regardless of who drafted, so the n-gram window stays
  updated even on MTP-won steps.
- **No shared n_max cap** between impls: each impl uses its own internal limit. The cascade
  lets n-gram propose up to 48 tokens on its steps while MTP stays at n_max=1.

## Benchmark results (TODO 117, 2026-06-12)

Host: gfx1103 (ROCm, `HSA_OVERRIDE_GFX_VERSION=11.0.2`).
Binary: `/opt/llama-yggdrasil-rocm/bin/llama-speculative-simple` v733 (`d4a3c802c`).
Model: `Qwen3.5-9B-MTP-Q4_K_M.gguf`. Prompt: 649-tok Wikipedia "Robert Boulter" passage.
Sampler: `--temp 0 --seed 1`. Context: `-n 256 -c 4096 -fa on -ngl 99 --no-mmap`.

| config | decode t/s | vs MTP-only | vs ngram-only | n_drafted | n_accept | accept% |
|---|---|---|---|---|---|---|
| pure decode (ref) | 13.0 | 0.89× | — | — | — | — |
| MTP-only, n_max=1 | 14.62 | 1.00× | 0.37× | 129 | 129 | 100% |
| ngram-only (default) | 39.57 | 2.71× | 1.00× | 240 | 215 | 90% |
| **cascade-default** (n=12→mtp) | **45.62** | **3.12×** | **+15.3%** | 266 | 240 | 90% |
| **cascade-aggressive** (n=4→mtp) | **49.10** | **3.36×** | **+24.1%** | 397 | 246 | 62% |

Results are 3-run means; variance <0.5%. Output is character-identical across all configs
(greedy temp=0 spec-decode is lossless). MTP-only re-confirmed in-session vs neighbor worker.

**Why cascade-default wins (+15%):** same 90% accept rate as ngram-only but 25 more accepted
tokens (+266 drafted vs 240). The extra accepts come from MTP filling steps where n-gram
found no context match — on those steps ngram-only falls back to pure 1-tok decode while the
cascade gets MTP's free +1 at 100% accept.

**Why cascade-aggressive wins (+24%):** size_n=4 fires n-gram far more often (397 vs 240
drafted), accept% drops to 62% but rejected n-gram drafts are CPU-only lookups, so the net
throughput is still higher.

## Caveats

- **Workload-dependent.** The 15–24% gain is on a repetitive Wikipedia passage where n-gram
  excels. On non-repetitive prose n-gram fires rarely; both cascade configs converge toward
  MTP-only (~14.6 t/s). The structural guarantee is: the cascade is **never worse** than the
  better of its two arms — it is Pareto-safe.
- **Default unchanged.** Do not change the global default to the cascade — n-gram's benefit is
  workload-specific and would surprise non-repetitive users.
- **Pick-longest (opt-in, `--spec-ensemble`).** A "true ensemble" running *both* arms every step
  and keeping the longer draft per seq (ties keep the higher-priority arm). Enabled with
  `--spec-ensemble` on top of a 2+ `--spec-type` list; the default remains the priority cascade.
  It forces the neural forward pass even on n-gram-hit steps, so for *throughput* it is generally
  dominated by the cascade (per the benchmark rationale above) — use it only when you specifically
  want max draft length per step rather than min latency. Correctness is identical: greedy output
  is character-for-character the same as the cascade. The losing arm each step is still broadcast
  `accept(is_other=true)`, and stateful arms (MTP) re-sync from the real accepted batch in
  `process()`, so discarding their proposal is safe. Implemented in
  `common_speculative_draft_ensemble()` (TODO 117).

## Related

- `common/speculative.cpp` — `common_speculative_draft()` dispatch → `_cascade` (default) /
  `_ensemble` (pick-longest, `--spec-ensemble`)
- `docs/features/mtp.md` — MTP speculative decoding
- TODO 117 measurement data: `kernel-work/worker-scratch/117-ngram-neural-ensemble-2026-06-12/`
