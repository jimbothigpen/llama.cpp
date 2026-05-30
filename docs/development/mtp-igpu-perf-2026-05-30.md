# MTP speculative decoding on iGPU (gfx1150 APU): perf characterization

**Date:** 2026-05-30. **Host:** ai00 (Strix Halo gfx1150 APU, ROCm).
**Status:** characterized; product default decision escalated to user.

## Summary

On the gfx1150 APU, MTP (multi-token-prediction / nextn) speculative decoding is a
**net throughput slowdown at every draft depth**, even at 70.8% accept and even at
100% draft acceptance. This is **inherent to the iGPU**, not a tuning miss or a
regression.

## Measurements (Qwen3.5-35B-A3B-MTP IQ4_XS; v-j-545 prompt; -c 2048 -ngl 99 --no-mmap -fa on --temp 0 -n 200)

OFF = `llama-cli` plain decode; ON = `llama-speculative-simple --spec-type draft-mtp`
(same methodology as the historical 0.737× anchor).

| config | t/s | accept |
|--------|-----|--------|
| OFF (pure decode) | 27.5 | — |
| ON n_max=1 | 20.05 | 100% |
| ON n_max=2 | 16.58 | 85.8% |
| ON n_max=3 (default) | 14.90 | 70.8% |
| ON n_max=4 | 13.15 | 55.6% |
| ON n_max=5 | 11.71 | 45.5% |

Ratio at default n_max=3 = **0.54×**. Throughput **falls monotonically** as draft
depth rises. Best case (n_max=1, perfect acceptance) is still below pure decode.

## Root cause: fixed per-decode overhead dominates

The MTP draft is compute-tiny — `qwen35moe.cpp` builds a separate `graph_mtp` running
only the **single** trailing MTP head block (`nextn_predict_layers==1`), ~1/47 of the
body. But on this APU each `llama_decode` carries a **fixed ~28 ms** dispatch/schedule/
sync/embd-upload overhead that dwarfs the draft's compute. Direct evidence from one run:

- prompt processing (batched, overhead amortized): **127 t/s = 7.9 ms/tok**
- single-token generation: **27.5 t/s = 36 ms/tok**

The ~28 ms gap is the per-decode overhead. A target decode ≈ 28 ms overhead + 8 ms
compute; an MTP-head draft decode ≈ 28 ms overhead + ~0 compute ≈ 30 ms — almost a full
target decode.

Per speculation cycle on Qwen3.5 (its MTP head keeps its **own** KV, so
`kv_shared_with_target=false` and a **catch-up decode runs every cycle**):
`1 target verify + 1 catch-up + n_max draft = n_max+2` overhead-bound decodes to produce
only ~`1+accepted` tokens. Overhead swamps the savings → slowdown. (Gemma4's assistant
shares KV and skips the catch-up, so it does not pay this.)

This is why the per-row bulk-sync fix (`ae5979c62`) did not close the gap: the dominant
cost is per-*decode* launch overhead × decode count, which that fix does not address.

## Implications

- On discrete GPUs the per-decode overhead is small relative to compute, so cheap MTP
  drafts win — MTP-on is appropriate there.
- On gfx1150 APU, MTP cannot beat pure decode without **reducing the number of decodes
  per cycle** (eliminate the Qwen catch-up decode, or use in-graph chain heads to emit
  multiple drafts per decode). Both are architectural changes requiring correctness
  validation.

See escalation `kernel-work/orchestrator-inbox/escalated/mtp-vj-perf-2026-05-30-igpu-inherent-default.md`
for the product options (default-off on iGPU / iGPU n_max=1 / invest in catch-up
elimination).

## Update 2026-05-30 — C1 (catch-up elimination) lands the iGPU win

The catch-up decode WAS eliminated from the steady-state cost (C1, "Path C"): the per-cycle
Qwen catch-up `llama_decode(ctx_dft)` is deferred and **batched into the next cycle's lead draft
decode** (one decode writes both the catch-up KV and the new lead), netting **−1 `llama_decode`
per speculation cycle**. Implemented in `common_speculative_impl_draft_mtp`
(`common/speculative.cpp`) — driver-only, no model/graph change.

Measured on the same rig (Qwen3.5-35B-A3B-MTP-IQ4_XS, v-j-545, -c2048 -ngl99 --no-mmap -fa on
--temp0 -n200, fresh ROCm build @ f83746fb1+C1):

| config | t/s | accept | vs pure |
|--------|-----|--------|---------|
| OFF (pure decode) | 28.0 | — | 1.00x |
| **ON C1 n_max=1** | **32.4** | **100%** | **1.16x** |
| ON C1 n_max=3 | 21.1 | 96.95% | 0.75x |

So **C1 + iGPU-default n_max=1 turns the 0.54x slowdown into a 1.16x speedup** — MTP is now a net
win on gfx1150 *at n_max=1*. n_max>=2 stays a slowdown even with C1 (still 4 decodes/cycle at
n_max=3), so the iGPU default is clamped to n_max=1 (`common_speculative_impl_draft_mtp` ctor;
explicit `--spec-draft-n-max` always honored). The startup note was updated accordingly.
(C2, in-graph chain heads, remains an orthogonal later lever — it amortizes the draft decodes,
not the catch-up.)

## Reproduction

`kernel-work/worker-scratch/mtp-vj-perf-2026-05-30/runs/` — `batch-p0.sh` (baseline),
`sweep-ndraft.sh` (depth sweep), summaries and per-run logs.
