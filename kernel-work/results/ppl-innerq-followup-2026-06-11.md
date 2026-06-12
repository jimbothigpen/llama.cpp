# InnerQ follow-up PPL results — 2026-06-11/12

Binary: f2f56232a (b732, /opt/llama-yggdrasil-rocm) — TODO-235 HSA fix
Model: Qwen3.5-9B-Q4_K_M, 50 chunks, c=4096
Backend: gfx1150-rocm (ai00), label=gfx1150-rocm

## 3 K=INNERQ cells (re-measured post-TODO-235 fix)

| kv_k | kv_v | PPL |
|---|---|---|
| turboq2_innerq | turboq2_innerq | 7.1385 |
| turboq3_innerq | turboq3_innerq | 6.9186 |
| turboq3_innerq | turboq2_innerq | 7.0722 |

All 3 in sanity band (9B Q4_K_M expected ~6.9–7.2). The K=INNERQ NaN bug (TODO-235)
is resolved. Results written to kv-quant-Qwen3.5-9B-gfx1150-rocm.csv (label=gfx1150-rocm).

## 236-L2 probe: TURBO_INNERQ=256

| cell | TURBO_INNERQ | PPL |
|---|---|---|
| turboq3_tcq/turboq3_tcq | 1 (prior) | 17.0266 |
| turboq3_tcq/turboq3_tcq | 256 (probe) | 17.3386 |

VERDICT: **ESCALATE** — more calibration tokens make no improvement (17.34 vs 17.03).
The TCQ×InnerQ scale_inv race is a real Q/K-map bug, not a single-token calibration
artifact. Phase 4 required.

Out-of-tree artifacts:
- /mnt/cephfs/0/Container/models/matrices/kv-quant-Qwen3.5-9B-gfx1150-rocm.csv (3 new rows)
- /mnt/cephfs/0/Container/models/matrices/MATRIX-Qwen3.5-9B-kv-quant-consolidated-2026-06-06.csv
- worker-scratch/ppl-innerq-followup-2026-06-11/logs/ (4 PPL run logs)
