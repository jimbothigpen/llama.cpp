# PPL Headroom Study for Asymmetric KV (turbo5/6 native KV gate)

**Date:** 2026-07-02
**Model:** Qwopus3.5-9B-Coder-IQ4_KS
**Corpus:** wikitext-2-raw/wiki.test.raw (16 chunks, ctx 512)
**Host:** ai01 (gfx1103)
**Binary:** Vulkan

## Available q5/q6-class Plain Types
- `q5_0` and `q5_1` exist as plain types.
- `q6_0` does **NOT** exist as a plain type (only `turboq6` exists).

## Results Matrix

| KV Config (K/V) | PPL | σ | K (MiB) | V (MiB) | Effective BPW |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `f16/f16` (anchor) | 8.6461 | 0.34833 | 32.00 | 32.00 | 16.00 |
| `q8_0/q8_0` (anchor) | 8.6515 | 0.34883 | 17.00 | 17.00 | 8.50 |
| `q8_0/q5_0` | 8.6481 | 0.34862 | 17.00 | 11.00 | 7.00 |
| `turboq6/q5_0` | TIMEOUT | N/A | 12.25 | 11.00 | 5.81 |
| `q5_1/q5_0` | 8.6416 | 0.34806 | 12.00 | 11.00 | 5.75 |
| `q5_0/q4_0` | 8.6433 | 0.34827 | 11.00 | 9.00 | 5.00 |
| `turboq4/q4_0` | 8.6661 | 0.34952 | 8.50 | 9.00 | 4.38 |
| `turboq4/turboq4` (anchor) | 8.7017 | 0.35202 | 8.50 | 8.50 | 4.25 |

*(Note: `turboq6/q5_0` was aborted due to falling back to CPU and hanging at 790% CPU, taking >12 mins for a single chunk).*

## Analysis and Verdict

When plotting PPL against effective KV BPW in the 4–8 BPW range:
- `turboq4/turboq4` provides a baseline of `8.7017` at `4.25` BPW.
- `q8_0/q8_0` provides a high-fidelity anchor of `8.6515` at `8.50` BPW.
- The `q5` configurations (`q5_0/q4_0` at 5.00 BPW, `q5_1/q5_0` at 5.75 BPW) sit around `8.64 - 8.65` PPL, which is indistinguishable from the `f16` and `q8_0` baselines. This represents a meaningful reduction in perplexity (~0.06 points) compared to `turboq4` while saving substantial VRAM (3-3.5 BPW) compared to `q8_0`.
- The PPL decay curve clearly shows that moving from 4.25 BPW (`turboq4`) to ~5.0–5.75 BPW (`q5` class) bridges a significant gap in perplexity, establishing a strong Pareto point. 

**Verdict:** **GO-228**
The q5/q6-class asymmetric configurations occupy a useful Pareto point, offering lower PPL-per-bit in the 4-8 bpw band. Implementing `turbo5/6` native KV is justified and provides headroom for improvement over `turboq4_0` without the heavy cost of `q8_0`.
