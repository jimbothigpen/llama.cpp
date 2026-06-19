# turboq5_0 / turboq6_0 — measurement results (TODO 250, 2026-06-19)

New KV-cache quant types invented this branch (slots 64/65), extending the turboq8
design (FWHT rotation + per-block absmax + uniform grid) to 5- and 6-bit with a
q5_0/q6_K-style lo/hi index split for cheap single-element device decode.

| type     | enum | bpv    | block bytes / 128 vals |
|----------|------|--------|------------------------|
| turboq5_0| 64   | 5.125  | 82  (norm2 + qs64 + qh16) |
| turboq6_0| 65   | 6.125  | 98  (norm2 + qs64 + qh32) |

## Encoder choice
Extended the **FWHT + uniform-grid** family (turboq8 pattern), not the PolarQuant
nonuniform-centroid family (turboq4). Rationale: (1) simplest correct device-side
decode — uniform grid is pure arithmetic, no centroid table to replicate on GPU;
(2) CPU↔GPU byte-identical encode is trivial; (3) proven design (turboq8 ships).

## Setup
- Model: Qwen3.5-9B-Q4_K_M (32 layers, head_dim 256, 4 KV heads, GQA 16:4).
- GPU: discrete RTX 2070 Super, sm_75, 8 GB, CUDA 12.4 (native), build FA_ALL_QUANTS=OFF.
- PPL: wikitext-2 wiki.test.raw, `-fa on -c 4096 -ub 512 -b 512 --no-mmap -ngl 99`, 20 chunks, symmetric KV.
- TPS: llama-bench `-fa 1 -mmp 0 -ngl 99`, symmetric KV.

## PPL (20 chunks, ±1σ)
| type      | bpv   | PPL ±σ            | vs comparable |
|-----------|-------|-------------------|---------------|
| q5_0      | 5.5   | 6.4052 ± 0.0769   | (baseline)    |
| q5_1      | 6.0   | 6.3960 ± 0.0768   | (baseline)    |
| q8_0      | 8.5   | 6.4040 ± 0.0769   | (baseline)    |
| **turboq5_0** | 5.125 | **6.4024 ± 0.0769** | vs q5_0: −0.0028 (tie, ≪σ); vs q5_1: +0.0064 (tie) |
| **turboq6_0** | 6.125 | **6.4033 ± 0.0769** | vs q5_1: +0.0073 (tie); vs q8_0: −0.0007 (tie) |

All five KV quants tie within σ — at 5–6 bit the KV cache is already near-lossless
on this model/context (the f16 KV floor is ~6.40), so there is no quality gap for
the WHT rotation to close.

## Throughput — PP512 (reliable, low variance)
| type      | PP512 t/s        | vs mainline |
|-----------|------------------|-------------|
| q5_0      | 160.82 ± 0.02    | —           |
| q5_1      | 161.03 ± 0.02    | —           |
| q8_0      | 160.94 ± 0.04    | —           |
| turboq5_0 | 110.29 ± 2.63    | **−31%**    |
| turboq6_0 | 118.65 ± 0.02    | **−26%**    |

Turbo types are VEC-only (no TILE/MMA prompt kernel) and the graph rotates Q /
un-rotates output every layer → a structural prompt-throughput penalty.

## Throughput — TG128 (UNRELIABLE on this host)
TG run-to-run variance was severe (e.g. q8_0 measured 48.83±0.42 then 10.23±4.37 on
identical isolated runs — model on networked FS + 8 GB tightness + GPU clock variance).
Indicative clean turbo numbers: turboq5_0 ≈ 46–48 t/s, turboq6_0 ≈ 38 t/s. Mainline
q5_0/q5_1 have **no FA-VEC kernel** in a FA_ALL_QUANTS=OFF build, so their single-token
(TG) decode aborts at `fattn.cu` — TG comparison vs q5_0/q5_1 is not available here.
TG is memory-bandwidth-bound on the 5.23 GiB model weights (KV is a few MB), so KV bpv
barely moves TG; no turbo TG win is expected or observed.

## VERDICT — NO-WIN (both widths)
- **turboq5_0: NO-WIN.** PPL ties q5_0/q5_1 (within σ); PP −31%; TG ≈ comparable.
- **turboq6_0: NO-WIN.** PPL ties q5_1/q8_0 (within σ); PP −26%; TG ≈ comparable.

Neither beats its comparable mainline KV quant on PPL **or** throughput. Per the
win-gate, **Phase 2 (Vulkan) was NOT entered.**

## Why (structural, not a bug)
The Turbo value proposition — WHT rotation buys near-f16 quality at LOW bit-width — has
no payoff at 5–6 bit because mainline q5/q8 KV is already near-lossless there. What
remains is turbo's VEC-only prompt-path cost. Turbo wins live at ≤4 bit (the
turboq2/3/4 niche), where mainline KV quants degrade; 5–6 bit is the wrong neighborhood.

## Functional correctness (verified)
- GPU set-rows encode + FA-vec decode produce coherent generation for both types
  (greedy smoke, no NaN/Inf).
- CPU reference codec roundtrip passes test-backend-ops GET_ROWS/SET_ROWS in isolation
  (a transient NaN seen once was traced to 3 concurrent test-backend-ops processes
  exhausting host RAM, not a codec defect).
- Full CUDA backend builds clean (sm_75), incl. all 12 hand-written FA-vec instance
  files (D=64/128/256/512) for the new types.

## Recommendation
**DORMANT** — keep the branch as a documented negative result + reusable wiring template
for future TurboQuant widths; do **not** merge to main, do **not** pay the Vulkan port cost.
