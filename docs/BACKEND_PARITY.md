# Backend parity policy

This fork targets **ROCm and Vulkan as first-class backends**. Every
feature is expected to reach parity across both; features that cannot are
explicitly documented as such.

This document defines the policy, lists the per-feature status, and tracks
the Vulkan-port backlog.

## Policy: parity-as-goal with two-track landing

A feature progresses through three states:

| State | Definition |
|---|---|
| **ROCm-landed** | Feature works correctly on ROCm. Passes PPL regression on at least one mandatory ROCm target. Merged to trunk. |
| **Vulkan-landed** | Vulkan implementation works correctly. Passes PPL regression on Vulkan. Merged to trunk (typically on a follow-up branch). |
| **Released** | Both ROCm-landed AND Vulkan-landed AND cross-backend PPL match within tolerance (defined per type in TYPE_ASSIGNMENTS.md). Listed in user-facing release notes. |

**Vulkan parity is high-priority, not a hard gate.** A feature ROCm-lands
first; the Vulkan port is scheduled as a follow-up sub-layer. The feature
does NOT appear as a released feature until both backends are
on trunk and cross-backend PPL matches.

**A feature is shipped ROCm-only if and only if** the Vulkan port has been
attempted and documented as non-viable (e.g., a fundamental compute-shader
limitation around Viterbi decoding or stateful KV reuse). Non-viability
requires a written rationale in this document.

## Hardware targets

### ROCm targets

| Target | Priority | Hardware | Notes |
|---|---|---|---|
| **gfx1150** | **Mandatory** | Strix Halo APU | Primary development target; all ROCm features must work here. |
| gfx1102 | Nice-to-have | RDNA3 mobile | AMD's upstream ROCm support is the real bottleneck (stock Tensile lacks GEMM kernels for many dtypes/shapes). Use Vulkan instead unless the side-quest below produces working ROCm. |
| gfx1103 | Nice-to-have | RDNA3 mobile | Same as gfx1102. |

A feature ROCm-lands when it works on **gfx1150**. gfx1102/1103 are not
release-blocking.

### Vulkan targets

Vulkan is expected to be the more portable backend across mixed
hardware. Targets:

- AMD RDNA3 (gfx1102/gfx1103) — primary Vulkan use case (systems where ROCm is blocked on upstream AMD)
- AMD RDNA3.5 (gfx1150) — first-class Vulkan target alongside ROCm
- Anything else that runs Mesa RADV — best-effort

A feature Vulkan-lands when it works on both RDNA3 and RDNA3.5 Vulkan.

## Cross-backend PPL parity

The PPL regression harness runs every supported quant type on BOTH ROCm
and Vulkan after both have landed, on the SAME pinned wikitext slice.
Cross-backend delta tolerance:

- **PPL delta < 0.5%** vs ROCm baseline → parity verified.
- **0.5% ≤ delta < 1.0%** → parity warning; investigate before release.
- **delta ≥ 1.0%** → parity failure; release blocked until fixed.

The harness pins the wikitext slice to avoid the [wikitext-mismatch trap](https://example.com/wikitext-pin)
that's burned past porting work. Same file, same byte offset, both backends.

## Per-feature parity status

Status updated per layer landing. Initial state derived from
[BACKEND_PARITY.md audit, 2026-05-12](#audit-baseline-2026-05-12).

| Feature | Layer | ROCm status | Vulkan status | Vulkan port priority |
|---|---|---|---|---|
| Zyphra ZAYA1-8B model arch (`LLM_ARCH_ZAYA`) | model port (not phased) | **RELEASED** on gfx1150; compiles on gfx1102/1103 but runtime dead per Tensile/hipBLAS gap | **RELEASED** on RDNA3 (gfx1103); single-seq + multi-seq PPL within ±0.5% across F16/Q8_0/Q5_K_M/IQ4_XS-imat-guq5k | n/a (released; pure-graph port, no new kernels or types) |
| TurboQuant KV (TURBOQ2/3/4_0) | 1 | **RELEASED** (gfx1150 first-class; gfx1102/1103 smoke-only via `HSA_OVERRIDE_GFX_VERSION=11.0.2`) | **RELEASED** (RDNA3 + RDNA3.5; cross-backend Δ ≤ +0.17%) | n/a (released) |
| TurboQuant8 KV (TURBOQ8_0, source: buun) | 1 | **wired** — CPU vec_dot + CUDA/HIP fattn-vec instances on main (`b22b6492d`); 8-bit uniform-grid codec; quality bench pending (measure-first) | not present — no Vulkan kernel | P2 — follow CUDA/HIP measure-first; port after benchmark justifies it |
| WHT weight quants (WHT3_0) | 1 | **RELEASED** | **RELEASED** | n/a (released) |
| WHT weight quants (WHT4_0) | 1 | **RELEASED** | **RELEASED** (cross-backend Δ +0.057%) | n/a (released) |
| GGML_OP_TURBO_WHT | 1 | **RELEASED** | **RELEASED** | n/a (released) |
| Boundary V / `TURBO_LAYER_ADAPTIVE` | 1 | **RELEASED** | **RELEASED** | n/a (default-off; backend-agnostic plumbing) |
| MTP spec-decode spine + Migration 0-3 | 2 | **RELEASED** (speculative driver + loader + graph converged to b9246; V-J gap closed `705ffccb8`) | no novel GPU kernels — Vulkan parity via inherited mainline paths | n/a (CPU/scheduler only; no backend-specific kernels) |
| NLD (Nemotron-Labs Diffusion) | model port (not phased) | **RELEASED** ROCm — CLI `49f88e18a`; server self-spec `1cb8c4218` | **RELEASED** Vulkan (smoke + 5ch PPL PASS on gfx1103 RADV, 2026-05-24; llama-cli load+gen + llama-perplexity validated) | n/a (released; backend-agnostic inference path, no new kernels) |
| TCQ KV (TURBOQ2/3_TCQ) | 3 | source has CUDA only, no HIP | source has none | P1 — Viterbi-in-shader is hard; investigate viability |
| TriAttention | 9 (revived) | **RELEASED** — Phase A+B in-graph K/V capture harness `6cbc9e06c` + HIP guard + Gemma-4 ISWA fix `cbd071632` + legacy compaction evictor `6f93b4e5d`; GQA CPU smoke GREEN 3/3; Phase C GPU GQA kernel HIP `51a64b43c` + Vulkan `0d13ac92b` SHIPPED; SWA-layer capture for Gemma-4 hybrid models SHIPPED `086c8508f` (Phase C Part 2) | **RELEASED** — Vulkan GPU GQA kernel `0d13ac92b` + SWA capture `086c8508f`; cross-backend parity achieved | n/a (released) |
| EAGLE3 | 5 | **RELEASED** — hidden-state extrapolation ported `c0f3c1486`; fc dtype-aware fix `4c38845c4` (BF16/F16→F32); struct rebase fixup `e109b17d8`; compact-vocab (SpecForge 32K-draft-vocab + d2t) `b2766ef47` (2026-05-31, PR #18039) | backend-agnostic; no novel GPU kernels | n/a (backend-agnostic) |
| PHANTOM-X | 5 | **RELEASED** — speculator ported `d6dc63224`; Phase 2 dispatch `388169995` | backend-agnostic; no novel GPU kernels | n/a (backend-agnostic) |
| TurboMind allocator | 5 | gfx1030-specific in source | source has none | Investigate; may not be needed |
| Wave32 RDNA2 kernels | 5 | ROCm-only by design (RDNA2 SIMD32) | not applicable | **ROCm-only** by design |
| IK quants base-K (IQ2_K, IQ3_K, IQ4_K) | 5 (5b-1a) | **RELEASED** — ROCm + Vulkan (PPL Δ < 0.0045 vs reference) | **RELEASED** — Vulkan batched mul_mat SEGV fixed `5fe804bcd` | n/a (released) |
| IK quants row-meta KS/KT (IQ4_KS, IQ4_KSS, IQ3_KS, IQ4_KT) | 5 (5b-1b) | **RELEASED** — ROCm + Vulkan (PPL gate 20-chunk Δ ≤ 0.043) | **RELEASED** — Vulkan SEGV fixed via `is_empty()` dequant-to-f16 fallback | n/a (released) |
| IK quants extended (IQ5_K, IQ6_K) | 5 (5b-2) | **RELEASED** — Phase 5b-2 CPU+CUDA/HIP `8e19be061` | **RELEASED** — Vulkan dequant + matvec shaders `0ade7ff86` 2026-05-25 | n/a (released) |
| IK quant IQ2_KL (2.6875 bpw) | 5 (5b-1c) | **RELEASED** — Phase 5b-1c CPU+CUDA/HIP `f18a92a42` | **RELEASED** — Vulkan dequant + matvec shaders `3723c1f61` 2026-05-25 | n/a (released) |
| IK Trellis IQ2_KT (Phase P3a) | 5 (Trellis P3a) | **RELEASED (§-FLAG do-not-use)** — CPU+ROCm + Vulkan; `0dac276d9` + cluster-accel `1e8501e46` 2026-05-25; known issues: 0.8B PPL anomaly open; cluster-accel PPL +8.3% above ≤+5% gate | **RELEASED** — Vulkan dequant + mul_mat_vec shaders (main) | Formal PPL parity gate pending |
| BitNet (IQ1_BN, IQ2_BN, I2_S) | 6 | source has CUDA + implicit HIP | source has none | P1 — ternary decode is simple |
| MLA / FlashMLA | 6 | source has CUDA | source has none | P2 — very high port cost |
| Fused MoE | 6 | source has CUDA | source has none | P2 |
| Trellis weight quants (IQ3_KT, IQ1_KT) | 5 (Trellis P3b, P3c) | **RELEASED** — IQ3_KT CPU+ROCm+Vulkan (`c809225f6`, CLOSED — gfx1150 GPU 99% util / 7.66s-pass; gfx1102 warmup crash is a separate Tensile confound); IQ1_KT queued | **RELEASED** — Vulkan dequant + mul_mat_vec shaders (main) | RESOLVED on Vulkan: IQ3_KT 8.4299 / IQ3_K 6.8348 (9B, 20ch); ROCm gfx1150 PASS (CLOSED) |
| Q\*_K row-interleaved (\_R4/\_R8) | 6 | CUDA only in ik_llama | none | P2 — CPU variants exist; GPU optional |
| RaBitQ TQ3 weights (RBQ3_\*) | 7 | source has CUDA; HIP branch not yet merged | source has none | P1 |
| DFlash spec-decode (S1+S2+S3+converter) | 7a | **RELEASED** — S1 loader `b6a75e524` + S2 dispatch `ef80c728c` + S3 GPU ring buffer + server `spec_type` `9b7ab4e83` + mask_token_id u32 fix `1436d1890` + DFlashDraftModel safetensors→GGUF converter `ee7d4f896`; end-to-end smoke GREEN @ `2726a56c0` | backend-agnostic at S3 dispatch level; GPU ring buffer is CUDA/HIP | n/a (released) |
| PFlash prompt compression | 8 | mostly CPU/scheduler | mostly CPU | P3 — backend-agnostic |
| --hugepages | 9 | Linux kernel feature; backend-agnostic | same | n/a |
| gfx1030 normalization | 9 | ROCm-only by design | not applicable | **ROCm-only** by design |

Priorities:
- **P0** = port concurrent with ROCm landing (target: same week)
- **P1** = port within 2 weeks of ROCm landing
- **P2** = port within 4 weeks of ROCm landing
- **P3** = port within 8 weeks of ROCm landing or backend-agnostic
- **declined** = will not port unless explicitly revisited

## Vulkan port methodology

For each Vulkan port, follow this recipe (informed by community experience
with the Vulkan SET_ROWS 5-place wiring pattern and the Vulkan
buffer-sizing pitfall for IK-family row-meta types):

1. **Audit upstream Vulkan dispatch.** Map the ROCm code path to its Vulkan
   counterpart in `ggml-vulkan.cpp`. Identify the 5 places SET_ROWS must
   be wired for any new quant type.
2. **Use `ggml_nbytes(src0)` for descriptor sizing** — never `type_size *
   x_ne / blck_size`. The latter silently undersizes IK-family row-meta
   types and produces silently-wrong results.
3. **Write the compute shader.** Start from the most similar existing
   shader as a template.
4. **Wire dispatch sizing.** Missing this for a new type produces silently
   wrong K cache and 0 tokens generated.
5. **PPL-test on a pinned wikitext slice.** Run the SAME slice on ROCm to
   confirm cross-backend parity. Different wikitext slices produce
   different absolute PPLs and have produced phantom 0.42-PPL "regressions"
   that were really file-mismatch artifacts.
6. **Submit as a follow-up topic branch.** Name: `vulkan/<feature>`.

## gfx1102 / gfx1103 ROCm — partial scope (smoke target)

AMD's upstream ROCm support for RDNA3 mobile (gfx1102/1103) is incomplete
— stock Tensile lacks GEMM kernels for many dtype/shape combinations
encountered in production workloads. ROCm calibration on RDNA3-mobile
systems is unworkable for production inference; Vulkan is the practical
alternative for those workloads.

**Project decision (2026-05-12, refined 2026-05-21):** This fork treats
gfx1102/1103 ROCm as **out of scope for production inference / calibration**,
**in scope as a regression-smoke target**. The build catches HIP-shim
breakage early (e.g., new `__shfl_xor_sync` call sites, missing
`cudaStreamCapture*` shims, undefined-symbol link errors when a new
fattn-vec template instance is added). Cross-host PPL parity is validated
against gfx1150 on models that fit gfx1103's GEMM coverage
(empirically: TurboQuant KV types + WHT weight quants pass; mainline
Q4_K_M passes; production-class quantize/calibrate workloads still hit
Tensile gaps).

**Build recipe.** Single-target gfx1102 build (dual-target gfx1102+gfx1103
install hangs at relink). Runtime requires the HSA override to load the
gfx1102 binary on gfx1103 hardware via RDNA3-family ISA compatibility:

```bash
cmake -B build-rocm-gfx1102 -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llama.cpp \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="gfx1102" \
    -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON -DGGML_AVX512_BF16=ON \
    -DCMAKE_C_FLAGS="-march=native -O3" \
    -DCMAKE_CXX_FLAGS="-march=native -O3"
cmake --build build-rocm-gfx1102 -j

# Runtime: ALWAYS set HSA_OVERRIDE
HSA_OVERRIDE_GFX_VERSION=11.0.2 ./build-rocm-gfx1102/bin/llama-perplexity ...
```

If AMD ships upstream support for gfx1102/1103 GEMM kernels at some
future point, this fork will inherit it via the standard cmake recipe
without project-side work, and this section will collapse back to
"first-class".

## Documenting Vulkan-non-viable features

If a feature's Vulkan port is attempted and abandoned, document it here
with:

- Feature name and layer
- What was attempted
- What failed (technical reason)
- ROCm-only marker added to feature documentation
- Date of decision

A feature shipped ROCm-only will be clearly marked in user-facing release
notes and the README's feature table.

### Currently ROCm-only by design

- **Wave32 RDNA2 kernels** (carlosfundora) — RDNA2-specific SIMD32 path.
  Vulkan equivalent would be a generic compute shader; the optimization
  doesn't translate.
- **gfx1030 normalization** (carlosfundora) — ROCm gfx1030 build-fixup
  patches. Vulkan-irrelevant.

### Vulkan-attempted-and-abandoned

(none yet)

## Known issues and workarounds

### MTP speculative decoding with multi-token drafts (Vulkan, n_max ≥ 2) — RESOLVED

**Feature:** MTP (Multi-Token Prediction) speculative decoding
**Backends affected:** Vulkan (historical)
**Status:** Resolved — `n_max >= 2` now works on Vulkan; the `--spec-draft-n-max 1` constraint is no longer required.
**Symptom (historical):** The inference process hung when running Vulkan MTP speculative decoding with `n_max >= 2` (multi-token draft batches). Single-token drafts (`n_max = 1`) were unaffected.

**Root cause:** A partial-acceptance checkpoint-restore livelock in `examples/speculative-simple`. On a partial accept the draft prefix was restored verbatim, but the MTP draft head's per-step state was not rolled back, so the resampled tail desynced the next verification trace and the restore loop never made progress (reachable only when `n_max >= 2` can leave more than one pending draft id). It presented as a GPU-consuming hang because every restore re-ran the verify decode. The equivalent fix already existed in the server speculative path; the standalone example was missing it.

**Fix:** Gate the checkpoint `pop_back` on MTP drafts that still have more than one pending id, so the restore loop converges. Vulkan MTP speculative decoding now completes correctly at `n_max >= 2`.

## PPL regression harness requirements

The harness MUST:

1. Run on both ROCm (gfx1150) and Vulkan from day one.
2. Pin the wikitext slice (specific file, specific byte range) and never
   compare across slices.
3. Store baseline PPLs per (model, type, backend) tuple.
4. Flag cross-backend deltas exceeding the parity tolerance bands above.
5. Run before any layer-landing merge.

A test that runs on only one backend is not a complete regression test.

## imatrix requirement (weight quants)

Every weight quantization type that lands in this fork must support the
imatrix (importance-matrix) mechanism. Adding a weight quant without
imatrix support is a layer-landing failure.

**Scope:** Applies to ALL weight quants (`RBQ3_1S/4S`,
`IQ*_K`, `IQ*_KS`, `IQ*_KSS`, BitNet `IQ1_BN/IQ2_BN/I2_S`,
trellis `IQ*_KT` if revived, and any future weight quant). Does NOT
apply to KV-cache quants (`TURBOQ*_0`, `TURBOQ*_TCQ`, `RQ_*`).

**Exception — WHT3_0/WHT4_0:** imatrix is intentionally **disabled** (quantizer audit `a6ccf0bfa`). The Walsh-Hadamard rotation mixes all 32 block columns, so post-rotation coefficient `buf[j]` no longer corresponds to original column `j`. Weighting the rotated residual by original-basis importance `iw[j]` misaligns importance and measurably degrades quality (−15.5 % PPL penalty). Both types quantize unweighted by design; `tensor_requires_imatrix()` returns `false`.

**Per-type integration requires:**

1. `quantize_row_<type>` (or equivalent mmq precursor) accepts an
   imatrix and uses it for per-tensor importance weighting.
2. `src/llama-quant.cpp` integrates with `qs.has_imatrix` /
   `requires_imatrix` per-ftype logic (matching the existing pattern
   for IQ3_S, IQ3_XXS, IQ4_NL).
3. `llama-imatrix` recorder computes the right activation statistic for
   the type.
4. `docs/TYPE_ASSIGNMENTS.md` documents which imatrix axis/group applies.
5. PPL regression run BOTH with and without imatrix; results recorded.

Rationale: weight-quant quality degrades sharply without activation-
weighted importance at low bit-rates (≤4-bit). Mainline already requires
imatrix for some quant types; this fork's contributing forks target ≤4-bit
quants where imatrix is not optional.

## Audit baseline (2026-05-12)

Initial backend coverage audit performed against fork tips:

- thetom/feature/turboquant-kv-cache (originally measured against `feature/alpha-scaling`; TQ-KV is a strict superset — see `recon/06-thetom-branches.md`. Re-audit Vulkan shader counts after Phase 1 cherry-pick.)
- buun/master
- carlosfundora/1-bit-turbo
- turbotan/main + turbotan/experiment/hip-tq3-support
- domvox/feature/triattention-scoring
- ikllama/main

Key finding: **no contributing fork has Vulkan implementations for its
distinctive features.** Vulkan shader counts at or below mainline for all
forks. ik_llama is 56 Vulkan shaders behind mainline.

The implication: this fork bears the entire Vulkan-port burden in-house.
Community work in the wider llama.cpp ecosystem (e.g., experiments outside
the seven audited forks) may have Vulkan implementations of some of these
features; sweep regularly.

## Version log

- **v1** (2026-05-12) — initial policy + audit baseline.
- **v2** (2026-05-21) — Phase 1 release: TurboQuant KV (TURBOQ2/3/4_0) +
  WHT4_0 released on both backends; WHT3_0 Vulkan released (`37737a197`).
  gfx1102/1103 ROCm scope refined from "out of scope" to
  "partial scope: smoke target" (HSA_OVERRIDE recipe documented).
- **v3** (2026-05-22 to 2026-05-24) — Phase 5b-1a (IQ2_K/IQ3_K/IQ4_K) +
  Phase 5b-1b (IQ4_KS/IQ4_KSS/IQ3_KS/IQ4_KT) released on both backends
  (Vulkan batched mul_mat SEGV fixed via `is_empty()` guard `5fe804bcd`).
  MTP spine rows updated: migration phases 0-3 complete; V-J accept-rate
  gap closed. NLD (Nemotron-Labs Diffusion) added as ROCm-released model port.
  Per-feature table expanded with 5b-1a/1b/2 + NLD + MTP migration rows.
- **v4** (2026-05-24) — Phase 5b-2 (IQ5_K/IQ6_K) and Phase 5b-1c (IQ2_KL)
  released rows updated. EAGLE3 + PHANTOM-X marked RELEASED (backend-agnostic).
  DFlash S1 model loader noted.
  NLD Vulkan RELEASED confirmed (gfx1103 RADV, `7da3a8378`). Vulkan base-K
  MUL_MAT_ID fix (`c4da029f3`) recorded.
