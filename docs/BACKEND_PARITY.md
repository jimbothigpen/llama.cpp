# Backend parity policy

llama-yggdrasil targets **ROCm and Vulkan as first-class backends**. Every
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
does NOT appear as a yggdrasil-released feature until both backends are
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
| Zyphra ZAYA1-8B model arch (`LLM_ARCH_ZAYA`) | model port (not phased) | **RELEASED** on gfx1150; compiles on gfx1102/1103 but runtime dead per Tensile/hipBLAS gap | **RELEASED** on RDNA3.5 (ai01); single-seq + multi-seq PPL within ±0.5% across F16/Q8_0/Q5_K_M/IQ4_XS-imat-guq5k | n/a (released; pure-graph port, no new kernels or types) |
| TurboQuant KV (TURBOQ2/3/4_0) | 1 | **RELEASED** (gfx1150 first-class; gfx1102/1103 smoke-only via `HSA_OVERRIDE_GFX_VERSION=11.0.2`) | **RELEASED** (RDNA3 + RDNA3.5; cross-backend Δ ≤ +0.17%) | n/a (released) |
| WHT weight quants (WHT3_0) | 1 | **RELEASED** | **deferred** — no TQ3_1S shaders in upstream; tracked as yggdrasil-original port | P2 — yggdrasil-original ~50-100 LOC; lands alongside RotorQuant Phase 5 |
| WHT weight quants (WHT4_0) | 1 | **RELEASED** | **RELEASED** (cross-backend Δ +0.057%) | n/a (released) |
| GGML_OP_TURBO_WHT | 1 | **RELEASED** | **RELEASED** | n/a (released) |
| Boundary V / `TURBO_LAYER_ADAPTIVE` | 1 | **RELEASED** | **RELEASED** | n/a (default-off; backend-agnostic plumbing) |
| MTP spec-decode spine | 2 | source has CPU/scheduler logic | source has none for novel kernels | P2 — depends on KV layer Vulkan |
| TCQ KV (TURBOQ2/3_TCQ) | 3 | source has CUDA only, no HIP | source has none | P1 — Viterbi-in-shader is hard; investigate viability |
| TriAttention | 4 | source has dedicated HIP scoring kernel | source has none | P1 — scoring kernel needs Vulkan port |
| RotorQuant (RQ_PLANAR/ISO3/4_0) | 5 | source has full HIP coverage | source has none | P0 — Hadamard/Givens map well to compute shaders |
| Q1_0_G128 (PrismML 1-bit) | 5 | source files not yet located | source has none | P1 |
| EAGLE3 | 5 | mostly CPU; scheduler logic | mostly CPU | P3 — backend-agnostic |
| PHANTOM-X | 5 | CPU n-gram | CPU n-gram | P3 — backend-agnostic |
| TurboMind allocator | 5 | gfx1030-specific in source | source has none | Investigate; may not be needed |
| Wave32 RDNA2 kernels | 5 | ROCm-only by design (RDNA2 SIMD32) | not applicable | **ROCm-only** by design |
| IK quants (IQ\*_K, IQ\*_KS) | 6 | source has CUDA + implicit HIP | **source has none** — ik_llama Vulkan is 56 shaders behind mainline | **P1** — large surface |
| BitNet (IQ1_BN, IQ2_BN, I2_S) | 6 | source has CUDA + implicit HIP | source has none | P1 — ternary decode is simple |
| MLA / FlashMLA | 6 | source has CUDA | source has none | P2 — very high port cost |
| Fused MoE | 6 | source has CUDA | source has none | P2 |
| Trellis weight quants (IQ2/3/4_KT) | 6 | ik_llama branches dormant | none | **declined** unless Phase 0.5 recon revives them |
| Q\*_K row-interleaved (\_R4/\_R8) | 6 | CUDA only in ik_llama | none | P2 — CPU variants exist; GPU optional |
| RaBitQ TQ3 weights (RBQ3_\*) | 7 | source has CUDA; HIP branch not yet merged | source has none | P1 |
| DFlash drafter | 8 | mostly CPU/scheduler | mostly CPU | P3 — backend-agnostic |
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

**Project decision (2026-05-12, refined 2026-05-21):** Yggdrasil treats
gfx1102/1103 ROCm as **out of scope for production inference / calibration**,
**in scope as a regression-smoke target**. The build catches HIP-shim
breakage early (e.g., new `__shfl_xor_sync` call sites, missing
`cudaStreamCapture*` shims, undefined-symbol link errors when a new
fattn-vec template instance is added). Cross-host PPL parity is validated
against ai00 gfx1150 on models that fit gfx1103's GEMM coverage
(empirically: TurboQuant KV types + WHT weight quants pass; mainline
Q4_K_M passes; production-class quantize/calibrate workloads still hit
Tensile gaps).

**Build recipe.** Single-target gfx1102 build (dual-target gfx1102+gfx1103
install hangs at relink). Runtime requires the HSA override to load the
gfx1102 binary on gfx1103 hardware via RDNA3-family ISA compatibility:

```bash
cmake -B build-rocm-gfx1102 -S . -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/opt/llama-yggdrasil-rocm-ai01 \
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
future point, yggdrasil will inherit it via the standard cmake recipe
without project-side work, and this section will collapse back to
"first-class".

## Documenting Vulkan-non-viable features

If a feature's Vulkan port is attempted and abandoned, document it here
with:

- Feature name and yggdrasil layer
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

Every weight quantization type that lands in yggdrasil must support the
imatrix (importance-matrix) mechanism. Adding a weight quant without
imatrix support is a layer-landing failure.

**Scope:** Applies to ALL weight quants (`WHT3_0/4_0`, `RBQ3_1S/4S`,
`Q1_0_G128`, `IQ*_K`, `IQ*_KS`, `IQ*_KSS`, BitNet `IQ1_BN/IQ2_BN/I2_S`,
trellis `IQ*_KT` if revived, and any future weight quant). Does NOT
apply to KV-cache quants (`TURBOQ*_0`, `TURBOQ*_TCQ`, `RQ_*`).

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
imatrix for some quant types; yggdrasil's contributing forks target ≤4-bit
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

The implication: yggdrasil bears the entire Vulkan-port burden in-house.
Community work in the wider llama.cpp ecosystem (e.g., experiments outside
the seven audited forks) may have Vulkan implementations of some of these
features; sweep regularly.

## Version log

- **v1** (2026-05-12) — initial policy + audit baseline.
- **v2** (2026-05-21) — Phase 1 release: TurboQuant KV (TURBOQ2/3/4_0) +
  WHT4_0 released on both backends; WHT3_0 Vulkan deferred (yggdrasil-original
  port pending). gfx1102/1103 ROCm scope refined from "out of scope" to
  "partial scope: smoke target" (HSA_OVERRIDE recipe documented).
