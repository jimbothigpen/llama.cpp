# llama-yggdrasil

> ## ⚠️ Disclaimer — please read before reviewing or using this repository
>
> **The repository owner is not a software developer.** This project is
> **vibe-coded** in collaboration with [Claude Code](https://claude.com/claude-code).
> Every line of source change against the upstream `ggml-org/llama.cpp`
> codebase — design, port, integration, build, test, documentation — is
> produced by Claude Code under conversational direction. **No code change
> in this repository is hand-written by a human.**
>
> Treat the contents accordingly:
>
> - Don't assume mainline-llama.cpp quality conventions. This is an
>   experimental consolidation project; correctness is verified empirically
>   (PPL parity, benchmarks, smoke tests), not by traditional code review.
> - Don't expect timely security patches, CVE response, or production-grade
>   support. If you need a hardened downstream, use mainline llama.cpp.
> - Don't open PRs expecting a developer-style review cycle. The owner can
>   discuss intent and shape but can't independently review code.
> - Cite upstream `ggml-org/llama.cpp` for everything not introduced by
>   yggdrasil layers; cite this repo for the consolidation work itself.
>
> If you're here to learn how a non-developer can drive a complex
> systems-software fork end-to-end with an AI agent, you're in the right
> place. If you're here for production-ready inference code, go upstream.

A unified downstream of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
that absorbs novel work from six sibling forks into a single coherent tree.

> Yggdrasil: the Norse world-tree where many branches converge at the trunk.

**Status:** Phases 0, 0.5, 0.7, 1, 2, 3, and 7b COMPLETE — **v355 (`4260989e8`)** on
`main`. Phases 3a (TCQ KV ROCm/CUDA), 3c (TCQ KV Vulkan), and 3d (InnerQ KV
types) are merged to `main`. Phase 4a (RotorQuant KV) is in-flight; Phase 7b
(PFlash prompt compression) shipped with HIP-optimized scorer. See [What's available now](#whats-available-now) and
[In-flight workstreams](#in-flight-workstreams) for detail.

## What yggdrasil is and isn't

**Is:** a long-lived downstream fork of mainline llama.cpp, syncing with
upstream on a regular cadence, layering vetted work from five contributing
forks plus selective backports from ik_llama.

**Isn't:** a patch-set distribution, a temporary branch, a competitor to
mainline, or a candidate for upstream contribution. Yggdrasil exists to
consolidate features that mainline doesn't yet absorb but that the community
has already implemented in disparate forks. Per project policy, no AI-generated
code is proposed for upstream submission.

The standing constraint is **mainline fidelity**: diverge only when measurably
better, document every deliberate divergence, and sync regularly. Most
yggdrasil commits are either mainline cherry-picks or mechanical ports from
sibling forks rebased onto mainline's architecture.

## Contributing forks

| Fork | Role in yggdrasil | Activity |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | **Base** — yggdrasil rebases against mainline regularly | upstream-of-everything |
| [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | TurboQuant KV cache (`TURBOQ{2,3,4}_0`), WHT weight quants, alpha-scaling, asymmetric K/V, InnerQ calibrated KV (`TURBOQ{2,3,4}_INNERQ`) | active |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | TCQ KV cache (`TURBOQ{2,3}_TCQ`), PFlash prompt compression | active (DFlash paused — see beellama) |
| [carlosfundora/llama.cpp-1-bit-turbo](https://github.com/carlosfundora/llama.cpp-1-bit-turbo) | RotorQuant KV V-cache (`RQ_*`), PrismML 1-bit (`Q1_0_G128`), EAGLE3, PHANTOM-X, TurboMind allocator, Wave32 RDNA2 kernels | active |
| [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) | RaBitQ TQ3 weight quants (`RBQ3_*`); MTP research | recent |
| [domvox/llama.cpp-turboquant-hip](https://github.com/domvox/llama.cpp-turboquant-hip) | TriAttention KV compression with GPU scoring, `--hugepages` | moderate |
| [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | IK quants (IQ\*_K, IQ\*_KS), BitNet, MLA / FlashMLA, fused MoE, ongoing MTP improvements | very active; **not a git merge source** — see [docs/IK_LLAMA_PORTS.md](docs/IK_LLAMA_PORTS.md) |
| [Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp) | DFlash spec-decode — monitoring only; hardened implementation + DDTree algorithm; revival candidate when drafter GGUF sourcing resolves | active; monitoring |

Forks deliberately excluded:

- **groxaxo/llama.cpp-tq3** — stale mirror of turbo-tan with no novel commits.
- **domvox's TurboQuant KV / HIP work** — superseded by TheTom catching up on HIP. Only domvox's triattention branch is tracked.

## Architecture: layer stack

Features land as discrete layers, each on its own topic branch. Each
layer follows a **two-track cadence**: ROCm-lands first (gating: PPL
regression on gfx1150), then Vulkan-lands as a follow-up sub-layer. A
feature is **released** only when both backends are on trunk and
cross-backend PPL matches within tolerance. See
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md) for the parity policy.

| Layer | Content | Sources | Status |
|---|---|---|---|
| 0 | Type-ID contract + PPL regression harness (dual-backend) | this project | **complete** |
| 0.5 | ik_llama architectural recon + EAGLE3 recon | this project | **complete** |
| 0.7 | Sidecar plugin engine (~355 LoC, backend-agnostic) — runtime adapters at residual-stream / MoE-expert / post-logits / weight-delta hook points; out-of-tree `.so` plugins | this project | **complete** |
| 1 | TurboQuant KV foundation (TURBOQ2/3/4_0 + WHT3/4_0 + layer-adaptive + Boundary V) | TheTom `feature/turboquant-kv-cache` | **complete (milestone `phase-1-turboquant-kv-foundation`)** |
| 2 | MTP spec-decode — mainline-aligned driver layer; internal Qwen3.5/MoE NextN-tail MTP; foreign-KV Gemma 4 external-assistant MTP | mainline PR #22673 + mainline `#22738` (gemma4-assistant) | **complete (milestone `phase-2-gemma4-mtp`)** |
| 3a | TCQ KV cache — ROCm/CUDA/HIP (`TURBOQ{2,3}_TCQ`) | buun `master` | **complete (main v291)** |
| 3c | TCQ KV cache — Vulkan (αA asymmetric pre-dequant FA path) | yggdrasil port | **complete (main v307)** |
| 3d | InnerQ KV — calibrated `TURBOQ{2,3,4}_INNERQ` types + CUDA calibration engine | TheTom calibration engine; yggdrasil port | **merged to main; RDC enabled broadly in v368 (commit 5e314b5f5) for ggml-hip and ggml-cuda; Vulkan gap documented** |
| 4a | RotorQuant V-cache — planar3/4 + iso3/4 (`RQ_*`) | carlosfundora | **in-flight — planar PPL gate passing; iso3/iso4 blocked on FA decoder (Phase 4a-1)** |
| 4 | Carlosfundora dense bundle (EAGLE3, PHANTOM-X, TurboMind allocator, Wave32 RDNA2) | carlosfundora `1-bit-turbo` | **pending (sequenced after RotorQuant completion)** |
| 5 | ik_llama subsystem backports (IK quants, BitNet, MLA, fused MoE, bf16 KV, MTP perf) | ik_llama (one subsystem at a time) | **pending** |
| 6 | RaBitQ TQ3 weight quants (`RBQ3_*`) | turbo-tan `main` | **pending** |
| 7a | DFlash spec-decode (drafter-model-based) | buun + beellama | **PAUSED — revival condition B satisfied (beellama active); drafter GGUF sourcing pending** |
| 7b | PFlash prompt compression (scorer-based KV compression) | buun SD-089-pflash | **shipped in v355 — HIP-optimized scorer (24× GPU speedup over CPU baseline)** |
| 8 | Polish (TURBO_ALPHA env-var defaults, `--hugepages`, asymmetric KV pair matrix completion) | mixed | **pending** |
| 9 | TriAttention KV compression with GPU scoring | domvox `feature/triattention-scoring` | **deferred post-Phase-8; halted on GGML backend bug** |

Each layer's Vulkan port is scheduled per its priority in
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). No upstream fork has
Vulkan implementations for novel features, so yggdrasil bears the Vulkan
port burden in-house.
## What's available now

As of **v355 (`4260989e8`)**, the following features are on `main`.

---

### TurboQuant KV cache types (`TURBOQ{2,3,4}_0`) — Phase 1

Calibration-free KV compression. Pass to `--cache-type-k` / `--cache-type-v`
on any GGUF whose `head_dim` is a multiple of 128. The KV cache is quantized
at runtime via `SET_ROWS`; model weights are unchanged.

| Type | Bits | Block | Compression vs fp16 | Notes |
|---|---|---|---|---|
| `turboq2` (slot 60) | 2.125 | 128 | ~7.5× | 4-centroid PolarQuant, no QJL |
| `turboq3` (slot 61) | 3.125 | 128 | ~5.1× | 2-bit PolarQuant + 1-bit QJL signs |
| `turboq4` (slot 62) | 4.25 | 128 | ~3.8× | 4-bit PolarQuant |

Example:
```bash
llama-cli --no-mmap -fa on \
    -m Qwen3.5-9B-Q4_K_M.gguf \
    --cache-type-k turboq3 --cache-type-v turboq3 \
    -c 4096 -ngl 99
```

PPL gates (Qwen3.5-9B-BF16, 32 chunks, c=512, wikitext-2-raw-test;
**legacy methodology** — current PPL harness uses c=4096, see [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md)):

| KV type | ROCm PPL | Vulkan PPL | Cross-backend Δ | vs F16 KV 6.8168 |
|---|---|---|---|---|
| `turboq2` | 7.8041 | 7.8059 | +0.023% | +14.5% |
| `turboq3` | 7.5939 | 7.6065 | +0.17% | +11.4% |

**Layer-adaptive KV precision** (optional). Set
`TURBO_LAYER_ADAPTIVE=N` to use higher-precision KV at boundary layers:
- `1` = q8_0 K+V for first-4 + last-4 layers, turbo elsewhere
- `2` = q8_0 K+V for last-8 layers, turbo elsewhere
- `5` = V=turboq4 at first-2+last-2 layers, V=turboq2 elsewhere (K unchanged)
- `6` = V=turboq4 at last-8 layers, V=turboq2 elsewhere (K unchanged)
- `7` = **Boundary V (recommended)**: V=q8_0 at first-2+last-2 layers,
  V=turboq2 elsewhere (K unchanged). Recovers ~1.2% PPL over pure turboq2.

Default is off (uniform precision); each non-zero mode is an explicit opt-in.

---

### TCQ KV cache types (`TURBOQ{2,3}_TCQ`) — Phase 3

Trellis Coded Quantization KV cache. Same CLI interface as TurboQuant_0 types;
higher per-bit quality from the Viterbi-coded codebook at the cost of a slower
encode step.

| Type | Bits | Block | Compression vs fp16 | Notes |
|---|---|---|---|---|
| `turboq2_tcq` (slot 66) | 2.125 | 128 | ~7.5× | TCQ2 Viterbi codebook |
| `turboq3_tcq` (slot 67) | 3.125 | 128 | ~5.1× | TCQ3 Viterbi codebook |

Example:
```bash
llama-cli --no-mmap -fa on \
    -m Qwen3.5-9B-Q4_K_M.gguf \
    --cache-type-k turboq2_tcq --cache-type-v turboq2_tcq \
    -c 4096 -ngl 99
```

PPL anchors (Qwen3.5-9B-Q4_K_M, n_seq=1, c=4096, wikitext-2-raw-test):

| KV type | ROCm PPL | vs F16 KV 6.49 | Vulkan notes |
|---|---|---|---|
| F16 (baseline) | 6.49 | — | Vulkan F16 ≈ 6.55 |
| `turboq2_tcq` | 6.53 ± 0.079 | +0.6% | Asymmetric K=TCQ2, V=F16 on Vulkan (RADV constraint) |
| `turboq3_tcq` | 6.90 ± 0.053 | +6.3% | Asymmetric K=TCQ3, V=F16 on Vulkan (RADV constraint) |

**Vulkan note:** The FA uber-shader on RADV PHOENIX (gfx1103) faults when
both K and V are TCQ-typed. The αA fix (Phase 3c) works around this by
pre-dequantizing V to F16 before FA dispatch, making Vulkan TCQ asymmetric
(`K=TCQ, V=F16`) by design. ROCm uses the symmetric `K=V=TCQ` path.

---

### InnerQ KV types (`TURBOQ{2,3,4}_INNERQ`) — Phase 3d

Calibrated KV quantization types. Unlike TurboQuant_0 and TCQ (which run on
any model), InnerQ uses per-model calibration data collected by a CUDA
calibration engine (`d_innerq_*` kernels from TheTom). Calibration is
one-time and stored alongside the GGUF.

| Type | Bits | Block | Notes |
|---|---|---|---|
| `turboq2_innerq` (slot 68) | 2.125 | 128 | Calibrated 2-bit |
| `turboq3_innerq` (slot 69) | 3.125 | 128 | Calibrated 3-bit |
| `turboq4_innerq` (slot 70) | 4.25 | 128 | Calibrated 4-bit |

**Backend support:** CUDA/HIP type traits, calibration engine, and FA-vec
dispatch are on `main`. RDC enabled broadly in v368 (commit 5e314b5f5) for
ggml-hip and ggml-cuda. Vulkan support is not yet implemented (gap documented).

---

### WHT-rotated weight quants — Phase 1

Weight quantization types requiring re-quantization and an imatrix.

| Type | Bits/value | Block | Backends | Notes |
|---|---|---|---|---|
| `WHT3_0` (slot 80) | ~3 | 32 | CPU + CUDA/HIP + Vulkan | |
| `WHT4_0` (slot 81) | ~4 | 32 | CPU + CUDA/HIP + Vulkan | 5.18 BPW; PPL beats `Q4_K_M` by ~1% at slightly higher BPW |

Example (Qwen3.5-9B-F16 → WHT4_0):
```bash
llama-imatrix -m Qwen3.5-9B-F16.gguf -f calibration.txt -o imatrix.dat

llama-quantize --imatrix imatrix.dat \
    Qwen3.5-9B-F16.gguf Qwen3.5-9B-WHT4_0.gguf WHT4_0
```

PPL gate (Qwen3.5-9B-WHT4_0, 32 chunks, c=512, wikitext-2-raw-test;
**legacy methodology**):

| Backend | PPL | vs F16 6.8168 | vs Q4_K_M 7.6278 (4.5 BPW) |
|---|---|---|---|
| ROCm | 7.5563 | +10.85% | **-0.94%** at 5.18 BPW |
| Vulkan | 7.5520 | +10.79% | — |

---

### Asymmetric KV cache

All types above support asymmetric K/V assignments — K and V can be different
types. This is useful to trade off quality vs compression on a per-cache-half
basis:

```bash
# K=turboq2_tcq (aggressive compression), V=turboq3_0 (higher quality)
llama-cli --no-mmap -fa on -m model.gguf \
    --cache-type-k turboq2_tcq --cache-type-v turboq3 -c 4096 -ngl 99
```

Asymmetric combinations are supported across the TURBOQ2/3/4_0, TURBOQ2/3_TCQ,
and F16/Q8_0 types. InnerQ asymmetric pairs are planned as part of Phase 8
(asymmetric KV pair matrix completion).

---

### Sidecar plugin engine — Phase 0.7

A backend-agnostic plugin runtime (~355 LoC) for hooking the forward graph
at residual-stream / MoE-expert / post-logits sites + weight deltas, via
out-of-tree `.so` plugins. Released alongside Phase 0.7; six companion
plugin tools are tracked separately. See `src/llama-sidecar.cpp` and the
plugin-engine commit `f99ad5df8`.

---

### MTP speculative decoding — Phase 2

Multi-token-prediction speculative decoding, aligned with the mainline
implementation (PR #22673). Two model families are supported.

**Internal NextN-tail MTP** — for Qwen3.5 / Qwen3.5-MoE GGUFs that carry
`nextn_predict_layers` MTP-tail blocks:

```bash
llama-server -m Qwen3.5-4B-MTP-BF16.gguf \
    --mtp --spec-type mtp --parallel 1 --no-mmap -fa on -ngl 999 -c 4096
```

**External-assistant MTP** — for the Gemma 4 family, whose drafter is a
separate "assistant" GGUF (foreign-KV, Q-only transformer that borrows the
backbone's K/V):

```bash
llama-server -m Gemma4-26B-A4B-it-IQ4_XS.gguf \
    -md Gemma4-26B-A4B-it-assistant-BF16.gguf \
    --spec-type mtp --parallel 1 --no-mmap -fa on -ngl 999 -ngld 999 -c 4096
```

Smoke-verified draft acceptance: ~89% on Qwen3.5-4B-MTP (internal), 85–89%
on Gemma 4 26B-A4B (external; ROCm + Vulkan). MTP changes the decode path,
not the output distribution, so there is no PPL gate — correctness is
verified by output coherence plus accept rate.

**CLI binaries:** Speculative MTP decoding requires an MTP-aware binary.
- `llama-server --spec-type mtp` (shown above) triggers MTP speculative.
- `llama-speculative-simple --mtp` (simple-speculative loop) also works for internal MTP.
- **`llama-cli --mtp` alone does NOT trigger speculative decoding** — the `--mtp` flag on `llama-cli` loads the MTP model but uses standard autoregressive generation. For ~2× speedup via draft acceptance, use `llama-speculative-simple --mtp` or `llama-server --spec-type mtp`.

**Divergence note:** Gemma 4 external-assistant MTP (`_external` context type,
666 LoC) has no mainline equivalent and is kept as a deliberate yggdrasil
divergence per `conventions/port-fidelity-to-mainline-llamacpp.md §D1`.

---

### Novel model architectures — in-tree ports

In addition to all mainline-supported architectures (inherited via upstream
sync), yggdrasil ships in-tree ports for novel hybrid architectures that
mainline does not yet recognize.

**Zyphra ZAYA1-8B** (`LLM_ARCH_ZAYA`) — 8.4B-param (760M active) hybrid MoE
with 80 layers alternating CCA (Mamba-cached convolutional attention) and
16-expert top-1 MoE, plus a depth-recurrent router state averaging (EDA)
second hidden stream, mixture-of-depths (MoD) skip routing, and per-layer
learned residual scaling. Gemma-family tokenizer (262 144 vocab), 131K
context, partial-RoPE 0.5, GQA 8/2. Runs end-to-end under default-flag
`llama-perplexity` / `llama-server` (both single-seq and multi-seq paths
validated). 3 shipping quants:

```bash
python3 convert_hf_to_gguf.py Zyphra/ZAYA1-8B \
    --outfile zaya1-8B-F16.gguf --outtype f16

llama-quantize --imatrix imatrix.dat --override-tensor zaya1-overrides.txt \
    zaya1-8B-F16.gguf zaya1-8B-IQ4_XS-imat-guq5k.gguf IQ4_XS
```

PPL gates (80 chunks, c=512, wikitext-2-raw-test, multi-seq `-np 4`):

| Quant | Bits | Multi-seq PPL | vs F16 30.5270 |
|---|---|---|---|
| F16 | 16 | 30.5270 | — |
| Q8_0 | 8.5 | 30.5231 | -0.01% |
| Q5_K_M | 5.5 | 29.9468 | -1.9% (in-noise) |
| IQ4_XS-imat-guq5k | 4.25 | 32.0073 | +4.9% |

See [docs/zaya1.md](docs/zaya1.md) for converter details, the
override-tensor list, multi-seq fix history, and the latent `ggml_conv_1d`
N>1 reshape workaround.

---

### Build flags

All shipped features are built unconditionally as part of the standard cmake
recipe; no new feature-gate flags are required. See [README.upstream.md](README.upstream.md)
for the unchanged mainline build instructions.

**InnerQ calibration:** RDC enabled broadly in v368 (commit 5e314b5f5) for
ggml-hip and ggml-cuda. The CUDA/HIP separable compilation flag
(`-fgpu-rdc` / `CUDA_SEPARABLE_COMPILATION`) is on by default; no manual
RDC build is required.

## In-flight workstreams

Active feature branches with work in progress; not yet merged to `main`.

| Workstream | Branch | Status |
|---|---|---|
| RotorQuant KV V-cache (planar3/4, iso3/iso4) | `feature/rotorquant-port-with-kv-alloc-fix` | planar4 PPL gate PASS; planar3 LIKELY PASS; iso3/iso4 blocked on FA decoder (Phase 4a-1 gap) |
| PFlash prompt compression — real scorer (1b) | pending | Quality validation and real scorer implementation pending |
| InnerQ RDC separable compilation | merged to main v368 (5e314b5f5) | **SHIPPED** |
| Mainline bug-fix cherry-pick batch | `feature/mainline-bugfixes-2026-05-18-am` | 2 new cherry-picks clean; 2 pending conflict triage |
| MTP migration phase G verify | `feature/mtp-migration-phase-g-verify` | verification run in progress |
| Vulkan BF16 FA allowlist | merged to main v324 | **SHIPPED** |
| KV alloc CPU fallback for types without GPU SET_ROWS | merged to main v324 | **SHIPPED** |

## Blocked / awaiting decision

| Item | Blocked on |
|---|---|
| DFlash spec-decode revival | Drafter GGUF sourcing (no Qwen3.5-9B DFlash drafter available from z-lab or community); beellama Criterion B satisfied, architecture verified |
| RotorQuant iso3/iso4 | Phase 4a-1 FA decoder gap — iso types currently fall to all-CPU (164s/pass); fix required before quality gate is meaningful |
| PFlash 1b (real scorer) | Quality validation smoke on existing 1a branch; user decision on 1b scope |
| Asymmetric KV pair matrix completion | Phase 8 planning; gap analysis approved |
| PolarQuant v2 evaluation | arXiv 2603.29078 withdrawn 2026-04-20 for errors; awaiting v2 repost or independent audit |

## Backend support

| Backend | Primary targets | Status |
|---|---|---|
| **ROCm** | gfx1150 (mandatory); gfx1102 / gfx1103 (regression-smoke target via single-target `-DAMDGPU_TARGETS=gfx1102` build + `HSA_OVERRIDE_GFX_VERSION=11.0.2` at runtime) | first-class on gfx1150; smoke-only on gfx1102/1103 |
| **Vulkan** | RDNA3 / RDNA3.5 (and broader — driver-portable) | first-class — high priority |
| CUDA, Metal, etc. | inherited from mainline | best-effort, not gated |

**Why these specific targets:** active development targets are gfx1150 and
gfx1103 (built single-target as gfx1102, run with `HSA_OVERRIDE_GFX_VERSION=11.0.2`
at runtime). Without hardware to measure perf, catch regressions, and sign off
on correctness, AMD targets outside this set are not actively supported.
**gfx1030, gfx900, gfx94X, gfx12XX, and other AMD GPUs are explicitly out of
scope** for active development; sibling-fork features targeting those GPUs are
SKIP-class by default.

Vulkan support is first-class because it's the cross-vendor path that lets
yggdrasil-novel work (TurboQuant KV, TCQ KV, sidecars, etc.) reach users on
hardware we don't own; the Vulkan port effort is a yggdrasil-owned burden for
each in-tree feature regardless of which fork it came from.

gfx1102/1103 ROCm is used as a regression-smoke target (catches HIP-shim
breakage early; cross-arch PPL parity is validated against gfx1150 ROCm builds).
Production-inference calibration on those hosts still defers to Vulkan due
to AMD upstream Tensile/hipBLAS GEMM gaps. See
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md).

## Key documents

- [**CHANGELOG.md**](CHANGELOG.md) — milestone-tagged change history (Phases 0,
  0.7, 1, 2, 3 to date).
- [**docs/TYPE_ASSIGNMENTS.md**](docs/TYPE_ASSIGNMENTS.md) — authoritative
  GGUF type-ID contract. Every cherry-pick renumbers to match. Resolves
  the five-fork collision space.
- [**docs/OP_ASSIGNMENTS.md**](docs/OP_ASSIGNMENTS.md) — yggdrasil-original
  `GGML_OP_*` registry (currently: `GGML_OP_TURBO_WHT`).
- [**docs/BACKEND_PARITY.md**](docs/BACKEND_PARITY.md) — ROCm/Vulkan
  parity policy, per-feature backend status, Vulkan port priorities,
  gfx1102/1103 partial-scope (smoke target) recipe, current PPL
  measurement methodology (c=4096, n_seq=1).
- [**docs/IK_LLAMA_PORTS.md**](docs/IK_LLAMA_PORTS.md) — subsystem tracker
  for ik_llama backports (not a git remote).
- [**docs/gemma4-assistant.md**](docs/gemma4-assistant.md) — Gemma 4 MTP
  assistant arch (`gemma4-assistant`): GGUF format, conversion, tensor schema.
- [**docs/zaya1.md**](docs/zaya1.md) — Zyphra ZAYA1-8B arch (`LLM_ARCH_ZAYA`):
  CCA / EDA / MoD architecture, conversion, tensor schema, quant overrides,
  multi-seq fix history.
- [**README.upstream.md**](README.upstream.md) — preserved mainline llama.cpp
  README for reference on build/usage docs that aren't yggdrasil-specific.

## Build / usage

Yggdrasil follows mainline's build system unchanged. All shipped features
(TurboQuant KV + TCQ KV + InnerQ KV + WHT weight quants + sidecar engine)
are built unconditionally — no new feature-gate flags. See
[README.upstream.md](README.upstream.md) and the upstream `docs/`
directory for build instructions.

For usage of the new types, see [What's available now](#whats-available-now)
above. For change history, see [CHANGELOG.md](CHANGELOG.md).

## Companion projects

The following projects are companion tools for yggdrasil, updated in lockstep
with every yggdrasil rebuild. They are designed to work with **any** llama.cpp
fork and contain no yggdrasil-specific type names or conditionals.

- **sidecar-abliteration, sidecar-control-vector, sidecar-logit-bias,
  sidecar-weight-delta** — out-of-tree `.so` plugins for the sidecar
  engine.
- **llama-quantize-cost** — quantization cost estimator. Installed as part
  of the main cmake build via `tools/quantize-cost` symlink.
- **prismaquant-llama** — Python-based prequantization pipeline with
  incremental probe, AWQ calibration, and GPTQ support. Invokes the
  installed `llama-quantize` binary; not included in the cmake install.

## Project shape

- Single long-lived downstream fork.
- Mainline sync cadence: every 2 weeks (target). Current merge base:
  `5d44db600` = mainline tag `b9133` (2026-05-13); rebase planned
  ~2026-05-24 to close ~80 commits of upstream drift.
- Trunk: `main` (HEAD `4260989e8` v355).
- Milestone tags on origin: `milestone/phase-0-foundation-complete`,
  `milestone/phase-0.7-sidecar-engine`,
  `milestone/phase-1-turboquant-kv-foundation`,
  `milestone/phase-2-mtp-foothold`,
  `milestone/phase-2-gemma4-mtp`.
- Feature work happens on `feature/<phase>-<scope>` topic branches and
  FF-merges back to `main` once all gates pass. See
  [conventions/git-workflow.md](conventions/git-workflow.md) in
  yggdrasil-context for the detailed workflow.
- ik_llama work is tracked subsystem-by-subsystem rather than as branches,
  because ik_llama's history is unrelated to mainline's. Cherry-pick
  individual commits or re-implement, never bulk-merge.

## Why yggdrasil (vs. picking one fork as base)

Mainline as base is the right choice for six of seven contributing forks
because their histories are GitHub-forks of mainline and their work
expresses as cherry-pickable topic branches. The seventh, ik_llama, has
independent history — porting subsystem-by-subsystem from it onto mainline
is a multi-month effort, but choosing ik_llama as base would orphan the
mainline-side improvements that arrive every week.

The trade-off: yggdrasil pays an ongoing ik_llama-port cost forever, in
exchange for staying mainline-current forever. The alternative (forking
ik_llama and pulling mainline in) would pay a giant one-time mainline
rebase cost upfront, then a forever cost of fighting ik_llama's
independent direction with mainline's.

The single-author velocity of mainline + ik_llama combined is too high to
choose either side as base and expect the other's improvements to arrive
cheaply. Yggdrasil's answer is to accept both as ongoing inputs.

## Attribution

yggdrasil is built on top of the [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) project (MIT) and incorporates work from several sibling forks. The yggdrasil-context conventions document the project's lift discipline. Sibling forks credited:

### Direct lifts (substantial code or design imported)

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — base mainline; rebased forward on a ~2-week cadence
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — TurboQuant KV cache quantization (Phase 1) + InnerQ calibrated KV types (Phase 3d) + WHT weight quants
- **[spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)** — TCQ KV cache types (Phase 3a, 3c) + PFlash prompt compression (Phase 7b, in-flight)
- **[carlosfundora/llama.cpp-1-bit-turbo](https://github.com/carlosfundora/llama.cpp-1-bit-turbo)** — RotorQuant KV V-cache variants (Phase 4, in-flight); EAGLE3, PHANTOM-X, TurboMind allocator
- **[Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)** — DFlash spec-decode hardening (Phase 7a, reference + monitoring, currently paused)
- **[turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)** — RaBitQ TQ3 weight quants (Phase 6, pending)
- **[domvox/llama.cpp-turboquant-hip](https://github.com/domvox/llama.cpp-turboquant-hip)** — TriAttention KV compression (Phase 9, pending post-Phase-8)
- **[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — IK quants, BitNet, MLA, fused MoE (Phase 5, subsystem-by-subsystem pending); ongoing MTP improvements
- **[Zyphra/transformers](https://github.com/Zyphra/transformers)** (zaya1 branch) — ZAYA1-8B model architecture (Phase 0, in-tree port)

### Inspiration / planned lifts

- **[Luce-Org/llama.cpp-dflash-ggml](https://github.com/Luce-Org/llama.cpp-dflash-ggml)** — FP64 RoPE theta precision fix + GGML_OP_FLASH_ATTN_SPARSE op (lift pending, user-approved 2026-05-18 PM)
- **[z-lab/dflash](https://github.com/z-lab/dflash)** — DFlash drafter training recipe reference

Per project policy, yggdrasil does NOT propose AI-generated contributions to mainline llama.cpp or any sibling forks. All yggdrasil ports and experiments remain in this repository.

---

## Contributing

This is currently a personal project. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the current PR / issue posture (TL;DR: the owner can discuss intent
but can't independently review code; please cite upstream
`ggml-org/llama.cpp` for everything not introduced by yggdrasil layers).
