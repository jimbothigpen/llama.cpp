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

**Status:** Phases 0, 0.5, 0.7, **and 1 COMPLETE** (as of milestone
[`phase-1-turboquant-kv-foundation`](../../releases/tag/milestone%2Fphase-1-turboquant-kv-foundation)).
Tree currently delivers the TurboQuant KV cache foundation (2-/3-/4-bit
PolarQuant KV types + WHT-rotated weight quants) on ROCm + Vulkan, plus
the sidecar plugin engine from Phase 0.7. Phase 2 (MTP spec-decode spine)
is the next entry. See [What's available now](#whats-available-now) for
usage.

## What yggdrasil is and isn't

**Is:** a long-lived downstream fork of mainline llama.cpp, syncing with
upstream on a regular cadence, layering vetted work from five contributing
forks plus selective backports from ik_llama.

**Isn't:** a patch-set distribution, a temporary branch, or a competitor to
mainline. Yggdrasil exists to consolidate features that mainline doesn't
yet absorb but that the community has already implemented in disparate
forks.

## Contributing forks

| Fork | Role in yggdrasil | Activity |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | **Base** — yggdrasil rebases against mainline regularly | upstream-of-everything |
| [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | TurboQuant KV cache (`TURBOQ{2,3,4}_0`), WHT weight quants, alpha-scaling, asymmetric K/V, Metal TurboFlash, Vulkan SET_ROWS | active |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | TCQ KV cache (`TURBOQ{2,3}_TCQ`), DFlash drafter, PFlash prompt compression | active |
| [carlosfundora/llama.cpp-1-bit-turbo](https://github.com/carlosfundora/llama.cpp-1-bit-turbo) | PrismML 1-bit (`Q1_0_G128`), RotorQuant KV (`RQ_*`), EAGLE3, PHANTOM-X, TurboMind allocator, Wave32 RDNA2 kernels | active |
| [turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3) | RaBitQ TQ3 weight quants (`RBQ3_*`), ik_llama MTP ported to mainline | recent |
| [domvox/llama.cpp-turboquant-hip](https://github.com/domvox/llama.cpp-turboquant-hip) | TriAttention KV compression with GPU scoring, `--hugepages` | moderate |
| [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) | IK quants (IQ\*_K, IQ\*_KS), BitNet, MLA / FlashMLA, fused MoE, ongoing MTP improvements | very active; **not a git merge source** — see [docs/IK_LLAMA_PORTS.md](docs/IK_LLAMA_PORTS.md) |

Forks deliberately excluded:

- **groxaxo/llama.cpp-tq3** — stale mirror of turbo-tan with no novel commits.
- **domvox's TurboQuant KV / HIP work** — superseded by TheTom catching up on HIP. Only domvox's triattention branch is pulled in.

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
| 1 | TurboQuant KV foundation (TURBOQ2/3/4_0 + WHT3/4_0 + Boundary V) | TheTom `feature/turboquant-kv-cache` | **complete (milestone tag `phase-1-turboquant-kv-foundation`)** |
| 2 | MTP spec-decode spine | turbo-tan `experiment/gemma4-mtp-upstream-pr` | pending |
| 3 | TCQ KV | buun `master` | pending |
| 4 | TriAttention | domvox `feature/triattention-scoring` | pending |
| 5 | Carlosfundora bundle (1-bit, RotorQuant, EAGLE3, PHANTOM-X, TurboMind, Wave32) | carlosfundora `1-bit-turbo` | pending |
| 6 | ik_llama subsystem backports (IK quants, MLA, BitNet, fused MoE, MTP perf) | ik_llama (one subsystem at a time) | pending |
| 7 | RaBitQ TQ3 weight quants | turbo-tan `main` | pending |
| 8 | DFlash/PFlash as spec-decode strategies | buun SD-* branches | pending |
| 9 | Polish (TURBO_ALPHA env-var defaults, --hugepages, gfx1030 norm) | mixed | pending |

Each layer's Vulkan port is scheduled per its priority in
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). No upstream fork has
Vulkan implementations for novel features, so yggdrasil bears the Vulkan
port burden in-house.

## What's available now

As of milestone `phase-1-turboquant-kv-foundation`, the following types
are usable on both ROCm and Vulkan (gfx1150 first-class; gfx1102/1103
Vulkan supported, ROCm regression-smoke only).

### TurboQuant KV cache types (no model re-quantization required)

These are KV cache types — pass them to `--cache-type-k` / `--cache-type-v`
on any existing GGUF model whose `head_dim` is a multiple of 128. The KV
cache is quantized at runtime via `SET_ROWS`; the weights of the model
stay whatever quantization the GGUF was built with.

| Type | Bits | Block | Compression vs fp16 KV | Notes |
|---|---|---|---|---|
| `turboq2` (`GGML_TYPE_TURBOQ2_0`, slot 60) | 2.125 | 128 (one block per rotation group) | ~7.5× | 4-centroid PolarQuant, no QJL |
| `turboq3` (`GGML_TYPE_TURBOQ3_0`, slot 61) | 3.125 | 128 | ~5.1× | 2-bit PolarQuant + 1-bit QJL signs |
| `turboq4` (`GGML_TYPE_TURBOQ4_0`, slot 62) | 4.25 | 128 | ~3.8× | 4-bit PolarQuant (default mode) |

Example:
```bash
llama-perplexity --no-mmap -fa on \
    -m Qwen3.5-9B-BF16.gguf \
    -f wikitext-2-raw-test.txt \
    --cache-type-k turboq3 --cache-type-v turboq3 \
    -c 512 --chunks 32 -ngl 99
```

PPL gates on Qwen3.5-9B-BF16 (32 chunks, c=512, wikitext-2-raw-test):

| KV type | ROCm PPL | Vulkan PPL | Cross-backend Δ | vs F16 KV baseline 6.8168 |
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

### WHT-rotated weight quants (requires re-quantization)

These are weight quantization types — re-quantize your model with
`llama-quantize` to get a smaller GGUF. **An imatrix is required** for
these types (ADR-016); see the `--imatrix` flag in `llama-imatrix` and
`llama-quantize`.

| Type | Bits/value | Block | Backends | Notes |
|---|---|---|---|---|
| `WHT3_0` (slot 80) | ~3 | 32 | CPU + CUDA/HIP | Vulkan port deferred (no TQ3_1S shaders in upstream sources; tracked as a yggdrasil-original future port) |
| `WHT4_0` (slot 81) | ~4 | 32 | CPU + CUDA/HIP + Vulkan | 5.18 BPW; PPL beats `Q4_K_M` by ~1% at slightly higher BPW |

Example (Qwen3.5-9B-F16 → WHT4_0):
```bash
# 1. Compute imatrix on a calibration corpus
llama-imatrix -m Qwen3.5-9B-F16.gguf -f calibration.txt -o imatrix.dat

# 2. Quantize with imatrix
llama-quantize --imatrix imatrix.dat \
    Qwen3.5-9B-F16.gguf Qwen3.5-9B-WHT4_0.gguf WHT4_0
```

PPL gate (Qwen3.5-9B-WHT4_0, 32 chunks, c=512, wikitext-2-raw-test):

| Backend | PPL | vs F16 weights 6.8168 | vs Q4_K_M 7.6278 (4.5 BPW) |
|---|---|---|---|
| ROCm | 7.5563 | +10.85% | **-0.94%** at 5.18 BPW |
| Vulkan | 7.5520 | +10.79% | — |

Cross-backend Δ +0.057% — well within the 0.5% release gate.

### Sidecar plugin engine (Phase 0.7)

A backend-agnostic plugin runtime (~355 LoC) for hooking the forward graph
at residual-stream / MoE-expert / post-logits sites + weight deltas, via
out-of-tree `.so` plugins. Released alongside Phase 0.7; six companion
plugin tools are tracked separately. See `src/llama-sidecar.cpp` and the
plugin-engine commit `f99ad5df8`.

### Build flags

Phase 1 features are built unconditionally as part of the standard cmake
recipe; no new feature-gate flags are required. See [README.upstream.md](README.upstream.md)
for the unchanged mainline build instructions.

## Backend support

| Backend | Primary targets | Status |
|---|---|---|
| **ROCm** | gfx1150 (mandatory); gfx1102 / gfx1103 (regression-smoke target via single-target `-DAMDGPU_TARGETS=gfx1102` build + `HSA_OVERRIDE_GFX_VERSION=11.0.2` at runtime) | first-class on gfx1150; smoke-only on gfx1102/1103 |
| **Vulkan** | RDNA3 / RDNA3.5 | first-class — high priority |
| CUDA, Metal, etc. | inherited from mainline | best-effort, not gated |

gfx1102/1103 ROCm is now used as a regression-smoke target (catches HIP-shim
breakage early; cross-host PPL parity is validated against ai00 gfx1150).
Production-inference calibration on those hosts still defers to Vulkan due
to AMD upstream Tensile/hipBLAS GEMM gaps. See
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md).

## Key documents

- [**CHANGELOG.md**](CHANGELOG.md) — milestone-tagged change history (Phase 0,
  0.7, 1 to date).
- [**docs/TYPE_ASSIGNMENTS.md**](docs/TYPE_ASSIGNMENTS.md) — authoritative
  GGUF type-ID contract. Every cherry-pick renumbers to match. Resolves
  the five-fork collision space.
- [**docs/OP_ASSIGNMENTS.md**](docs/OP_ASSIGNMENTS.md) — yggdrasil-original
  `GGML_OP_*` registry (currently: `GGML_OP_TURBO_WHT`).
- [**docs/BACKEND_PARITY.md**](docs/BACKEND_PARITY.md) — ROCm/Vulkan
  parity policy, per-feature backend status, Vulkan port priorities,
  gfx1102/1103 partial-scope (smoke target) recipe.
- [**docs/IK_LLAMA_PORTS.md**](docs/IK_LLAMA_PORTS.md) — subsystem tracker
  for ik_llama backports (not a git remote).
- [**README.upstream.md**](README.upstream.md) — preserved mainline llama.cpp
  README for reference on build/usage docs that aren't yggdrasil-specific.

## Build / usage

Yggdrasil follows mainline's build system unchanged. Phase 1 features
(TurboQuant KV + WHT weight quants + sidecar engine) are built
unconditionally — no new feature-gate flags. See
[README.upstream.md](README.upstream.md) and the upstream `docs/`
directory for build instructions.

For usage of the new types, see [What's available now](#whats-available-now)
above. For change history, see [CHANGELOG.md](CHANGELOG.md).

## Project shape

- Single long-lived downstream fork.
- Mainline sync cadence: every 2 weeks (target).
- Trunk: `main` (mainline-tracking; current HEAD at milestone
  `phase-1-turboquant-kv-foundation`).
- Feature work happens on `feature/<phase>-<scope>` topic branches and
  FF-merges back to `main` once all gates pass. See
  [conventions/git-workflow.md](conventions/git-workflow.md) in
  yggdrasil-context for the detailed workflow.
- Each completed phase is tagged `milestone/<phase>-...` on origin.
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

## Contributing

This is currently a personal project. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the current PR / issue posture (TL;DR: the owner can discuss intent
but can't independently review code; please cite upstream
`ggml-org/llama.cpp` for everything not introduced by yggdrasil layers).
