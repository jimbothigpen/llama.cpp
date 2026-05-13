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

**Status:** Phase 0 in progress. No feature layers landed yet. Tree currently
tracks mainline `ggml-org/llama.cpp` verbatim, with documented intent for what
lands next.

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

| Layer | Content | Sources |
|---|---|---|
| 0 | Type-ID contract + PPL regression harness (dual-backend) | this project |
| 0.5 | ik_llama architectural recon + EAGLE3 recon | this project |
| 0.7 | Sidecar plugin engine (~355 LoC, backend-agnostic) — runtime adapters at residual-stream / MoE-expert / post-logits / weight-delta hook points; out-of-tree `.so` plugins | this project |
| 1 | TurboQuant KV foundation | TheTom `feature/alpha-scaling` |
| 2 | MTP spec-decode spine | turbo-tan `experiment/gemma4-mtp-upstream-pr` |
| 3 | TCQ KV | buun `master` |
| 4 | TriAttention | domvox `feature/triattention-scoring` |
| 5 | Carlosfundora bundle (1-bit, RotorQuant, EAGLE3, PHANTOM-X, TurboMind, Wave32) | carlosfundora `1-bit-turbo` |
| 6 | ik_llama subsystem backports (IK quants, MLA, BitNet, fused MoE, MTP perf) | ik_llama (one subsystem at a time) |
| 7 | RaBitQ TQ3 weight quants | turbo-tan `main` |
| 8 | DFlash/PFlash as spec-decode strategies | buun SD-* branches |
| 9 | Polish (alpha-scaling defaults, --hugepages, gfx1030 norm) | mixed |

Each layer's Vulkan port is scheduled per its priority in
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). No upstream fork has
Vulkan implementations for novel features, so yggdrasil bears the Vulkan
port burden in-house.

## Backend support

| Backend | Primary targets | Status |
|---|---|---|
| **ROCm** | gfx1150 (mandatory); gfx1102 / gfx1103 (nice-to-have, blocked on upstream AMD) | first-class |
| **Vulkan** | RDNA3 / RDNA3.5 | first-class — high priority |
| CUDA, Metal, etc. | inherited from mainline | best-effort, not gated |

A side-quest tracks community fixes that may unblock gfx1102/gfx1103 ROCm
support; see [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md#gfx1102--gfx1103-rocm-side-quest).
Until then, RDNA3-mobile systems run Vulkan as the practical alternative.

## Key documents

- [**docs/TYPE_ASSIGNMENTS.md**](docs/TYPE_ASSIGNMENTS.md) — authoritative
  GGUF type-ID contract. Every cherry-pick renumbers to match. Resolves
  the five-fork collision space.
- [**docs/BACKEND_PARITY.md**](docs/BACKEND_PARITY.md) — ROCm/Vulkan
  parity policy, per-feature backend status, Vulkan port priorities,
  gfx1102/1103 side-quest tracker.
- [**docs/IK_LLAMA_PORTS.md**](docs/IK_LLAMA_PORTS.md) — subsystem tracker
  for ik_llama backports (not a git remote).
- [**README.upstream.md**](README.upstream.md) — preserved mainline llama.cpp
  README for reference on build/usage docs that aren't yggdrasil-specific.

## Build / usage

Yggdrasil follows mainline's build system unchanged for now (Phase 1 work
has not yet landed). Use [README.upstream.md](README.upstream.md) and the
upstream `docs/` directory for build instructions until that changes.

When new build options become required for yggdrasil-specific features
(e.g., `GGML_TURBOQUANT=ON`, `GGML_ROTORQUANT=ON`), they will be documented
here.

## Project shape

- Single long-lived downstream fork.
- Mainline sync cadence: every 2 weeks (target).
- Trunk: `master` (mainline-tracking; current).
- Feature work happens on `port/<source>/<feature>` topic branches.
- Each port lands as a tagged commit (`ported/<source>/<feature>-<date>`)
  recording the source-fork commit SHAs it captures.
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

This is currently a personal project. Contribution guidelines will follow
once Phase 1 lands and the layer-stack pattern is stable enough to onboard
others against.
