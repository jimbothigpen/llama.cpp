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

**Status:** Phases 0, 0.5, 0.7, 1, 2, and 3 COMPLETE — **v326 (`9ffaa0967`)** on
`main`. Phases 3a (TCQ KV ROCm/CUDA), 3c (TCQ KV Vulkan), and 3d (InnerQ KV
types) are merged to `/opt`. Phase 4a (RotorQuant KV) is in-flight; Phase 7b
(PFlash prompt compression) has a working placeholder implementation on a feature
branch. See [What's available now](#whats-available-now) and
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
| 3d | InnerQ KV — calibrated `TURBOQ{2,3,4}_INNERQ` types + CUDA calibration engine | TheTom calibration engine; yggdrasil port | **merged to main; ROCm RDC-enable pending (`feature/innerq-rdc-enable`); Vulkan gap documented** |
| 4a | RotorQuant V-cache — planar3/4 + iso3/4 (`RQ_*`) | carlosfundora | **in-flight — planar PPL gate passing; iso3/iso4 blocked on FA decoder (Phase 4a-1)** |
| 4 | Carlosfundora dense bundle (EAGLE3, PHANTOM-X, TurboMind allocator, Wave32 RDNA2) | carlosfundora `1-bit-turbo` | **pending (sequenced after RotorQuant completion)** |
| 5 | ik_llama subsystem backports (IK quants, BitNet, MLA, fused MoE, bf16 KV, MTP perf) | ik_llama (one subsystem at a time) | **pending** |
| 6 | RaBitQ TQ3 weight quants (`RBQ3_*`) | turbo-tan `main` | **pending** |
| 7a | DFlash spec-decode (drafter-model-based) | buun + beellama | **PAUSED — revival condition B satisfied (beellama active); drafter GGUF sourcing pending** |
| 7b | PFlash prompt compression (scorer-based KV compression) | buun SD-089-pflash | **in-flight — placeholder scorer mode on `feature/pflash-1a` (20× TTFT speedup); real scorer (1b) pending** |
| 8 | Polish (TURBO_ALPHA env-var defaults, `--hugepages`, asymmetric KV pair matrix completion) | mixed | **pending** |
| 9 | TriAttention KV compression with GPU scoring | domvox `feature/triattention-scoring` | **deferred post-Phase-8; halted on GGML backend bug** |

Each layer's Vulkan port is scheduled per its priority in
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). No upstream fork has
Vulkan implementations for novel features, so yggdrasil bears the Vulkan
port burden in-house.
