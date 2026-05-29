# llama.cpp

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
>   this fork's layers; cite this repo for the consolidation work itself.
>
> If you're here to learn how a non-developer can drive a complex
> systems-software fork end-to-end with an AI agent, you're in the right
> place. If you're here for production-ready inference code, go upstream.

A unified downstream of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)
that absorbs novel work from six sibling forks into a single coherent tree.


**Status:** Phases 0, 0.5, 0.7, 1, 2, 3, 5b-1a, 5b-1b, 5b-1c, 5b-2, 7a, 7b, MTP Migration 0-3, NLD COMPLETE, **MTP Convergence Phase A** — **HEAD `d738c4341`** on
`main` (post-mainline-rebase to `b745`). Recent ships (2026-05-29 cascade): **IQ3_KT trellis 3-bit quant** — 3-bit PPL +23.5% vs IQ3_K (inherent to single-codebook design); cluster-accel k=60 CPU/ROCm/Vulkan; imatrix required (`623835cc9`); **IK weight-quant feature docs** — base-K (IQ2/3/4_K), high-bit-K (IQ5/6_K), row-meta (IQ4_KS/IQ3_KS/IQ4_KSS/IQ2_KL) + family primer (docs/features/ik-*.md); IQ2_KL phase fix 5b-2a→5b-1c (`7ca3e0e8c`). Prior 2026-05-28 wave: **Mainline rebase b745** — 68 mainline commits integrated; FWHT dual-pipeline resolution (`cf70bbd33`, `3caf1caa0`); ZAYA/TALKIE arch slot + Q1_0_G128 Vulkan dequant conflicts resolved; PPL 6.5453 PASS (gfx1103); **domvox SWA KV** — per-layer `--cache-type-k-swa` / `--cache-type-v-swa` for hybrid SWA-models; Gemma 4 PPL 27.7k vs >100k all-turbo3 (`30472d827`); **buun-3-fixes** — tensor-split with quantized KV unblocked (`6774410fa`) + TURBO_WHT added to split planner (`340f6fe21`); **ccee426 revert shipped** — KV cache reuse regression on multi-turn Qwen3.6-35B-A3B fixed, loader-smoke TODO 147 PASS (`f92e515f2`); **MTP convert fixes** — `attn_norm.weight` emission for bundled-MTP GGUFs (`c0d71d750`, TODO 145) + `block_count`/`nextn` metadata for `--no-mtp` GGUFs (`36164e428`, TODO 146). Prior 2026-05-27 wave: **OScaR Phase 2** (`c892e62a3`); **TriAttention Phase B** (`6f93b4e5d`); **MTP Convergence Phase A** (`fd44da73f`); MTP perf + draft-simple gate + DFlash converter + FWHT op + OScaR INT2 KV + Kaggle T4 guard. Prior 2026-05-25 wave: mainline rebase `b9310` (`1191e48fc`); MTP M-RoPE fix; MTP→`draft-mtp` rename + GGML enum convergence; IQ2_KT P3a + cluster-accel; IQ2_KL + IQ5_K/IQ6_K Vulkan; EAGLE3 fc fix; DFlash S2+S3; TriAttention Phase A. See [What's available now](#whats-available-now) and
[In-flight workstreams](#in-flight-workstreams) for detail.

## What this fork is and isn't

**Is:** a long-lived downstream fork of mainline llama.cpp, syncing with
upstream on a regular cadence, layering vetted work from five contributing
forks plus selective backports from ik_llama.

**Isn't:** a patch-set distribution, a temporary branch, a competitor to
mainline, or a candidate for upstream contribution. This fork exists to
consolidate features that mainline doesn't yet absorb but that the community
has already implemented in disparate forks. Per project policy, no AI-generated
code is proposed for upstream submission.

The standing constraint is **mainline fidelity**: diverge only when measurably
better, document every deliberate divergence, and sync regularly. Most
commits are either mainline cherry-picks or mechanical ports from
sibling forks rebased onto mainline's architecture.

## Contributing forks

| Fork | Role in this fork | Activity |
|---|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | **Base** — this fork rebases against mainline regularly | upstream-of-everything |
| [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) | TurboQuant KV cache (`TURBOQ{2,3,4}_0`), WHT weight quants, alpha-scaling, asymmetric K/V, InnerQ calibrated KV (`TURBOQ{2,3,4}_INNERQ`) | active |
| [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp) | TCQ KV cache (`TURBOQ{2,3}_TCQ`), PFlash prompt compression, DFlash S1 model loader | active |
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
| 2 | MTP spec-decode — mainline-aligned driver layer; internal Qwen3.5/MoE NextN-tail MTP; foreign-KV Gemma 4 external-assistant MTP; MTP Migration 0-3 (mainline-convergent speculative driver, loader split, graph embed-norm masked) | mainline PR #22673 + mainline `#22738` (gemma4-assistant); migration to mainline b9246 arch | **complete — V-J accept-rate gap closed `705ffccb8` (38% → 70.28%); Migration phases 0-3 complete through `4a9977f49`** |
| 3a | TCQ KV cache — ROCm/CUDA/HIP (`TURBOQ{2,3}_TCQ`) | buun `master` | **complete (main v291)** |
| 3c | TCQ KV cache — Vulkan (αA asymmetric pre-dequant FA path) | this fork's port | **complete (main v307)** |
| 3d | InnerQ KV — calibrated `TURBOQ{2,3,4}_INNERQ` types + CUDA calibration engine | TheTom calibration engine; this fork's port | **merged to main; RDC enabled broadly in v368 (commit 5e314b5f5) for ggml-hip and ggml-cuda; Vulkan gap documented** |
| 4a | RotorQuant KV cache — iso3/4 + planar3/4 (`iso3`, `iso4`, `planar3`, `planar4`) | carlosfundora | **complete — 34/34 K×V pairs shipped (`88afd0b5a`); iso3-K cross-V hang (4 pairs) RESOLVED 2026-05-24 (fix landed `16c1a0012` on 2026-05-19; previously open as TODO 68; 4/4 smokes PASS exit:0)** |
| 4 | Carlosfundora dense bundle (Q1_0_G128, EAGLE3, PHANTOM-X, DFlash S1, TurboMind allocator, Wave32 RDNA2) | carlosfundora `1-bit-turbo` / buun | **Q1_0_G128 ported (`87d3705e0`); EAGLE3 ported (`c0f3c1486` + fc dtype-aware fix `4c38845c4`); PHANTOM-X ported + Phase 2 dispatch (`d6dc63224`, `388169995`); DFlash S1 model loader ported (`b6a75e524`); TurboMind allocator queued for opportunistic port (PORT-LATER); Wave32 RDNA2 out of scope (`[[supported-rocm-hardware-targets]]`)** |
| 5 | ik_llama subsystem backports (IK quants, BitNet, MLA, fused MoE, bf16 KV, MTP perf) | ik_llama (one subsystem at a time) | **5b-1a (IQ2_K/IQ3_K/IQ4_K) complete (`c12d37dbc`); 5b-1b (IQ4_KS/IQ4_KSS/IQ3_KS/IQ4_KT) complete (`63b754e84..a25ee1cf7`); 5b-1c (IQ2_KL type-157) complete CPU+CUDA/HIP+Vulkan (`f18a92a42` + `3723c1f61`); 5b-2 (IQ5_K/IQ6_K) complete CPU+CUDA/HIP+Vulkan (`8e19be061` + `0ade7ff86`); Trellis P3a IQ2_KT complete (`0dac276d9` + cluster-accel `1e8501e46`); P3b IQ3_KT complete CPU/ROCm/Vulkan, cluster-accel k=60, imatrix required (`623835cc9`, 2026-05-29); P3c IQ1_KT queued; MLA declined** |
| 6 | RaBitQ TQ3 weight quants (`RBQ3_*`) | turbo-tan `main` | **pending — imatrix retrofit required per PM-15 (~6h) before port** |
| 7a | DFlash spec-decode (drafter-model-based) | buun + beellama | **DFlash S1 loader (`b6a75e524`) + S2 `common_speculative_state_dflash` + dispatch (`ef80c728c`) + S3 GPU ring buffer + server spec_type wiring (`9b7ab4e83`) + mask_token_id u32 fix (`1436d1890`) + DFlashDraftModel safetensors→GGUF converter (`ee7d4f896`) all shipped; end-to-end smoke gate GREEN @ `2726a56c0` with 33.3% combined accept (dual-spec mode); isolated DFlash-only accept unmeasured (known-limitation)** |
| 7b | PFlash prompt compression (scorer-based KV compression) | buun SD-089-pflash | **base shipped in v355 — HIP-optimized scorer (24× GPU speedup over CPU baseline); 4b bulk-upload shipped in v365; 4c LRU scorer cache shipped (`38d6b7dea`); NEW-D Vulkan GPU scorer fix shipped (`276508aaa`) — IGPU fallback enables ~0.20s GPU scoring on Strix Halo Vulkan (was 3-5s CPU fallback); NEW-E on-disk persistence (`6930e37e9`)** |
| 8 | Polish (TURBO_ALPHA env-var defaults, `--hugepages`, asymmetric KV pair matrix completion) | mixed | **partial — asymmetric KV production pairs all shipped (X-2/X-3-s1/X-InnerQ-s1/s2/s3/xrq-wave2); TURBO_ALPHA / hugepages / gfx1030-norm pending; X-3-s2/s3 K-aggressive pairs deferred per policy** |
| 9 | TriAttention KV compression with GPU scoring | domvox `feature/triattention-scoring` | **REVIVED — Phase A in-graph K/V capture harness shipped (`6cbc9e06c`); HIP guard + safe null fixes (`eea5e25f5`, `2ad2564f1`); Phase B GQA CPU smoke GREEN 3/3 models (Qwen3.5-9B, Llama-3.1-8B, Gemma-4-E2B); Gemma-4 ISWA capture fix shipped (`cbd071632`); Phase C GPU GQA kernel extension queued (needs `--cache-type-k q8_0` for GPU scoring; SWA-layer capture extension for Gemma-4 also pending)** |

Each layer's Vulkan port is scheduled per its priority in
[docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). No upstream fork has
Vulkan implementations for novel features, so this fork bears the Vulkan
port burden in-house.
## What's available now

As of **HEAD `d738c4341`**, the following features are on `main`.

---

### TurboQuant KV cache types (`TURBOQ{2,3,4}_0`) — Phase 1

> **Full feature doc:** [docs/features/turboquant-kv-base.md](docs/features/turboquant-kv-base.md)

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

> **Full feature doc:** [docs/features/tcq-kv.md](docs/features/tcq-kv.md)

Trellis Coded Quantization KV cache. Same CLI interface as TurboQuant_0 types;
higher per-bit quality from the Viterbi-coded codebook at the cost of a slower
encode step (GPU required for encode).

| Type | Bits | Block | Compression vs fp16 | Notes |
|---|---|---|---|---|
| `turboq2_tcq` (slot 66) | 2.25 | 128 | ~7.1× | TCQ2 Viterbi codebook (k=2, 256 states) |
| `turboq3_tcq` (slot 67) | 3.25 | 128 | ~4.9× | TCQ3 Viterbi codebook (k=3, 512 states) |

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

### InnerQ KV types (`TURBOQ{2,3}_INNERQ`) — Phase 3d

> **Full feature doc:** [docs/features/innerq-kv.md](docs/features/innerq-kv.md)

Online-calibrated KV quantization types (CUDA/HIP only). InnerQ reuses the same
block structs as `turboq2` / `turboq3` (identical memory footprint) and adds a
per-session per-channel equalization pass that improves quality when K channels
have unequal variance. No calibrated GGUF file is needed — calibration runs
automatically during the first N tokens of inference when `TURBO_INNERQ=N` is set.

| Type | Bits | Block | Notes |
|---|---|---|---|
| `turboq2_innerq` (slot 68) | 2.125 | 128 | Per-channel equalization + 2-bit PolarQuant |
| `turboq3_innerq` (slot 69) | 3.125 | 128 | Per-channel equalization + 3-bit PolarQuant |

> **No 4-bit InnerQ.** Per-channel equalization regresses quality at 4-bit (observed PPL
> regression vs plain `turboq4`); slot 70 is retired. Use `turboq4` for 4-bit KV cache.

Example:
```bash
TURBO_INNERQ=256 llama-cli --no-mmap -fa on \
    -m Qwen3.5-9B-Q4_K_M.gguf \
    --cache-type-k turboq3_innerq --cache-type-v turboq3_innerq \
    -c 4096 -ngl 99
```

**Backend support:** CUDA/HIP only. No Vulkan encode (InnerQ types on Vulkan fall
back to plain PolarQuant without equalization). Flash attention (`-fa on`) required.

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

### RotorQuant KV cache types (`iso3`, `iso4`, `planar3`, `planar4`) — Phase 4a

1-bit quantization for K and V caches with iso (isotropic) and planar variants.

| Type | Bits | Block | Compression vs fp16 | Notes |
|---|---|---|---|---|
| `iso3` (slot 71) | 1.0 | 128 | ~16× | Isotropic 1-bit (3 codebook vectors) |
| `iso4` (slot 72) | 1.0 | 128 | ~16× | Isotropic 1-bit (4 codebook vectors) |
| `planar3` (slot 73) | 1.0 | 128 | ~16× | Planar 1-bit (3 codebook vectors) |
| `planar4` (slot 74) | 1.0 | 128 | ~16× | Planar 1-bit (4 codebook vectors) |

All 34 asymmetric K/V pairs are shipped as of `88afd0b5a`. Quality gate (PPL)
validates planar variants. The iso3-K cross-V family (4 pairs: iso3×{iso4, f16,
q8_0, planar4}) was historically gated by a HIP kernel hang on gfx1102 (TODO 68);
that fix landed in `16c1a0012` (replaces `#ifndef GGML_USE_HIP` with
`if constexpr (!V_is_turbo) { __syncthreads(); }` at `fattn-vec.cuh:424`) plus
dispatch entries in `71fe14e26` — 2026-05-24 retro-smoke verified 4/4 PASS
(exit:0, ~14.0 t/s each; PPL sanity 6.5896 ±0.16099 on 5 chunks).

Example:
```bash
llama-cli --no-mmap -fa on -m model.gguf \
    --cache-type-k iso3 --cache-type-v iso3 -c 4096 -ngl 99
```

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

**Shipped asymmetric coverage (~85+ pairs):**
- Q4_0 / Q4_1 K × TURBOQ V (X-2b-s2, `46c5dec9c`)
- F16 / BF16 / Q8_0 K × TURBOQ V (X-2a)
- TURBOQ_0 × TURBOQ_TCQ cross-family (X-2c, `305901807`)
- RotorQuant K-side (iso3/4, planar3/4) × RotorQuant V-side (34/34 pairs, `88afd0b5a`)
- InnerQ asymmetric (7 pairs, X-InnerQ-s1, shipped `42078ec1b`)
- TURBOQ/TCQ × Q4/Q5 K (10 lower-priority, X-3-s1, shipped `52b316453`)
- TURBOQ/TCQ/Q4/Q5 × INNERQ (10 HIGH-priority, X-InnerQ-s2, shipped `88afd0b5a`)
- RQ K × INNERQ V (8 pairs, X-InnerQ-s3, shipped `4684c13c8`)
- PFlash NEW-D Vulkan GPU scorer fix (IGPU fallback, `276508aaa`)

Remaining pairs (X-3-s2, X-3-s3) pending.

---

### Per-layer SWA KV cache type (`--cache-type-k-swa` / `--cache-type-v-swa`) — domvox port

For hybrid SWA (Sliding Window Attention) models such as Gemma 4, the full-attention
and SWA layers can use different KV cache types. Without this, applying turbo KV
uniformly collapses Gemma 4 PPL from 24.9k (F16 baseline) to >100k.

```bash
# Gemma 4 26B-A4B: turboq3 for full-attn layers, f16 for SWA layers
llama-cli --no-mmap -fa on \
    -m Gemma-4-26B-A4B-Q4_K_M.gguf \
    --cache-type-k turboq3 --cache-type-v turboq3 \
    --cache-type-k-swa f16 --cache-type-v-swa f16 \
    -c 8192 -ngl 99
```

PPL gate (Gemma 4 26B-A4B, Qwen-format wikitext): 27.7k vs 24.9k F16 baseline
(vs >100k all-turboq3). Ported from domvox `feature/turboquant-hip-port-clean`
commit `5c59d773f` (`d8ec65064`).

---

### ik_llama weight quants (IQ2_K, IQ3_K, IQ4_K) — Phase 5b-1a

Ported IQ-family weight quantization types from ik_llama.cpp. All three types are production-ready. These join the existing IQ*_KS types in providing a rich gradient of quality/compression tradeoffs for weight quantization.

| Type | Bits | Block | Backends | Notes |
|---|---|---|---|---|
| `IQ2_K` (slot 137) | 2.375 bpw | 256 | CPU + CUDA/HIP + Vulkan | imatrix-aware; nonlinear values + per-group scales |
| `IQ3_K` (slot 138) | 3.44 bpw | 256 | CPU + CUDA/HIP + Vulkan | imatrix-aware; nonlinear values + per-group scales |
| `IQ4_K` (slot 139) | 4.50 bpw | 256 | CPU + CUDA/HIP + Vulkan | imatrix-aware; nonlinear values + per-group scales |

Example:
```bash
# Generate imatrix first (required):
llama-imatrix -m Qwen3.5-9B-F16.gguf -f calibration-data.txt -o Qwen3.5-9B.imatrix
# Then quantize:
llama-quantize --imatrix Qwen3.5-9B.imatrix Qwen3.5-9B-F16.gguf Qwen3.5-9B-IQ3_K.gguf IQ3_K
```

**imatrix required.** These are imatrix-aware quants; without `--imatrix` the quantizer hard-errors for all tensors except `token_embd` and `output`. PPL parity Δ < 0.0045 vs ik_llama across multiple quant/model pairs. Slots are in the ik_llama compatibility zone (96–199) per [docs/TYPE_ASSIGNMENTS.md](docs/TYPE_ASSIGNMENTS.md).

**Docs:** [IK Base-K weight quants](docs/features/ik-base-k.md) · [IK quantization family primer](docs/features/concepts/ik-quantization-family.md)

---

### ik_llama row-meta weight quants (IQ4_KS, IQ4_KSS, IQ3_KS, IQ4_KT) — Phase 5b-1b

Row-meta IK-family weight quantization types requiring a `row_meta_size` infrastructure
prerequisite (`d91059253`). Vulkan batched `mul_mat` SEGV fixed via `is_empty()` guard
(`5fe804bcd`) — all 4 types work on both ROCm and Vulkan.

| Type | Slot | Block | Backends | Notes |
|---|---|---|---|---|
| `IQ4_KS` | 144 | 256 | CPU + CUDA/HIP + Vulkan | 4-bit row-meta small; float row-scale |
| `IQ4_KSS` | 146 | 256 | CPU + CUDA/HIP + Vulkan | 4-bit row-meta small-small |
| `IQ3_KS` | 156 | 256 | CPU + CUDA/HIP + Vulkan | 3-bit row-meta small; uint16_t half-row-scale |
| `IQ4_KT` | 155 | 256 | CPU + CUDA/HIP + Vulkan | 4-bit IK trellis weight quant |

PPL gate (Qwen3.5-9B, 20 chunks, c=4096, wikitext-2-raw-test):

| Type | Vulkan PPL | ROCm anchor | Δ |
|---|---|---|---|
| IQ4_KS | 6.4131 | 6.4390 | −0.026 |
| IQ3_KS | 6.7325 | 6.7488 | −0.016 |
| IQ4_KSS | 6.5773 | 6.6202 | −0.043 |
| IQ4_KT | 6.5364 | 6.5701 | −0.034 |

Slots are ik_llama compatibility zone IDs (preserved verbatim). Row-meta types store per-row scale metadata alongside the quantized block; `ggml_nbytes` must be used for buffer sizing (not `type_size × ne`).

**Docs:** [IK Row-Meta weight quants](docs/features/ik-ks-row-meta.md) · [IK quantization family primer](docs/features/concepts/ik-quantization-family.md)

---

### ik_llama extended weight quants (IQ5_K, IQ6_K) — Phase 5b-2

Ported IQ5_K and IQ6_K from ik_llama.cpp (CPU+CUDA/HIP `8e19be061`; Vulkan dequant + matvec shaders `0ade7ff86`). Higher-quality extensions of the IK-K family.

| Type | Slot | Backends | Notes |
|---|---|---|---|
| `IQ5_K` | 140 | CPU + CUDA/HIP + Vulkan | 5-bit imatrix-aware weight quant |
| `IQ6_K` | 141 | CPU + CUDA/HIP + Vulkan | 6-bit imatrix-aware weight quant |

Imatrix required per PM-15 mandate. Slots are in the ik_llama compatibility zone (96–199) per [docs/TYPE_ASSIGNMENTS.md](docs/TYPE_ASSIGNMENTS.md).

**Docs:** [IK High-Bit-K weight quants](docs/features/ik-high-bit-k.md) · [IK quantization family primer](docs/features/concepts/ik-quantization-family.md)

---

### ik_llama IQ2_KL weight quant — Phase 5b-1c

Ported IQ2_KL (ik_llama type 157, 2.6875 bpw) from ik_llama.cpp (CPU+CUDA/HIP `f18a92a42`; Vulkan dequant + matvec shaders `3723c1f61`). Ultra-low bitrate imatrix-aware weight quantization.

| Type | Slot | bpw | Backends | Notes |
|---|---|---|---|---|
| `IQ2_KL` | 157 | 2.6875 | CPU + CUDA/HIP + Vulkan | Low-bpw variant of the IK-K family |

Imatrix required.

---

### ik_llama Trellis IQ2_KT weight quant — Phase 5 Trellis P3a

Ported IQ2_KT (Trellis P3a) from ik_llama via the new `ggml-iqk-kt-family.hpp` template header (`e9520caac`) and shipped at `0dac276d9`. Cluster-acceleration via 8D base-3 hash with `k_neighbours=60` shipped at `1e8501e46` (~30× quantize speedup vs brute-force; recovers most of the IQ4_KT scaffold's accel pattern for the GROUP_SIZE=8 case).

| Type | Slot | bpw | Backends | Notes |
|---|---|---|---|---|
| `IQ2_KT` | 153 | 2.125 | CPU + CUDA/HIP | 2-bit IK trellis with 65536-entry codebook (`IQKTParams<8, 16, false>`); imatrix required |

**Known limitations:**

- PPL on Qwen3.5-0.8B is 107.87 vs IQ2_KL=26.12 and IQ4_KT=11.43 (anomaly OPEN — under investigation; scope-TBD: scale-dependent vs general). Status: ship-with-§-FLAG per `[[iq2-kt-known-issues]]`.
- Cluster acceleration overshoots the ≤+5% PPL gate (+8.3% vs brute-force baseline); k=80–100 retune is planned for late-stage polish.
- Vulkan path not yet ported.
- IQ3_KT / IQ1_KT (Trellis P3b / P3c) queued behind P3a; will use the same `iqkt_cooked_book_init` site for cluster acceleration.

---

### Q1_0_G128 (PrismML 1-bit weight quant) — Phase 4 carlosfundora

Ported `Q1_0_G128` from carlosfundora `1-bit-turbo` (`67edf5eec`). 1-bit weight quantization with 128-element groups (Bonsai-family). Placed at ygg canonical slot 96 (first slot of ik_llama compat zone) to resolve the three-way collision between ik_llama (slot 41), carlosfundora (slot 43), and mainline `Q1_0` (slot 41).

| Type | Slot | Backends | Notes |
|---|---|---|---|
| `Q1_0_G128` | 96 | CPU + CUDA/HIP | 1-bit with 128-element group; imatrix required per PM-15 |

---

### PFlash NEW-E: on-disk persistence for scorer cache — Phase 7b

The PFlash LRU scorer cache (4c) now supports on-disk persistence. Cache entries are serialized to disk and restored on subsequent runs, eliminating the penalty of cache cold-start on repeat queries. The on-disk format is opaque (versioned sidecar); format breakage triggers a cache rebuild.

Example (scorer cache persisted to `./mymodel.pf-cache`):
```bash
llama-server -m model.gguf --pf-cache mymodel.pf-cache
```

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
666 LoC) has no mainline equivalent and is kept as a deliberate divergence per
`conventions/port-fidelity-to-mainline-llamacpp.md §D1`.

**MTP Migration (phases 0-3, 2026-05-23):** The fork's MTP speculative driver, Qwen3.5/MoE
loader, and graph-builder have been migrated to align with mainline b9246 architecture:
Phase 0 (preflight recon), Phase 1 (arch constants + server loader to LLAMA_CONTEXT_TYPE_MTP),
Phase 2 (loader split into `load_block_trunk` + `load_block_mtp` lambdas),
Phase 3 (graph convergence to `embeddings_pre_norm_masked`). The fork's bundled-MTP
semantics are preserved with inverted polarity as documented in the Phase 3 brief.

**V-J accept-rate gap closed (2026-05-23, `705ffccb8`):** `llama-speculative-simple` was
missing the `common_speculative_process(spec, batch_tgt)` call after target decode —
the server already had it but the standalone binary did not. Without it, the MTP head's
`pending_h` stayed zeroed and drafts were garbage. Adding the call lifted acceptance on
Qwen3.5-35B-A3B-MTP from 38% → 70.28% (mainline anchor: 71.3%). Throughput: +45% e2e.

---

### Nemotron-Labs Diffusion (NLD) — CLI + server self-spec

Nemotron-Labs Diffusion 14B (`LLM_ARCH_DREAM`) — a masked diffusion LLM that
generates tokens by iterative block-wise refinement (fill-in-the-blank at masked
positions). Ported from buun `f339dbebe` (TODO 80: CLI Tier-B port, `49f88e18a`;
TODO 86: server self-spec loop, `1cb8c4218`).

```bash
# Block-mode generation (32-step decode)
llama-diffusion-cli -m nemotron-diffusion-14b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Write a function to compute fibonacci numbers." \
    -n 256 --diffusion-block-length 32 --diffusion-steps 32

# Self-speculative decoding (3.7× speedup, 68% draft acceptance)
llama-diffusion-cli -m nemotron-diffusion-14b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Write a python function for fibonacci." \
    -n 128 --diffusion-self-spec --diffusion-draft-length 8
```

The NLD server auto-detects diffusion models via `llama_model_is_diffusion()` and
activates the self-spec rejection-sampling loop automatically — no `--spec-type` flag.
Server self-spec measured at 4.49 t/s (128 tokens); CLI self-spec at 7.0 t/s.
MTP regression gate confirmed clean (84.6% accept on Qwen3.5-35B-A3B-MTP after NLD port).

---

### EAGLE3 hidden-state extrapolation speculative decoder — Phase 4

Ported EAGLE3 from carlosfundora `1-bit-turbo` (core port `c0f3c1486`; fc dtype-aware
read with BF16/F16→F32 conversion `4c38845c4`; post-rebase struct fixup `e109b17d8`).
EAGLE3 uses the target model's hidden-state residuals to extrapolate draft tokens
rather than running a separate full drafter model. Primary test target: Gemma 4 26B-A4B
with the paired assistant checkpoint.

Backend-agnostic — no novel GPU kernels; operates entirely within the existing
speculative-decode scheduling path. Smoke gate PASSED: EAGLE3 init OK, 6.3 t/s
generation, EXIT:0. End-to-end accept-rate gate runs as part of the full spec-decode
matrix (see [In-flight workstreams](#in-flight-workstreams)).

---

### PHANTOM-X speculative decoder — Phase 4

Ported PHANTOM-X from carlosfundora `1-bit-turbo` (`2199e8445`; Phase 2 factory
dispatch adapter `4fd52ddc0`). PHANTOM-X uses a learned per-model n-gram pattern
lookup for draft generation. The Phase 2 adapter wires it into the `--spec-type`
factory dispatch so it can be selected alongside other spec-decode mechanisms.

Backend-agnostic — CPU n-gram lookup; no novel GPU kernels.

---

### DFlash speculative decode (S1 loader + S2 dispatch + S3 GPU ring buffer) — Phase 7a

DFlash drafter spec-decode lifted from buun `master` + z-lab drafter:

- **S1 model loader** (`b6a75e524`) — drafter model architecture + GGUF loader in-tree.
- **S2 dispatch** (`ef80c728c`) — `common_speculative_state_dflash` + factory dispatch wired into `--spec-type dflash`.
- **S3 GPU ring buffer + bulk argmax + server `spec_type` wiring** (`9b7ab4e83`).
- **`mask_token_id` GGUF type fix** (`1436d1890`; int32_t → uint32_t to match `llama-hparams.h`).
- **DFlashDraftModel safetensors→GGUF converter** (`ee7d4f896`) — converts the z-lab dflash drafter family; smoke GREEN on Qwen3.6-DFlash @ 915 MB GGUF.

End-to-end smoke gate PASSED on Qwen3.6-35B-A3B-MTP-IQ4_XS target + Qwen3.6 DFlash-draft Q8_0 (1.8 GB):

```bash
llama-server --spec-type dflash -m target.gguf -md dflash-draft.gguf \
    -fa on -ngl 999 --no-mmap
```

**Known limitations (open):**

- `--spec-type dflash` with `-md` also activates draft-simple — measured 33.3% combined accept; isolated DFlash-only accept-rate unmeasured (estimated ≥60% if dual-spec disabled).
- Gemma-4 DFlash converter path exists but is not yet smoke-tested (z-lab Gemma-4 directory missing tokenizer files at last check).
- Build must be ≥ `2726a56c0` (`mask_token_id` type fix is required to load any z-lab DFlash drafter).

---

### TriAttention KV compression with in-graph K/V capture harness — Phase 9 (revived)

TriAttention from domvox `feature/triattention-scoring` was originally Phase 4 work, then deferred post-Phase-8 in 2026-05-15 (halted on a GGML/ROCm backend bug — sub-alloc zero-read in `ggml_backend_tensor_get`). The 2026-05-24 deferred-subsystem recon confirmed mainline never fixed the basic non-2D `tensor_get` path and identified a viable workaround: an in-graph K/V capture harness that reads from a persistent graph-side buffer instead of the backend-broken path.

**What's shipped (2026-05-25 revival):**

- **Phase A in-graph K/V capture harness** (`6cbc9e06c`) — CPU-backed capture buffers via `llama_tria_capture_alloc`, populated by `ggml_set_rows` nodes in `build_attn()`. Phase A `tria_compact_kv` is a no-op (zero evictions; harness-only).
- **`TRIA_HIP_BACKEND` guard** (`eea5e25f5`) — Phase B GPU path gated behind undefined macro until Phase C lands.
- **Safe null-return in `get_layer_k/v_raw`** (`2ad2564f1`) — handles hybrid Qwen3.5-0.8B models where only 6/24 layers have full-attn KV cache.
- **Gemma-4 ISWA capture fix** (`cbd071632`) — handles `llama_kv_cache_iswa` (doesn't inherit `llama_kv_cache`) via a third `dynamic_cast` branch using `iswa->get_base()`; Gemma-4 now allocates 35-layer K/V capture buffers.

**Validation (Phase B GQA CPU smoke):** 3/3 GQA models GREEN (Qwen3.5-9B, Llama-3.1-8B, Gemma-4-E2B). 4-cell build PASS (ROCm gfx1150 + gfx1102 + Vulkan gfx1150 + Vulkan gfx1103). PPL gate: baseline 15.2055 vs triattention 15.1913, Δ=0.09% — within ±10% (no-op compact is expected for Phase A).

**Open Phase C work:** GPU GQA kernel extension to aggregate query heads for `nh != nkv` GQA models, SWA-layer capture for Gemma-4 (`kv_swa` not currently captured), and GPU scoring path (requires `--cache-type-k q8_0`).

---

### Novel model architectures — in-tree ports

In addition to all mainline-supported architectures (inherited via upstream
sync), this fork ships in-tree ports for novel hybrid architectures that
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

Active feature branches / queued workers; not yet merged to `main`.

| Workstream | Branch | Status |
|---|---|---|
| Trellis IQ3_KT (Phase P3b) | `main` | **Complete** — CPU/ROCm/Vulkan shipped 2026-05-29 (`623835cc9`); PPL +23.5% vs IQ3_K inherent to single-codebook design; cluster-accel k=60; imatrix required |
| Trellis IQ1_KT (Phase P3c) | — | Queued behind P3b; IQKTParams<8,13,false> per `[[trellis-p3-prep-landed]]` |
| MTP Gemma4 §-FLAG-B fix | `feat/mtp-gemma4-guided-port-2026-05-29` | Fix validated (`96b487c1c`) — move "mtp." tensor rename AFTER load_all_data to fix accept 0%→33.9–61.8%; land pending on convergence bridge |
| IQ2_KT cluster-accel PPL retune (k=80–100) | — | Late-stage-polish queue; current ship at k=60 has §-FLAG PPL +8.3% above ≤+5% clean threshold |
| Full spec-decode validation matrix | — | 40-cell matrix (4 backends × 2 main models × 5 mechanisms); gated on EAGLE3 + DFlash + PHANTOM-X all landed (they are) + clean MTP V-J reverify (queued) |
| MTP V-J clean re-measurement | — | Post-mrope-fix throughput re-measure on a quiet host (`e8e767347` shipped fix but smoke measurement was concurrent with peer builds; current ratio 0.737× below 0.78-0.85× projection — clean number pending) |
| TriAttention Phase C GPU GQA kernel | — | Phase A+B + Gemma-4 ISWA fix shipped; Phase C extends `tria_score` GPU path to GQA (`nh != nkv`) and adds SWA-layer capture for Gemma-4 |
| RBQ3 imatrix retrofit + port | — | turbo-tan RBQ3 family is sole pending weight quant needing imatrix retrofit (~6h) per PM-15; port follows |
| Vulkan MTP UPDATE_ACCEPTED SIGSEGV diagnostic | — | Opus-class diagnostic queued (`gf_res_prev` shared-pointer use-after-free hypothesis; `[[vulkan-mtp-bugs]]` revised 2026-05-24) |

## Blocked / awaiting decision

| Item | Blocked on |
|---|---|
| Full spec-decode matrix (40 cells) | Sequencing: needs clean MTP V-J reverify + opportunistic GPU-time window (~16-24h walltime) |
| PFlash 1b (real scorer) | Quality validation smoke on existing 1a branch; user decision on 1b scope |
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
novel work (TurboQuant KV, TCQ KV, sidecars, etc.) reach users on
hardware we don't own; the Vulkan port effort is a burden for
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
- [**docs/OP_ASSIGNMENTS.md**](docs/OP_ASSIGNMENTS.md) — original
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
  README for reference on build/usage docs that aren't fork-specific.

## Build / usage

This fork follows mainline's build system unchanged. All shipped features
(TurboQuant KV + TCQ KV + InnerQ KV + WHT weight quants + sidecar engine)
are built unconditionally — no new feature-gate flags. See
[README.upstream.md](README.upstream.md) and the upstream `docs/`
directory for build instructions.

For usage of the new types, see [What's available now](#whats-available-now)
above. For change history, see [CHANGELOG.md](CHANGELOG.md).

## Companion projects

The following projects are companion tools for this fork, updated in lockstep
with every rebuild. They are designed to work with **any** llama.cpp
fork and contain no fork-specific type names or conditionals.

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
  mainline `b745` (`751ebd17a`); rebased 2026-05-28 (68 new mainline commits
  since `b9310`; PPL 6.5453 GREEN (gfx1103 RDNA3). Next sync ~2026-06-11.
- Trunk: `main` (HEAD `d738c4341`).
- Milestone tags on origin: `milestone/phase-0-foundation-complete`,
  `milestone/phase-0.7-sidecar-engine`,
  `milestone/phase-1-turboquant-kv-foundation`,
  `milestone/phase-2-mtp-foothold`,
  `milestone/phase-2-gemma4-mtp`,
  `milestone/phase-5b-1a-iq-quants` (shipped `09c0d1d6c`).
- Feature work happens on `feature/<phase>-<scope>` topic branches and
  FF-merges back to `main` once all gates pass. See
  [conventions/git-workflow.md](conventions/git-workflow.md) for the
  detailed workflow.
- ik_llama work is tracked subsystem-by-subsystem rather than as branches,
  because ik_llama's history is unrelated to mainline's. Cherry-pick
  individual commits or re-implement, never bulk-merge.

## Why this fork exists (vs. picking one fork as base)

Mainline as base is the right choice for six of seven contributing forks
because their histories are GitHub-forks of mainline and their work
expresses as cherry-pickable topic branches. The seventh, ik_llama, has
independent history — porting subsystem-by-subsystem from it onto mainline
is a multi-month effort, but choosing ik_llama as base would orphan the
mainline-side improvements that arrive every week.

The trade-off: this fork pays an ongoing ik_llama-port cost forever, in
exchange for staying mainline-current forever. The alternative (forking
ik_llama and pulling mainline in) would pay a giant one-time mainline
rebase cost upfront, then a forever cost of fighting ik_llama's
independent direction with mainline's.

The single-author velocity of mainline + ik_llama combined is too high to
choose either side as base and expect the other's improvements to arrive
cheaply. The answer is to accept both as ongoing inputs.

## Attribution

This fork is built on top of the [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) project (MIT) and incorporates work from several sibling forks. The conventions document the project's lift discipline. Sibling forks credited:

### Direct lifts (substantial code or design imported)

- **[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)** — base mainline; rebased forward on a ~2-week cadence
- **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)** — TurboQuant KV cache quantization (Phase 1) + InnerQ calibrated KV types (Phase 3d) + WHT weight quants
- **[spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)** — TCQ KV cache types (Phase 3a, 3c) + PFlash prompt compression (Phase 7b, Vulkan GPU scorer fix shipped in NEW-D) + DFlash S1 model loader (`b8bf27eda`); tensor-split with quantized KV unblocked + TURBO_WHT split-planner fix (shipped `6774410fa`, `340f6fe21`)
- **[carlosfundora/llama.cpp-1-bit-turbo](https://github.com/carlosfundora/llama.cpp-1-bit-turbo)** — RotorQuant KV V-cache variants (Phase 4a, shipped); Q1_0_G128 (shipped `87d3705e0`); EAGLE3 (shipped `c0f3c1486` + fc dtype-aware fix `4c38845c4`); PHANTOM-X (shipped `d6dc63224` + Phase 2 dispatch `388169995`); TurboMind allocator queued (PORT-LATER); Wave32 RDNA2 out of scope per `[[supported-rocm-hardware-targets]]`
- **[Anbeeld/beellama.cpp](https://github.com/Anbeeld/beellama.cpp)** — DFlash spec-decode hardening (Phase 7a; DFlashDraftModel safetensors→GGUF converter ported `ee7d4f896`; S2 dispatch + S3 GPU ring buffer shipped)
- **[turbo-tan/llama.cpp-tq3](https://github.com/turbo-tan/llama.cpp-tq3)** — RaBitQ TQ3 weight quants (Phase 6, pending imatrix retrofit per PM-15)
- **[domvox/llama.cpp-turboquant-hip](https://github.com/domvox/llama.cpp-turboquant-hip)** — TriAttention KV compression (Phase 9, REVIVED 2026-05-25 — Phase A in-graph capture harness shipped, Phase B GQA CPU smoke GREEN, Phase C GPU GQA kernel pending); per-layer SWA KV cache type `--cache-type-{k,v}-swa` (shipped `d8ec65064`)
- **[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)** — IK quants (5b-1a/1b/1c/5b-2 all shipped CPU+CUDA/HIP+Vulkan; IQ2_KT Trellis P3a shipped with cluster-accel; P3b/P3c queued), BitNet (pending), MLA/FlashMLA (declined), fused MoE (pending), bf16 KV (pending); ongoing MTP improvements
- **[Zyphra/transformers](https://github.com/Zyphra/transformers)** (zaya1 branch) — ZAYA1-8B model architecture (Phase 0, in-tree port)

### Inspiration / planned lifts

- **[Luce-Org/llama.cpp-dflash-ggml](https://github.com/Luce-Org/llama.cpp-dflash-ggml)** — FP64 RoPE theta precision fix (shipped `e54d0aadd`) + GGML_OP_FLASH_ATTN_SPARSE op (shipped `bc3387bd3`)
- **[z-lab/dflash](https://github.com/z-lab/dflash)** — DFlash drafter training recipe + safetensors checkpoint family (Qwen3.6, Gemma-4); GGUF converter ported (`ee7d4f896`)

Per project policy, this fork does NOT propose AI-generated contributions to mainline llama.cpp or any sibling forks. All ports and experiments remain in this repository.

---

## Contributing

This is currently a personal project. See [CONTRIBUTING.md](CONTRIBUTING.md)
for the current PR / issue posture (TL;DR: the owner can discuss intent
but can't independently review code; please cite upstream
`ggml-org/llama.cpp` for everything not introduced by this fork).
