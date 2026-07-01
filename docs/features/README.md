# Feature Docs

End-user documentation for fork-specific features. Each doc covers one feature family: what it is, how to use it, and how it works.

> **Provenance:** [PROVENANCE.md](PROVENANCE.md) is the canonical map of where every ported feature comes from (code / converter / weights upstreams, tracked remote, and synced ref for drift-checking). Each feature doc also carries its own §Provenance section.

## KV Cache Quantization

Runtime KV cache compression — apply to any GGUF via `--cache-type-k`/`--cache-type-v`, no model changes required.

| Feature | Status | Types | Compression vs fp16 |
|---|---|---|---|
| [TurboQuant KV base](turboquant-kv-base.md) | Stable (`turboq8` Preview, CPU + CUDA/HIP only) | `turboq2`, `turboq3`, `turboq4`, `turboq8` | ~7.5× / ~5.1× / ~3.8× / ~1.97× |
| [TurboQuant high-bit KV](turboquant-hibit-kv.md) | Functional (perf TBD; CPU + CUDA/HIP only) | `turboq5`, `turboq6` | ~3.12× / ~2.61× (dormant slots 64-65) |
| [TCQ KV cache](tcq-kv.md) | Stable | `turboq2_tcq`, `turboq3_tcq` | ~7.1× / ~4.9× |
| [OScaR INT2 K-cache](oscar-kv.md) | Experimental (CUDA/HIP only, Phase 1) | `kv_oscar_int2` | ~8× K-only; FHT+INT2; residual window (`--cache-oscar-residual-window`); gate PASS 9B (+9.1% PPL); **DO NOT USE sub-1B** (architectural issue) |

## KV Cache Architecture

Per-layer or per-class configuration that composes with the KV type flags above.

| Feature | Status | Applicable models | Summary |
|---|---|---|---|
| [SWA per-layer KV types](swa-per-layer-kv.md) | Stable | Gemma 4 / Gemma 2/3, Llama 4, MiMo2 | Assign separate KV types to global and SWA sub-caches; avoids catastrophic PPL collapse under uniform aggressive quant |

## KV Cache Eviction

Token-level eviction — drops low-importance cached entries to fit a budget, complementary to quantization.

| Feature | Status | Applicable models | Summary |
|---|---|---|---|
| [TriAttention KV eviction](triattention.md) | Experimental — requires `.tria` calibration file | All GQA: Qwen3.x, Llama-3.1, Gemma-4, hybrid SSM+attn | Score cached tokens by attention-trajectory importance; evict below a budget; 100% retrieval @25% budget on Qwen3-8B; GPU scoring (HIP+Vulkan) for hd≤128; `--triattention`, `--tri-budget` |

## Concept Primers

Short standalone explanations of techniques used across multiple feature families. Feature docs link here instead of repeating the same background.

- [IK quantization family](concepts/ik-quantization-family.md) — shared IK concepts: block structure, imatrix mandate, Vulkan dispatch split, and the four-sub-family doc map
- [WHT / Hadamard rotation](concepts/hadamard-wht-rotation.md) — the randomized Walsh-Hadamard Transform used by TurboQuant and TCQ
- [Trellis-coded quantization](concepts/trellis-coded-quantization.md) — Viterbi trellis, codebook, and why TCQ beats nearest-centroid at the same bit-width
- [Asymmetric KV cache & K×V pairing](concepts/asymmetric-kv-cache.md) — why K and V behave differently and how to pick a pair
- [Feature maturity levels & backend support](concepts/feature-maturity-levels.md) — what Stable / Experimental / Preview mean; CPU/CUDA/HIP/Vulkan notation

## Weight Quantization

Offline quantization — produce a smaller GGUF from an F16/BF16 source with
`llama-quantize`. All IK weight quants require an imatrix.

| Feature | Status | Types | Notes |
|---|---|---|---|
| [IK Base-K weight quants](ik-base-k.md) | Stable | `IQ2_K`, `IQ3_K`, `IQ4_K` | 2–4.5 bpw; imatrix required; better PPL than mainline K-quants at matched bpw |
| [IK High-Bit-K weight quants](ik-high-bit-k.md) | Stable | `IQ5_K`, `IQ6_K` | 5.5–6.625 bpw; imatrix required; near-lossless quality below Q8_0's footprint |
| [IK Row-Meta weight quants](ik-ks-row-meta.md) | Stable | `IQ4_KS`, `IQ3_KS`, `IQ4_KSS`, `IQ2_KL`, `IQ2_KS`, `IQ5_KS` | 2.6875–4.25 bpw; per-row scale prefix; imatrix required; near-twin IQ4_KS/IQ4_KSS differ by 0.25 bpw |
| [IK KT/Trellis weight quants](ik-kt-trellis.md) | IQ4_KT/IQ3_KT: Stable; IQ2_KT: **§-FLAG DO NOT USE** | `IQ4_KT` (4.0 bpw), `IQ3_KT` (3.0 bpw), `IQ2_KT` (2.0 bpw §-FLAG), `IQ1_KT` (1.75 bpw) | No stored codebook — LCG hash regenerates 65,536-entry implicit codebook; cluster-accel NN search at quantize time; imatrix required; IQ2_KT has general codebook defect (PPL 33.96 at 9B) — use `IQ2_KL` instead |
| [WHT weight quants](wht-weight-quants.md) | Stable | `WHT3_0` (~4.0 bpw), `WHT4_0` (~5.0 bpw), `WHT5_0`, `WHT6_0`, `WHT8_0` | TurboQuant weight family (TheTom); WHT rotation → Lloyd-Max codebook; calibration-free (imatrix not used); WHT4_0 peers Q5_K_M, WHT3_0 peers Q4_0/IQ4_XS |
| [WQ3 TCQ](wq3-tcq.md) | Experimental | `WQ3_TCQ` (slot 92) | Trellis-coded weight quant, not recommended |

More IK sub-family docs are in progress — see the
[IK quantization family primer](concepts/ik-quantization-family.md) for the full
four-sub-family map.

## Novel Model Architectures

In-tree ports for hybrid model architectures not present in mainline llama.cpp.
GGUF files produced from these converters load only in this fork.

| Feature | Status | Architecture | Summary |
|---|---|---|---|
| [Zyphra ZAYA1-8B](zaya1.md) | Stable | `LLM_ARCH_ZAYA` | 8.4B hybrid MoE — 80 layers alternating CCA attention and 16-expert top-1 MoE; CPU/ROCm gfx1150/Vulkan RDNA3 validated |
| [DiffusionGemma](nld-diffusion-self-spec.md) | Stable | `LLM_ARCH_DIFFUSION_GEMMA` | Diffusion-LM architecture |

## Speculative Decode

Faster inference via draft-and-verify strategies. Each entry describes its own trigger mechanism — check the doc for whether it uses `--spec-type` or a model-specific flag.

| Feature | Status | Models | Summary |
|---|---|---|---|
| [MTP speculative decode](mtp.md) | Stable | Qwen3.5/3.6 (internal NextN-tail), Gemma 4 26B-A4B (external assistant) | MTP head drafts tokens from inside the target GGUF or a separate assistant; 75.56% accept (Qwen3.5/3.6 9B), 47.3% (Gemma 4); ~1.16× on iGPU, 1.5–2.5× on dGPU; `--spec-type draft-mtp` |
| [EAGLE3 speculative decode](eagle3.md) | Preview — functional, **architectural ceiling at 1/n_draft for n_draft≥2; use n_draft=1** | Any target + matching EAGLE3 drafter GGUF | Aux-layer hidden-state drafter; 100% accept at n_draft=1 (recommended), 33.333% at n_draft=3 (1/n_draft ceiling); server single-slot only; `--spec-type draft-eagle3`; EAGLE 3.1 future-watch in §5 |
| [NLD diffusion self-spec](nld-diffusion-self-spec.md) | Stable | Dream / LLaDA / LLaDA-MoE / RND1 | Bidirectional draft + causal verify on shared KV; ~3.7× over block-mode; CLI flag `--diffusion-self-spec`; server auto-detects |
| [Qwen3.5/3.6 MTP converter](qwen35-mtp-converter.md) | Stable | Qwen3.5/3.6 dense + MoE | Three converter modes (bundled / `--no-mtp` / `--mtp` split-export); 75.6% draft accept; `--spec-type draft-mtp`; upstream mainline PR #22673 |
| [PHANTOM-X self-speculative n-gram drafter](phantom-x.md) | Stable | Any causal-LM GGUF | Bloom-filtered n-gram tables; no separate draft model; `--spec-type phantom`; +34% on repetitive code (86.6% accept); flat on prose; ported from carlosfundora 1-bit-turbo |
| [Cascade Ensemble](spec-decode-ensemble.md) | Stable | Any | Prioritized cascade of n-gram and MTP drafters; `--spec-type ngram-map-k` |
| [DFlash drafter spec-decode](dflash.md) | Preview — correct, **no speedup yet** | Any target + z-lab DFlash drafter | Cross-attention-ring drafter; 28.7% solo accept (gfx1150, Qwen3.6); `--spec-type dflash`; S3 GPU ring complete (BUUN port retained) |

## Prompt Compression

Reduces the token budget presented to the target model's prefill via scorer-guided
importance ranking. Complementary to KV cache quantization and eviction.

| Feature | Status | Summary |
|---|---|---|
| [PFlash prompt compression](pflash.md) | Experimental — Qwen3.x scorer validated; **§-FLAG (PFL-1): non-Qwen scorer (Gemma3/Llama/Qwen2/Mistral) UNVALIDATED** | Scorer assigns importance weights to each prompt token; top fraction retained up to `--pflash-keep-ratio`; activates only above `--pflash-min-tokens`; CUDA/HIP GPU scorer; Vulkan falls back to CPU; `--pflash-scorer PATH` |

## Adding a new doc

1. Create `docs/features/<feature-name>.md` using the template in `turboquant-kv-base.md` as the exemplar.
2. Add an entry to the appropriate section in this index.
3. Link out to relevant primers rather than re-explaining them inline.
4. Add a cross-link from the top-level `README.md` features table.
