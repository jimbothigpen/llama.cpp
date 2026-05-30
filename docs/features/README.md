# Feature Docs

End-user documentation for fork-specific features. Each doc covers one feature family: what it is, how to use it, and how it works.

## KV Cache Quantization

Runtime KV cache compression — apply to any GGUF via `--cache-type-k`/`--cache-type-v`, no model changes required.

| Feature | Status | Types | Compression vs fp16 |
|---|---|---|---|
| [TurboQuant KV base](turboquant-kv-base.md) | Stable | `turboq2`, `turboq3`, `turboq4` | ~7.5× / ~5.1× / ~3.8× |
| [TCQ KV cache](tcq-kv.md) | Stable | `turboq2_tcq`, `turboq3_tcq` | ~7.1× / ~4.9× |
| [InnerQ KV cache](innerq-kv.md) | Experimental (CUDA/HIP only) | `turboq2_innerq`, `turboq3_innerq` | ~7.5× / ~5.1× (same memory as base; quality improvement) |

## KV Cache Architecture

Per-layer or per-class configuration that composes with the KV type flags above.

| Feature | Status | Applicable models | Summary |
|---|---|---|---|
| [SWA per-layer KV types](swa-per-layer-kv.md) | Stable | Gemma 4 / Gemma 2/3, Llama 4, MiMo2 | Assign separate KV types to global and SWA sub-caches; avoids catastrophic PPL collapse under uniform aggressive quant |

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
| [IK Row-Meta weight quants](ik-ks-row-meta.md) | Stable | `IQ4_KS`, `IQ3_KS`, `IQ4_KSS`, `IQ2_KL` | 2.6875–4.25 bpw; per-row scale prefix; imatrix required; near-twin IQ4_KS/IQ4_KSS differ by 0.25 bpw |

More IK sub-family docs are in progress — see the
[IK quantization family primer](concepts/ik-quantization-family.md) for the full
four-sub-family map.

## Novel Model Architectures

In-tree ports for hybrid model architectures not present in mainline llama.cpp.
GGUF files produced from these converters load only in this fork.

| Feature | Status | Architecture | Summary |
|---|---|---|---|
| [Zyphra ZAYA1-8B](zaya1.md) | Stable | `LLM_ARCH_ZAYA` | 8.4B hybrid MoE — 80 layers alternating CCA attention and 16-expert top-1 MoE; CPU/ROCm gfx1150/Vulkan RDNA3 validated |

## Speculative Decode

Faster inference via draft-and-verify strategies. Each entry describes its own trigger mechanism — check the doc for whether it uses `--spec-type` or a model-specific flag.

| Feature | Status | Models | Summary |
|---|---|---|---|
| [NLD diffusion self-spec](nld-diffusion-self-spec.md) | Stable | Dream / LLaDA / LLaDA-MoE / RND1 | Bidirectional draft + causal verify on shared KV; ~3.7× over block-mode; CLI flag `--diffusion-self-spec`; server auto-detects |
| [Qwen3.5/3.6 MTP converter](qwen35-mtp-converter.md) | Stable | Qwen3.5/3.6 dense + MoE | Three converter modes (bundled / `--no-mtp` / `--mtp` split-export); 75.6% draft accept; `--spec-type draft-mtp`; upstream mainline PR #22673 |

## Adding a new doc

1. Create `docs/features/<feature-name>.md` using the template in `turboquant-kv-base.md` as the exemplar.
2. Add an entry to the appropriate section in this index.
3. Link out to relevant primers rather than re-explaining them inline.
4. Add a cross-link from the top-level `README.md` features table.
