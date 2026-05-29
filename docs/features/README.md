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

- [WHT / Hadamard rotation](concepts/hadamard-wht-rotation.md) — the randomized Walsh-Hadamard Transform used by TurboQuant and TCQ
- [Trellis-coded quantization](concepts/trellis-coded-quantization.md) — Viterbi trellis, codebook, and why TCQ beats nearest-centroid at the same bit-width
- [Asymmetric KV cache & K×V pairing](concepts/asymmetric-kv-cache.md) — why K and V behave differently and how to pick a pair
- [Feature maturity levels & backend support](concepts/feature-maturity-levels.md) — what Stable / Experimental / Preview mean; CPU/CUDA/HIP/Vulkan notation

## Adding a new doc

1. Create `docs/features/<feature-name>.md` using the template in `turboquant-kv-base.md` as the exemplar.
2. Add an entry to the appropriate section in this index.
3. Link out to relevant primers rather than re-explaining them inline.
4. Add a cross-link from the top-level `README.md` features table.
