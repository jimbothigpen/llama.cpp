# Primer: Asymmetric KV Cache & K×V Pairing

This primer explains what asymmetric KV cache quantization means, why K and V behave differently, and how to choose a K×V pair.

## What "asymmetric" means

llama.cpp allows K and V caches to be quantized independently via `--cache-type-k` and `--cache-type-v`. Setting them to different types is called an **asymmetric** configuration. Example:

```bash
--cache-type-k turboq4 --cache-type-v turboq2
```

This is different from weight quantization, where mixed precision applies to individual tensors in the model file.

## Why K and V respond differently to compression

K (key) and V (value) play different roles in the attention mechanism:

- **K** is used in the dot-product comparison `Q·Kᵀ`. Errors in K shift attention scores globally, affecting which tokens are attended to. Small K errors can produce systematic attention-weight distortion.
- **V** is used as the weighted average output. Errors in V appear as additive noise in the output representation, but they do not change *which* tokens are attended to — only *what* their contribution looks like.

In practice, lowering V precision costs less model quality than lowering K precision by the same amount. This asymmetry is why **K-bpw ≥ V-bpw** is the quality-preserving direction.

## Choosing a K×V pair

| K type | V type | Approximate memory vs fp16 KV | Use case |
|---|---|---|---|
| F16/BF16 (unquantized) | turboq3 | ~2.5× reduction in V | K quality preserved, moderate V saving |
| Q8_0 | turboq3 | ~3× total | Good balance |
| Q8_0 | turboq2 | ~4× total | More aggressive, K still decent |
| turboq4 | turboq3 | ~4.4× total | Fully quantized, good quality |
| turboq3 | turboq3 | ~5.1× total | Symmetric sweet spot |
| turboq3 | turboq2 | ~6× total | High compression, monitor quality |
| turboq2 | turboq2 | ~7.5× total | Maximum compression |

## Per-layer differentiation

The layer-adaptive knob (`TURBO_LAYER_ADAPTIVE`) provides a third axis: vary precision by layer position rather than keeping it uniform. Boundary layers (first and last few) tend to be more sensitive; see the [TurboQuant feature doc](../turboquant-kv-base.md#layer-adaptive-precision-optional) for mode descriptions.

## Memory math

KV cache memory per token per layer is roughly:

```
bytes = 2 × n_heads × head_dim × (bpw_k + bpw_v) / 8
```

At turboq3 symmetric (3.125 bpw each), a 32-layer model with 32 heads × 128 head_dim uses about 1 MB per 1024 tokens, vs ~5 MB for fp16.

## Further reading

- [TurboQuant KV base feature doc](../turboquant-kv-base.md)
- [Feature maturity levels & backend support](feature-maturity-levels.md)
