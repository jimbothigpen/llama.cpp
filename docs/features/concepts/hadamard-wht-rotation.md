# Primer: Hadamard / WHT Rotation

This primer explains the Walsh-Hadamard Transform (WHT) rotation used by the TurboQuant KV cache family and why it improves quantization quality.

## The basic idea

When quantizing a vector with a fixed codebook (like PolarQuant's small set of centroids), elements that happen to fall near a centroid midpoint are poorly approximated. If a single element is a large outlier, it absorbs most of the quantization error and degrades the whole block.

A **randomized orthogonal rotation** spreads that problem: rotate the vector first, quantize in the rotated domain, then un-rotate on decode. Because the rotation is orthogonal (length-preserving), the approximation quality is the same on average — but the worst-case error is distributed across all elements rather than concentrated on one. For a fixed codebook, this variance-reduction almost always improves perplexity.

## Why WHT?

The Walsh-Hadamard Transform is a natural choice:
- **O(d log d)** per group (compared to O(d²) for a dense rotation matrix), which matters at inference speed.
- **Integer arithmetic friendly** — the butterfly involves only additions and subtractions.
- **Exact inverse** — the inverse WHT has the same structure as the forward pass, just with sign sequences swapped.

The "randomized" part comes from pre-multiplying and post-multiplying by fixed ±1 diagonal matrices (`s1`, `s2`), drawn once from a seeded PRNG. This makes the transform look like a random dense orthogonal matrix to the input distribution, without sacrificing the O(d log d) cost.

## The TurboQuant implementation

The forward transform is `turbo_cpu_fwht` (`ggml/src/ggml-turbo-quant.c:216`):

```
y = D(s2) · (1/√d) · H · D(s1) · x
```

where `H` is the standard Hadamard butterfly, `D(s)` means pointwise multiply by the sign vector `s`, and `1/√d` normalizes the output to unit variance under a unit-norm input.

The sign vectors `s1` and `s2` are fixed constants (`turbo_cpu_s1`, `turbo_cpu_s2`) drawn once with seed 42, and are shared across CPU, CUDA/HIP, and Vulkan backends — ensuring every backend encodes and decodes identically.

## KV cache specifics

For KV compression, the rotation group size matches `head_dim` (128 elements for all supported models). The KV vector is L2-normalized before rotation so that the distribution fed to PolarQuant is approximately N(0, 1/√128) — which is what the fixed codebook centroids were optimized for.

On decode, the inverse WHT is applied by a graph op (`GGML_OP_TURBO_WHT`) rather than inside the dequantize kernel, so it can be fused with the attention computation.

## Further reading

- [TurboQuant KV base feature doc](../turboquant-kv-base.md)
- arXiv 2504.19874 — original paper
