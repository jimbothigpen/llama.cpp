# Trellis-Coded Quantization (TCQ)

Trellis-Coded Quantization is a sequence-quantization technique that finds the
globally optimal assignment of quantization symbols to a group of values using
a Viterbi search over a state trellis. It consistently outperforms nearest-centroid
scalar quantization at the same nominal bit-width.

## Why nearest-centroid falls short

Scalar quantization maps each element independently to its nearest codebook entry.
This misses the structure across adjacent elements: two elements close in value may
land in different centroids even when a small global shift would have reduced the
total reconstruction error. Each element's quantization is locally optimal but the
sequence as a whole is not.

## The trellis

A TCQ trellis has **S states** and **k bits per symbol**. For each time step `t`
(one input element), every state has two valid transitions, each producing an output
symbol. The output symbol maps to a codebook vector; the reconstruction error for
step `t` is `(input[t] − codebook[output_symbol])²`.

The **Viterbi algorithm** finds the path through the trellis (one transition per
step, for all 128 steps of a block) that minimizes the sum of per-step reconstruction
errors. This is a dynamic-programming sweep:
1. Forward pass: for each step `t`, compute the minimum-cost path arriving at each
   state, storing the best predecessor.
2. Backward pass (traceback): starting from the minimum-cost terminal state, follow
   predecessor pointers to recover the optimal symbol sequence.

The result is a sequence of k-bit symbols that, when decoded, minimizes the total
L2 reconstruction error for the block — a guarantee no per-element method can make.

## Parameters in this implementation

| Type | Bits (k) | States (S) | GPU threads/block |
|---|---|---|---|
| `turboq3_tcq` | 3 | 512 (L=9) | 512 |
| `turboq2_tcq` | 2 | 256 (L=8) | 256 |

Each GPU thread is responsible for exactly one trellis state. All threads advance
in lockstep through the 128 time steps with a `__syncthreads()` barrier at each
step. Double-buffered cost arrays reduce the sync count from 384 to 128 per block.

## Why the codebook beats Lloyd-Max

A standard Lloyd-Max codebook is optimal for independent draws from a known
distribution. TCQ achieves approximately +3.0 dB MSE improvement over Lloyd-Max
for `turboq3_tcq` and +1.75 dB for `turboq2_tcq` (per the buun upstream) because
the joint optimization over the full sequence exploits the inter-symbol structure
that Lloyd-Max ignores.

## Decode

Decoding is a simple left-to-right scan: for symbol index `t`, read k bits starting
at bit `t*k` from the stored bitstream, look up `codebook[state]`, multiply by the
block norm. No Viterbi required; O(n) in block size, branch-free, and suitable for
CPU or GPU inline use.

## References

- [TCQ KV feature doc](../tcq-kv.md) — usage, flags, and backend details
- [WHT / Hadamard rotation primer](hadamard-wht-rotation.md) — the pre-rotation
  stage that precedes TCQ encoding
