# IK Quantization Family

The **IK family** is a suite of weight quantization types ported from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp). All IK types
are **imatrix-aware**: they require an importance matrix as a mandatory offline input
to the quantizer. In exchange they achieve better quality-per-bit than the comparable
mainline K-quants at matched bit-widths.

This primer covers the shared design concepts — block structure, the imatrix
requirement, shared code infrastructure, and the Vulkan dispatch split — that all four
IK sub-family docs build on. Each sub-family doc links here instead of repeating this
background.

---

## What "K-quant" means in the IK context

Mainline llama.cpp "K-quants" (Q2_K, Q3_K_M, Q4_K_M, …) use 256-element super-blocks
where each block stores a single block scale plus per-group (32-element) subscales. The
quantization indices point into a fixed symmetric codebook.

IK types extend this in two ways:

1. **Nonlinear value tables.** Instead of a symmetric codebook, each type uses a set of
   nonlinear integer centroid values hand-tuned for the distribution of transformer
   weight tensors. Two variants of each table exist — standard and shifted — and a
   per-sub-block bit (`extra` field) in the block header selects which variant applies
   to each 16-element sub-block. This lets the quantizer adapt the effective codebook
   on a fine-grained basis at zero index overhead.

2. **Imatrix-aware scale optimization.** During quantization the per-group scales are
   jointly optimized using element-wise importance weights from the calibration matrix,
   rather than using the L2-optimal scale for each group in isolation. This is where the
   quality gain comes from: more important weights (those the model's activations
   actually exercise) receive tighter quantization.

The "K" suffix signals the super-block layout (256 elements + per-group scales), and
the "I" prefix signals the ik_llama origin and imatrix dependency.

---

## Imatrix is mandatory

> **The most important fact about every IK weight quant.**

**Without `--imatrix`, the quantizer hard-errors** for every quantizable tensor:

```
ERROR: this quantization requires an importance matrix!
        - offending tensor: blk.0.attn_q.weight
        - target type: IQ3_K
```

This is enforced in the quantizer (`src/llama-quant.cpp`) — it is not a warning and
there is no fallback. The only exempt tensors are `token_embd` (embedding table) and
`output` (logit projection), which the quantizer handles without imatrix data.

### Generating an imatrix

An imatrix is generated once from a calibration corpus and then reused across
quantization runs of the same model:

```bash
# Step 1 — generate the importance matrix
llama-imatrix \
    -m model-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o model.imatrix
```

Use a calibration corpus that represents your target workload. For MoE models, ensure
the corpus activates a broad mix of experts — a short, repetitive corpus will
under-cover rarely-used experts and degrade quality on the tasks that use them.

---

## Shared infrastructure

| File | Role |
|---|---|
| `ggml/src/ggml-iqk-quants.c` | CPU dequant for the base-K family (IQ2_K / IQ3_K / IQ4_K) |
| `ggml/src/ggml-iqk-kt.cpp` | CPU dequant + quantize for the KT trellis family |
| `ggml/src/ggml-iqk-kt-family.hpp` | Template header shared by all KT variants |
| `ggml/src/ggml-cuda/mmvq-iqk.cu` | CUDA/HIP MMVQ (mat-vec) dot kernels for all IK types |

Vulkan shaders live under `ggml/vulkan-shaders/` (one `.comp` per type); pipeline
registration is in `ggml/src/ggml-vulkan/ggml-vulkan.cpp`.

---

## Vulkan dispatch: decode vs. prefill

All IK types have native Vulkan **decode** (single-token mat-vec) shaders — quantized
values are dequantized in-shader during the dot product, the same bandwidth-preserving
path that mainline Q4_K and Q6_K use. Decode throughput on Vulkan is therefore
**full-speed and unpenalized**.

**Prefill (mat-mat, batch > 1) is different.** The IK base-K and KS types do not have
native Vulkan GEMM tiles. When a batch of prompt tokens forces a mat-mat dispatch, the
Vulkan backend:

1. Dequantizes the weight tensor to a **transient fp16 scratch buffer** (the weights
   themselves remain stored in their quantized format — no permanent storage increase).
2. Dispatches a generic fp16 × fp16 GEMM using the scratch buffer.

The practical impact: **long-prompt prefill on Vulkan is slower than on ROCm/CUDA** and
also slower than mainline Q4_K/Q6_K on Vulkan (which have native quantized GEMM tiles).
The decode (token generation) path is unaffected — no Vulkan prefill penalty applies
there.

The benchmark matrices in the per-family docs include a Vulkan PP (prefill) column that
will quantify the gap once benchmarks are collected.

> Native IK Vulkan GEMM tiles are a tracked future improvement; the dequant-fallback
> path is correct and lossless in the interim.

---

## The IK family — four sub-families

| Sub-family | Doc | Types | Encoding |
|---|---|---|---|
| **Base-K** (this primer's primary family) | [IK Base-K weight quants](../ik-base-k.md) | `IQ2_K`, `IQ3_K`, `IQ4_K` | Global-scale super-block; nonlinear value tables + codebook-shift |
| **High-bit-K** | _(doc pending)_ | `IQ5_K`, `IQ6_K` | Global-scale super-block; nonlinear value tables + codebook-shift |
| **KS / row-meta** | _(doc pending)_ | `IQ4_KS`, `IQ3_KS`, `IQ4_KSS`, `IQ2_KL` | Per-row floating-point meta-scale prepended to each row; sub-block structure |
| **KT / trellis** | _(doc pending)_ | `IQ4_KT`, `IQ2_KT`, `IQ3_KT` | Per-row scale + trellis-coded quantization (Viterbi search over per-block groups) |

`IQ2_KL` is grouped with KS because it is structurally a row-meta type (a floating-point
per-row scale prepended to the row data), not a global-scale type — despite being 2-bit
like the base-K `IQ2_K`. The split is by encoding structure, not by bit-width.

All IK types are in the **ik_llama compatibility zone (enum slots 96–199)** defined in
`docs/TYPE_ASSIGNMENTS.md`. All require imatrix for quantization.
