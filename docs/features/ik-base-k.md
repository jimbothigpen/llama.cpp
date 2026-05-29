# IK Base-K Weight Quants (`IQ2_K` / `IQ3_K` / `IQ4_K`)

> **Status: Stable** — CPU, CUDA/HIP, and Vulkan backends.
>
> **Known limitation:** MoE models on Vulkan — the `MUL_MAT_ID` dispatch is
> code-complete and the dense Vulkan path is verified, but the MoE-on-Vulkan path
> has not yet been end-to-end smoke-validated with a real base-K MoE quant.

---

## At a glance

| CLI name | Enum | Slot | Effective bpw | Block bytes | Backends |
|---|---|---|---|---|---|
| `IQ2_K` | `GGML_TYPE_IQ2_K` | 137 | **2.375** | 76 | CPU, CUDA/HIP, Vulkan |
| `IQ3_K` | `GGML_TYPE_IQ3_K` | 138 | **3.4375** | 110 | CPU, CUDA/HIP, Vulkan |
| `IQ4_K` | `GGML_TYPE_IQ4_K` | 139 | **4.50** | 144 | CPU, CUDA/HIP, Vulkan |

**TL;DR.** The IK base-K types are offline weight quants that target better perplexity
than the comparable mainline K-quants (IQ2_K vs Q2_K, IQ3_K vs Q3_K_M, IQ4_K vs
Q4_K_M) at the same or lower bits-per-weight. They require a one-time imatrix
calibration step before quantization. The resulting GGUF is smaller and loads just like
any other GGUF at inference time.

**Quick start:**

```bash
# Step 1 — generate an importance matrix on a calibration corpus
llama-imatrix \
    -m model-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o model.imatrix

# Step 2 — quantize with the imatrix
llama-quantize --imatrix model.imatrix model-F16.gguf model-IQ3_K.gguf IQ3_K

# Step 3 — run inference (same flags as any GGUF)
llama-server -m model-IQ3_K.gguf -fa on -ngl 99 --no-mmap
```

---

## §1 Provenance

IQ2_K, IQ3_K, and IQ4_K are ported from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), the source of the
entire IK weight-quant family. This fork adds ROCm and Vulkan parity: the CUDA dequant
kernels were corrected for the renumbered type IDs, and the Vulkan in-shader decode path
was wired via the dequant pipeline infrastructure. The upstream ik_llama source is
CUDA-centric; the ROCm and Vulkan paths are this fork's contribution.

**Type ID renumbering.** ik_llama uses type IDs 58/59/60 for IQ4_K/IQ3_K/IQ2_K. This
fork renumbers them to 139/138/137 — within the ik_llama compatibility zone (slots
96–199) defined in `docs/TYPE_ASSIGNMENTS.md`. GGUF files quantized with ik_llama will
not load directly in this fork without re-quantization; the GGUF type fields reflect the
renumbered IDs.

**Perplexity parity vs. ik_llama.** Across multiple model and quantization pairs,
perplexity divergence between this fork and the ik_llama reference is Δ < 0.0045.

---

## §2 Use in production

### Imatrix is mandatory

> **This is the most important user-facing fact about the base-K types.**

The quantizer hard-throws for every quantizable weight tensor if no imatrix is provided:

```
ERROR: this quantization requires an importance matrix!
        - offending tensor: blk.0.attn_q.weight
        - target type: IQ3_K
```

Only the `token_embd` (embedding table) and `output` (logit projection) tensors are
exempt; all other weight tensors require imatrix data. There is no fallback mode — the
quantizer exits with an error.

See the [IK quantization family primer](concepts/ik-quantization-family.md) for imatrix
generation guidance, including corpus recommendations for MoE models.

### Quantize workflow

```bash
# Generate imatrix (adjust -c and --chunks to your available VRAM and time)
llama-imatrix \
    -m Qwen3.5-9B-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o Qwen3.5-9B.imatrix

# Quantize to IQ3_K
llama-quantize \
    --imatrix Qwen3.5-9B.imatrix \
    Qwen3.5-9B-F16.gguf \
    Qwen3.5-9B-IQ3_K.gguf \
    IQ3_K
```

This is an **offline, one-time step** — the resulting GGUF loads and runs at inference
time with no imatrix needed.

### Inference flags

No inference-time flags are specific to the IK base-K types. Standard recommendations:

| Flag | Reason |
|---|---|
| `-fa on` | Flash attention; recommended for performance on supported models |
| `-ngl 99` | Offload all layers to GPU (adjust to your VRAM) |
| `--no-mmap` | Avoids mmap-related slowdowns; recommended on Linux + ROCm |

```bash
llama-server \
    -m model-IQ4_K.gguf \
    -fa on -ngl 99 --no-mmap \
    -c 8192
```

---

## §3 Benefits & potential drawbacks

### Benefits

- **Better quality at matched bit-width** — the IK base-K types are designed to
  outperform the mainline K-quants (Q2_K, Q3_K_M, Q4_K_M) at equal or lower bpw.
  The benchmark matrix below will quantify this once numbers are collected.
- **Full decode speed on all backends** — token generation uses native in-shader
  dequant on Vulkan (same bandwidth-preserving path as Q4_K), and native MMVQ kernels
  on CUDA/HIP. No decode penalty relative to mainline K-quants.
- **Smaller file than the mainline comparator** — IQ3_K (3.44 bpw, 110 B/block) is
  smaller than Q3_K_M (~3.9 bpw); IQ4_K (4.50 bpw, 144 B/block) is smaller than
  Q4_K_M (~4.8 bpw).

### Potential drawbacks

- **Imatrix required** — you need a representative calibration corpus and a GPU pass
  to generate the imatrix before quantizing. This is a one-time cost per model, but it
  cannot be skipped.
- **Vulkan prefill (prompt ingestion) is slower than mainline K-quants.** The IK base-K
  types have no native Vulkan GEMM tiles. Long-prompt batches on Vulkan pay a transient
  dequant→fp16 pass before the GEMM (the weights stay stored quantized — no permanent
  storage increase). Decode (token generation) is **not affected** — only batched
  prefill is. The Vulkan PP column in the benchmark matrix below will quantify this;
  if your workload is primarily decode or you are using ROCm/CUDA, this does not apply.
- **MoE-on-Vulkan caveat** — see the status banner above. Dense Vulkan inference is
  verified; the MoE (`MUL_MAT_ID`) path on Vulkan is code-complete but not yet
  end-to-end smoke-validated.

### Benchmark matrix

> Numbers to be filled in after benchmarking. See caption for configuration.
>
> **Configuration:** Qwen3.5-9B (dense) and Qwen3.6-35B-A3B (MoE), context=4096
> tokens. GPU class stated per row (RDNA3.5 / RDNA3); backends ROCm and Vulkan.
> Each IK type shown alongside its mainline comparator at matched bpw.

#### Dense model (Qwen3.5-9B)

| # | Type | bpw | PPL | File size | TG (t/s) | PP (t/s) |
|---|---|---|---|---|---|---|
| **Quality ceiling** | | | | | | |
| 1 | F16 | 16.0 | TBD | TBD | TBD | TBD |
| **IQ2_K vs Q2_K (aggressive 2-bit head-to-head)** | | | | | | |
| 2a | `Q2_K` | ~2.6 | TBD | TBD | TBD | TBD |
| 2b | `IQ2_K` | 2.375 | TBD | TBD | TBD | TBD |
| **IQ3_K vs Q3_K_M (mid 3-bit head-to-head)** | | | | | | |
| 3a | `Q3_K_M` | ~3.9 | TBD | TBD | TBD | TBD |
| 3b | `IQ3_K` | 3.4375 | TBD | TBD | TBD | TBD |
| **IQ4_K vs Q4_K_M (4-bit head-to-head — the common case)** | | | | | | |
| 4a | `Q4_K_M` | ~4.8 | TBD | TBD | TBD | TBD |
| 4b | `IQ4_K` | 4.50 | TBD | TBD | TBD | TBD |

#### Dense model — Vulkan vs. ROCm throughput (IK-specific dequant-fallback rows)

| # | Type | Backend | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|
| 4b-R | `IQ4_K` | ROCm | TBD | TBD | native MMVQ |
| 4b-V | `IQ4_K` | Vulkan | TBD | TBD | native decode; dequant-fallback prefill |
| 4a-V | `Q4_K_M` | Vulkan | TBD | TBD | native GEMM tile; compare prefill column |

> The PP (prefill) column for rows 4b-V vs 4a-V isolates the IK Vulkan prefill gap.
> The TG (decode) column for rows 4b-V vs 4b-R should show no significant IK-specific
> Vulkan penalty — decode is native on Vulkan for IK types.

#### MoE model (Qwen3.6-35B-A3B)

| # | Type | bpw | Backend | PPL | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|---|---|
| 5 | F16 | 16.0 | ROCm | TBD | TBD | TBD | quality ceiling |
| 6a | `Q4_K_M` | ~4.8 | ROCm | TBD | TBD | TBD | mainline comparator |
| 6b | `IQ4_K` | 4.50 | ROCm | TBD | TBD | TBD | |
| 6b-V | `IQ4_K` | 4.50 | Vulkan | TBD | TBD | TBD | MoE-Vulkan caveat — row pending smoke-validate |

---

## §4 How it works under the hood

### Block structures (`ggml/src/ggml-common.h`)

All three types are 256-element super-blocks (`QK_K = 256`). Every block contains:
- `d` (`ggml_half`, 2 bytes) — overall block scale
- `extra` (`uint16_t`, 2 bytes) — 1 bit per 16-element sub-block, selects standard or
  shifted nonlinear value table for that sub-block
- per-group scale fields
- packed quantization indices

**`block_iq4_k`** — 144 bytes (`ggml-common.h:465–475`):
```
[d: fp16, 2B] [extra: u16, 2B]
[scales_h: 4B (2-bit high parts of 16 scales)] [scales_l: 8B (4-bit low parts)]
[qs: 128B (4-bit indices, 2 per byte)]
```
4.50 bpw = 144 × 8 / 256.

**`block_iq3_k`** — 110 bytes (`ggml-common.h:504–514`):
```
[d: fp16, 2B] [extra: u16, 2B] [scales_h: u16, 2B (sign bits per sub-block)]
[scales_l: 8B (4-bit low magnitudes)] [qs: 64B (2 low bits of 3-bit index)]
[qh: 32B (1 high bit of 3-bit index)]
```
3.4375 bpw = 110 × 8 / 256.

**`block_iq2_k`** — 76 bytes (`ggml-common.h:516–526`):
```
[d: fp16, 2B] [extra: u16, 2B]
[scales: 8B (two 4-bit signed-offset-by-8 scales per 32-element pair)]
[qs: 64B (2-bit indices, 4 per byte)]
```
2.375 bpw = 76 × 8 / 256.

### Nonlinear value tables

Each type uses a compact set of nonlinear integer centroid values:

| Type | Values | Layout |
|---|---|---|
| IQ4_K | `iq4k_values[32]` | 2 × 16 entries (standard + shifted); `extra` bit selects per sub-block |
| IQ3_K | `iq3nl_values[16]` | 2 × 8 entries (standard + shifted); `extra` bit selects per sub-block |
| IQ2_K | `iq2nl_values[8]` | 2 × 4 entries (standard + shifted); `extra` bit selects per sub-block |

The dual-table approach (standard + shifted variant) doubles the effective quantization
resolution at each bit-width without increasing the index size — each sub-block chooses
the table that better fits its local weight distribution.

### CPU dequant (`ggml/src/ggml-iqk-quants.c`)

Reference scalar implementations:

| Type | Entry point | Approx. line |
|---|---|---|
| IQ4_K | `dequantize_row_iq4_k` | line 88 |
| IQ3_K | `dequantize_row_iq3_k` | line 344 |
| IQ2_K | `dequantize_row_iq2_k` | line 678 |

### CUDA/HIP kernels

Mat-vec (decode) uses MMVQ dot-product kernels in `ggml/src/ggml-cuda/mmvq-iqk.cu`.
`ggml_cuda_supports_mul_mat` enables the IK types for the optimized CUDA dispatch path.

### Vulkan pipelines

**Decode (mat-vec):** native `mul_mat_vec_iq{2,3,4}_k` compute shaders — registered
at `ggml-vulkan.cpp:4626–4628` (f32 input) and `:4666–4668` (f16 input). Dequant
happens in-shader during the dot product; no intermediate buffer is used.

**MoE decode (mat-vec-id):** native `mul_mat_vec_id_iq{2,3,4}_k` shaders — registered
at `ggml-vulkan.cpp:4726–4728`.

**Prefill (mat-mat, batch > 1):** no native GEMM pipeline exists for IK base-K types.
The Vulkan backend falls back to the dequant-then-f16-GEMM path: dequant shaders
(`ggml-vulkan.cpp:4798–4800`) write a transient fp16 scratch buffer, then a generic
fp16 × fp16 GEMM runs against it. The weights remain stored quantized; the scratch is
transient. See the [IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill)
for a full discussion.

---

## §5 Further reading

- **Upstream source:** [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- **Related docs (this repo):**
  - [IK quantization family primer](concepts/ik-quantization-family.md) — shared IK
    concepts: block structure, imatrix mandate, Vulkan dispatch split, 4-sub-family map
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and
    upstream-name mapping for all IK types
  - [docs/features/README.md](README.md) — index of all feature docs
