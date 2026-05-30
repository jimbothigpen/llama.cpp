# IK High-Bit-K Weight Quants (`IQ5_K` / `IQ6_K`)

> **Status: Stable** — CPU, CUDA/HIP, and Vulkan backends.

---

## At a glance

| CLI name | Enum | Slot | Effective bpw | Block bytes | Backends |
|---|---|---|---|---|---|
| `IQ5_K` | `GGML_TYPE_IQ5_K` | 140 | **5.50** | 176 | CPU, CUDA/HIP, Vulkan |
| `IQ6_K` | `GGML_TYPE_IQ6_K` | 141 | **6.625** | 212 | CPU, CUDA/HIP, Vulkan |

**TL;DR.** IQ5_K and IQ6_K are the high-quality end of the IK weight-quant family — 5- and
6-bit, imatrix-aware, 256-element super-block quants. The value proposition is
**near-lossless quality at fewer bits than Q8_0**: if you want PPL close to F16/Q8_0 but
need the model to fit in less VRAM, and you are willing to do a one-time imatrix calibration
pass, these are the types to consider.

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
llama-quantize --imatrix model.imatrix model-F16.gguf model-IQ6_K.gguf IQ6_K

# Step 3 — run inference
llama-server -m model-IQ6_K.gguf -fa on -ngl 99 --no-mmap
```

---

## §1 Provenance

IQ5_K and IQ6_K are ported from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), the source of the
entire IK weight-quant family. This fork adds ROCm and Vulkan parity: the CUDA dequant
kernels were adapted for the renumbered type IDs, and the Vulkan in-shader decode path and
dequant shaders were wired via the same dequant pipeline infrastructure used by the IK base-K
types. The upstream ik_llama source is CUDA-centric; the ROCm and Vulkan paths are this
fork's contribution.

**Type ID renumbering.** ik_llama uses type IDs 56/57 for IQ5_K/IQ6_K. This fork
renumbers them to 140/141 — within the ik_llama compatibility zone (slots 96–199) defined
in `docs/TYPE_ASSIGNMENTS.md`. GGUF files quantized with ik_llama will not load directly in
this fork without re-quantization.

**Structural family.** IQ5_K and IQ6_K are **global-scale K-quants**: each block carries a
single `ggml_half d` block scale, with no per-row floating-point meta-scale prefix. This
places them in the same structural family as the base-K types (IQ2_K / IQ3_K / IQ4_K), just
at higher bit-widths. See the
[IK quantization family primer](concepts/ik-quantization-family.md) for the four-sub-family
map and the shared K-quant design background.

---

## §2 Use in production

### Imatrix is mandatory

> **This is the most important user-facing fact about IQ5_K and IQ6_K.**

The quantizer hard-throws for every quantizable weight tensor if no imatrix is provided
(`src/llama-quant.cpp:793–794`):

```
ERROR: this quantization requires an importance matrix!
        - offending tensor: blk.0.attn_q.weight
        - target type: IQ6_K
```

Only the `token_embd` (embedding table) and `output` (logit projection) tensors are exempt;
all other weight tensors require imatrix data. There is no fallback — the quantizer exits
with an error.

See the [IK quantization family primer](concepts/ik-quantization-family.md#imatrix-is-mandatory)
for imatrix generation guidance, including corpus recommendations for MoE models.

### Quantize workflow

```bash
# Generate imatrix (adjust -c and --chunks to your available VRAM and time)
llama-imatrix \
    -m Qwen3.5-9B-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o Qwen3.5-9B.imatrix

# Quantize to IQ6_K
llama-quantize \
    --imatrix Qwen3.5-9B.imatrix \
    Qwen3.5-9B-F16.gguf \
    Qwen3.5-9B-IQ6_K.gguf \
    IQ6_K

# Or to IQ5_K
llama-quantize \
    --imatrix Qwen3.5-9B.imatrix \
    Qwen3.5-9B-F16.gguf \
    Qwen3.5-9B-IQ5_K.gguf \
    IQ5_K
```

This is an **offline, one-time step** per model. The resulting GGUF loads and runs at
inference time with no imatrix needed.

### Inference flags

No inference-time flags are specific to the IK high-bit-K types. Standard recommendations:

| Flag | Reason |
|---|---|
| `-fa on` | Flash attention; recommended for performance on supported models |
| `-ngl 99` | Offload all layers to GPU (adjust to your VRAM) |
| `--no-mmap` | Avoids mmap-related slowdowns; recommended on Linux + ROCm |

```bash
llama-server \
    -m model-IQ6_K.gguf \
    -fa on -ngl 99 --no-mmap \
    -c 8192
```

---

## §3 Benefits & potential drawbacks

### Benefits

- **Near-lossless quality below Q8_0's footprint.** IQ6_K at 6.625 bpw targets PPL close
  to F16/Q8_0 while occupying significantly less VRAM than Q8_0 (8.5 bpw). IQ5_K at
  5.50 bpw extends this further for tighter memory budgets.
- **Better quality at matched bit-width than mainline K-quants.** At comparable bpw, the
  IK imatrix-aware scale optimization typically delivers lower PPL than the mainline
  Q5_K_M / Q6_K. The benchmark matrix below will quantify the gap; see the honest framing
  note under Drawbacks.
- **Full decode speed on all backends.** Token generation uses native in-shader dequant on
  Vulkan (same bandwidth-preserving path as Q4_K_M and Q6_K), and native MMVQ kernels on
  CUDA/HIP. No decode penalty relative to mainline K-quants.
- **Smaller file than Q8_0.** IQ6_K at 6.625 bpw and IQ5_K at 5.50 bpw are meaningfully
  smaller than Q8_0 at 8.5 bpw.

### Potential drawbacks

- **Imatrix required** — a calibration corpus and a GPU pass to generate the imatrix are
  prerequisites. This is a one-time cost per model, but it cannot be skipped.
- **Diminishing returns at high bit-widths.** At 5.5–6.6 bpw the absolute quality gap to
  F16 is already small, so the relative win of the imatrix-tuned IK quants over the
  well-regarded mainline Q5_K_M / Q6_K may be modest. The benchmark matrix will
  quantify the actual gap; do not assume a large PPL advantage without measuring.
- **Vulkan prefill (prompt ingestion) is slower than mainline K-quants.** Like all IK types,
  IQ5_K and IQ6_K have no native Vulkan GEMM tiles. Long-prompt batches on Vulkan pay a
  transient dequant→fp16 pass before the GEMM. Decode (token generation) is **not
  affected**. See the
  [IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill)
  for the full discussion; the Vulkan PP column in the matrix below will quantify the
  prefill cost.

### Benchmark matrix

*TBD (pending benchmark)*

**Configuration:** Qwen3.5-9B (dense) and Qwen3.6-35B-A3B (MoE), context=4096 tokens.
GPU class stated per row (RDNA3.5 / RDNA3); backends ROCm and Vulkan.
The **Q8_0 anchor** is the relevant near-lossless reference for high-bit quants.

#### Dense model (Qwen3.5-9B)

| # | Type | bpw | PPL | File size | TG (t/s) | PP (t/s) |
|---|---|---|---|---|---|---|
| **Quality ceiling** | | | | | | |
| 1 | F16 | 16.0 | TBD | TBD | TBD | TBD |
| **Near-lossless anchor** | | | | | | |
| 2 | `Q8_0` | 8.5 | TBD | TBD | TBD | TBD |
| **6-bit head-to-head** | | | | | | |
| 3a | `Q6_K` | ~6.5625 | TBD | TBD | TBD | TBD |
| 3b | `IQ6_K` | 6.625 | TBD | TBD | TBD | TBD |
| **5-bit head-to-head** | | | | | | |
| 4a | `Q5_K_M` | ~5.5 | TBD | TBD | TBD | TBD |
| 4b | `IQ5_K` | 5.50 | TBD | TBD | TBD | TBD |

#### Dense model — Vulkan vs. ROCm throughput (IK-specific dequant-fallback rows)

| # | Type | Backend | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|
| 3b-R | `IQ6_K` | ROCm | TBD | TBD | native MMVQ |
| 3b-V | `IQ6_K` | Vulkan | TBD | TBD | native decode; dequant-fallback prefill |
| 3a-V | `Q6_K` | Vulkan | TBD | TBD | native GEMM tile; compare prefill column |

> The PP (prefill) column for rows 3b-V vs 3a-V isolates the IK Vulkan prefill gap.
> The TG (decode) column for rows 3b-V vs 3b-R should show no significant IK-specific
> Vulkan penalty — decode is native on Vulkan for IK types.

#### MoE model (Qwen3.6-35B-A3B)

| # | Type | bpw | Backend | PPL | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|---|---|
| 5 | F16 | 16.0 | ROCm | TBD | TBD | TBD | quality ceiling |
| 6 | `Q8_0` | 8.5 | ROCm | TBD | TBD | TBD | near-lossless anchor |
| 7a | `Q6_K` | ~6.5625 | ROCm | TBD | TBD | TBD | mainline comparator |
| 7b | `IQ6_K` | 6.625 | ROCm | TBD | TBD | TBD | |
| 8a | `Q5_K_M` | ~5.5 | ROCm | TBD | TBD | TBD | mainline comparator |
| 8b | `IQ5_K` | 5.50 | ROCm | TBD | TBD | TBD | |

---

## §4 How it works under the hood

### Block structures (`ggml/src/ggml-common.h`)

Both types are 256-element super-blocks (`QK_K = 256`). Every block contains a global scale
`d` (`ggml_half`, 2 bytes), a codebook-select `extra` field (`uint16_t`, 2 bytes, 1 bit per
16-element sub-block), per-group scale fields, and packed quantization indices.

**`block_iq5_k`** — 176 bytes (`ggml-common.h:481–489`):
```
[d: fp16, 2B] [extra: u16, 2B]
[scales_h: 4B (2-bit high parts of 16 scales)]
[scales_l: 8B (4-bit low parts → 6-bit composite scale per 16-element group)]
[qs: 128B (4 low bits of 5-bit index, 2 per byte)]
[qh: 32B (1 high bit per element, 8 per byte)]
```
5.50 bpw = 176 × 8 / 256.

**`block_iq6_k`** — 212 bytes (`ggml-common.h:495–502`):
```
[d: fp16, 2B] [extra: u16, 2B]
[scales: 16B (direct int8 scale per 16-element group)]
[qs: 128B (4 low bits of 6-bit index, 2 per byte)]
[qh: 64B (2 high bits per element, 4 per byte)]
```
6.625 bpw = 212 × 8 / 256.

The two types diverge in their scale layout: IQ5_K uses the same composite (2-bit high +
4-bit low = 6-bit) scale format as IQ4_K, while IQ6_K uses a wider direct `int8_t` per
16-element group — freeing budget for the extra high-bit storage in `qh`. Both use the `extra`
field's per-sub-block codebook-shift bit to double effective quantization resolution, the same
mechanism as all IK base-K types (see the
[IK family primer](concepts/ik-quantization-family.md) for background on the dual-table
approach).

### CPU dequant (`ggml/src/ggml-iqk-quants.c`)

Reference scalar implementations:

| Type | Entry point | Approx. line |
|---|---|---|
| IQ5_K | `dequantize_row_iq5_k` | line 1767 |
| IQ6_K | `dequantize_row_iq6_k` | line 2297 |

### CUDA/HIP kernels

Mat-vec (decode) uses MMVQ dot-product kernels in `ggml/src/ggml-cuda/mmvq-iqk.cu`.
`ggml_cuda_supports_mul_mat` enables the IK types for the optimized CUDA dispatch path.

### Vulkan pipelines

**Decode (mat-vec):** native `mul_mat_vec_iq5_k` / `mul_mat_vec_iq6_k` compute shaders —
registered at `ggml-vulkan.cpp:4630–4631` (f32 input) and `:4671–4672` (f16 input). Dequant
happens in-shader during the dot product; no intermediate buffer is used.

**MoE decode (mat-vec-id):** native `mul_mat_vec_id_iq5_k` / `mul_mat_vec_id_iq6_k` shaders
— registered at `ggml-vulkan.cpp:4731–4732`.

**Prefill (mat-mat, batch > 1):** no native GEMM pipeline exists for IK high-bit-K types.
The Vulkan backend falls back to the dequant-then-f16-GEMM path: dequant shaders
(`ggml-vulkan.cpp:4804–4805`) write a transient fp16 scratch buffer, then a generic
fp16 × fp16 GEMM runs against it. The weights remain stored quantized; the scratch is
transient. See the
[IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill)
for a full discussion.

---

## §5 Further reading

- **Upstream source:** [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- **Related docs (this repo):**
  - [IK quantization family primer](concepts/ik-quantization-family.md) — shared IK
    concepts: block structure, imatrix mandate, Vulkan dispatch split, four-sub-family map
  - [IK Base-K weight quants](ik-base-k.md) — the lower-bit siblings (IQ2_K / IQ3_K /
    IQ4_K) with the same structural family and Vulkan characteristic
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and
    upstream-name mapping for all IK types
  - [docs/features/README.md](README.md) — index of all feature docs
