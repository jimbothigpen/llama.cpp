# IK Row-Meta Weight Quants (`IQ4_KS` / `IQ3_KS` / `IQ4_KSS` / `IQ2_KL`)

> **Status: Stable** — CPU, CUDA/HIP, and Vulkan backends; all four types.

---

## At a glance

| CLI name | Enum | Slot | Effective bpw | Block bytes | Row-meta | Backends |
|---|---|---|---|---|---|---|
| `IQ2_KL` | `GGML_TYPE_IQ2_KL` | 157 | **2.6875** | 86 | 2-byte half-float | CPU, CUDA/HIP, Vulkan |
| `IQ3_KS` | `GGML_TYPE_IQ3_KS` | 156 | **3.1875** | 102 | 2-byte half-float | CPU, CUDA/HIP, Vulkan |
| `IQ4_KSS` | `GGML_TYPE_IQ4_KSS` | 146 | **4.0** | 128 | 4-byte float | CPU, CUDA/HIP, Vulkan |
| `IQ4_KS` | `GGML_TYPE_IQ4_KS` | 144 | **4.25** | 136 | 4-byte float | CPU, CUDA/HIP, Vulkan |

**TL;DR.** The KS/KL ("row-meta") members of the IK weight-quant family. Instead of
storing the quantization scale inside each 256-element block, these types prepend a
single per-row scale — called row metadata — before the block data, freeing block bytes
for tighter payload packing. The result is better perplexity than the mainline K-quants
at matched or smaller file sizes. A one-time imatrix calibration step is required before
quantizing.

**Pick `IQ4_KSS` for standard 4-bit work** — it shaves 0.25 bpw below `IQ4_KS` at
near-identical quality by packing scale bits directly into the quantization indices
(see [§3 near-twin callout](#iq4_ks-vs-iq4_kss-near-twin)). `IQ3_KS` and `IQ2_KL`
extend the ladder down to 3.2 and 2.7 bpw for aggressive compression.

**Quick start:**

```bash
# Step 1 — generate an importance matrix on a calibration corpus
llama-imatrix \
    -m model-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o model.imatrix

# Step 2 — quantize with the imatrix (substitute IQ3_KS / IQ4_KS / IQ2_KL as needed)
llama-quantize --imatrix model.imatrix model-F16.gguf model-IQ4_KSS.gguf IQ4_KSS

# Step 3 — run inference (same flags as any GGUF)
llama-server -m model-IQ4_KSS.gguf -fa on -ngl 99 --no-mmap
```

---

## §1 Provenance

IQ4_KS, IQ3_KS, IQ4_KSS, and IQ2_KL are ported from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp), the source of the
entire IK weight-quant family. This fork adds ROCm and Vulkan parity on top of the
upstream CUDA-centric implementation.

### Differences from upstream

**Type ID renumbering.** ik_llama uses different type IDs for these slots. This fork
places all four within the ik_llama compatibility zone (slots 96–199) defined in
`docs/TYPE_ASSIGNMENTS.md`. GGUF files quantized with ik_llama will not load directly
in this fork without re-quantization; the GGUF type fields reflect the renumbered IDs.

**CUDA dequant hardening.** The original KS CUDA dequant kernels left some elements
uninitialised per 256-element block, causing NaN propagation through matrix
multiplication. This fork ships a corrected full-coverage implementation. Perplexity
parity with the ik_llama reference (Δ < 0.5%) was confirmed across CPU, ROCm, and
Vulkan before release.

**Imatrix mandate.** All four types are imatrix-aware; the imatrix is a hard requirement
(not optional) for quantizing any weight tensor other than `token_embd` and `output`.

---

## §2 Use in production

### Imatrix is mandatory

> **This is the most important user-facing fact about the row-meta types.**

The quantizer hard-throws for every quantizable weight tensor if no imatrix is provided:

```
ERROR: this quantization requires an importance matrix!
        - offending tensor: blk.0.attn_q.weight
        - target type: IQ4_KSS
```

Only `token_embd` (embedding table) and `output` (logit projection) are exempt; all
other weight tensors require imatrix data. There is no fallback mode — the quantizer
exits with an error.

See the [IK quantization family primer](concepts/ik-quantization-family.md) for imatrix
generation guidance and corpus recommendations for MoE models.

### Quantize workflow

```bash
# Generate imatrix (adjust -c and --chunks to your available VRAM and time)
llama-imatrix \
    -m Qwen3.5-9B-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o Qwen3.5-9B.imatrix

# Quantize to IQ4_KSS
llama-quantize \
    --imatrix Qwen3.5-9B.imatrix \
    Qwen3.5-9B-F16.gguf \
    Qwen3.5-9B-IQ4_KSS.gguf \
    IQ4_KSS
```

This is an **offline, one-time step** — the resulting GGUF loads and runs at inference
time with no imatrix needed.

### Inference flags

No inference-time flags are specific to the IK row-meta types. Standard recommendations:

| Flag | Reason |
|---|---|
| `-fa on` | Flash attention; recommended for performance on supported models |
| `-ngl 99` | Offload all layers to GPU (adjust to your VRAM) |
| `--no-mmap` | Avoids mmap-related slowdowns; recommended on Linux + ROCm |

```bash
llama-server \
    -m model-IQ4_KSS.gguf \
    -fa on -ngl 99 --no-mmap \
    -c 8192
```

---

## §3 Benefits & potential drawbacks

### Benefits

- **More quality per bit than in-block-scale alternatives.** Moving the scale out of
  the block and into a per-row prefix frees block bytes for payload, letting KS types
  target better perplexity than the mainline K-quants at the same or smaller file size.
  The benchmark matrix below will quantify this once numbers are collected.
- **Full decode speed on all backends.** Token generation uses native in-shader dequant
  on Vulkan (same bandwidth-preserving path as Q4_K), and native MMVQ kernels on
  CUDA/HIP. No decode penalty relative to mainline K-quants.
- **Four bpw points in one family.** 2.6875 → 3.1875 → 4.0 → 4.25 lets you dial
  size-vs-quality precisely, including the convenient 0.25 bpw step between IQ4_KSS and
  IQ4_KS.

### IQ4_KS vs IQ4_KSS near-twin

Both types sit at ~4 bits per weight and share the same per-row float scale, but differ
by 0.25 bpw:

- **`IQ4_KS`** stores an explicit `scales[8]` byte array inside each block (136 B/block,
  4.25 bpw). Straightforward decode.
- **`IQ4_KSS`** ("super-squeezed") **Gray-code-packs the scale bits directly into the
  `uint32 qs[32]` words** (128 B/block, 4.0 bpw) — no separate scale bytes exist in
  the block struct. The 0.25 bpw savings comes at the cost of a slightly more involved
  decode.

**Practical guidance:** prefer `IQ4_KSS` for the smaller file at near-identical quality.
Choose `IQ4_KS` if you need the marginally simpler decode path or have a reason to stay
at exactly 4.25 bpw. The benchmark matrix will quantify any PPL difference.

### Potential drawbacks

- **Imatrix required** — a calibration corpus and a GPU pass to generate the imatrix.
  One-time cost per model; cannot be skipped.
- **Vulkan prefill slower than mainline K-quants.** The IK row-meta types share the
  family-wide Vulkan characteristic: decode (single-token mat-vec) is native full-speed;
  long-prompt prefill on Vulkan pays a transient dequant→fp16 pass before the GEMM. ROCm
  and CUDA prefill are unaffected. See the
  [IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill)
  for a full discussion. The Vulkan PP column in the matrix below will quantify the gap.
- **IQ4_KSS decode complexity.** The Gray-code scale-unpacking step is more work per
  element than IQ4_KS. Usually negligible on decode (memory-bound), but noted for
  completeness.

### Benchmark matrix

*TBD (pending benchmark)*

**Configuration:** Qwen3.5-9B (dense) and Qwen3.6-35B-A3B (MoE), context=4096 tokens.
GPU class stated per row (RDNA3.5 / RDNA3); backends ROCm and Vulkan.
Each IK type shown alongside its mainline comparator at matched bpw.

#### Dense model (Qwen3.5-9B)

| # | Type | bpw | PPL | File size | TG (t/s) | PP (t/s) |
|---|---|---|---|---|---|---|
| **Quality ceiling** | | | | | | |
| 1 | F16 | 16.0 | TBD | TBD | TBD | TBD |
| **IQ2_KL vs Q2_K (aggressive 2-bit head-to-head)** | | | | | | |
| 2a | `Q2_K` | ~2.63 | TBD | TBD | TBD | TBD |
| 2b | `IQ2_KL` | 2.6875 | TBD | TBD | TBD | TBD |
| **IQ3_KS vs IQ3_S / Q3_K_M (3-bit head-to-head)** | | | | | | |
| 3a | `IQ3_S` / `Q3_K_M` | ~3.4 | TBD | TBD | TBD | TBD |
| 3b | `IQ3_KS` | 3.1875 | TBD | TBD | TBD | TBD |
| **IQ4_KSS vs IQ4_XS (squeezed 4-bit head-to-head)** | | | | | | |
| 4a | `IQ4_XS` | ~4.25 | TBD | TBD | TBD | TBD |
| 4b | `IQ4_KSS` | 4.0 | TBD | TBD | TBD | TBD |
| **IQ4_KS vs Q4_K_M (standard 4-bit head-to-head)** | | | | | | |
| 5a | `Q4_K_M` | ~4.5 | TBD | TBD | TBD | TBD |
| 5b | `IQ4_KS` | 4.25 | TBD | TBD | TBD | TBD |

#### Dense model — Vulkan vs. ROCm throughput (IK-specific dequant-fallback rows)

| # | Type | Backend | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|
| 5b-R | `IQ4_KS` | ROCm | TBD | TBD | native MMVQ |
| 5b-V | `IQ4_KS` | Vulkan | TBD | TBD | native decode; dequant-fallback prefill |
| 5a-V | `Q4_K_M` | Vulkan | TBD | TBD | native GEMM tile; compare prefill column |

> The PP (prefill) column for rows 5b-V vs 5a-V isolates the IK Vulkan prefill gap.
> The TG (decode) column for rows 5b-V vs 5b-R should show no significant IK-specific
> Vulkan penalty — decode is native on Vulkan for IK types.

#### MoE model (Qwen3.6-35B-A3B)

| # | Type | bpw | Backend | PPL | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|---|---|
| 6 | F16 | 16.0 | ROCm | TBD | TBD | TBD | quality ceiling |
| 7a | `Q4_K_M` | ~4.5 | ROCm | TBD | TBD | TBD | mainline comparator |
| 7b | `IQ4_KS` | 4.25 | ROCm | TBD | TBD | TBD | |
| 7b-V | `IQ4_KS` | 4.25 | Vulkan | TBD | TBD | TBD | Vulkan prefill-fallback row |

---

## §4 How it works under the hood

### The row-meta concept

Every KS/KL type stores a **per-row scale prefix** — called `row_meta_size` bytes —
immediately before each row's block data on disk and in memory. This per-row scale
captures the overall magnitude of each weight row so the in-block data can be packed
more tightly. The block structs reflect this: they carry no overall block scale field;
the scale lives in the row prefix instead.

The infrastructure to handle the prefix is the per-type `row_meta_size` field in
`ggml.c` type traits:

| Type | `row_meta_size` | Precision |
|---|---|---|
| `IQ4_KS` | 4 bytes | `float` |
| `IQ4_KSS` | 4 bytes | `float` |
| `IQ3_KS` | 2 bytes | `ggml_half` |
| `IQ2_KL` | 2 bytes | `ggml_half` |

All buffer-sizing calls must use `ggml_nbytes()` or `ggml_row_size()` rather than
`type_size × ne` — the per-row prefix bytes are included by those functions but not
by a naive element-count calculation.

### Block structures (`ggml/src/ggml-common.h`)

All four types use 256-element super-blocks (`QK_K = 256`). The block struct is the
per-block payload **excluding** the row-meta prefix (the prefix is stored before the
first block in each row and amortised over all blocks in that row).

**`block_iq4_ks`** — 136 bytes (`ggml-common.h:532–536`):
```
[scales[8]: 8B — 1 codebook-shift bit + 7-bit signed-offset scale per sub-block]
[qs[128]: 128B — 4-bit indices, 2 per byte]
```
Row prefix: 4-byte `float` scale.  
4.25 bpw = 136 × 8 / 256.

**`block_iq4_kss`** — 128 bytes (`ggml-common.h:561–564`):
```
[qs[32]: 128B — 32 uint32_t words; 4-bit indices + Gray-code-packed scale bits]
```
No separate scale bytes — scales are packed into the `qs` words.  
Row prefix: 4-byte `float` scale.  
4.0 bpw = 128 × 8 / 256.

**`block_iq3_ks`** — 102 bytes (`ggml-common.h:569–575`):
```
[extra: u16, 2B] [scales[4]: 4B] [qs[64]: 64B] [qh[32]: 32B]
```
Row prefix: 2-byte `ggml_half` scale.  
3.1875 bpw = 102 × 8 / 256.

**`block_iq2_kl`** — 86 bytes (`ggml-common.h:580–587`):
```
[scales_h: u16, 2B] [scales_l[4]: 4B] [qs[64]: 64B] [qh[16]: 16B]
```
Row prefix: 2-byte `ggml_half` scale.  
2.6875 bpw = 86 × 8 / 256.

### CPU dequant (`ggml/src/ggml-iqk-quants.c`)

Each dequant function reads the row-meta prefix first, then processes the blocks:

| Type | Entry point | Line | Row-meta read |
|---|---|---|---|
| IQ4_KS | `dequantize_row_iq4_ks` | 913 | `const float *dptr → d` |
| IQ3_KS | `dequantize_row_iq3_ks` | 1152 | `GGML_FP16_TO_FP32` |
| IQ4_KSS | `dequantize_row_iq4_kss` | 1452 | `const float *dptr → d` |
| IQ2_KL | `dequantize_row_iq2_kl` | 2041 | `GGML_FP16_TO_FP32` |

### CUDA/HIP kernels

Mat-vec (decode) uses MMVQ dot-product kernels in `ggml/src/ggml-cuda/mmvq-iqk.cu`.
Mat-mat (batch) uses per-type dequant kernels in `ggml/src/ggml-cuda/convert.cu`.

### Vulkan pipelines

**Decode (mat-vec):** native `mul_mat_vec_iq{4_ks,3_ks,4_kss,2_kl}` compute shaders —
in-shader dequant including row-meta read; no intermediate buffer. Full-speed decode on
all Vulkan-capable hardware.

**Prefill (mat-mat, batch > 1):** the IK row-meta types share the family-wide
dequant-fallback path. An `is_empty()` guard in the Vulkan backend
(`ggml-vulkan.cpp:6895–6897`) routes the batch dispatch through a transient
dequant→fp16 scratch buffer + generic fp16 GEMM. Weights remain stored quantized;
decode is unaffected. See the
[IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill)
for the full explanation.

---

## §5 Further reading

- **Upstream source:** [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- **Related docs (this repo):**
  - [IK quantization family primer](concepts/ik-quantization-family.md) — shared IK
    concepts: block structure, imatrix mandate, Vulkan dispatch split, four-sub-family map
  - [IK Base-K weight quants](ik-base-k.md) — `IQ2_K`, `IQ3_K`, `IQ4_K` (in-block
    scale; the global-scale counterpart to this doc)
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and
    upstream-name mapping for all IK types
  - [docs/features/README.md](README.md) — index of all feature docs
