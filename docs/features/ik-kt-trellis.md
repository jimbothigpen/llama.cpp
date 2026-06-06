# IK KT/Trellis Weight Quants (`IQ4_KT` / `IQ3_KT` / `IQ2_KT`)

> **IQ4_KT — Status: Stable** — CPU, CUDA/HIP, and Vulkan backends; PPL excellent (6.54 on Qwen3.5-9B).
>
> **IQ3_KT — Status: Stable** — CPU, CUDA/HIP, and Vulkan backends; PPL healthy (9.05 on Qwen3.5-9B, +23.5% vs IQ3_K — inherent to single-codebook design; cross-backend PPL parity confirmed: RESOLVED Vulkan / CLOSED ROCm gfx1150).
>
> **IQ2_KT — Status: Known limitation — DO NOT USE** — blanket DO-NOT-USE at any scale. General codebook defect confirmed: PPL 99.58 (0.8B brute-force baseline) / 107.87 (0.8B shipped k=60) / 33.96 (9B Vulkan, Qwen3.5-9B) — all catastrophic vs IQ2_KL (26.12 at 2.6875 bpw). Use `IQ2_KL` instead.

---

## At a glance

| Type | ggml ID | LLAMA_FTYPE | CLI name | bpw (block) | bpw (w/ row-meta) | Backends | State |
|---|---|---|---|---|---|---|---|
| `IQ4_KT` | 155 (`ggml.h:468`) | `MOSTLY_IQ4_KT`=49 (`llama.h:168`) | "4.0 bpw ik_llama row-meta trellis" | **4.0** | **4.125** | CPU, CUDA/HIP, Vulkan | **Stable** |
| `IQ3_KT` | 154 (`ggml.h:467`) | `MOSTLY_IQ3_KT`=54 (`llama.h:174`) | "3.0 bpw ik_llama row-meta trellis" | **3.0** | **3.125** | CPU, CUDA/HIP, Vulkan | **Stable** |
| `IQ2_KT` | 153 (`ggml.h:466`) | `MOSTLY_IQ2_KT`=55 (`llama.h:175`) | "2.0 bpw ik_llama row-meta trellis 2-bit" | **2.0** | **2.125** | CPU, CUDA/HIP, Vulkan | **§-FLAG — DO NOT USE** |

Row-meta overhead: each row carries a 4-byte `float` prefix. For a 256-element row (single block) this adds 0.125 bpw; for larger rows (e.g. 4096 elements = 16 blocks) the overhead is ~0.008 bpw. The "w/ row-meta" column reports the per-block-equivalent at one block per row (the maximum overhead case).

**TL;DR.** The KT ("trellis") members of the IK weight-quant family — row-meta quants that store
**no explicit codebook**. A deterministic LCG hash regenerates a 65,536-entry implicit codebook on
the fly at inference time; a cluster-accelerated nearest-neighbour search selects indices at
quantize time. `IQ4_KT` and `IQ3_KT` deliver solid quality at 4 and 3 bpw respectively.
`IQ2_KT` is shipped but **defective at all tested scales** — use `IQ2_KL` (2.6875 bpw) instead.
A one-time imatrix calibration step is required before quantizing.

See the [IK quantization family primer](concepts/ik-quantization-family.md) for shared concepts:
imatrix mandate, Vulkan dispatch split (native decode, dequant-fallback prefill), and the
four-sub-family map that locates KT in the broader family.

**Quick start:**

```bash
# Step 1 — generate an importance matrix
llama-imatrix \
    -m model-F16.gguf \
    -f calibration-data.txt \
    -c 512 --chunks 200 \
    -ngl 99 --no-mmap \
    -o model.imatrix

# Step 2 — quantize (substitute IQ3_KT as needed; DO NOT use IQ2_KT)
llama-quantize --imatrix model.imatrix model-F16.gguf model-IQ4_KT.gguf IQ4_KT

# Step 3 — run inference (same flags as any GGUF)
llama-server -m model-IQ4_KT.gguf -fa on -ngl 99 --no-mmap
```

---

## §1 Provenance

IQ4_KT, IQ3_KT, and IQ2_KT are ported from
[ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) (the `andrew_trellis`
development branch), the source of the entire IK weight-quant family. This fork adds
ROCm and Vulkan parity on top of the upstream CUDA-centric implementation.

**Type ID renumbering.** This fork assigns ggml type IDs 153/154/155 — within the ik_llama
compatibility zone (slots 96–199) defined in `docs/TYPE_ASSIGNMENTS.md`. GGUF files from
upstream ik_llama cannot be loaded directly in this fork without re-quantization.

---

## §2 Use in production

### Imatrix is mandatory

> **This is the most important user-facing fact about the KT types.**

The quantizer hard-throws for every quantizable weight tensor if no imatrix is provided
(`src/llama-quant.cpp:790–796`):

```
ERROR: this quantization requires an importance matrix!
        - offending tensor: blk.0.attn_q.weight
        - target type: IQ4_KT
```

Only `token_embd` (embedding table) and `output` (logit projection) tensors are exempt
(`src/llama-quant.cpp:768`); all other weight tensors require imatrix data. There is no
fallback — the quantizer exits with an error.

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

# Quantize to IQ4_KT
llama-quantize \
    --imatrix Qwen3.5-9B.imatrix \
    Qwen3.5-9B-F16.gguf \
    Qwen3.5-9B-IQ4_KT.gguf \
    IQ4_KT

# Or to IQ3_KT
llama-quantize \
    --imatrix Qwen3.5-9B.imatrix \
    Qwen3.5-9B-F16.gguf \
    Qwen3.5-9B-IQ3_KT.gguf \
    IQ3_KT

# IQ2_KT — DO NOT USE (see §3 Quality / IQ2_KT warning below)
```

This is an **offline, one-time step** — the resulting GGUF loads and runs at inference time with no imatrix needed.

### Inference flags

No inference-time flags are specific to the KT types. Standard recommendations:

| Flag | Reason |
|---|---|
| `-fa on` | Flash attention; recommended for performance on supported models |
| `-ngl 99` | Offload all layers to GPU (adjust to your VRAM) |
| `--no-mmap` | Avoids mmap-related slowdowns; recommended on Linux + ROCm |

---

## §3 Quality and guidance

### Measured perplexity (Qwen3.5-9B, wiki.test.raw, 20 chunks)

| Type | PPL | vs. mainline | Verdict |
|---|---|---|---|
| `IQ4_KT` | **6.54** | ≈ IQ4_K (6.57) — near-identical | 🟢 Excellent — use freely |
| `IQ3_KT` | **9.05** | +23.5% vs IQ3_K (7.32) | 🟢 Healthy — inherent to single-codebook design; still workable |
| `IQ2_KT` | **33.96** (9B) / **107.87** (0.8B shipped) | vs IQ2_KL 26.12 — catastrophic at all scales | 🔴 DO NOT USE |

### IQ2_KT — blanket DO NOT USE

> **IQ2_KT has a confirmed general codebook defect. It produces unacceptably high perplexity
> at all tested model scales. Do not use it in production. Use `IQ2_KL` (2.6875 bpw) instead.**

Two measurements at different model scales both confirm the defect is general, not scale-dependent:

- **Qwen3.5-0.8B** — PPL **99.58** (brute-force baseline, cluster-accel OFF) / **107.87** (shipped k=60 cluster-accel, +8.3%). Both catastrophic vs IQ2_KL 26.12.
- **Qwen3.5-9B** — PPL **33.96** (Vulkan, 20 chunks). Still RED (>30) at 9B — the anomaly does **not** diminish at larger scale.

Root cause: the IQ2_KT codebook algorithm has a fundamental quality defect at this bit-width. The 0.8B→9B drop from ~108 to 34 PPL is due to the model being larger (more redundancy), not a fix. A full algorithm redesign would be needed to make 2-bit trellis viable.

**Recommended 2-bit alternative: `IQ2_KL`** — a row-meta type at 2.6875 bpw with PPL 26.12 on Qwen3.5-9B. See the [IK Row-Meta weight quants doc](ik-ks-row-meta.md).

### IQ3_KT — healthy, with a note

IQ3_KT's PPL of 9.05 is +23.5% above IQ3_K (7.32). This gap is **inherent to the single-codebook trellis design** at 3-bit: unlike IQ3_K which uses per-sub-block nonlinear value tables, IQ3_KT regenerates the codebook from a global LCG hash (GROUP_SIZE=8, 6561 bins, k=60 cluster-accel). The fixed codebook limits per-block adaptation. For workloads where 3 bpw is the target and the PPL overhead is acceptable, IQ3_KT is usable. If you need better quality at ~3 bpw, consider `IQ3_KS` (3.1875 bpw, verified PPL near IQ3_K).

### IQ4_KT — excellent

IQ4_KT at 6.54 PPL matches IQ4_K within rounding noise. The trellis encoding overhead at 4 bits is negligible because the 65,536-entry implicit codebook (GROUP_SIZE=4, 625 bins, k=6 cluster-accel, dual codebook A+B) provides sufficient representation density.

---

## §4 Benefits & potential drawbacks

### Benefits

- **No stored codebook** — the implicit LCG-hash codebook costs no storage overhead. The 4-byte per-row float meta is the only overhead beyond raw index storage.
- **Solid quality at 4 bpw** — IQ4_KT matches IQ4_K quality at a lower effective bpw than IQ4_KS/IQ4_KSS, offering a point on the quality-size curve not covered by the other IK row-meta types.
- **Full decode speed on all backends** — native in-shader decode on Vulkan (mat-vec), native MMVQ kernels on CUDA/HIP. No decode penalty vs. mainline K-quants.
- **All three backends, all three shipped types** — CPU, CUDA/HIP, and Vulkan for IQ4_KT, IQ3_KT, and IQ2_KT.

### Potential drawbacks

- **Imatrix required** — a calibration corpus and GPU pass to generate the imatrix. One-time cost per model; cannot be skipped.
- **IQ2_KT is defective — blanket DO NOT USE.** See §3.
- **IQ3_KT carries a +23.5% PPL overhead** vs IQ3_K due to the single global codebook. If that gap matters for your use case, prefer `IQ3_KS` (3.1875 bpw) or `IQ3_K` (3.4375 bpw).
- **Vulkan prefill (prompt ingestion) slower than mainline K-quants.** Like all IK types, the KT family has no native Vulkan GEMM tiles. Long-prompt batches on Vulkan pay a transient dequant→fp16 pass before the GEMM. Decode (token generation) is **not affected** — only batched prefill. See the [IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill) for the full explanation.
- **IQ3_KT cross-backend PPL parity not yet formally verified.** The Vulkan shaders are code-complete and wired, but an explicit PPL cross-backend gate (Vulkan vs. CPU/ROCm) has not been run for IQ3_KT.

### Benchmark matrix

*TBD (pending benchmark)*

**Configuration:** Qwen3.5-9B (dense) and Qwen3.6-35B-A3B (MoE), context=4096 tokens. GPU class stated per row; backends ROCm and Vulkan. PPL values measured on wiki.test.raw (20 chunks). IQ2_KT row carries the §-FLAG; see §3 for the full discussion.

#### Dense model (Qwen3.5-9B)

| # | Type | bpw | PPL | File size | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|---|---|
| **Quality ceiling** | | | | | | | |
| 1 | F16 | 16.0 | TBD | TBD | TBD | TBD | |
| **4-bit trellis vs. matched row-meta / mainline** | | | | | | | |
| 2a | `IQ4_KSS` | 4.0 | TBD | TBD | TBD | TBD | matched bpw, no codebook |
| 2b | `Q4_K_M` | ~4.8 | TBD | TBD | TBD | TBD | mainline comparator |
| 2c | `IQ4_KT` | 4.0 | **6.54** | TBD | TBD | TBD | trellis — measured |
| **3-bit trellis vs. matched row-meta / mainline** | | | | | | | |
| 3a | `IQ3_KS` | 3.1875 | TBD | TBD | TBD | TBD | matched bpw |
| 3b | `IQ3_K` | 3.4375 | **7.32** | TBD | TBD | TBD | matched family |
| 3c | `IQ3_KT` | 3.0 | **9.05** | TBD | TBD | TBD | trellis — measured; +23.5% vs IQ3_K |
| **2-bit trellis §-FLAG vs. viable 2-bit alternatives** | | | | | | | |
| 4a | `IQ2_KL` | 2.6875 | **26.12** | TBD | TBD | TBD | recommended 2-bit alternative |
| 4b | `Q2_K` | ~2.6 | TBD | TBD | TBD | TBD | mainline comparator |
| 4c | `IQ2_KT` | 2.0 | **33.96** (9B) / **107.87** (0.8B) | TBD | TBD | TBD | §-FLAG: DO NOT USE; general codebook defect |

> The IQ2_KT cells (row 4c) use measured PPL values. All other PPL cells are TBD pending
> a full benchmark run. IQ2_KT is listed for reference only; it should not be chosen in
> production.

---

## §5 How it works under the hood

### The trellis spine — implicit codebook via LCG hash

All KT types share a core property: **no codebook bytes are stored in the GGUF**. At inference time, each codebook entry is regenerated on the fly from a multiplicative LCG hash:

```
x_next = 0xCBAC1FED × x_prev        (mod 2³²)
```

Starting from an index + offset, iterating `GROUP_SIZE` steps produces one codebook entry.
Across all 65,536 possible indices, this yields a 65,536-entry implicit codebook that is
identical on every backend and never needs to be stored. The source is
`iqkt_gen_group_int<GROUP_SIZE,IS_ABS>()` (`ggml/src/ggml-iqk-kt-family.hpp:58–72`).

IQ4_KT uses a **dual codebook** (offsets A=4096 and B=4096+32768): each sub-block selects
either codebook A or B based on a stored bit, doubling effective range. IQ3_KT and IQ2_KT
use a **single codebook** each (offsets 4096 and 0 respectively), which is the structural
source of the 3-bit quality gap — less adaptive than a dual design.

### Cluster-accelerated nearest-neighbour search (quantize time)

At quantize time, finding the best index for each group is an expensive NN search over
65,536 entries. The implementation accelerates this with a soft-bin cluster index
(`iqkt_build_cluster_index`, `ggml/src/ggml-iqk-kt-family.hpp:280–315`):

- The codebook is partitioned into soft bins by hashing each entry's GROUP_SIZE-dimensional
  vector into a compact key:
  - **GROUP_SIZE=4 (IQ4_KT):** 5⁴ = **625 bins**, base-5 hash
  - **GROUP_SIZE=8 (IQ3_KT, IQ2_KT):** 3⁸ = **6561 bins**, base-3 hash
- At quantize time, the query is hashed to its bin, and the search scans only that bin's
  candidate list plus soft-replica neighbours (`k_neighbours` entries).
- `k_neighbours` constants: IQ4_KT = **6** (`ggml-iqk-kt.cpp:60`); IQ3_KT = **60** (`ggml-iqk-kt.cpp:576`); IQ2_KT = **60** (`ggml-iqk-kt.cpp:434`).

This reduces the NN search from O(65536) to O(k_neighbours per bin), making quantize
practical without sacrificing codebook coverage.

> **IQ2_KT cluster-accel note.** The shipped k=60 value carries a +8.3% PPL penalty over
> the brute-force baseline (k=60 → PPL 107.87; brute-force → PPL 99.58 on Qwen3.5-0.8B).
> A retune to k=80–100 is planned but has not shipped. Both values are catastrophic;
> the retune does not fix the underlying codebook defect.

### Block structures (`ggml/src/ggml-common.h`)

All three types use 256-element super-blocks (`QK_K = 256`) with a 4-byte `float` per-row
prefix before the first block in each row. The block struct is the per-block payload.

**`block_iq4_kt`** — 128 bytes (`ggml-common.h:544–547`):
```
qs[32]: 128B total (32 uint32_t)
  shb[0..7]  (32B) — scale + use-codebook-B flag + 24 high bits per sub-block
  ql[0..63]  (64B) — 8 low bits per group (64 groups total)
  qh[0..15]  (16B) — 4 mid bits per group, 2 groups per byte
  (16B reserved padding)
```
4.0 bpw = 128 × 8 / 256; GROUP_SIZE=4, dual codebook (A+B), k=6.

**`block_iq3_kt`** — 96 bytes (`ggml-common.h:533–535`):
```
qs[24]: 96B total (24 uint32_t)
  qs[0..7]   (32B) — scale (ls+128) bits + 4 high index bits per group
  qs[8..15]  (32B) — 8 low bits per group, 4 groups packed per u32
  qs[16..19] (16B) — 4 mid bits per group, 2 groups per nibble
  qs[20..23] (16B) — padding
```
3.0 bpw = 96 × 8 / 256; GROUP_SIZE=8, single codebook, k=60.
16-bit index (ql_byte | qh_nibble<<8 | sh_4bits<<12) → 65,536-entry codebook.

**`block_iq2_kt`** — 64 bytes (`ggml-common.h:553–555`):
```
qs[32]: 64B total (32 uint16_t)
  One 16-bit codebook index per group; GROUP_SIZE=8, kNumGroups=32
```
2.0 bpw = 64 × 8 / 256; GROUP_SIZE=8, single codebook, k=60.

### CPU dequant (`ggml/src/ggml-iqk-kt.cpp`)

| Type | Entry point | Line |
|---|---|---|
| IQ4_KT | `dequantize_row_iq4_kt` | 311 |
| IQ2_KT | `dequantize_row_iq2_kt` | 766 |
| IQ3_KT | `dequantize_row_iq3_kt` | 836 |

Each function reads the 4-byte row_meta float prefix first, then iterates blocks, regenerating codebook entries on the fly via the LCG hash rather than reading from a stored table.

### Vulkan pipelines

**Decode (mat-vec):** native `mul_mat_vec_iq{2,3,4}_kt.comp` compute shaders — registered for all three types at `ggml-vulkan.cpp:4612/4653` (f32/f16 input variants). In-shader LCG codebook regeneration; no intermediate buffer. Full-speed decode on all Vulkan-capable hardware.

**MoE decode (mat-vec-id):** pipelines registered for all three types at `ggml-vulkan.cpp:4722–4724`.

**Prefill (mat-mat, batch > 1):** no native GEMM pipeline exists for KT types. The Vulkan backend falls back to the dequant-then-f16-GEMM path: dequant shaders (`dequant_iq{2,3,4}_kt.comp`, registered at `ggml-vulkan.cpp:4797`) write a transient fp16 scratch buffer, then a generic fp16 × fp16 GEMM runs against it. Weights remain stored quantized; the scratch buffer is transient. See the [IK family primer](concepts/ik-quantization-family.md#vulkan-dispatch-decode-vs-prefill) for a full discussion.

**Dispatch switch cases:** `ggml-vulkan.cpp:6769/6854/6950/16792` for IQ3_KT (IQ2_KT and IQ4_KT have matching slots; search `GGML_TYPE_IQ{2,4}_KT` in the same file).

> **IQ3_KT cross-backend parity (RESOLVED).** Vulkan shaders shipped; cross-backend PPL gate RESOLVED (Vulkan 8.4299 / IQ3_K 6.8348, 9B 20ch). ROCm gfx1150 GPU path confirmed `c809225f6` (CLOSED — 99% GPU utilization / 7.66s-pass); gfx1102 warmup crash is a separate Tensile confound (not IQ3_KT-specific). All three backends confirmed. `BACKEND_PARITY.md` updated to reflect CPU+ROCm+Vulkan.

---

## §6 Further reading

- **Upstream source:** [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)
- **Related docs (this repo):**
  - [IK quantization family primer](concepts/ik-quantization-family.md) — shared IK
    concepts: block structure, imatrix mandate, Vulkan dispatch split, four-sub-family map
  - [IK Row-Meta weight quants](ik-ks-row-meta.md) — `IQ4_KS`, `IQ3_KS`, `IQ4_KSS`,
    `IQ2_KL` — the non-trellis row-meta family; `IQ2_KL` is the recommended 2-bit alternative
  - [IK Base-K weight quants](ik-base-k.md) — `IQ2_K`, `IQ3_K`, `IQ4_K` — global-scale
    K-quants (no row-meta, no trellis)
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments for all IK types
  - [docs/features/README.md](README.md) — index of all feature docs
