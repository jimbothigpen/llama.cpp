# WHT Weight Quantization (`WHT3_0` / `WHT4_0` / `WHT5_0` / `WHT6_0` / `WHT8_0`)

> **Status: Stable (WHT3_0/WHT4_0)** — CPU, CUDA/HIP, and Vulkan backends. WHT3_0/WHT4_0 do **not** use imatrix — calibration-free.
>
> **Status: Functional + fused decode (WHT5_0/WHT6_0/WHT8_0, 2026-06-22)** — wider Lloyd-Max codebooks (32/64/256 levels) at index bit-widths 5/6/8. CPU (quant + dequant + vec_dot) and CUDA/HIP are wired. As of 2026-06-22 single-token decode uses a **fused mmvq vec_dot kernel** (native, decode-then-MAC in registers) instead of the old dequant→cuBLAS path — a large decode-TPS win (WHT5_0: **0.9 → 10.6 t/s, 11.8×** on RDNA3.5/gfx1103) at PPL-parity. Prefill (`ne[1] > 8`) still uses dequant→cuBLAS/rocBLAS. **No Vulkan backend or SIMD CPU vec_dot yet** (deferred). Like the narrower siblings they are calibration-free (imatrix integrated but ignored). Slots 82/83/84.
>
> ✅ **Test-GGUF resolved (2026-06-23):** the earlier garbage `Qwen3.5-9B-WHT6_0.gguf` / `-WHT8_0.gguf` (PPL ≈ 1020 / 512k) were **stale files produced by a superseded pre-commit WIP quantizer** (written 2026-06-19, three days *before* the WHT5/6/8 quantize code landed in `4fd735f2c` on 2026-06-22). The committed quantizer **and** dequantizer are correct — a CPU round-trip test (production `ggml_quantize_chunk` → `to_float`) reconstructs N(0,1) cleanly with monotonic NRMSE 3.7 % / 1.9 % / 0.5 % (WHT5/6/8). **Re-quantizing from BF16 with the committed code restored healthy PPL:** WHT6_0 **7.6542** (b=512) / **7.6631** (b=1, mmvq); WHT8_0 **7.6892** / **7.6825** (Qwen3.5-9B, wikitext c=512, 6 chunks; WHT5_0 reference 8.6988). mmvq (b=1) tracks the dequant→cuBLAS path (b=512) within ≪1σ — the deferred WHT6/8 mmvq parity gate is now **closed**. No code change was required; the defect was a stale data file, not a dequant bug.

---

## At a glance

**Important:** the digit in `WHT3_0` / `WHT4_0` is the *index bit-width*, not the bits-per-weight. Each block also stores two fp16 half-block scales, which add roughly 1 bpw. Effective costs are:

| CLI string | GGML enum | FTYPE | Index bits | **Effective bpw** | Block bytes | Backends |
|---|---|---|---|---|---|---|
| `WHT3_0` | `GGML_TYPE_WHT3_0` (slot 80) | `MOSTLY_WHT3_0` (41) | 3-bit (8 centroids) | **4.0** | 16 (QK=32) | CPU, CUDA/HIP, Vulkan |
| `WHT4_0` | `GGML_TYPE_WHT4_0` (slot 81) | `MOSTLY_WHT4_0` (42) | 4-bit (16 centroids) | **5.0** | 20 (QK=32) | CPU, CUDA/HIP, Vulkan |
| `WHT5_0` | `GGML_TYPE_WHT5_0` (slot 82) | `MOSTLY_WHT5_0` (59) | 5-bit (32 centroids) | **6.0** | 24 (QK=32) | CPU, CUDA/HIP (mmvq decode + cuBLAS prefill) |
| `WHT6_0` | `GGML_TYPE_WHT6_0` (slot 83) | `MOSTLY_WHT6_0` (60) | 6-bit (64 centroids) | **7.0** | 28 (QK=32) | CPU, CUDA/HIP (mmvq decode + cuBLAS prefill) |
| `WHT8_0` | `GGML_TYPE_WHT8_0` (slot 84) | `MOSTLY_WHT8_0` (61) | 8-bit (256 centroids) | **9.0** | 36 (QK=32) | CPU, CUDA/HIP (mmvq decode + cuBLAS prefill) |

> `WHT5_0`/`WHT6_0`/`WHT8_0` (2026-06-22): single-token decode runs through a **fused mmvq `vec_dot` kernel** (CUDA/HIP); prefill (`ne[1] > 8`) uses dequant→cuBLAS/rocBLAS. No Vulkan backend or SIMD CPU `vec_dot` yet — deferred perf legs. See the §Backend support section for the decode-TPS table and the WHT6_0/WHT8_0 broken-test-GGUF caveat.

**TL;DR.** WHT4_0 competes with Q5_K_M — not Q4_K_M — because its true cost is ~5 bpw. A Walsh-Hadamard rotation applied before quantization flattens weight outliers and lets a compact fitted codebook achieve better quality than a plain low-bit quant at the same cost. No imatrix calibration is needed.

**Quick start:**

```bash
# Quantize to WHT4_0 (~5 bpw, competes with Q5_K_M)
llama-quantize model-F16.gguf model-WHT4_0.gguf WHT4_0

# Quantize to WHT3_0 (~4 bpw, competes with Q4_0 / IQ4_XS)
llama-quantize model-F16.gguf model-WHT3_0.gguf WHT3_0

# Wider variants (functional; ~6/7/9 bpw). 8-bit ≈ Q8_0-class quality.
llama-quantize model-F16.gguf model-WHT5_0.gguf WHT5_0
llama-quantize model-F16.gguf model-WHT6_0.gguf WHT6_0
llama-quantize model-F16.gguf model-WHT8_0.gguf WHT8_0

# Run inference
llama-cli -m model-WHT4_0.gguf --no-mmap -fa on -p "Hello"
```

---

## §1 Provenance

WHT3_0 and WHT4_0 are the **weight-quantization half of TheTom's TurboQuant family** — the offline GGUF weight counterpart to the `turboq2`/`turboq3`/`turboq4` KV-cache quants. Both halves originate from the same upstream source: [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant), `feature/turboquant-kv-cache`. See also: [`turboquant-kv-base.md`](turboquant-kv-base.md) for the KV-cache side of the family.

The upstream names were `TQ3_1S` and `TQ4_1S`. They were renamed `WHT3_0` / `WHT4_0` on port to avoid colliding with mainline llama.cpp's unrelated ternary types `TQ1_0` / `TQ2_0`. The name "WHT" reflects the Walsh-Hadamard Transform rotation at the core of the method — the same mathematical family as the KV-cache quants, but a distinct in-tree implementation (see disambiguation in §4).

### Differences from upstream

This fork adds ROCm and Vulkan backend support. An imatrix path was initially ported (ADR-016), but a quantizer audit confirmed it measurably hurts quality: the forward RHT rotation mixes all 32 block columns, so weighting the rotated residual by the original-basis importance misaligns importance with the rotated coefficients. Both types now quantize unweighted, byte-for-byte matching TheTom's upstream reference. The core quantization algorithm and block layout are unchanged. Model files quantized with TheTom's build are compatible; only the type names on the command line differ.

---

## §2 Use in production

### imatrix is not used

WHT3_0 and WHT4_0 do **not** use an imatrix. The forward Walsh-Hadamard rotation mixes
all 32 columns of each block, so post-rotation coefficient `buf[j]` no longer corresponds
to original column `j`. Weighting the rotated residual by original-basis importance `iw[j]`
misaligns importance with the rotated coefficient and measurably degrades quality. Both
types quantize unweighted — the scale search and WLS refinement ignore the imatrix
entirely. `tensor_requires_imatrix()` returns false for both types.

**A/B PPL confirming the fix (Qwen3.5-9B WHT3_0, wikitext-2-raw, 30 chunks, ROCm):**
- with-imatrix (defect): PPL 8.6105 ±0.092
- no-imatrix (fixed): PPL **7.2728** ±0.074 — −15.5%; matches and beats the upstream reference (7.6776)

### Quantization

```bash
llama-quantize \
    model-F16.gguf \
    model-WHT4_0.gguf \
    WHT4_0
```

### Inference flags

| Flag | Effect |
|---|---|
| `--no-mmap` | Recommended — avoids page-fault stalls during weight decode |
| `-fa on` | Flash attention (improves throughput; required for some KV-cache quant combinations) |

### Choosing between WHT3_0 and WHT4_0

- **WHT4_0** (~5.0 bpw) — compare against `Q5_K_M` (~5.5 bpw) and `Q4_K_M` (~4.5 bpw). Its real peer is Q5_K_M, not Q4_K_M.
- **WHT3_0** (~4.0 bpw) — compare against `Q4_0` (4.5 bpw) and `IQ4_XS` (4.25 bpw); `Q3_K_M` (~3.35 bpw) as a smaller secondary reference.

---

## §3 Benefits & potential drawbacks

### Benefits

- **Better quality than a plain quant at matched bpw** — the Hadamard rotation spreads each weight's energy across the block, preventing a few large outliers from inflating the per-block scale. A compact fitted codebook then covers the now-Gaussian coefficient distribution more efficiently.
- **Cross-backend parity** — measured cross-backend perplexity divergence < 0.5% between CUDA/ROCm and Vulkan for WHT4_0 (`docs/BACKEND_PARITY.md`).
- **All three backends** — CPU, CUDA/HIP, and Vulkan for both types.

### Potential drawbacks

- **Name understates the cost** — `WHT4_0` is a 4-bit index quant that costs ~5 bpw; its honest peer is Q5_K_M. Evaluate against the right baseline.
- **At 4–5 bpw the gap to F16 is already small** — the Hadamard edge over K-quants is real but modest; the benchmark matrix (below) will quantify it.

### Performance

Decode throughput on RDNA3 ROCm (Qwen3.5-9B, tg128, scalar/half path):

| Type | tg128 | Notes |
|---|---|---|
| WHT4_0 | **10.49 t/s** | +52.7% vs prior fp32 kernel |
| WHT3_0 | **9.42 t/s** | +65.3% vs prior fp32 kernel |

NVIDIA WHT4_0 (dp4a int8 path): T4×2 tg128 ~39.88 t/s ≈ IQ4_XS. AMD RDNA dp4a is not
available; the scalar/half path is used instead. Standard prefill (`-ub512`) throughput
is unchanged (cuBLAS/rocBLAS path). Small-batch prefill (`-ub8`) throughput on RDNA3:
WHT3_0 +290%, WHT4_0 unchanged (dp4a path is NVIDIA-only; scalar/half ties cuBLAS here).

### Benchmark matrix

*TBD (PPL + full backend matrix pending)*

**Configuration:** model=TBD, context=TBD tokens, GPU class=TBD (RDNA3.5 / RDNA3), backends=ROCm + Vulkan.

| Type | Effective bpw | PPL | TG (t/s) | PP (t/s) | Notes |
|---|---|---|---|---|---|
| F16 | 16 | TBD | TBD | TBD | ceiling |
| `WHT3_0` | 4.0 | TBD | TBD | TBD | primary peer: Q4_0 / IQ4_XS |
| `Q4_0` | 4.5 | TBD | TBD | TBD | matched-bpw reference |
| `IQ4_XS` | 4.25 | TBD | TBD | TBD | matched-bpw reference |
| `Q3_K_M` | ~3.35 | TBD | TBD | TBD | smaller secondary reference |
| `WHT4_0` | 5.0 | TBD | TBD | TBD | primary peer: Q5_K_M |
| `Q5_K_M` | ~5.5 | TBD | TBD | TBD | matched-bpw reference |
| `Q4_K_M` | ~4.5 | TBD | TBD | TBD | secondary reference |

---

## §4 How it works under the hood

### Block structures (`ggml/src/ggml-common.h`)

Each block covers 32 weights (`QK_TQ3_0 = QK_WHT4_0 = 32`) with two fp16 half-block scales — one for elements 0–15 (`d0`) and one for elements 16–31 (`d1`). The dual-scale design is what adds ~1 bpw over the nominal index bit-width.

**`block_wht3_0`** — 16 bytes (`ggml-common.h:429–438`):
```
[d0: fp16, 2B] [d1: fp16, 2B] [qs[12]: 3-bit indices, 8 per 3 bytes]
```
= 4.0 bits/value (`static_assert(sizeof(block_wht3_0) == 16)`)

**`block_wht4_0`** — 20 bytes (`ggml-common.h:441–450`):
```
[d0: fp16, 2B] [d1: fp16, 2B] [qs[16]: 4-bit indices nibble-packed]
```
= 5.0 bits/value (`static_assert(sizeof(block_wht4_0) == 20)`)

The wider variants follow the identical `d0`/`d1` dual-scale layout (`QK_WHT5_0 = QK_WHT6_0 = QK_WHT8_0 = 32`), differing only in codebook size and index packing:

**`block_wht5_0`** — 24 bytes: `[d0][d1][qs[20]: 5-bit indices, 8 packed into 5 bytes per group]` = 6.0 bpw.
**`block_wht6_0`** — 28 bytes: `[d0][d1][qs[24]: 6-bit indices, 4 packed into 3 bytes per group]` = 7.0 bpw.
**`block_wht8_0`** — 36 bytes: `[d0][d1][qs[32]: 8-bit indices, 1 per byte]` = 9.0 bpw.

The CPU encode/decode share a single parameterized core (`wht_quant_core` / `wht_dequant_core` in `ggml-turbo-quant.c`) over the per-type Lloyd-Max codebook; the CUDA dequant LUTs in `ggml-cuda/turbo-quant.cuh` are kept byte-identical (quantize runs on CPU, dequant on GPU).

### Encode pipeline (`ggml/src/ggml-turbo-quant.c`)

For each 32-weight block:

1. **Forward Walsh-Hadamard rotation** (`tq3_0_rht_forward`, `:823–835`) — a ±1 sign pattern applied before and after log₂ butterfly stages, normalized by 1/√32. This spreads weight energy across the block, making the distribution approximately Gaussian.
2. **Per-half-block RMS normalization** — independently for elements 0–15 and 16–31 (producing `d0` and `d1`).
3. **Scale search** — WLS scale refinement over candidate scales; importance weights are **not** applied (imatrix ignored; see §2).
4. **Centroid quantization** — rotated coefficients are mapped to the nearest Lloyd-Max centroid fitted for a unit Gaussian: 8 levels (3-bit, WHT3_0) or 16 levels (4-bit, WHT4_0).

Decode (`tq3_0_rht_inverse`, `:838–849`): unpack indices → centroid lookup → inverse WHT to restore the original weight domain.

### Disambiguation: three Hadamard things in this codebase

These weight quants use their **own inline Hadamard** (`tq3_0_rht_forward`/`tq3_0_rht_inverse`):

- **NOT** `GGML_OP_TURBO_WHT` — that is the KV-cache-side runtime op used by `turboq*` cache quantization during attention computation.
- **NOT** `GGML_OP_FWHT` — that standalone op was removed as dead code (no graph consumers).

The three share the same mathematical family (Walsh-Hadamard Transform) but are fully independent implementations serving different roles.

### Backend support

- **CPU** — fully implemented in `ggml-turbo-quant.c`
- **CUDA/HIP** — three-tier dispatch in `ggml/src/ggml-cuda/mmvq-tq.cu`:
  - **`ne1 == 1` (single-token decode):** fused `ggml_cuda_mul_mat_tq_multi<1>` — dp4a int8 on NVIDIA WHT4_0; scalar/half on AMD RDNA (dp4a not available on RDNA).
  - **`ne1 = 2–8` (small-batch / low `-ub`):** fused `ggml_cuda_mul_mat_tq_multi`, reusing each weight block across all tokens.
  - **`ne1 > 8` (standard prefill):** dequant-to-Q8_0 → cuBLAS/rocBLAS.
- **Vulkan** — mul_mat_vec pipelines for WHT3_0/WHT4_0 wired in `ggml-vulkan.cpp` (`dequant_wht3_0.comp` / `mul_mat_vec_wht3_0.comp` shaders)

**WHT5_0/WHT6_0/WHT8_0 backends (functional + fused decode, 2026-06-22):**

- **CPU** — full quant + dequant + `vec_dot` (dequant-then-dot against Q8_0), parameterized over the codebook.
- **CUDA/HIP — single-token decode (`ne1 == 1`, also `ne1 ≤ 8`):** fused **mmvq** `vec_dot_wht{5,6,8}_0_q8_1` (in `ggml-cuda/mmvq.cu`). The dot reconstructs the full 32-element block in registers (centroid LUT + inverse-RHT via the shared `wht{5,6,8}_0_reconstruct` helpers in `dequantize.cuh`, byte-identical to the dequant path) then float-MACs against the q8_1-quantized activations: `d8 · Σ w[k]·a_q[k]`. The whole block is processed by one thread (type traits `qi == vdr == 2` ⇒ `iqs == 0`); **no dp4a/WMMA** (the inverse Hadamard is not separable into integer sub-dots). All three are **native** (no WHT4_0-style load-time Q8_0 conversion). WHT5_0/WHT6_0 were chosen native (6/7 bpw — Q8_0 conversion would inflate); WHT8_0 was also kept native (uniform code, avoids the extra upload-hook + allocation reasoning of the q8_0-conversion route).
- **CUDA/HIP — prefill (`ne1 > 8`):** unchanged **dequant→cuBLAS/rocBLAS**. WHT5/6/8 have no mmq kernel, so they are kept out of mmq (`is_wht_no_mmq` guard in `ggml-cuda.cu`) and large batches fall through to cuBLAS. (`mul_mat_id`/MoE + CUDA-graph paths still take the dequant route — deferred; the test model is dense.)
- **Decode TPS (Qwen3.5-9B, RDNA3.5/gfx1103, `llama-cli -n 128 -fa on`):**

  | Type | Before (dequant→cuBLAS, every token) | After (fused mmvq) | Speedup | PPL parity (BLAS vs mmvq, c=512) |
  |------|------|------|------|------|
  | WHT5_0 | 0.9 t/s | **10.6 t/s** | **11.8×** | 8.6988 vs 8.7221 (+0.27 %, ≪1σ) ✅ |
  | WHT6_0 | 1.0 t/s | 19.5 t/s | — | **7.6542 vs 7.6631** (+0.12 %, ≪1σ) ✅ |
  | WHT8_0 | — | — | — | **7.6892 vs 7.6825** (−0.09 %, ≪1σ) ✅ |

  PPL re-measured 2026-06-23 on the **re-quantized** WHT6_0/WHT8_0 GGUFs (wikitext c=512, 6 chunks; the earlier 1020 / 512k figures were stale pre-commit files — see the resolved caveat at the top). WHT8_0 now generates coherently (64-tok smoke OK); both legs confirm the mmvq fused decode is PPL-faithful to the dequant→cuBLAS path.
- **Deferred (perf legs):** Vulkan shaders and SIMD CPU `vec_dot`; mmvq fusion (bias/gate) and `mul_mat_id` mmvq for these types — none are required for the decode win above.

---

## §5 Further reading

- **TurboQuant KV-cache counterpart:** [`turboquant-kv-base.md`](turboquant-kv-base.md) — the `turboq2`/`turboq3`/`turboq4` runtime KV-cache quants from the same upstream source
- **Upstream source:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — `feature/turboquant-kv-cache`
- **Slot assignments:** [`docs/TYPE_ASSIGNMENTS.md`](../TYPE_ASSIGNMENTS.md) — slots 80–85: WHT weight family
- **Concept primers:**
  - [WHT / Hadamard rotation](concepts/hadamard-wht-rotation.md) — the Walsh-Hadamard Transform and why it improves quantization
  - [IK quantization family](concepts/ik-quantization-family.md) — imatrix workflow and calibration guidance
  - [Feature maturity levels & backend support](concepts/feature-maturity-levels.md)
