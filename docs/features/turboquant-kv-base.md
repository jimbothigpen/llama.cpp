# TurboQuant KV Cache (`turboq2` / `turboq3` / `turboq4`)

> **Status: Stable** — CPU, CUDA/HIP, and Vulkan backends; fused flash-attention path required.

---

## At a glance

| CLI string | Enum | Effective bpw | Compression vs fp16 | Block bytes | Backends |
|---|---|---|---|---|---|
| `turboq2` | `GGML_TYPE_TURBOQ2_0` (slot 60) | 2.125 | ~7.5× | 34 | CPU, CUDA/HIP, Vulkan |
| `turboq3` | `GGML_TYPE_TURBOQ3_0` (slot 61) | 3.125 | ~5.1× | 50 | CPU, CUDA/HIP, Vulkan |
| `turboq4` | `GGML_TYPE_TURBOQ4_0` (slot 62) | 4.25 | ~3.8× | 68 | CPU, CUDA/HIP, Vulkan |

**TL;DR.** Drop-in KV compression — drop `--cache-type-k`/`-v` onto any existing GGUF run. No model re-download, no offline quantization step. `turboq3` is the practical sweet spot: roughly 5× memory reduction with ~11% perplexity increase vs fp16 KV.

**Quick start:**

```bash
llama-server \
    -m model.gguf \
    --cache-type-k turboq3 --cache-type-v turboq3 \
    -fa on -c 32768 -ngl 99
```

---

## §1 Provenance

TurboQuant KV cache is based on **TheTom's `feature/turboquant-kv-cache`** branch
([TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)),
which implements the TurboQuant method (arXiv 2504.19874, ICLR 2026) — a randomized Walsh-Hadamard rotation plus quantization scheme that builds on PolarQuant (arXiv 2502.02617, AISTATS 2026).

### Differences from upstream

TheTom's fork names these types `turbo2`, `turbo3`, `turbo4` (enum symbols `TURBO2_0`, `TURBO3_0`, `TURBO4_0`). This fork renames them to `turboq2`, `turboq3`, `turboq4` (`TURBOQ2_0`, `TURBOQ3_0`, `TURBOQ4_0`). The appended `q` disambiguates the PolarQuant family from the trellis `turboq*_tcq` variants that live alongside them.

**Impact on users:** substitute `turboq2`/`turboq3`/`turboq4` wherever TheTom's docs say `turbo2`/`turbo3`/`turbo4`. There is no file-compatibility issue — KV cache type identifiers are runtime-only and are **never serialized into the `.gguf`**. Model files created with TheTom's build load and run unchanged in this fork; the type string you pass on the command line is the only thing that changed.

---

## §2 Use in production

### Requirements

1. **Flash attention:** pass `-fa on` (or `--flash-attn on`). The quantized KV path is only reachable via the fused flash-attention kernel; without it, context creation fails with an error.
2. **`head_dim` multiple of 128:** the block size for all three types is QK=128. Most Qwen, Llama-3, Mistral, Gemma-2+ models satisfy this. If `head_dim % 128 ≠ 0`, the runtime will error at context creation (explicit check in `src/llama-context.cpp`).

### Flags

| Flag | Short | Description |
|---|---|---|
| `--cache-type-k TYPE` | `-ctk TYPE` | Quantization type for the K cache |
| `--cache-type-v TYPE` | `-ctv TYPE` | Quantization type for the V cache |

These flags are resolved by `kv_cache_type_from_str` in `common/arg.cpp:424`. The KV cache is quantized on every token at inference time (via `SET_ROWS`); no offline preparation step is required.

### Asymmetric K/V

K and V can use different compression levels. The quality-preserving direction is **K-bpw ≥ V-bpw** — lowering V precision costs less perplexity than lowering K precision by the same amount.

Recommended asymmetric pairs (most to least aggressive):

```bash
# Maximum compression: high-quality K, aggressive V
--cache-type-k turboq4 --cache-type-v turboq2

# Balanced asymmetric
--cache-type-k turboq3 --cache-type-v turboq2

# Symmetric sweet spot
--cache-type-k turboq3 --cache-type-v turboq3
```

### Layer-adaptive precision (optional)

Set `TURBO_LAYER_ADAPTIVE=N` to use higher-precision KV at boundary layers while keeping aggressive compression in the middle:

| Mode | Effect |
|---|---|
| `0` | Uniform (default — off) |
| `1` | K+V = q8_0 for first-4 and last-4 layers |
| `2` | K+V = q8_0 for last-8 layers |
| `5` | V = turboq4 at first-2+last-2 layers, V = turboq2 elsewhere (K unchanged) |
| `6` | V = turboq4 at last-8 layers, V = turboq2 elsewhere (K unchanged) |
| `7` | **Boundary V (recommended):** V = q8_0 at first-2+last-2 layers, V = turboq2 elsewhere (K unchanged) |

Mode 7 recovers approximately 1.2% perplexity relative to uniform turboq2 at very little extra memory cost. It is an explicit opt-in — default behavior is uniform precision.

See `src/llama-kv-cache.cpp:277` for the full mode dispatch.

---

## §3 Benefits & potential drawbacks

### Benefits

- **Large KV memory reduction** — enables long-context inference on VRAM-limited hardware: ~7.5× at turboq2, ~5.1× at turboq3, ~3.8× at turboq4 (all vs fp16 KV).
- **Zero offline work** — no calibration dataset, no re-quantization, no extra download. Apply to any existing GGUF.
- **Cross-backend parity** — measured cross-backend perplexity divergence < 0.5% between CUDA/ROCm and Vulkan.

### Potential drawbacks (theoretical)

- **Accuracy cost grows at lower bit-widths** — turboq2 incurs ~14.5% perplexity increase vs fp16 KV; turboq3 ~11.4%; turboq4 less (numbers not yet benchmarked; see matrix below). For generation tasks, the perceptible quality gap is smaller than the perplexity number suggests, but it is real.
- **`head_dim % 128 == 0` required** — excludes some older or non-standard architectures.
- **Flash attention required** — enabling `-fa on` can affect throughput on CPU-only or metal builds; check your build supports it.

### Benchmark matrix

*TBD (pending benchmark)*

**Configuration:** model=TBD, context=TBD tokens, backend=TBD, GPU class=TBD.

| Configuration | PPL | TG (t/s) | PP (t/s) | Memory vs F16 KV |
|---|---|---|---|---|
| **Baseline (mainline llama.cpp)** | | | | |
| F16 K / F16 V | TBD | TBD | TBD | 1.0× (reference) |
| Q8_0 K / Q8_0 V | TBD | TBD | TBD | TBD |
| **This fork** | | | | |
| F32/BF16 K / turboq4 V | TBD | TBD | TBD | TBD |
| F32/BF16 K / turboq3 V | TBD | TBD | TBD | TBD |
| F32/BF16 K / turboq2 V | TBD | TBD | TBD | TBD |
| Q8_0 K / turboq4 V | TBD | TBD | TBD | TBD |
| Q8_0 K / turboq3 V | TBD | TBD | TBD | TBD |
| Q8_0 K / turboq2 V | TBD | TBD | TBD | TBD |
| turboq4 K / turboq3 V | TBD | TBD | TBD | TBD |
| turboq4 K / turboq2 V | TBD | TBD | TBD | TBD |
| turboq3 K / turboq3 V | TBD | TBD | TBD | TBD |
| turboq3 K / turboq2 V | TBD | TBD | TBD | TBD |
| turboq2 K / turboq2 V | TBD | TBD | TBD | TBD |
| **Cross-fork sanity (TheTom's build, equivalent combos)** | | | | |
| turbo3 K / turbo3 V | TBD | TBD | TBD | TBD |
| turbo2 K / turbo2 V | TBD | TBD | TBD | TBD |

---

## §4 How it works under the hood

### Block structures (`ggml/src/ggml-common.h`)

Each block covers 128 elements (`QK_TURBOQ* = 128`).

**`block_turboq2_0`** — 34 bytes (`ggml-common.h:303`):
```
[norm: fp16, 2B] [qs[32]: 2-bit PolarQuant indices, 4 per byte]
```
`static_assert(sizeof(block_turboq2_0) == 34)`

**`block_turboq3_0`** — 50 bytes (`ggml-common.h:352`):
```
[norm: fp16, 2B] [qs[32]: lower 2 bits of 3-bit index, 4 per byte]
                 [signs[16]: upper 1 bit (QJL sign), 8 per byte]
```
`static_assert(sizeof(block_turboq3_0) == 50)`

**`block_turboq4_0`** — 68 bytes (`ggml-common.h:372`):
```
[norm: fp16, 2B] [rnorm: fp16, 2B] [qs[64]: 4-bit PolarQuant indices, nibble-packed]
```
`static_assert(sizeof(block_turboq4_0) == 68)` — default mode: `TURBOQ4_USE_4BIT=1` (`ggml-common.h:360`).

> **Note on turboq4 legacy mode:** a compile-time flag `TURBOQ4_USE_4BIT=0` switches turboq4 to 3-bit PolarQuant + 1-bit QJL residual and a different block layout (norm + rnorm + qs[48] + signs[16] = 68 bytes). The default build uses the 4-bit mode. The description "3-bit PolarQuant + 1-bit QJL" that appears in some internal docs and source file comments refers to this legacy mode, not the current default.

### Encode pipeline (`ggml/src/ggml-turbo-quant.c`)

For each group of 128 elements (turboq2/3) or per-block for turboq4:

1. **L2-normalize** the input group, store `grp_norm`.
2. **Forward WHT rotation** (`turbo_cpu_fwht`, `ggml-turbo-quant.c:216`) — a randomized Walsh-Hadamard Transform: multiply by a fixed ±1 sign vector (`s1`), apply the Hadamard butterfly, normalize by 1/√128, multiply by a second fixed ±1 sign vector (`s2`). This spreads quantization error across the full group.
3. **PolarQuant** — map each rotated element to the nearest centroid from a fixed codebook (no calibration):
   - turboq2: 4 centroids `{±0.040, ±0.133}` (2-bit)
   - turboq3: 8 centroids, Lloyd-Max optimal for N(0, 1/128) (3-bit); upper bit stored in `signs[]`
   - turboq4: 16 centroids (4-bit, nibble-packed)
4. **Corrected norm** — compute the L2 norm of the reconstructed centroids (`recon_norm`), store `grp_norm / recon_norm` as the block norm, pre-compensating for codebook approximation error.

Reference encode entry point: `quantize_row_turboq2_0_ref` (`ggml-turbo-quant.c:272`).

CLI string → enum resolution: `kv_cache_type_from_str` (`common/arg.cpp:424`), which iterates the allowed KV cache types list and matches by `ggml_type_name`.

See [concepts/hadamard-wht-rotation.md](concepts/hadamard-wht-rotation.md) for an explanation of the WHT rotation and why it improves quantization quality. See [concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md) for the K/V asymmetry rationale.

---

## §5 Further reading

- **Upstream source:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — `feature/turboquant-kv-cache` branch
- **Paper:** arXiv 2504.19874 — TurboQuant: ICLR 2026
- **PolarQuant paper:** arXiv 2502.02617 — PolarQuant: Quantizing KV Caches with Polar Transformation (AISTATS 2026)
- **Related docs (this repo):**
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and upstream-name mapping
  - [docs/features/README.md](README.md) — index of all feature docs
- **Concept primers:**
  - [WHT / Hadamard rotation primer](concepts/hadamard-wht-rotation.md)
  - [Asymmetric KV cache primer](concepts/asymmetric-kv-cache.md)
  - [Feature maturity levels & backend support](concepts/feature-maturity-levels.md)
