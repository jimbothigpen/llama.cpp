# InnerQ KV Cache (`turboq2_innerq` / `turboq3_innerq`)

> **Status: Experimental** — CUDA/HIP only (no Vulkan); requires a short online calibration.

---

## At a glance

| CLI string | Enum | Slot | Effective bpw | Compression vs fp16 | Block bytes | Backends |
|---|---|---|---|---|---|---|
| `turboq2_innerq` | `GGML_TYPE_TURBOQ2_INNERQ` (slot 68) | 68 | 2.125 | ~7.5× | 34 | CUDA/HIP only — no Vulkan; CPU dequant |
| `turboq3_innerq` | `GGML_TYPE_TURBOQ3_INNERQ` (slot 69) | 69 | 3.125 | ~5.1× | 50 | CUDA/HIP only — no Vulkan; CPU dequant |

**TL;DR.** InnerQ is not a new compression format — it reuses the same on-disk block structs as
`turboq2` / `turboq3` (identical memory footprint) and adds a per-session, online per-channel
equalization pass before quantization. When K-cache channels have unequal variance, InnerQ
reduces quantization error at zero extra memory cost, delivering better quality than plain
`turboq2` / `turboq3` at equal bit-width.

**Quick start (CUDA/HIP):**

```bash
TURBO_INNERQ=256 llama-server \
    -m model.gguf \
    --cache-type-k turboq3_innerq --cache-type-v turboq3_innerq \
    -fa on -c 4096 -ngl 99
```

> Without `TURBO_INNERQ`, InnerQ types are accepted but the equalization layer is inactive
> (scales remain identity), so quality equals plain `turboq2` / `turboq3`.

---

## §1 Provenance

InnerQ KV cache originates in **TheTom's `feature/turboquant-kv-cache`** branch
([TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)).
The per-channel calibration engine (`d_innerq_*` device state, `turbo_innerq_init` /
`turbo_innerq_finalize`) is part of TheTom's original implementation.

### Differences from upstream

**Naming.** TheTom's fork calls these types `TURBO2_INNERQ` and `TURBO3_INNERQ`. This fork
renames them to `turboq2_innerq` / `turboq3_innerq` (enum symbols `GGML_TYPE_TURBOQ2_INNERQ` = 68
and `GGML_TYPE_TURBOQ3_INNERQ` = 69, `ggml/include/ggml.h:440-441`), consistent with the
`turboq`-prefix family convention. KV cache type identifiers are runtime-only — they are never
serialized into `.gguf`. Model files from TheTom's build load and run unchanged; substitute
`turboq2_innerq` / `turboq3_innerq` wherever TheTom's docs say `TURBO2_INNERQ` / `TURBO3_INNERQ`.

**No 4-bit InnerQ.** TheTom's fork included `TURBO4_INNERQ`; this fork removes it entirely
(slot 70 is retired and reserved). Per-channel equalization at 4-bit regresses quality — the
4-bit PolarQuant codebook is fine-grained enough that channel rescaling hurts more than it helps
(observed PPL regression: 9.08 vs 7.47 with `turboq4` baseline). Use plain `turboq4` for 4-bit
KV cache compression instead.

**Calibration is online, not GGUF-shipped.** TheTom's README describes storing calibration data
alongside the GGUF. In the code as shipped, calibration is entirely online and per-session: it
starts when you set the `TURBO_INNERQ` environment variable and completes after the first N
tokens of the current inference session. Nothing is written to disk and no calibrated GGUF file
is needed or produced (see §4 for details).

---

## §2 Use in production

### Requirements

1. **CUDA or HIP GPU.** The InnerQ calibration kernels and scale-apply path are CUDA/HIP-only
   (`ggml/src/ggml-cuda/turbo-quant.cuh`). Without a CUDA or HIP backend, encode falls back to a
   CPU path that performs only basic PolarQuant quantization without equalization. Vulkan is not
   supported.
2. **Flash attention:** pass `-fa on` (or `--flash-attn on`). The quantized KV path requires the
   fused flash-attention kernel; context creation will fail with an error without it.
3. **`head_dim` multiple of 128:** both types use QK=128 blocks. Most Qwen, Llama-3, Mistral, and
   Gemma-2+ models satisfy this. If `head_dim % 128 ≠ 0` the runtime errors at context creation.
   Models where `head_dim > 128` (non-standard architectures with multi-group-per-head WHT) cause
   InnerQ to auto-disable with a log warning; plain PolarQuant encoding is used instead.

### Calibration

Set `TURBO_INNERQ=N` where N is the number of tokens to use as the calibration window (100–500
is typical). Calibration is automatic:

1. On startup InnerQ enters **calibrating** mode and begins accumulating per-channel
   sum-of-squares across the first N encoded K tokens.
2. After N tokens InnerQ **finalizes**: computes per-channel RMS, derives equalization scales
   (`scale[i] = (mean_rms / rms[i])^strength`, clamped to [0.5, 2.0]), and activates them.
3. From that point forward, scales are applied to every encoded InnerQ token (K and V both, if V
   is also an innerq type).
4. **Auto-disable:** if the maximum channel variance ratio is below 1.2 (channels are already
   balanced), InnerQ logs a notice and disables itself — no quality impact, just a hint that InnerQ
   is unnecessary for this model.

Optional tuning:

| Env var | Controls | Default |
|---|---|---|
| `TURBO_INNERQ` | Calibration token count; `0` or unset = disabled | (unset) |
| `TURBO_INNERQ_STRENGTH` | Exponent for scale derivation | `0.5` |

`TURBO_INNERQ_STRENGTH` values: `0.0` = no equalization (identity); `1.0` = full equalization to
mean RMS; `0.5` (default) = geometric mean, which balances correction against over-equalization.

### Flags

| Flag | Short | Description |
|---|---|---|
| `--cache-type-k TYPE` | `-ctk TYPE` | Quantization type for the K cache |
| `--cache-type-v TYPE` | `-ctv TYPE` | Quantization type for the V cache |

Resolved by `kv_cache_type_from_str` in `common/arg.cpp` (same lookup as all TurboQuant types).
The allowed KV cache types list includes `GGML_TYPE_TURBOQ2_INNERQ` and
`GGML_TYPE_TURBOQ3_INNERQ` only — `turboq4_innerq` is not accepted and will produce an error.

### K-only vs K+V equalization

InnerQ is a K-cache equalization technique — K channels in attention tend to exhibit higher
variance imbalance than V channels, and this is where the quality gain comes from.

When V is also set to an InnerQ type, the same equalization scales (derived from K/V calibration
statistics) are applied to V. This is allowed and causes no errors, but the benefit on V is
model-dependent and typically smaller than on K.

For the purest K-only equalization, use a mixed K/V pair:

```bash
# K-only InnerQ — V uses plain turboq3 (unscaled)
TURBO_INNERQ=256 llama-server \
    -m model.gguf \
    --cache-type-k turboq3_innerq --cache-type-v turboq3 \
    -fa on -c 4096 -ngl 99
```

This combination is also row 7 of the benchmark matrix (§3) and isolates the K-side equalization
effect in controlled comparisons.

### Asymmetric K/V

InnerQ types can be paired with any other KV type, including the base PolarQuant types and TCQ:

```bash
# Symmetric InnerQ @ 3-bit
--cache-type-k turboq3_innerq --cache-type-v turboq3_innerq

# K-only equalization with plain V (isolates K-side effect)
--cache-type-k turboq3_innerq --cache-type-v turboq3

# High-precision K, InnerQ-equalized V (unusual but valid)
--cache-type-k q8_0 --cache-type-v turboq3_innerq
```

As with all KV quantization, lowering V precision costs less perplexity than lowering K precision
by the same amount. See [concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md) for
the K/V asymmetry rationale.

---

## §3 Benefits & potential drawbacks

### Benefits

- **Better quality than plain `turboq2` / `turboq3` at the same memory** — block structs are
  identical; InnerQ is a pure quality improvement for models with imbalanced K-channel variance.
- **Online calibration, no offline work** — set `TURBO_INNERQ=256` and the calibration runs
  during the first 256 tokens of inference; no calibration dataset, no extra download, no GGUF
  modification.
- **Auto-adapts and auto-disables** — if channels are already balanced, InnerQ disables itself
  gracefully with no quality penalty.
- **Configurable equalization strength** — `TURBO_INNERQ_STRENGTH` lets you dial between no
  correction (0.0) and full equalization (1.0).

### Potential drawbacks

- **CUDA/HIP only** — Vulkan and CPU-only builds get no equalization benefit (InnerQ types still
  work but behave like plain `turboq2` / `turboq3`).
- **Model-dependent benefit** — InnerQ helps only when K channels have measurable variance
  imbalance. It auto-disables on balanced models; on others the benefit varies by architecture.
- **Calibration window** — the first N tokens are used for calibration; very short contexts (fewer
  tokens than `TURBO_INNERQ`) will never finalize and scales remain identity.
- **Same `head_dim % 128 == 0` requirement** as the base TurboQuant types.

### Benchmark matrix

> Numbers to be filled in after benchmarking. See caption for configuration.
>
> **Configuration:** model TBD, context TBD tokens, CUDA/HIP backend, GPU class TBD.
> InnerQ rows measured with `TURBO_INNERQ=256` active (or as noted). Results are model-dependent;
> benefit is largest on models with high K-channel variance imbalance.
> No Vulkan section — InnerQ has no Vulkan encode support.

| # | K / V | PPL | TG (t/s) | PP (t/s) | Memory vs F16 KV |
|---|---|---|---|---|---|
| **Baselines (mainline llama.cpp)** | | | | | |
| 1 | F16 / F16 | TBD | TBD | TBD | 1.0× (reference) |
| 2 | Q8_0 / Q8_0 | TBD | TBD | TBD | TBD |
| **Base PolarQuant (equal-memory comparison)** | | | | | |
| 3 | turboq2 / turboq2 | TBD | TBD | TBD | TBD |
| 4 | turboq3 / turboq3 | TBD | TBD | TBD | TBD |
| **InnerQ (same memory as rows 3–4)** | | | | | |
| 5 | turboq2_innerq / turboq2_innerq | TBD | TBD | TBD | TBD |
| 6 | turboq3_innerq / turboq3_innerq | TBD | TBD | TBD | TBD |
| 7 | turboq3_innerq / turboq3 (K-only) | TBD | TBD | TBD | TBD |
| **Cross-fork (TheTom build, equivalent types — or N/A if not measured)** | | | | | |
| 8 | turbo2_innerq / turbo2_innerq | TBD | TBD | TBD | TBD |
| 9 | turbo3_innerq / turbo3_innerq | TBD | TBD | TBD | TBD |

---

## §4 How it works under the hood

### Block structures — identical to base TurboQuant (`ggml/src/ggml-common.h`)

InnerQ uses the **same block structs** as `turboq2` / `turboq3`. There is no InnerQ-specific
block layout; the equalization is applied as a pre-processing step before the block is encoded.

**`block_turboq2_0`** — 34 bytes (`ggml-common.h:301-305`):
```
[norm: fp16, 2B] [qs[32]: 2-bit PolarQuant indices, 4 per byte]
```
`static_assert(sizeof(block_turboq2_0) == sizeof(ggml_half) + QK_TURBOQ2/4)`

**`block_turboq3_0`** — 50 bytes (`ggml-common.h:351-355`):
```
[norm: fp16, 2B] [qs[32]: lower 2 bits of 3-bit index, 4 per byte]
                 [signs[16]: upper 1 bit (QJL sign), 8 per byte]
```
`static_assert(sizeof(block_turboq3_0) == sizeof(ggml_half) + QK_TURBOQ3/4 + QK_TURBOQ3/8)`

Both types cover 128 elements per block (`QK_TURBOQ2 = QK_TURBOQ3 = 128`).

### Calibration engine (`ggml/src/ggml-cuda/turbo-quant.cuh:187-334`)

The calibration state lives in CUDA device-side `__device__` globals:

- `d_innerq_sq_accum[128]` — running per-channel sum-of-squares
- `d_innerq_count` — tokens accumulated so far
- `d_innerq_calibrating` — 1 while accumulating, 0 after finalization
- `d_innerq_active` — 1 once scales are uploaded and live
- `d_innerq_scale[128]` / `d_innerq_scale_inv[128]` — finalized per-channel scales

**Init (`turbo_innerq_init`, `turbo-quant.cuh:206-232`):** reads `TURBO_INNERQ` from the
environment; if set and positive, zeros the accumulators on device and sets `d_innerq_calibrating=1`.

**Accumulation:** during each SET_ROWS kernel call while `d_innerq_calibrating` is 1, each
thread atomically adds its element's squared value to `d_innerq_sq_accum[channel_index]`.

**Finalization (`turbo_innerq_finalize`, `turbo-quant.cuh:235-299`):** triggered when
`d_innerq_count ≥ TURBO_INNERQ`. Reads accumulators from device, computes per-channel RMS,
then for each channel:

```
scale[i] = clamp( (mean_rms / rms[i])^strength, 0.5, 2.0 )
```

Before uploading scales the engine checks the maximum channel variance ratio. If `max_ratio < 1.2`
(all channels within 20% of each other), equalization is skipped and InnerQ auto-disables.
Otherwise, scales are uploaded to `d_innerq_scale` / `d_innerq_scale_inv`, `d_innerq_active` is
set to 1, and `turbo_innerq_publish` copies `scale_inv` to host-side state for the KV cache
tensor update.

### Scale-apply path (`ggml/src/ggml-cuda/set-rows.cu`)

During each SET_ROWS kernel invocation, before the L2 normalization and WHT rotation:

```c
// InnerQ calibration accum
if (d_innerq_calibrating && sid < 128)
    atomicAdd(&d_innerq_sq_accum[sid], x[sid] * x[sid]);

// InnerQ equalization
if (d_innerq_active && sid < 128) x[sid] *= d_innerq_scale[sid];
```

This appears at `set-rows.cu:388-400` (turboq2 path) and `set-rows.cu:1174-1182` /
`set-rows.cu:1476-1483` (turboq3 and TCQ paths that InnerQ also gates through). The equalization
scales the channel value **before** the WHT rotation, so the rotation operates on already-balanced
data. Math: `⟨Q/s, s·K⟩ = ⟨Q, K⟩` — dot products with Q are preserved; only quantization
error distribution changes.

### Dispatch (`ggml/src/ggml-cuda/set-rows.cu:2187-2193`)

```c
} else if (dst->type == GGML_TYPE_TURBOQ2_INNERQ) {
    turbo_innerq_check_finalize(QK_TURBOQ2, ne00);
    set_rows_cuda_turboq2<idx_t>(ctx, src0, src1, dst);
} else if (dst->type == GGML_TYPE_TURBOQ3_INNERQ) {
    set_rows_cuda_turboq3<idx_t>(ctx, src0, src1, dst);
```

`TURBOQ2_INNERQ` and `TURBOQ3_INNERQ` call the same encode kernels as the base PolarQuant
types — InnerQ is not wired into TCQ (`turboq2_tcq` / `turboq3_tcq`) at all.

### KV cache tensor update (`src/llama-kv-cache.cpp:994-998`)

After calibration finalizes, the host-side `scale_inv` array needs to reach a `turbo_innerq_scale_inv`
tensor stored in the KV cache context (for potential dequant reference):

```c
if (turbo_innerq_scale_inv && turbo_innerq_needs_tensor_update()) {
    ggml_backend_tensor_set(turbo_innerq_scale_inv, g_innerq_scale_inv_host, 0, sizeof(float) * INNERQ_MAX_CHANNELS);
    turbo_innerq_mark_tensor_updated();
}
```

This is a one-time update per session; no scale data is serialized to GGUF.

CLI string → enum resolution: `kv_cache_type_from_str` (`common/arg.cpp:409-419`). The allowed
KV types list contains `GGML_TYPE_TURBOQ2_INNERQ` and `GGML_TYPE_TURBOQ3_INNERQ`; requesting
`turboq4_innerq` produces an error (slot 70 is retired).

See [concepts/hadamard-wht-rotation.md](concepts/hadamard-wht-rotation.md) for the WHT rotation
shared with the base PolarQuant types.

---

## §5 Further reading

- **Upstream source:** [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) — `feature/turboquant-kv-cache` branch
- **Related docs (this repo):**
  - [TurboQuant KV base](turboquant-kv-base.md) — base PolarQuant types (`turboq2` / `turboq3` / `turboq4`); the equal-memory comparison baseline for InnerQ
  - [TCQ KV cache](tcq-kv.md) — Trellis-Coded Quantization; higher quality per bit via Viterbi trellis (independent of InnerQ; not composable)
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and upstream-name mapping
  - [docs/features/README.md](README.md) — index of all feature docs
- **Concept primers:**
  - [WHT / Hadamard rotation primer](concepts/hadamard-wht-rotation.md) — the WHT rotation stage shared with base PolarQuant and TCQ
  - [Asymmetric KV cache primer](concepts/asymmetric-kv-cache.md) — K/V asymmetry rationale and pair selection
  - [Feature maturity levels & backend support](concepts/feature-maturity-levels.md)
