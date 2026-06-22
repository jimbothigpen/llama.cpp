# TurboQuant high-bit KV Cache (`turboq5` / `turboq6`)

> **Status: Functional (perf TBD)** — CPU + CUDA/HIP backends; fused flash-attention path required. Vulkan and perf tuning deferred (see §3).

---

## At a glance

| CLI string | Enum | Effective bpw | Compression vs fp16 | Block bytes | Grid | Backends |
|---|---|---|---|---|---|---|
| `turboq5` | `GGML_TYPE_TURBOQ5_0` (slot 64) | 5.125 | ~3.12× | 82 | 32-level uniform | CPU, CUDA/HIP |
| `turboq6` | `GGML_TYPE_TURBOQ6_0` (slot 65) | 6.125 | ~2.61× | 98 | 64-level uniform | CPU, CUDA/HIP |

`turboq5` and `turboq6` extend the **`turboq8` (buun) FWHT + uniform-grid + per-block absmax** design — no
QJL, no learned codebook — down to 5- and 6-bit precision. They fill the gap between the aggressive
PolarQuant `turboq2/3/4` family and the near-lossless `turboq8`, giving high-fidelity KV options at
roughly 3× / 2.6× memory reduction vs fp16 KV. They are the natural choice when `turboq4` loses too much
quality but `turboq8` leaves memory on the table.

**TL;DR.** Drop-in KV compression — add `--cache-type-k`/`-v` to any existing GGUF run. No model
re-download, no offline quantization step.

**Quick start:**

```bash
llama-server \
    -m model.gguf \
    --cache-type-k turboq6 --cache-type-v turboq6 \
    -fa on -c 32768 -ngl 99
```

---

## §1 Provenance

The TurboQuant KV-cache lineage is **TheTom's `feature/turboquant-kv-cache`** branch
([TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)), implementing the
TurboQuant method (arXiv 2504.19874, ICLR 2026) — a randomized Walsh-Hadamard rotation plus
quantization scheme that builds on PolarQuant (arXiv 2502.02617, AISTATS 2026). The 8-bit member
`turboq8` (FWHT + uniform 256-level grid + per-block absmax, no QJL) was contributed by **buun** as
`TURBO8_0`.

`turboq5` / `turboq6` are **this fork's own extension** (ygg, TODO 250): they apply the same
FWHT + uniform-grid + per-block-absmax codec as `turboq8` at 5- and 6-bit precision, using a
`q5_0`/`q6_K`-style low/high index split (low nibble in `qs`, high bit(s) in `qh`) so that a single
element can be decoded cheaply on-device inside the flash-attention vector kernel.

Slots **64** (`turboq5`) and **65** (`turboq6`) sit in the gap between `GGML_TYPE_TURBOQ8_0 = 63` and
`GGML_TYPE_TURBOQ2_TCQ = 66`.

> **Prior measurement (TODO 250 / TODO 228).** An earlier evaluation of 5/6-bit KV against lossless
> KV found no perplexity headroom over the existing regime on Qwen3.5-9B-Q4_K_M (within ~1σ), i.e. these
> types are **PP-negative** relative to cheaper members and are not the throughput-optimal choice. They
> are implemented and wired for completeness and correctness; perf tuning is explicitly deferred.

---

## §2 Use in production

### Requirements

1. **Flash attention:** pass `-fa on`. The quantized KV path is only reachable via the fused
   flash-attention kernel; without it, context creation fails.
2. **`head_dim` multiple of 128:** the block size for both types is QK=128 (`QK_TURBOQ5 == QK_TURBOQ6 ==
   128`). Most Qwen, Llama-3, Mistral, Gemma-2+ models satisfy this.

### Flags

| Flag | Short | Description |
|---|---|---|
| `--cache-type-k TYPE` | `-ctk TYPE` | Quantization type for the K cache |
| `--cache-type-v TYPE` | `-ctv TYPE` | Quantization type for the V cache |

Resolved by `kv_cache_type_from_str` in `common/arg.cpp`, which matches the CLI string against
`ggml_type_name` (`"turboq5"`, `"turboq6"`). The KV cache is quantized on every token at inference time
(via `SET_ROWS`); no offline preparation is required.

### Asymmetric K/V

K and V can use different types. The quality-preserving direction is **K-bpw ≥ V-bpw**. These high-bit
types pair naturally as the high-precision K anchor against a more aggressive V, e.g.:

```bash
# High-fidelity K, aggressive V
--cache-type-k turboq6 --cache-type-v turboq3
--cache-type-k turboq5 --cache-type-v turboq4
```

---

## §3 Status, benefits & drawbacks

### Benefits

- **High-fidelity KV at meaningful compression** — ~3.12× (`turboq5`) / ~2.61× (`turboq6`) vs fp16 KV,
  filling the fidelity gap above `turboq4` and below `turboq8`.
- **Zero offline work** — no calibration dataset, no re-quantization, no extra download.
- **Cheap on-device decode** — `q5_0`/`q6_K`-style lo/hi index split keeps single-element dequant inside
  the FA-vec kernel inexpensive.

### Drawbacks / current limitations

- **PP-negative vs cheaper members** — see the prior-measurement note in §1; not the throughput-optimal
  KV choice. Implemented for completeness; perf tuning deferred.
- **Backends: CPU + CUDA/HIP only** — no Vulkan kernel yet (deferred to a future win-gated phase, matching
  `turboq8`).
- **`head_dim % 128 == 0` and `-fa on` required.**

### Functional validation

Both types pass a functional smoke on ai00 (gfx1150 ROCm/HIP) with Qwen3.5-9B-Q4_K_M, `-fa on
--no-mmap -ngl 99`, symmetric K=V: coherent generation, no assert/crash, RC=0. This is a **functional**
check only — not a TPS/PPL measurement.

---

## §4 How it works under the hood

### Block structures (`ggml/src/ggml-common.h`)

Each block covers 128 elements.

**`block_turboq5_0`** — 82 bytes (5.125 bpv):
```
[norm: fp16, 2B]  grp_norm * per-block absmax scale
[qs[64]: low 4 bits of the 5-bit index, nibble-packed]
[qh[16]: high 1 bit of the 5-bit index, 8 per byte]
```
`static_assert(sizeof(block_turboq5_0) == 82)`. Grid: uniform 32-level, `centroid[i] = (i − 15.5)/15.5`.

**`block_turboq6_0`** — 98 bytes (6.125 bpv):
```
[norm: fp16, 2B]  grp_norm * per-block absmax scale
[qs[64]: low 4 bits of the 6-bit index, nibble-packed]
[qh[32]: high 2 bits of the 6-bit index, 4 per byte]
```
`static_assert(sizeof(block_turboq6_0) == 98)`. Grid: uniform 64-level, `centroid[i] = (i − 31.5)/31.5`.

### Encode pipeline (`ggml/src/ggml-turbo-quant.c`)

For each group of 128 elements (identical to `turboq8`, differing only in grid resolution and the
index-split packing):

1. **Forward FWHT rotation** (`turbo_cpu_fwht`) — randomized Walsh-Hadamard Transform spreading
   quantization error across the full group.
2. **Per-block absmax scale** — find the max absolute rotated value; the uniform grid is scaled by it.
3. **Uniform-grid quantize** — map each rotated element to the nearest of the 32 (`turboq5`) / 64
   (`turboq6`) uniform levels; split the index into low nibble (`qs`) and high bit(s) (`qh`).
4. **Norm** — store `grp_norm × absmax_scale` as the block `norm` (fp16).

Device-side single-element decode and the FA-vec KQ/V paths live in `ggml/src/ggml-cuda/turbo-quant.cuh`,
`set-rows.cu` (GPU encode, shared-mem index staging to avoid packed-byte write hazards), and the
`fattn-vec-instance-turboq{5,6}_0-*.cu` template instances.

See [concepts/hadamard-wht-rotation.md](concepts/hadamard-wht-rotation.md) for the WHT rotation and
[concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md) for the K/V asymmetry rationale.

---

## §5 Further reading

- **Sibling doc:** [turboquant-kv-base.md](turboquant-kv-base.md) — the PolarQuant `turboq2/3/4` + `turboq8` base family.
- **Paper:** arXiv 2504.19874 — TurboQuant (ICLR 2026).
- **Related docs (this repo):**
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and upstream-name mapping.
  - [docs/features/README.md](README.md) — index of all feature docs.
