# TCQ KV Cache (`turboq2_tcq` / `turboq3_tcq`)

> **Status: Stable** — CUDA/HIP and Vulkan backends; fused flash-attention path required;
> GPU required for encode.

---

## At a glance

| CLI string | Enum | Slot | Effective bpw | Compression vs fp16 | Block bytes | Backends |
|---|---|---|---|---|---|---|
| `turboq2_tcq` | `GGML_TYPE_TURBOQ2_TCQ` | 66 | 2.25 | ~7.1× | 36 | CUDA/HIP, Vulkan; CPU-decode only |
| `turboq3_tcq` | `GGML_TYPE_TURBOQ3_TCQ` | 67 | 3.25 | ~4.9× | 52 | CUDA/HIP, Vulkan; CPU-decode only |

**TL;DR.** Drop-in KV compression via Trellis-Coded Quantization — higher per-bit quality
than the base PolarQuant types (`turboq2`/`turboq3`) at the same nominal bit-width, at
the cost of a Viterbi encode pass that requires a CUDA or HIP GPU. Apply via
`--cache-type-k`/`-v` exactly as you would any other KV type.

**Quick start (ROCm/CUDA):**

```bash
llama-server \
    -m model.gguf \
    --cache-type-k turboq3_tcq --cache-type-v turboq3_tcq \
    -fa on -c 4096 -ngl 99
```

---

## §1 Provenance

TCQ KV cache is ported from the **buun** fork
([spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)),
which names these types `TURBO2_TCQ` and `TURBO3_TCQ`.

### Differences from upstream

**Naming.** This fork renames them to `turboq2_tcq` / `turboq3_tcq`
(enum symbols `GGML_TYPE_TURBOQ2_TCQ` = 66 / `GGML_TYPE_TURBOQ3_TCQ` = 67,
`ggml/include/ggml.h:438-439`). The `q`-prefix and `_tcq` suffix
unambiguously distinguish these from the base PolarQuant family
(`turboq2`/`turboq3`, slots 60–61) and from each other. Because KV cache types
are runtime-only identifiers (never serialized into `.gguf`), the rename has no
file-compatibility impact.

**Alpha default.** The buun fork ships with norm-scaling defaults K=1.1 / V=1.3
(`TURBO_TCQ_ALPHA` / `TURBO_TCQ_ALPHA_V`). This fork defaults both to **1.0 / 1.0**
(`ggml/src/ggml-cuda/turbo-quant.cuh:651-652`). Probes on Qwen3.5-9B Q4_K_M showed
buun's defaults regress short-context PPL by 5–6%; the gains buun reports appear to
be long-context-only. Long-context users can opt in via environment variables — see
[§2 Alpha tuning](#alpha-tuning-optional) below.

---

## §2 Use in production

### Requirements

1. **CUDA or HIP GPU for encode** — The Viterbi encode kernels are GPU-only
   (`ggml/src/ggml-cuda/set-rows.cu`). A CPU path exists but performs only the
   norm-only stub; actual quantization requires CUDA or HIP. Decode is trivial
   sliding-window bitstream access and works on CPU.
2. **Flash attention:** pass `-fa on` (or `--flash-attn on`). The TCQ decode path
   is reached only via the fused flash-attention kernel; without it, context creation
   fails with an error.
3. **`head_dim` multiple of 128:** both types use QK=128. Most Qwen, Llama-3,
   Mistral, and Gemma-2+ models satisfy this. If `head_dim % 128 ≠ 0`, the
   runtime will error at context creation.

### Flags

| Flag | Short | Description |
|---|---|---|
| `--cache-type-k TYPE` | `-ctk TYPE` | Quantization type for the K cache |
| `--cache-type-v TYPE` | `-ctv TYPE` | Quantization type for the V cache |

These flags are resolved by `kv_cache_type_from_str` in `common/arg.cpp` (same
lookup as the base TurboQuant types — see [TurboQuant KV base](turboquant-kv-base.md)
for details).

### Asymmetric K/V

K and V can use different types. As with all KV quantization, the quality-preserving
direction is **K-bpw ≥ V-bpw**. Recommended TCQ pairs:

```bash
# Symmetric — higher quality at each bit-width
--cache-type-k turboq3_tcq --cache-type-v turboq3_tcq

# Asymmetric — aggressive V with quality K
--cache-type-k turboq3_tcq --cache-type-v turboq2_tcq

# Mixed — high-precision K, TCQ-compressed V
--cache-type-k q8_0 --cache-type-v turboq3_tcq
```

TCQ types are also compatible with the base PolarQuant types in any K/V pair. See
[concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md) for the K/V
asymmetry rationale.

### Vulkan note

On Vulkan (RADV PHOENIX gfx1103 and other tested cards), the RADV shader-variant
specialization has a known bug: the flash-attention shader faults when both K and V
are TCQ-typed at sequence length N≥512 or KV>2048.

The **αA workaround** (shipped in this fork) pre-dequantizes each TCQ KV tensor to
a **transient F16 scratch buffer** immediately before the FA dispatch
(`ggml/src/ggml-vulkan/ggml-vulkan.cpp:9853-9883`). This means:

- **Storage**: KV cache is **stored as TCQ** — full memory savings are preserved.
- **During FA**: one F16 scratch buffer is allocated per TCQ-typed K and V. Peak
  VRAM at FA time = TCQ storage + temporary F16 window (≈ context × n_heads ×
  head_dim × sizeof(fp16) per tensor).
- **Cross-backend PPL**: because the FA kernel sees the same dequantized values as
  ROCm, cross-backend PPL should be within measurement noise.

> **Clarification:** the README currently describes the αA behavior as
> "K=TCQ, V=F16 asymmetric." This is outdated — the current implementation
> stores **both K and V as TCQ** and dequants both to transient F16 before FA.
> There is no permanent V-side savings forfeit on Vulkan.

---

### Alpha tuning (optional)

TCQ encodes a scaled version of the per-block L2 norm via two runtime knobs:

| Env var | Controls | Default |
|---|---|---|
| `TURBO_TCQ_ALPHA` | K norm scale | 1.0 |
| `TURBO_TCQ_ALPHA_V` | V norm scale (independent) | 1.0 |

If `TURBO_TCQ_ALPHA` is set without `TURBO_TCQ_ALPHA_V`, V tracks K
(backwards-compatible with single-var usage).

**When to use.** The defaults (1.0/1.0) are calibrated for short-to-mid-range
contexts on typical Qwen3.5-class models. For long-context workloads (>16K
tokens), buun's original K=1.1/V=1.3 may recover quality — test on your
specific model before adopting:

```bash
# Long-context opt-in (buun defaults)
TURBO_TCQ_ALPHA=1.1 TURBO_TCQ_ALPHA_V=1.3 llama-server \
    -m model.gguf --cache-type-k turboq3_tcq --cache-type-v turboq3_tcq \
    -fa on -c 32768 -ngl 99
```

The knob is applied at encode time
(`ggml/src/ggml-cuda/set-rows.cu` — the `corrected_norm *= iq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v` path).

---

## §3 Benefits & potential drawbacks

### Benefits

- **Better quality per bit than base PolarQuant** — the Viterbi trellis finds the
  globally optimal quantization path through each group, giving a
  ~+3 dB / ~+1.75 dB MSE improvement over Lloyd-Max scalar quantization at the
  same bit-width for `turboq3_tcq` / `turboq2_tcq` respectively.
- **Large KV memory reduction** — ~7.1× at `turboq2_tcq`, ~4.9× at `turboq3_tcq`
  (vs fp16 KV, struct-derived; see §4 for block-struct details).
- **Zero offline work** — no calibration dataset, no re-quantization, no extra
  download. Apply to any existing GGUF.
- **Vulkan memory savings preserved** — the αA workaround keeps KV stored as TCQ
  on Vulkan; only the transient FA scratch is F16.

### Potential drawbacks (theoretical)

- **GPU-only encode** — CUDA or HIP required; Vulkan encode is not supported
  (Viterbi kernel is a CUDA/HIP-specific implementation).
- **Slower encode than base PolarQuant** — the Viterbi trellis search (512 states
  for `turboq3_tcq`, 256 for `turboq2_tcq`) is more compute-intensive than
  PolarQuant's nearest-centroid lookup.
- **Vulkan FA overhead** — transient F16 dequant adds a dispatch overhead and a
  temporary peak-VRAM cost per FA call (see §2 Vulkan note).
- **`head_dim % 128 == 0` required** — same constraint as the base TurboQuant types.

### Qualitative observations

In measurement data not controlled for session-to-session variance, `turboq2_tcq` has
occasionally scored better perplexity than `turboq3_tcq` despite fewer bits, reflecting
model and prompt sensitivity at these compression levels. The benchmark matrix will resolve
this under controlled conditions.

### Benchmark matrix

*TBD (pending benchmark)*

**Configuration:** Qwen3.5-9B-Q4_K_M, context=4096 tokens. Separate Vulkan
rows use the same logical K/V intent but annotated with the actual Vulkan KV
format as stored. GPU class will be stated per row.

#### ROCm — primary matrix

| # | K / V | PPL | TG (t/s) | PP (t/s) | Memory vs F16 KV |
|---|---|---|---|---|---|
| 1 | F16 / F16 (mainline baseline) | TBD | TBD | TBD | 1.0× |
| 2 | Q8_0 / Q8_0 (mainline) | TBD | TBD | TBD | TBD |
| 3 | turboq2 / turboq2 (base PolarQuant) | TBD | TBD | TBD | TBD |
| 4 | turboq3 / turboq3 (base PolarQuant) | TBD | TBD | TBD | TBD |
| 5 | turboq2_tcq / turboq2_tcq | TBD | TBD | TBD | TBD |
| 6 | turboq3_tcq / turboq3_tcq | TBD | TBD | TBD | TBD |
| 7 | turboq3_tcq / turboq2_tcq (asymmetric) | TBD | TBD | TBD | TBD |
| 8 | Q8_0 / turboq3_tcq (high-precision K) | TBD | TBD | TBD | TBD |
| 9 | Q8_0 / turboq2_tcq (high-precision K, aggressive V) | TBD | TBD | TBD | TBD |
| 10 | F16 / turboq3_tcq (V-only compression) | TBD | TBD | TBD | TBD |
| 11 | F16 / turboq2_tcq (V-only, most aggressive) | TBD | TBD | TBD | TBD |

#### Cross-fork — buun origin build (K=1.1/V=1.3 default alpha)

| # | K / V | PPL | TG (t/s) | PP (t/s) | Memory vs F16 KV |
|---|---|---|---|---|---|
| 12 | turboq3_tcq / turboq3_tcq | TBD | TBD | TBD | TBD |
| 13 | turboq2_tcq / turboq2_tcq | TBD | TBD | TBD | TBD |
| 14 | turboq3_tcq / turboq2_tcq (asymmetric) | TBD | TBD | TBD | TBD |

#### Vulkan — rows 5–7 as shipped (αA transient-dequant path)

Stored format is TCQ in both cases; "effective FA format" column states the
actual format the FA shader reads from.

| # | K / V (configured) | Effective FA format | PPL | TG (t/s) | PP (t/s) | Stored memory vs F16 KV |
|---|---|---|---|---|---|---|
| 5V | turboq2_tcq / turboq2_tcq | F16 (transient) / F16 (transient) | TBD | TBD | TBD | TBD |
| 6V | turboq3_tcq / turboq3_tcq | F16 (transient) / F16 (transient) | TBD | TBD | TBD | TBD |
| 7V | turboq3_tcq / turboq2_tcq | F16 (transient) / F16 (transient) | TBD | TBD | TBD | TBD |

---

## §4 How it works under the hood

### Block structures (`ggml/src/ggml-common.h:307-329`)

Each block covers 128 elements (`QK_TURBOQ3_TCQ = QK_TURBOQ2_TCQ = 128`).

**`block_turboq3_tcq`** — 52 bytes (`ggml-common.h:312-316`):
```
[norm: fp16, 2B] [qs[49]: 390-bit trellis bitstream (k=3, 128 symbols + 6-bit prefix)] [pad: 1B]
```
3.25 bits/value → ~4.9× compression vs fp16.
`static_assert(sizeof(block_turboq3_tcq) == sizeof(ggml_half) + 50)`

**`block_turboq2_tcq`** — 36 bytes (`ggml-common.h:324-328`):
```
[norm: fp16, 2B] [qs[33]: 262-bit trellis bitstream (k=2, 128 symbols + 6-bit prefix)] [pad: 1B]
```
2.25 bits/value → ~7.1× compression vs fp16.
`static_assert(sizeof(block_turboq2_tcq) == sizeof(ggml_half) + 34)`

> **Note on code comments vs struct:** the comment above `block_turboq3_tcq` in
> `ggml-common.h` states "3.1875 bits/value → 5.0×." These figures are incorrect;
> the struct static_assert is authoritative. The correct values are 3.25 bpv and
> ~4.9× as derived above.

### Trellis-coded quantization overview

Both `turboq2_tcq` and `turboq3_tcq` use a **right-shift Viterbi trellis** to find
the globally optimal sequence of quantization symbols for each 128-element group.
This is fundamentally different from the nearest-centroid approach used by the base
PolarQuant types: instead of quantizing each element independently, the Viterbi
search minimizes total reconstruction error across all 128 elements jointly.

See [concepts/trellis-coded-quantization.md](concepts/trellis-coded-quantization.md)
for a self-contained explanation of the algorithm.

### Encode pipeline (CUDA/HIP — `ggml/src/ggml-cuda/set-rows.cu`)

For each 128-element group:

1. **L2-normalize** — compute the group L2 norm; divide each element by it.
2. **Forward WHT rotation** (in-kernel, `set-rows.cu:1208-1234`) — a randomized
   Walsh-Hadamard Transform spreads quantization error across the full group. Same
   rotation as the base PolarQuant types; see
   [concepts/hadamard-wht-rotation.md](concepts/hadamard-wht-rotation.md).
3. **Viterbi trellis search** — a GPU-parallel Viterbi search finds the minimum-cost
   sequence of symbols. Each GPU thread handles one trellis state:
   - `turboq3_tcq`: 512 states (k=3, L=9), 512 threads/block (`set-rows.cu:1109`)
   - `turboq2_tcq`: 256 states (k=2, L=8), 256 threads/block (`set-rows.cu:1414`)
   Path costs use a double-buffered scheme (reduces syncs from 384 → 128 per group);
   the backtrace lives in preallocated global memory or dynamic shared memory
   depending on device capability.
4. **Corrected norm** — after Viterbi, compute the L2 norm of the reconstructed
   values, divide the input norm by it (pre-compensating codebook approximation
   error), then scale by the alpha knob. Store as `block.norm`.
5. **Bitpack** — pack the 128 output symbols and the 6-bit initial-state prefix into
   the `qs[]` bitstream.

### Decode pipeline (CPU and GPU)

Decode is a sliding-window bitstream read: for element `t`, read `k` bits starting
at bit `t*k`, look up `codebook[state]`, multiply by the stored norm. No SIMD or
Viterbi required; decode works on CPU and is used in the GPU FA kernel inline
(`ggml/src/ggml-cuda/turbo-quant.cuh:662` — `dequantize_turboq2_tcq`).

### Vulkan αA pre-dequant path (`ggml/src/ggml-vulkan/ggml-vulkan.cpp:9853-9883`)

Before dispatching the FA shader, any TCQ-typed K or V tensor is:
1. Detected (`k_is_tcq` / `v_is_tcq` flags).
2. Dequantized into a **preallocated F16 scratch buffer**
   (`prealloc_k_tcq_dequant` / `prealloc_v_tcq_dequant`).
3. The FA shader is dispatched against the F16 scratch buffer with the
   `effective_k_type` / `effective_v_type` set to `GGML_TYPE_F16`.

The scratch buffers are persistent across FA calls (preallocated to the maximum
expected KV window) and reused each decode step.

---

## §5 Further reading

- **Upstream source:** [spiritbuun/buun-llama-cpp](https://github.com/spiritbuun/buun-llama-cpp)
- **Related docs (this repo):**
  - [TurboQuant KV base](turboquant-kv-base.md) — base PolarQuant types (`turboq2`/`turboq3`/`turboq4`); comparison baseline for TCQ
  - [docs/TYPE_ASSIGNMENTS.md](../TYPE_ASSIGNMENTS.md) — slot assignments and upstream-name mapping
  - [docs/features/README.md](README.md) — index of all feature docs
- **Concept primers:**
  - [Trellis-coded quantization primer](concepts/trellis-coded-quantization.md) — Viterbi trellis, codebook, and why TCQ beats nearest-centroid at the same bit-width
  - [WHT / Hadamard rotation primer](concepts/hadamard-wht-rotation.md) — the WHT rotation stage shared with the base PolarQuant types
  - [Asymmetric KV cache primer](concepts/asymmetric-kv-cache.md) — K/V asymmetry rationale and pair selection
  - [Feature maturity levels & backend support](concepts/feature-maturity-levels.md)
