# OScaR INT2 K-cache (`kv_oscar_int2`)

> **Status: Experimental — CUDA/HIP only (Phase 1 prototype)**
>
> OScaR INT2 (`GGML_TYPE_KV_OSCAR_INT2`, slot 71) is a Phase 1 CUDA-only
> K-cache quantization type. Vulkan and ROCm backends are deferred to Phase 2.
> Phase 1 gate was run on Qwen3.5-9B; smaller models (0.8B) show architectural
> underperformance — see PPL notes below.

---

## At a glance

| | Value |
|---|---|
| **What it is** | 2-bit K-cache via Fast Hadamard Transform (FHT) + per-block min-max INT2 quantization |
| **Slot** | 71 (`GGML_TYPE_KV_OSCAR_INT2`) |
| **Effective bpw** | ~2.0 bits per element (INT2 with per-block scale + offset) |
| **Block size** | `QK_OSCAR_INT2` (128 elements per block) |
| **Backend** | CUDA/HIP only — Phase 2 (Vulkan/ROCm) deferred |
| **Paper** | arXiv:2605.19660 |
| **Ship commit** | `e1f3e7083` |
| **CLI flag** | `--cache-type-k kv_oscar_int2` |
| **Optional flag** | `--cache-oscar-residual-window N` (default: 128) |

---

## Quick start

```bash
llama-speculative-simple \
    -m model.gguf \
    -fa on -ngl 999 --no-mmap \
    --cache-type-k kv_oscar_int2 \
    --cache-oscar-residual-window 512
```

OScaR applies only to the K-cache. The V-cache uses the type specified by
`--cache-type-v` (default: `f16`). OScaR does not replace or depend on
TurboQuant; both can be selected independently.

---

## How OScaR works

OScaR applies a Fast Walsh–Hadamard Transform (FWHT) to each K-cache block
before quantizing to INT2. The FWHT redistributes energy across the block,
making the distribution closer to uniform and reducing the quantization error
of a simple min-max INT2 scheme.

**Residual window.** OScaR maintains an F16 sidecar tensor (`k_res`) holding
the most-recent `--cache-oscar-residual-window` K-rows in full precision. Flash
attention performs a split read: F16 for positions in the tail window, INT2/FWHT
for older blocks. This improves quality for recent tokens at a modest memory
cost proportional to the window size.

```
┌──────────────────────────────┬─────────────────────┐
│  Older K tokens: INT2 + FWHT │  Recent N: F16 sidecar│
│  (compressed, ~2 bpw)        │  (residual window)  │
└──────────────────────────────┴─────────────────────┘
```

---

## PPL results

**Qwen3.5-9B-Q4_K_M** (Phase 1 gate, gfx1150, wikitext-2-raw-test):

| KV type | PPL | vs F16 KV (baseline ~6.6) | Notes |
|---|---|---|---|
| F16 (baseline) | ~6.60 | — | |
| `kv_oscar_int2` | **7.2005** | +9.1% | R=128; **PASS** (gate ≤17% over baseline) |
| `kv_oscar_int2` R=512 | lower (sweep) | −2.77% vs R=128 | R=512 recommended deployment default |

**Known limitation — small models (0.8B):**
Qwen3.5-0.8B showed flat PPL (~21) across the full R=128/256/512/1024 sweep,
failing the ≤17.0 gate. This architectural non-response to the residual window
is tracked as `FOLLOWUP-F` (full-dimension FWHT). Do not use `kv_oscar_int2`
for sub-1B models until FOLLOWUP-F is resolved.

---

## Residual window tuning

| `--cache-oscar-residual-window` | Effect |
|---|---|
| 0 | Disable residual sidecar; pure INT2/FWHT throughout |
| 128 (default) | Conservative; near-minimal memory overhead |
| 512 | Recommended for 9B+ models; −2.77% PPL vs R=128 in sweep |
| 1024 | Diminishing returns in sweep; higher VRAM cost |

The sidecar allocates `window × n_head × head_dim × sizeof(float16)` bytes per
layer. For a 9B model (32 layers, 8 K-heads, head_dim=128): R=512 adds ~134 MB.

---

## Phase status

| Phase | What | Status |
|---|---|---|
| **Phase 1 — CUDA prototype** | `GGML_TYPE_KV_OSCAR_INT2` type, FWHT encode/decode, residual window, CUDA flash-attn dispatch | ✅ Shipped `e1f3e7083` (2026-05-26) |
| **Phase 2 — Vulkan/ROCm backends** | Port FWHT + INT2 decode to Vulkan compute shaders; ROCm HIP path | 🔄 Deferred |
| **FOLLOWUP-F — full-dimension FWHT** | Architectural fix for 0.8B under-performance | 🔄 Identified, not scheduled |

---

## Backend notes

- **CUDA (gfx1150):** Validated. Template instances for `kv_oscar_int2×{f16, bf16, q8_0}` query types in `ggml/src/ggml-cuda/template-instances/fattn-vec-instance-kv_oscar_int2-*.cu`.
- **Vulkan:** Blocked at type dispatch (`ggml-vulkan.cpp`); flag raises a warning and falls back to F16 K-cache.
- **ROCm:** Phase 1 is CUDA-only; no gfx1150/gfx1102 gate has been run. Use F16 or TurboQuant for ROCm K-cache until Phase 2 lands.
- **`head_dim`:** CUDA dispatch requires `head_dim ≤ 128`. Models with larger head dimensions are not supported in Phase 1.

---

## Relationship to other KV types

OScaR occupies slot 71 and is independent of the TurboQuant (`TURBOQ*`, slots 60–69) family (the former RotorQuant family at slots 72–75 was removed; those slots are now reserved). It can coexist with any V-cache type:

```bash
# OScaR K + TurboQuant V
--cache-type-k kv_oscar_int2 --cache-type-v turboq3 --cache-oscar-residual-window 512
```

---

## Further reading

- **Paper:** [arXiv:2605.19660](https://arxiv.org/abs/2605.19660)
- **Type definition:** `ggml/include/ggml.h` line 442 — `GGML_TYPE_KV_OSCAR_INT2 = 71`
- **FWHT encode/decode:** `ggml/src/ggml-quants.c` — `quantize_row_kv_oscar_int2_ref`, `dequantize_row_kv_oscar_int2`
- **Flash-attn dot product:** `ggml/src/ggml-cuda/fattn-vec.cuh` — `// OScaR INT2 K dot product` comment block
- **Residual window management:** `src/llama-kv-cache.cpp` — `llama_kv_cache::alloc` (k_res allocation), `src/llama-graph.cpp` — `ggml_flash_attn_ext_set_oscar_res()`
- **CLI flag:** `common/arg.cpp` — `--cache-oscar-residual-window`
- **Feature index:** [docs/features/README.md](README.md)
- **Related docs (this repo):**
  - [TurboQuant KV base](turboquant-kv-base.md) — 2/3/4-bit KV cache (CUDA/HIP/Vulkan)
  - [TCQ KV cache](tcq-kv.md) — Viterbi-coded KV cache
  - [InnerQ KV cache](innerq-kv.md) — InnerQ calibrated equalization
