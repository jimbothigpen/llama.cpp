# WQ3 TCQ — 3-bit Trellis-Coded Weight Quant (`WQ3_TCQ`)

**Status: CUDA-only, Phase 1 of 4 (adopt + build-verified).** The CUDA runtime (dequant + native
decode + FFN fusion) is ported and builds clean on CUDA sm_75. **No quantizer ships yet** — a WQ3
GGUF cannot be produced in-tree until Phase 2 (see [Limitations](#limitations)). CPU, HIP/ROCm and
Vulkan backends are Phases 2–4 (see the port plan).

Source: buun `feat/tcq-wq3-ffn-fusion` (commits `704cc7780` + `77355efa2` + `d0e929b0e`). TCQ lineage
credit: TheTom (TurboQuant TCQ). See `PROVENANCE.md`.

---

## At a glance

| Property | Value |
|---|---|
| `ggml_type` | `GGML_TYPE_WQ3_TCQ` = **92** (`ggml/include/ggml.h`) |
| gguf-py | `GGMLQuantizationType.WQ3_TCQ` = 92, block `(128, 52)` |
| Block layout | reuses `block_turboq3_tcq` — 52 bytes / 128 values, **3.25 bpv** (~4.9× vs fp16) |
| Quantizes | **weights** (FFN/attention projections), not KV cache |
| Dequant | **GPU-only** (`to_float = NULL`, `from_float_ref = NULL` in type-traits) |
| Backends | CUDA ✅ (Ph1) · CPU ⏳ (Ph2) · HIP ⏳ (Ph3) · Vulkan ⏳ (Ph4) |
| CLI ftype | **none yet** — no `llama-quantize` path (Ph2) |

## Mechanism

WQ3_TCQ stores weights in the **FWHT-rotated domain** using a 3-bit trellis-coded quantizer
(k=3, L=9, 512 trellis states, Viterbi-encoded). It shares the exact wire format of the TCQ KV-cache
type (`block_turboq3_tcq`; see [`tcq-kv.md`](tcq-kv.md)): one block = one 128-element rotation group =
2-byte corrected L2 norm + 49-byte 390-bit trellis bitstream + 1 pad byte.

Decode (per block):
1. Read 9-bit trellis state per 128 outputs (`state_t = read_9_bits(qs, t*3)`).
2. Look up a **1024-entry codebook** (stored in GGUF metadata, loaded to CUDA `__constant__`).
3. Apply the **inverse FWHT rotation**: `signs2 → butterfly → signs1 → normalize` (×1/√128).

Two CUDA paths:
- **Dequant-to-fp16/fp32 → cuBLAS** for prefill / batch > 1 (`dequantize_wq3_tcq_to_fp16/fp32`,
  wired into `convert.cu`).
- **Native mmvq** (fused dequant + dot-product, TILE_M register tiling) for batch-1 decode, plus a
  **fused gate+up GLU** launcher for FFN fusion (`ggml_cuda_wq3_tcq_mmvq_fused_gate_up_glu`).

### Required GGUF metadata
The loader (`src/llama.cpp`, under `GGML_USE_CUDA`) requires, for any model containing WQ3_TCQ tensors:
- `turbo.tcq.codebook.weight` **or** `turbo.tcq.codebook.default` — 512 or 1024 floats.
- FWHT signs: either `turbo.tcq.signs1` + `turbo.tcq.signs2` (128 floats each) **or**
  `turbo.tcq.sign_seed` (u32; default 42).

## Files (CUDA Phase 1)

| File | Role |
|---|---|
| `ggml/src/ggml-cuda/wq3-tcq.cu` / `.cuh` | dequant + native-decode + fused-GLU kernels, codebook/sign setup |
| `ggml/src/ggml-cuda/template-instances/mmq-instance-wq3_tcq.cu` | MMQ tile loader (q8_1 path) |
| `ggml/src/ggml-cuda/{mmq,convert,getrows,common,ggml-cuda}.c*` | type dispatch wiring |
| `ggml/include/ggml.h`, `ggml/src/ggml.c` | enum slot 92 + type-traits |
| `gguf-py/gguf/{constants,quants}.py` | Python type registration (quant/dequant raise `NotImplementedError`) |
| `src/llama.cpp` | codebook/sign loading from GGUF metadata |
| `src/llama-graph.{cpp,h}`, `src/llama-model.h`, `src/models/qwen35.cpp` | FFN-fusion + `act_scale` graph wiring |

## Slot decision

buun's upstream inserts `WQ3_TCQ` mid-enum at **46**, which renumbers `TURBO2_TCQ`/`TURBO8_0`/`COUNT`.
Our tree has long relocated all TurboQuant types into the 60–95 fork zone, so we re-slotted WQ3_TCQ to
**92** (the unanticipated-weight-quant reserve), avoiding a GGUF-breaking renumber and dodging the
in-flight `turboq5/6` (64/65) and `WHT5/6/8` (83–85) reservations. See `docs/TYPE_ASSIGNMENTS.md`.

## Limitations

- **No quantizer in-tree.** quants.py raises `NotImplementedError`; type-traits `from_float_ref = NULL`;
  there is no `llama-quantize` ftype. buun produces WQ3 GGUFs via an external research pipeline. A WQ3
  GGUF therefore **cannot be produced or PPL-validated in Phase 1** — building the quantizer + codebook
  generator + GGUF metadata writer is the first Phase-2 deliverable.
- **CUDA-only.** Non-CUDA backends abort on WQ3 tensors until Phases 2–4 land.
- **MMQ batched path** asserts `ne12 == 1 && ne13 == 1` (batched src1 not implemented).

## Port plan

Multi-backend roadmap (Ph2 CPU+imatrix, Ph3 HIP/ROCm, Ph4 Vulkan) tracked in the orchestrator's
`WQ3-PORT-PLAN.md`. Phase 1 verification was a **full CUDA build on ai02 (0 errors)**; the end-to-end
quantize+decode smoke is deferred to after the Phase-2 quantizer.
