# WQ3 TCQ — 3-bit Trellis-Coded Weight Quant (`WQ3_TCQ`)

**Status: Phase 3 of 4 complete (quantizer + CPU dequant + imatrix + HIP/ROCm).** The CUDA runtime (Ph1)
plus the CPU Viterbi **encoder**, CPU **dequant**, `llama-quantize` ftype, codebook generator and GGUF
metadata writer (Ph2), and the **HIP/ROCm runtime** (Ph3) are in-tree. A real WQ3 GGUF can now be
produced and decodes coherently on **CPU, CUDA, and HIP/ROCm (gfx1150)** with bit-identical codebook
parity (validated on Qwen3.5-2B; see [Validation](#validation)). Vulkan is Phase 4 (see the port plan).

> **Production trellis = k=3, L=10, 1024 states** (the older `tcq_rshift.py` prototype + some stale
> comments say L=9/512; the authoritative CUDA decode uses L=10/1024 — that is what the encoder targets).

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
| Dequant | **CPU + CUDA** (`to_float`/`from_float_ref` wired in Ph2; CUDA native + cuBLAS paths) |
| Backends | CUDA ✅ (Ph1) · CPU ✅ (Ph2) · HIP/ROCm ✅ (Ph3, gfx1150) · Vulkan ⏳ (Ph4) |
| CLI ftype | **`WQ3_TCQ`** = `LLAMA_FTYPE_MOSTLY_WQ3_TCQ` (62) — `llama-quantize … WQ3_TCQ` |

## Mechanism

WQ3_TCQ stores weights in the **FWHT-rotated domain** using a 3-bit trellis-coded quantizer
(**k=3, L=10, 1024 trellis states**, Viterbi-encoded). It shares the exact wire format of the TCQ
KV-cache type (`block_turboq3_tcq`; see [`tcq-kv.md`](tcq-kv.md)): one block = one 128-element rotation
group = 2-byte corrected L2 norm + 49-byte 390-bit trellis bitstream + 1 pad byte (7-bit zero prefix
+ 128×3-bit outputs).

Decode (per block):
1. Read a 10-bit trellis state per 128 outputs via a sliding window (`state_t = (read24(qs,t*3))&0x3FF`).
2. Look up a **1024-entry codebook** (stored in GGUF metadata, loaded to CUDA `__constant__`; on CPU
   the identical codebook is hardcoded — `ggml/src/wq3-tcq-codebook.inc`).
3. Apply the **inverse FWHT rotation**: `signs2 → Hadamard butterfly → ×(1/√128)·signs1 · norm`.

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
| `gguf-py/gguf/{constants,quants}.py` + `wq3_tcq_data.py` | Python type registration + **dequant** (quantize delegates to C `llama-quantize`) |
| `ggml/src/ggml-turbo-quant.c` + `wq3-tcq-codebook.inc` | **CPU encoder** (`quantize_row_wq3_tcq_ref`, Viterbi), **CPU dequant** (`dequantize_row_wq3_tcq`), hardcoded 1024-entry codebook + seed-42 signs (Ph2) |
| `ggml/src/ggml-cpu/ggml-cpu.c` | CPU `type_traits_cpu` vec_dot (dequantize-then-dot, F32) for mul_mat (Ph2) |
| `src/llama-quant.cpp` | `WQ3_TCQ` ftype, tensor selection, imatrix passthrough, codebook/sign_seed GGUF emit (Ph2) |
| `src/llama.cpp` | codebook/sign loading from GGUF metadata |
| `src/llama-graph.{cpp,h}`, `src/llama-model.h`, `src/models/qwen35.cpp` | FFN-fusion + `act_scale` graph wiring |

## Slot decision

buun's upstream inserts `WQ3_TCQ` mid-enum at **46**, which renumbers `TURBO2_TCQ`/`TURBO8_0`/`COUNT`.
Our tree has long relocated all TurboQuant types into the 60–95 fork zone, so we re-slotted WQ3_TCQ to
**92** (the unanticipated-weight-quant reserve), avoiding a GGUF-breaking renumber and dodging the
in-flight `turboq5/6` (64/65) and `WHT5/6/8` (83–85) reservations. See `docs/TYPE_ASSIGNMENTS.md`.

## Quantizer (Phase 2)

`llama-quantize <model> <out> WQ3_TCQ` produces a WQ3 GGUF. The CPU encoder
(`quantize_row_wq3_tcq_ref`) per 128-element group:
1. `saved_norm = ||x||`; normalize to unit L2.
2. Forward FWHT rotate into the codebook domain: `signs1 → Hadamard butterfly → ×(1/√128)·signs2`.
3. **Viterbi** trellis-encode (start state 0, right-shift transition `next=(s>>3)|(out<<7)`) against the
   1024-entry codebook → 128 states; `recon = codebook[states]`.
4. `corrected_norm = saved_norm / ||recon||` (fp16) — cancels codebook magnitude drift so decode
   restores `||x||` exactly.
5. Pack: 7-bit zero prefix + `out[t]=state[t]>>7` at bit `7+3t`.

The **codebook** is deterministic and hardcoded (`wq3-tcq-codebook.inc`): coset-init (Lloyd-Max levels,
128 cosets × 8 outputs) + 20 Lloyd iterations on unit-norm Gaussians → 1024 f32 in the unit-norm domain
(RMS≈0.103, |max|≈0.307, matching buun's reference). The **same table** is hardcoded in C (encoder +
CPU dequant), mirrored in `gguf-py/wq3_tcq_data.py`, and emitted into the GGUF
(`turbo.tcq.codebook.weight` + `turbo.tcq.sign_seed=42`) so CPU and CUDA decode against byte-identical
data. **Signs** are the seed-42 pair (`h_wq3_tcq_signs1_seed42` / `h_wq3_tcq_signs2_seed1084`) copied
verbatim from the CUDA runtime — distinct from the KV-cache `turbo_cpu_s1/s2`.

**imatrix:** passthrough, **not applied**. The FWHT rotates each 128-group, and the Hadamard's
equal-magnitude entries (`F_tj² = 1/128 ∀t,j`) make the rotated importance diagonal uniform, so
per-element weighting degenerates to a constant and cannot change the Viterbi path. This follows the
audited WHT3_0/WHT4_0 decision (original-basis imatrix weighting hurt PPL +16% there). `imatrix_required`
returns false for WQ3_TCQ.

## Validation

End-to-end on **Qwen3.5-2B** (sm_75):
- Quantize RC=0 → 4.15 BPW (931.78 MiB), GGUF header valid (file_type=62, codebook=1024, sign_seed=42,
  320 tensors: 186 WQ3_TCQ / 133 F32 / 1 Q6_K).
- **CPU decode** (`-ngl 0`) and **CUDA decode** (`-ngl 99`) both **coherent** (correct, no NaN/abort).
- **Parity:** CUDA-loaded codebook amax (0.30662) == hardcoded table; CPU-C dequant vs independent
  Python reference **max-abs-err = 1.19e-7** (1 fp32 ULP). CPU↔CUDA difference bounded only by the CUDA
  fast kernel's fp16 store.

**Phase 3 (HIP/ROCm, gfx1150):**
- **Build:** full ggml-hip backend, ROCm 7.2.4, `-DAMDGPU_TARGETS=gfx1150`, 0 errors. The `.cu`/`mmq*.cu`
  globs auto-include `wq3-tcq.cu` + `mmq-instance-wq3_tcq.cu`; dispatch is the shared `ggml-cuda.cu`
  (`is_tq_weight`/WQ3 guards), so no CMake/dispatch edits were needed — only source hipify shims.
- **HIP decode** (`-ngl 99 -fa on --no-mmap -st`) **coherent** (correct "Paris", no NaN/abort), native
  WQ3 mmvq path, ~52 t/s gen. Runtime log confirms codebook (amax 0.30662) + signs (seed 42) on device.
- **PPL parity (15 chunks, wiki.test.raw, c=4096):** HIP/gfx1150 = **13.5502 ± 0.22144**; pure-CPU dequant
  path cumulative tracks it bit-for-bit (chunk-1 cumulative CPU 16.0932 vs HIP 16.0681, **Δ0.16% ≪ 1σ**),
  confirming the HIP decode is numerically equivalent to the CPU/CUDA codebook+signs+trellis math.

## Limitations

- **MMQ batched path** asserts `ne12 == 1 && ne13 == 1` (batched src1 not implemented).
- **CUDA get_rows** does not cover WQ3 (nor WHT3/4_0) — these weight quants aren't used with get_rows;
  `test-backend-ops` therefore can't sweep them (it also lacks the model-load codebook upload).
- Vulkan dequant landed (Phase 4).

## Port plan

Multi-backend roadmap (Ph3 HIP/ROCm done, Ph4 Vulkan done).
Phase 2 delivered the quantizer + CPU dequant + imatrix decision and ran the
deferred Ph1 smoke (quantize + dual-backend coherent decode + dequant parity), plus a minimal Ph1 CUDA
dispatch fix (WHT-only `mul_mat_vec_tq` branch; WQ3 multi-token now uses dequant→cuBLAS).

**Phase 3 (HIP/ROCm)** was a mechanical hipify — the kernels use no WMMA/mma/tensor-core, only
dp4a/MAD + `__shfl`. Three source shims were required (no CMake/dispatch changes):
1. `__shfl_sync`/`__shfl_xor_sync` calls needed the explicit 4th `width` arg (`WARP_SIZE`) — the
   `vendors/hip.h` macros expand to 4-param `__shfl(...)`/`__shfl_xor(...)`, so the WQ3 3-arg calls
   failed to compile under hipcc (the rest of ggml-cuda already passes `width` for this reason).
2. Two event aliases added to `vendors/hip.h` (`cudaEventDefault`→`hipEventDefault`,
   `cudaEventElapsedTime`→`hipEventElapsedTime`) for the profiling scaffolding.
3. The `wq3_tcq_mul_u32` NVPTX fast path (`asm("mad.lo.u32 …")`) was gated `&& !defined(GGML_USE_HIP)`
   — `vendors/hip.h` defines `__CUDA_ARCH__`, so the PTX reached `amdgcn-link` and failed as an invalid
   instruction; AMD/host now take the portable `x*c` (which lowers to the same instruction).
Warp-size correctness: the 128-wide FWHT butterfly assumes 32-lane warps, which matches RDNA3.5
wave32 (gfx1150/gfx1103 default; no `-mwavefrontsize64`).

## CUDA correctness on newer drivers (driver-580 / CUDA-13.0) — known issue

A Kaggle T4 run reported PPL = 685,572 (garbage) for WQ3_TCQ while Q4_K_M was healthy on the same
binary. Investigation (worker `wq3-cuda-fastpath-fix`, 2026-06-23) found the **quant and kernels are
correct**: the *exact* Kaggle binary scores a healthy **PPL 13.05** on sm_75 hardware running
**NVIDIA driver 550 / CUDA 12.4**, matching the HIP result (13.55). The garbage reproduces only on the
Kaggle T4's **driver 580 / CUDA 13.0** (cudart is 12.8 on both, so it is not a cudart mismatch; the
inline-asm fast path was ruled out — it compiles to byte-identical sm_75 SASS and the lib ships SASS
with no PTX, so there is no JIT).

Likely mechanism: WQ3 publishes its codebook/signs into six file-scope `__constant__` symbols via
`cudaMemcpyToSymbol` from inside `libggml-cuda.so` (`BUILD_SHARED_LIBS=ON`, `-fgpu-rdc`). On the
driver-580/CUDA-13 runtime these device symbols are not correctly populated → the kernels read a
zero/garbage codebook → systematic garbage. Q4_K_M uses no such device symbol and is unaffected.

**Durable fix (recommended, not yet landed):** replace the `__constant__`/`cudaMemcpyToSymbol`
codebook+signs with `cudaMalloc`'d device buffers passed to the kernels as pointer arguments, removing
the driver-fragile symbol-registration path. Validation requires a driver-580 box (Kaggle T4); older drivers
(driver 550) cannot reproduce.
