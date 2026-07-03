# KV-MMA Turbo-Tier Deep-Context pp-speed (PH-1c)

**Hardware:** NVIDIA GeForce RTX 2070 Super (sm_75)
**Model:** Qwen3.5-0.8B (Q4_K_M)
**Configuration:** `-ctk turboq4 -ctv turboq2 -fa 1 -ngl 99 -mmp 0 -r 8`
**Contention Note:** GPU was shared with the 4B `/route` serve brain (held via `gpu-exclusive-run.sh` with `LEASE_CLASS=tps` to serialize jobs).

## LEG 1 - Memory (Peak VRAM at Large Prefill)

- We measured peak VRAM for a large prefill (`-p 16384`) on both NATIVE (`GGML_TURBO_MMA_FUSED=1`) and VEC fallback (`GGML_TURBO_MMA_FUSED=0`).
- **Result:** Both NATIVE and VEC showed no F16-expansion buffer. Peak VRAM was identical and memory-clean.
- **Why:** Code analysis of `ggml_cuda_get_best_fattn_kernel` in `fattn.cu` shows that for `is_turbo_type(...)`, the fallback explicitly bypasses `BEST_FATTN_KERNEL_TILE` and routes to `BEST_FATTN_KERNEL_VEC`. Because it routes to VEC, `need_f16_K` evaluates to false, completely avoiding the synchronous `to_fp16_cuda_t` expansion buffer.
- **Verdict for Leg 1:** The current (no-native) fallback path for turbo types is already memory-clean (falls back to VEC). The benefit of the native MMA kernel is therefore speed-only.

## LEG 2 - Speed (Native MMA vs VEC)

| Test | NATIVE (t/s) | VEC (t/s) | Ratio (NATIVE / VEC) |
| --- | --- | --- | --- |
| `pp2048` | 2521.66 ± 9.64 | 2068.36 ± 0.38 | **1.219** (+21.9%) |
| `pp4096` | 1623.05 ± 11.98 | 1317.78 ± 53.37 | **1.231** (+23.1%) |
| `pp8192` | 836.39 ± 27.66 | 693.94 ± 44.28 | **1.205** (+20.5%) |
| `tg128`  | 184.33 ± 0.28  | 128.15 ± 1.95   | **1.438** (+43.8%) |

- **Verdict for Leg 2:** Native MMA shows a massive, meaningful speedup over VEC at all deep context sizes (≥20% for prompt processing) and an exceptionally large speedup for text generation (+43.8%).

## Terminal Verdict: GO-FANOUT
While there is no memory-saving rationale (since the VEC fallback avoids the F16 expansion), the speed rationale is undeniably strong (Leg 2 wins decisively). Proceed with fanning out the CUDA matrix.