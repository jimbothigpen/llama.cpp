# Repo context STATE — off-git (do NOT git add)

**Current branch:** `main` (HEAD **`8c27d346e`** — fix: port crossfork P1 correctness/crash fixes (WHT get_rows + RPC op-count + Vulkan cm1 f16vec4))
**Origin synced:** yes (main...origin/main 0/0)

Full orchestrator state in `kernel-work/context/STATE.md`.

---

## crossfork P1 landing — 2026-06-29

Commit `8c27d346e` ports three correctness/crash fixes from the crossfork onto main:

1. **WHT get_rows CUDA** (`ggml/src/ggml-cuda/getrows.cu`): Added `GGML_TYPE_WHT4_0` and `GGML_TYPE_WHT3_0` cases to `ggml_cuda_get_rows_switch_src0_type()`, enabling CUDA get_rows for WHT quant types.
2. **RPC op-count** (`ggml/include/ggml-rpc.h`): Bumped `RPC_PROTO_PATCH_VERSION` from 1→2 and updated `static_assert(GGML_OP_COUNT == 97)` to `== 99` to match current op enum size.
3. **Vulkan cm1 f16vec4** (`ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm1.comp`): Wrapped `dequantize4()` return in `f16vec4()` at four call sites (K-path lines 238,281; V-path lines 408,459) to fix implicit narrowing warnings / potential type mismatch on certain drivers.

**OWED follow-up:** Build-verify `ggml/src/ggml-cuda/getrows.cu` changes on a real CUDA host (ai02/Kaggle). The WHT4_0/WHT3_0 get_rows additions were only HIP-compile-verified so far.

---

## § 27B VRAM campaign — measured 2026-06-26

**Model:** Qwopus3.6-27B-Coder-IQ4_KS.gguf · ai00 ROCm · hd=256 hybrid
(64 layers: 16 full-attn at `full_attention_interval=4`, 48 gated-delta/SSM fixed-state)

### Measured VRAM (ai00 ROCm, IQ4_KS weights ~12.6 GiB loaded)

| Config | ctx | VRAM peak |
|---|---|---|
| `-ctk turboq6 -ctv turboq3` (q6q3) | 256K | 19.10 GiB |
| `-ctk turboq6 -ctv turboq3` (q6q3) | 512K | 23.98 GiB |
| `-ctk turboq6 -ctv turboq3` (q6q3) | 768K | 28.85 GiB |
| `-ctk turboq6 -ctv turboq3` (q6q3) | **1M** | **33.73 GiB** |
| `-ctk q6_0 -ctv q5_0` (q6q5 ref) | 1M | 37.6 GiB |
| IQ4_KSS `-ctk turboq5 -ctv turboq3` (q5q3) | **1M** | **31.02 GiB** (measured; decode 5.7 t/s, coherent) |

**Analytic key:** only 16 full-attn layers scale with ctx (KV/token ≈ 18.5 KiB);
48 gated-delta layers = fixed SSM state (~const), so VRAM growth is ~16/64 of a pure-transformer.

### Kaggle 2×T4 (16 GiB × 2) verdict

- **27B@768K = OOM** — root cause: compute-buffer on the hot card (~3.4 GiB unsplit), not total capacity.
  q6q3 768K total ~28.85 GiB (29 GiB) exceeds single-card 16 GiB + residual room.
- **27B@8K fits** (confirmed from earlier probe).
- **Max-ctx ceiling between 8K and 768K — UNPROBED.** On-demand 27B Kaggle serve @768K NOT viable.

### Conclusion

- ai00 (96 GiB carve) comfortably fits 27B@1M q6q3 (33.73 GiB) with ~62 GiB headroom.
- Kaggle 27B is low-ctx-only; ceiling needs a targeted probe if needed.
- turboq5/turboq6 as K-cache confirmed viable for 27B serving (no abort/fatal — FA matrix fix landed `393307e58`).

Sources: `kernel-work/orchestrator-handoff-fattn-kv-matrix-2026-06-26.md` +
`kernel-work/orchestrator-inbox/completed/processed/orchestrator-brief-iq4kss-q5q3-1m-vram-2026-06-26.md`
