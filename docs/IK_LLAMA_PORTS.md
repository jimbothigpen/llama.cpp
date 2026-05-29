# ik_llama subsystem port tracker

ik_llama is NOT a git remote we merge from — it forked from mainline years
ago and has independent history. We port from it subsystem-by-subsystem,
each as its own topic branch with its own PR. This document tracks the
status of each subsystem port.

The companion type-ID contract lives at [TYPE_ASSIGNMENTS.md](TYPE_ASSIGNMENTS.md).

## Status legend

- **pending-recon** — Phase 0.5 recon has not yet classified this subsystem.
- **recon-additive** — Recon classified as additive. Can land any time.
- **recon-structural** — Recon classified as structural. Must inform
  architecture decisions before downstream layers land.
- **port-in-progress** — Active porting work.
- **ported** — Landed on this fork, regression-tested.
- **dormant** — ik_llama work itself is shelved; we may skip the port.
- **declined** — We chose not to port; record the reason.

## Subsystem inventory

### Quantization types

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| IK "K" family (IQ2_K, IQ3_K, IQ4_K) | ygg canonical IDs: IQ2_K=137, IQ3_K=138, IQ4_K=139; ft2 IDs 58/59/60 (renumbered at landing) | **ported** (Phase 5b-1a, 2026-05-20 to 2026-05-22, `aed6d2965`) | CPU + CUDA/HIP + Vulkan on main. PPL parity Δ < 0.0045 vs ft2 reference. Vulkan batched mul_mat SEGV also fixed (`5fe804bcd`). No row_meta; standard `ggml_type_size`-only layout. |
| IK "KS" row-meta family (IQ3_KS, IQ4_KS, IQ4_KSS) | ygg canonical IDs: IQ4_KS=144, IQ4_KSS=146, IQ3_KS=156; ft2 IDs 61/63/62 | **ported** (Phase 5b-1b, 2026-05-22, `026671689` + `5fe804bcd`) | P0 prereq `row_meta_size` infra landed first (`d91059253`). CPU + CUDA/HIP + Vulkan on main. Vulkan batched mul_mat SEGV fixed via `is_empty()` guard. PPL gate (20-chunk Vulkan): Δ ≤ 0.043 vs ROCm anchors. Row-meta sizes: IQ3_KS=2 bytes (uint16_t half-row-scale), IQ4_KS=4 bytes (float row-scale), IQ4_KSS=4 bytes. |
| IK trellis weight quant (IQ4_KT) | ygg canonical ID: IQ4_KT=155; ft2 ID 64 | **ported** (Phase 5b-1b, 2026-05-22, bundled with KS family) | CPU + CUDA/HIP + Vulkan on main. Depends on row_meta_size infra (P0). PPL gate: Vulkan 6.5364 vs ROCm 6.5701 anchor (Δ −0.034). IQ4_KT is a weight quant; buun TCQ is a separate KV-cache quant. The ik_llama trellis branches (IQ2_KT, IQ3_KT, IQ1_KT) remain **dormant**. |
| IK "K" extended (IQ5_K, IQ6_K) | ygg canonical IDs: IQ5_K=140, IQ6_K=141 | **ported** (Phase 5b-2; CPU+CUDA/HIP `8e19be061` 2026-05-24; Vulkan shaders `0ade7ff86` 2026-05-25) | CPU + CUDA/HIP + Vulkan on main. Imatrix required per PM-15. |
| IK "K-L" extended (IQ2_KL) | ygg canonical ID: IQ2_KL=157 | **ported** (Phase 5b-1c; CPU+CUDA/HIP `f18a92a42` 2026-05-24; Vulkan shaders `3723c1f61` 2026-05-25) | 2.6875 bpw ultra-low-bitrate. CPU + CUDA/HIP + Vulkan on main. Imatrix required per PM-15. |
| BitNet (IQ1_BN, IQ2_BN) | type IDs 134–135 | pending-recon | First-class BitNet support — ik_llama is the only fork with this. |
| Q8 K-block variants (Q8_K16/K32/K64/K128/KR8/KV) | type IDs 136, 147–151 | pending-recon | Workhorses for K-quant intermediate compute. |
| Q6_0 (revived legacy) | type ID 133 | pending-recon | ik_llama kept this after mainline removed it. |
| Q8_*_X4 interleaved | type IDs 97–99 | pending-recon | |
| Row-interleaved R-suffix variants | type IDs 202–230 | pending-recon | Wide surface. May be one big port or split per family. |
| Trellis weight quants — IQ2_KT (Phase P3a) | ygg canonical ID: IQ2_KT=153 | **ported** (Phase 5 Trellis P3a; template refactor `e9520caac` + port `0dac276d9` + cluster-accel `1e8501e46`, all 2026-05-25) | 2.125 bpw `IQKTParams<8, 16, false>` via new `ggml/src/ggml-iqk-kt-family.hpp` template family header. CPU + CUDA/HIP on main. Imatrix required per PM-15. PPL §-FLAG on Qwen3.5-0.8B (107.87 vs IQ2_KL 26.12, IQ4_KT 11.43 — anomaly under investigation). Cluster-accel via 8D base-3 hash + `k_neighbours=60` (~30× quantize speedup over brute-force; PPL +8.3% vs brute-force baseline, §-FLAG). Vulkan path not yet ported. |
| Trellis weight quants — IQ3_KT, IQ1_KT (Phase P3b, P3c) | ygg canonical IDs: IQ3_KT=154, IQ1_KT=158 | port-in-progress (P3b on branch `feature/trellis-iq3-kt-port`; P3c queued) | ADR-018 supersedes the earlier ADR-010 trellis-decline verdict (ADR-010 economic analysis flipped once IQ4_KT shipped; PM-15 directive 2026-05-24 authorizes IQ2/3/1_KT port). Template params: IQ3_KT `IQKTParams<8, 16, true>` (kNumVal=65536, is_abs); IQ1_KT `IQKTParams<8, 13, false>` (kNumVal=8192). Cluster-accel `k_neighbours=60` to be applied at each `iqkt_cooked_book_init` call site. |

### IK quant lift-disposition summary (2026-05-22)

```
Prereq commit (must land first, ~1 session):
  P0: row_meta_size infra — add field to ggml_type_traits struct + extend ggml_row_size()

Phase 5b-1a — Base K family (no row_meta, simplest):
  IQ2_K (ft2 60→ygg 137) — LIFT-WITH-GLUE
  IQ3_K (ft2 59→ygg 138) — LIFT-WITH-GLUE
  IQ4_K (ft2 58→ygg 139) — LIFT-WITH-GLUE
  Source files: ggml-iqk-quants.c, ggml-iqk-kt.cpp (CPU); 6 .comp shaders + types.glsl + ggml-vulkan.cpp wiring
  HIP: convert.cu + mmvq-iqk.cu (IQ2_K/IQ3_K/IQ4_K sections)

Phase 5b-1b — Row-meta KS family + IQ4_KT (requires P0 first):
  IQ4_KS  (ft2 61→ygg 144) — LIFT-WITH-GLUE
  IQ4_KSS (ft2 63→ygg 146) — LIFT-WITH-GLUE
  IQ3_KS  (ft2 62→ygg 156) — LIFT-WITH-GLUE
  IQ4_KT  (ft2 64→ygg 155) — LIFT-WITH-GLUE
  Source files: same ggml-iqk-quants.c + ggml-iqk-kt.cpp; 8 more .comp shaders
  HIP: convert.cu + mmvq-iqk.cu (row-meta sections)
```

Type ID renumber map (ft2 ID → ygg canonical ID):
| ft2 slot | ft2 name | ygg canonical ID | ygg name |
|---|---|---|---|
| 58 | IQ4_K | 139 | GGML_TYPE_IQ4_K |
| 59 | IQ3_K | 138 | GGML_TYPE_IQ3_K |
| 60 | IQ2_K | 137 | GGML_TYPE_IQ2_K |
| 61 | IQ4_KS | 144 | GGML_TYPE_IQ4_KS |
| 62 | IQ3_KS | 156 | GGML_TYPE_IQ3_KS |
| 63 | IQ4_KSS | 146 | GGML_TYPE_IQ4_KSS |
| 64 | IQ4_KT | 155 | GGML_TYPE_IQ4_KT |

Note: IDs 137–156 are in the ik_llama compatibility zone (96–199). GGML_TYPE_COUNT must expand from 82 to at least 157 to hold all the ik_llama canonical IDs. ygg TYPE_ASSIGNMENTS.md explicitly reserves these IDs.

Vulkan architecture compatibility: ft2 uses the SAME per-type standalone .comp shader pattern for these weight quants (separate mul_mat_vec_iq4_k.comp etc.) which matches how ygg handles other IQ types (IQ1_S, IQ1_M, etc.). The weight-quant matvec shaders are architecture-neutral relative to ft2's FA per-type vs ygg's FA uber-shader difference — FA is separate and these types explicitly skip FA in vulkan-shaders-gen.cpp. LIFT is clean on the Vulkan side; only the type-ID enum values and wiring need updating.

**All 7 types are now ported as of 2026-05-22.** CPU + CUDA/HIP + Vulkan for all 7. The lift-disposition notes above (LIFT-WITH-GLUE analysis, type-ID renumber map) are preserved as historical record of the porting work.

### Attention / KV cache

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| MLA (Multi-head Latent Attention) | 20+ branches `ik/cuda_mla*`, `ik/FlashMLA-3`, `ik/cpu_mla_all_quants`, `ik/deepseek_*` | **declined** (2026-05-22, ik-quant-lift) | Post-Phase-1 architecture window passed; requires per-layer conditional v_cache + 4 new hparams (`n_lora_q`, `n_lora_kv`, `n_embd_head_k_full`, `n_embd_head_v_full`); cost exceeds benefit at current project stage. |
| FlashMLA CUDA kernels | `ik/cuda_flash_mla*` family | **declined** (2026-05-22, ik-quant-lift) | Depends on MLA; MLA declined. |
| CPU MLA | `ik/cpu_mla_all_quants`, `ik/cpu_deepseek_fa` | **declined** (2026-05-22, ik-quant-lift) | Depends on MLA; MLA declined. |
| bf16 KV cache | `ik/bf16_kv_cache` | pending-recon | Likely additive. |
| Better Q4_0 KV cache | `ik/better_q40_kv_cache`, `ik/better_q40_kv_cache_cpu` | pending-recon | Likely additive. |
| Fused MoE | `GGML_OP_MOE_FUSED_UP_GATE` op; `iqk_moe_fused_up_gate()` in `ggml/src/iqk/iqk_mul_mat.cpp`; graph builder in `src/llama-build-context.cpp:1014`; PR #1707 + scattered. | **recon-additive** (2026-05-12, session 2) | Adds new ggml op enum (`GGML_OP_MOE_FUSED_UP_GATE`). CPU path depends on IK quants (`IQK_IMPLEMENT`); Metal/Vulkan stubs are fork-original. Port-order: IK quants → fused MoE CPU → fused MoE GPU. |

### Speculative decoding (MTP)

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| MTP foundation (Qwen3.5 MoE, Gemma 4, GLM, Mistral3) | PRs #1736, #1741, #1744, #1745 (Gemma 4), #1758 (multimodal), #1771 (GLM fix) | pending-recon | **Already partially-ported to mainline by turbo-tan's `experiment/gemma4-mtp-upstream-pr`.** Phase 2 of this fork's layer plan uses turbo-tan's port as the foothold. ik_llama subsystem-port becomes "backport ongoing improvements" rather than "port from scratch." |
| MTP graph reuse | PRs #1713, #1728, #1780 | pending-recon | Sits on top of turbo-tan's foundation. |
| MTP per-step SSM optimizations | PRs #1713, #1718, #1724, #1728, #1767, #1773, #1778 | **recon-structural** (2026-05-12, session 2) | Confirmed structural risk: requires ik_llama's `split_s_l_shadow`, dual-graph reuse (`prev_mtp`), and `ggml_delta_net` 6-src-tensor signature. Forces Phase 2 choice: mainline-style MTP foothold (Path α, first-pass recommended) vs ik_llama-style foundation (Path β). |
| MTP async copies | PR #1781 | pending-recon | |
| MTP target slot position | PR #1781 | pending-recon | |
| MTP discard fix | PR #1757 | pending-recon | |

### Backend / infrastructure

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| Scheduler / fattn dispatch tweaks | (recon to enumerate) | pending-recon | Whether these are pervasive enough to be structural is the key question. |
| Type-traits extensions | scattered across many branches | pending-recon | Mostly additive once new ggml_type slots are reserved per [TYPE_ASSIGNMENTS.md](TYPE_ASSIGNMENTS.md). |
| Metal kernels for IK quants | `ik/metal_new_trellis`, scattered | pending-recon | May depend on whether trellis quants are ported. |
| NEON optimizations | `ik/trellis_neon`, scattered | pending-recon | |

## Port methodology

Each subsystem port follows this template:

1. **Recon (Phase 0.5)** — classify additive/structural. ~1 day per subsystem.
2. **Topic branch** — create `port/ik_llama/<subsystem>` off the current
   trunk. Cherry-pick commits one-by-one; do NOT bulk-merge a
   range of ik_llama commits (their history is rebased and unrelated).
3. **Renumber types** — any new ggml_type from this port follows
   [TYPE_ASSIGNMENTS.md](TYPE_ASSIGNMENTS.md).
4. **Type-traits + CPU vecdot + ftype in same commit** — partial landings
   rejected.
5. **PPL regression** — must run before merge to trunk.
6. **Tag** — once merged, tag as `ported/ik_llama/<subsystem>-<date>`. The
   tag records the ik_llama commit SHAs included.
7. **Update this document.**

## Forward-tracking ik_llama upstream

ik_llama lands ~3 PRs/week (mostly MTP). For each new ik_llama PR:

1. Triage: relevant to this fork? (yes/no/maybe)
2. If yes: open a tracking issue with the ik_llama PR URL.
3. Schedule the backport based on which subsystem it targets and what
   already-ported subsystems it depends on.

A weekly ik_llama sweep (every Monday) catches new PRs.

## Declined ports

| Subsystem | Date | Reason |
|---|---|---|
| MLA (Multi-head Latent Attention) | 2026-05-22 | Post-Phase-1 architecture window passed; requires per-layer v_cache refactor + 4 new hparams; cost exceeds benefit at current project stage |
| FlashMLA CUDA kernels | 2026-05-22 | Depends on MLA; MLA declined |
| CPU MLA | 2026-05-22 | Depends on MLA; MLA declined |

## Version log

- **v1** (2026-05-12) — initial inventory. Recon (Phase 0.5) has not yet
  started; all entries marked pending-recon.
- **v2** (2026-05-12, session 2) — first-pass recon completed for 3
  high-priority subsystems (MLA, MTP per-step SSM, fused MoE). Carlosfundora
  EAGLE3 + PHANTOM-X also recon'd.
  Remaining subsystems (IK quants, BitNet, bf16 KV, R-suffix variants,
  CPU MLA, etc.) still pending-recon but not blocking Phase 1 entry.
- **v3** (2026-05-22, ik-quant-lift-recon) — Full IK quant subsystem recon
  from ik_llama. All 7 target quant types
  classified recon-additive (LIFT-WITH-GLUE). Critical blocker identified:
  `row_meta_size` field absent from ygg's `ggml_type_traits` — must land as
  P0 prereq commit before row-meta types (IQ3_KS, IQ4_KS, IQ4_KSS, IQ4_KT).
  MLA + FlashMLA + CPU MLA marked declined. Port ladder and type-ID renumber
  map added to this file. HIP/CUDA implementations confirmed present in ft2.
- **v4** (2026-05-20, mainline-rebase-optd) — Fork forward-synced to mainline
  `ggml-org/llama.cpp` b9246 (`871b0b70f`). All MTP-zone conflicts resolved
  fork-side (fork's bundled MTP driver, E3b chain, and all KV-quant layers
  preserved intact). No MLA tier story — MLA predates the merge-base b9133
  and is already in the fork. This doc is unaffected by the rebase.
- **v5** (2026-05-22 to 2026-05-24) — All 7 IK quant types (IQ2_K/IQ3_K/IQ4_K base-K + IQ4_KS/IQ4_KSS/IQ3_KS/IQ4_KT row-meta KS/KT) now **ported** to main. Phase 5b-1a (`aed6d2965`) shipped base-K family; Phase 5b-1b (`026671689` + `5fe804bcd`) shipped KS/KT family with row_meta_size prereq and Vulkan batched-mul_mat SEGV fix. Status rows updated from recon-additive → ported. Phase 5b-2 (IQ5_K/IQ6_K) recon in-flight.
- **v6** (2026-05-24) — Phase 5b-2 S1 (`f7a489de5`): IQ5_K/IQ6_K **ported** (CPU + CUDA/HIP + Vulkan). Phase 5b-1c S1 (`e404274b9`): IQ2_KL type-157 **ported** (CPU + CUDA/HIP). Both rows added/updated. Vulkan parity in-flight for new types.
