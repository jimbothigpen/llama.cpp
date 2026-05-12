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
- **ported** — Landed on yggdrasil, regression-tested.
- **dormant** — ik_llama work itself is shelved; we may skip the port.
- **declined** — We chose not to port; record the reason.

## Subsystem inventory

### Quantization types

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| IK "K" family (IQ2_K..IQ6_K) | type IDs 137–141; commits across many `ik/cuda_iq*` branches | pending-recon | Foundation for the IK quant ecosystem. Additive in principle. |
| IK "KS" small family (IQ2_KS, IQ3_KS, IQ4_KS, IQ4_KSS, IQ5_KS, IQ2_KL) | type IDs 144–146, 152, 156–157 | pending-recon | Refinements of K family. |
| BitNet (IQ1_BN, IQ2_BN, I2_S) | type IDs 36, 134–135 | pending-recon | First-class BitNet support — ik_llama is the only fork with this. |
| Q8 K-block variants (Q8_K16/K32/K64/K128/KR8/KV) | type IDs 136, 147–151 | pending-recon | Workhorses for K-quant intermediate compute. |
| Q6_0 (revived legacy) | type ID 133 | pending-recon | ik_llama kept this after mainline removed it. |
| Q8_*_X4 interleaved | type IDs 97–99 | pending-recon | |
| Row-interleaved R-suffix variants | type IDs 202–230 | pending-recon | Wide surface. May be one big port or split per family. |
| Trellis weight quants (IQ2_KT, IQ3_KT, IQ4_KT, IQ1_KT) | type IDs 153–155, 158; branches `ik/andrew_trellis`, `ik/new_trellis_2`, etc. | **dormant** | ik_llama trellis branches frozen since June 2025. May be ports we skip permanently. Buun's TCQ (KV) is the active TCQ work. |

### Attention / KV cache

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| MLA (Multi-head Latent Attention) | 20+ branches `ik/cuda_mla*`, `ik/FlashMLA-3`, `ik/cpu_mla_all_quants`, `ik/deepseek_*` | **recon-structural** (2026-05-12, session 2) | Requires per-layer conditional v_cache + 4 new hparams (`n_lora_q`, `n_lora_kv`, `n_embd_head_k_full`, `n_embd_head_v_full`). Cannot be added post-Phase-1 without refactoring memory + hparams. See `yggdrasil-context/recon/01-mla.md`. |
| FlashMLA CUDA kernels | `ik/cuda_flash_mla*` family | pending-recon | Depends on MLA recon outcome. |
| CPU MLA | `ik/cpu_mla_all_quants`, `ik/cpu_deepseek_fa` | pending-recon | |
| bf16 KV cache | `ik/bf16_kv_cache` | pending-recon | Likely additive. |
| Better Q4_0 KV cache | `ik/better_q40_kv_cache`, `ik/better_q40_kv_cache_cpu` | pending-recon | Likely additive. |
| Fused MoE | `GGML_OP_MOE_FUSED_UP_GATE` op; `iqk_moe_fused_up_gate()` in `ggml/src/iqk/iqk_mul_mat.cpp`; graph builder in `src/llama-build-context.cpp:1014`; PR #1707 + scattered. | **recon-additive** (2026-05-12, session 2) | Adds new ggml op enum (`GGML_OP_MOE_FUSED_UP_GATE`). CPU path depends on IK quants (`IQK_IMPLEMENT`); Metal/Vulkan stubs are yggdrasil-original. Port-order: IK quants → fused MoE CPU → fused MoE GPU. See `yggdrasil-context/recon/03-fused-moe.md`. |

### Speculative decoding (MTP)

| Subsystem | ik_llama refs | Status | Notes |
|---|---|---|---|
| MTP foundation (Qwen3.5 MoE, Gemma 4, GLM, Mistral3) | PRs #1736, #1741, #1744, #1745 (Gemma 4), #1758 (multimodal), #1771 (GLM fix) | pending-recon | **Already partially-ported to mainline by turbo-tan's `experiment/gemma4-mtp-upstream-pr`.** Phase 2 of yggdrasil layer plan uses turbo-tan's port as the foothold. ik_llama subsystem-port becomes "backport ongoing improvements" rather than "port from scratch." |
| MTP graph reuse | PRs #1713, #1728, #1780 | pending-recon | Sits on top of turbo-tan's foundation. |
| MTP per-step SSM optimizations | PRs #1713, #1718, #1724, #1728, #1767, #1773, #1778 | **recon-structural** (2026-05-12, session 2) | Confirmed structural risk: requires ik_llama's `split_s_l_shadow`, dual-graph reuse (`prev_mtp`), and `ggml_delta_net` 6-src-tensor signature. Forces Phase 2 choice: mainline-style MTP foothold (Path α, first-pass recommended) vs ik_llama-style foundation (Path β). See `yggdrasil-context/recon/02-mtp-per-step-ssm.md`. |
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
   yggdrasil trunk. Cherry-pick commits one-by-one; do NOT bulk-merge a
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

1. Triage: relevant to yggdrasil? (yes/no/maybe)
2. If yes: open a yggdrasil tracking issue with the ik_llama PR URL.
3. Schedule the backport based on which subsystem it targets and what
   already-ported subsystems it depends on.

A weekly ik_llama sweep (every Monday) catches new PRs.

## Declined ports

(none yet)

## Version log

- **v1** (2026-05-12) — initial inventory. Recon (Phase 0.5) has not yet
  started; all entries marked pending-recon.
- **v2** (2026-05-12, session 2) — first-pass recon completed for 3
  high-priority subsystems (MLA, MTP per-step SSM, fused MoE). Carlosfundora
  EAGLE3 + PHANTOM-X also recon'd (see `yggdrasil-context/recon/04-eagle3-phantom-x.md`).
  Remaining subsystems (IK quants, BitNet, bf16 KV, R-suffix variants,
  CPU MLA, etc.) still pending-recon but not blocking Phase 1 entry.
  Output: `yggdrasil-context/phase-0.5-recon-constraints.md`.
