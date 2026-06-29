# Changelog

All notable fork-specific changes. Tagged milestones correspond to
phase completions. For mainline llama.cpp changes inherited via upstream
sync, see the [ggml-org/llama.cpp release notes](https://github.com/ggml-org/llama.cpp/releases).

The format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versioning is milestone-driven (one tag per phase completion), not semver.

## [Unreleased]

HEAD: `1abca5d4d` (2026-06-29 — perf: remove n_acc_tokens_per_pos hot-path vector). Prior: `67120d2e4` (2026-06-29 — adopt: turboq4/turboq3 Lloyd-Max centroid corrections). Prior: `1289ee3cc`/`459e0b8b7`/`43919f495`/`e3f11fc9c` (2026-06-29 — docs hygiene passes P0–P3 + WQ3_TCQ ftype). Prior: `0372c10e4` (2026-06-29 — Merge: server variance checkpoint-eviction). Prior: `83e7a17e3` (2026-06-29 — Merge: fwd-sync). Prior: `e6fc143f8` (2026-06-29 — provenance audit). Prior: `393307e58`/`fc1f979c2` (2026-06-29 — asym-KV FA matrix). Prior: `dc7236725` (2026-06-29 — OSCAR-V CUDA/HIP KV FA matrix). Prior: `e5c328187` (2026-06-18 — Merge: InnerQ Q-rotation gate on g_innerq_finalized, TODO 236). Prior: `6e5087c35` (2026-06-18 — Merge: opt-in --spec-ensemble pick-longest drafter, TODO 117). Prior: `3ff8220f3` (2026-06-14 — Merge: Vulkan `--no-mmap` MoE load deadlock fix, gfx1150/RADV `9447292d1`). Prior: `a4a92c62b`/`b22b6492d` (2026-06-14 — TURBOQ8_0 8-bit KV codec, port of buun `TURBO8_0`). Prior: `e40b79091`/`d487cece8` (2026-06-14 — EAGLE3 fc_norm converter variant + dead-entry cleanup). Prior: `5944b2de8`/`5bd9b5212`/`d66c1886e`/`df1d68f69` (2026-06-14 — EAGLE3 `t_layer_inp` per-layer hidden-state capture across 10 archs, TODO 240c). Prior: `50b747212` (2026-06-14 — vulkan: fix mul_mat_id expert offset for ik_llama IQK quants, qwen35moe MoE garbage gfx1150). Prior: `e54bdd9d3` (2026-06-14 — port buun batch: recurrent state-view fast-path + GDN tensor-split fix). Prior: `d1eda75f4` (2026-06-13 — Merge mainline ggml-org @ `c2ba3e47a`: cohere2-MoE arch #24260, jinja fixes, vulkan non-contig unary #24215). Prior: `aa395f8eb` (2026-06-13 — llama-grammar: clamp oversized repetition upper-bound instead of aborting). Prior: `3431f29d5` (2026-06-13 — gemma4-assistant: proportional RoPE fallback for spec-decode accuracy). Prior: `dedc7787e`/`c3b5c1b82` (2026-06-13 — docs: scrub private-infra leaks, reconcile drifted feature-doc code refs). Prior: `2ef6c9d3a` (2026-06-10 — fix(merge): GGML_OP_COUNT static_assert 98→99). Prior: `0d08feeaf` (2026-06-10 — cuda: guard mul_mat_id fast path on pascal). Prior: `ea5f6c658` (2026-06-10 — graph: extend iswa kq_mask null-buffer guard to local src_kv_iswa, MTP draft). Prior: `abe20ebea` (2026-06-10 — context: output_reorder() use n_embd_out() for embd/embd_nextn stride, TODO 221). Prior: `e65fe2ae6` (2026-06-10 — Merge mainline-38 / b1144 into fork main; ggml-vulkan.cpp + ggml.c conflicts resolved; 0/519 behind, GitHub behind-banner cleared). Prior: `d964e3a2f` (2026-06-09 — tria-gen: support qwen35 hybrid arch — Qcur_full de-interleave + linear-layer zero-fill; Qwen3.5/3.6 calibration now captures full-attention layers correctly). Prior: `81ca6b749` (2026-06-07 — WHT3_0/WHT4_0 Vulkan MoE expert dispatch). Prior: `d8393c386` (2026-06-07 — iq4_nl FA-vec KV CUDA/HIP kernel; Kaggle-T4 PPL 7.3941 ≈ f16 7.4047). Prior: `4f39662dd` (2026-06-06 — TODO 217: Vulkan mul_mat_vec_id adds IQ5_KS/IQ2_KS/IQ1_KT). Prior: `ae6bc152c` (2026-06-06 — TODO 207 scrub: private infra purged). Prior: `031e87b57` (2026-06-06 — TODO 212: Vulkan TURBOQ{2,3}_INNERQ KV alias support). Prior: `6fcd17fce` (2026-06-05 — WHT `ne1=1` decode to fused `*_multi<1>` kernel, retire fp32 v12). Prior: `3abe1c048` (WHT3_0/WHT4_0 small-batch throughput: route ne1≤8 to fused TQ kernel, +290% WHT3_0 pp at -ub8 on RDNA3). Prior: `a7a2a1d0d` (2026-06-04 — weight-quant matrix PPL-reference + bench-only methodology). Prior: `55bb0d418` (2026-06-02 — remove RotorQuant iso/planar KV family (slots 72-75) — zero-rotation scalar dup, strictly dominated). Prior: `d0773ae2d` IQ2_KT: fix GS=8 cluster-index Phase 2 + k=256; `38c8ce589` port carlosfundora#109: ROCm KV-guardrails tests and arg docs; `9fdf82344` port carlosfundora#108: bounds-check multi-token extraction tensor copy; `cf81fa92b` common: warn when mmap + -ngl>0 is used with an integrated GPU; `a937c23f6` feat(bench/ppl): wire TriAttention + PFlash enable flags into bench + perplexity tools; `570953782` chore: remove external companion-project references; `48dd0b3b8` speculative-simple: allow self-spec types without external draft model; `851b3a88d` Merge mainline ggml-org/llama.cpp @95b8b8ec1 (forward-sync); `7337523e6` oscar: full-dim D=256 WHT for INT2 KV + GGML_OP_FWHT removed. /opt: b1144 shipped 2026-06-10.

In-flight: EAGLE3 catch-up-decode PORT (C1 stash+prepend, ~80-110 LOC); Trellis P3c (IQ1_KT weight quant port); IQ2_KT cluster-accel PPL retune to k=80–100 (late-stage polish); mainline PORT-NOW fixes; PFlash non-Qwen live-scorer validation (§-FLAG). §-FLAG-ATTN_ROT_KSHIFT: OScaR INT2 K-shift for streaming inference unverified. Known-issue TODO 213: gfx1103 ROCm PPL rc=134 (transient; no-repro; no code fix). Vulkan TURBOQ INNERQ KV DONE (`031e87b57`); 207 scrub DONE (`ae6bc152c`); 217 Vulkan mul_mat_vec_id IQ5_KS/IQ2_KS/IQ1_KT DONE (`4f39662dd`); iq4_nl FA-vec KV DONE (`d8393c386`); WHT3_0/WHT4_0 mul_mat_vec_id DONE (`81ca6b749`); RotorQuant iso/planar removal DONE (was in-flight; now `55bb0d418`).

### Performance — remove `n_acc_tokens_per_pos` hot-path vector (turbotan ADOPT perf-half) (2026-06-29)

`1abca5d4d`. Port of turbotan `44da65ea1` (perf-half only). The
`n_acc_tokens_per_pos` vector — a per-draft-position acceptance counter
used only for the `'#acc rate/pos'` log suffix — was being resized and
incremented on every accepted token in the MTP hot path, costing ~4 tok/s.
The vector and the log suffix are removed; the `'#mean acc len'` stat is
retained. Behavior-neutral: no change to draft generation, acceptance
logic, or output.

The server checkpoint-gate half of turbotan `44da65ea1` is **held** pending
server-side spec-decode acceptance validation (`feat/adopt1-mtp-tps-fix-2026-06-29`,
`0d02fb8ce`).

Changed: `common/speculative.cpp` (+1/−19).

### Fixed — turboq4/turboq3 Lloyd-Max centroid corrections (thetom ADOPT-2 + ADOPT-4) (2026-06-29)

`67120d2e4`. Ports two centroid table corrections from TheTom
(`77ab7e988` + `545092c36`).

**ADOPT-2** (`77ab7e988`): turboq4 extreme centroid −0.173926f → −0.241529f;
full 16-entry table + 15 midpoints corrected in both CPU
(`ggml-turbo-quant.c`, 3×) and CUDA/HIP (`turbo-quant.cuh`, 1×). Also
drops a dead `rnorm` zero-write.

**ADOPT-4** (`545092c36`): turboq3 outer centroid −0.190685f → −0.190207f;
8-entry table + 7 midpoints corrected in both backends.

Quality (pure-CPU, Qwen3.5-9B-Q4_K_M, c=512, 10 chunks):

| KV type  | PPL         | vs f16 |
|----------|-------------|--------|
| f16      | 9.3831±0.488 | —      |
| turboq4  | 9.4491±0.491 | +0.7%  |
| turboq3  | 9.4699±0.494 | +0.9%  |

Mean KLD @2048 −4.3% (turboq3). GPU PPL and KV-quant matrix rerun owed
post-merge (tracked separately).

Attribution: TheTom (`tturney1@gmail.com`).

Changed: `ggml/src/ggml-turbo-quant.c` (+29/−28),
`ggml/src/ggml-cuda/turbo-quant.cuh` (+13/−15).

### Added — opt-in `--spec-ensemble` pick-longest ensemble drafter for speculative decode (TODO 117) (2026-06-18)

`19832d892` + `115ad9aa9` (feature + docs) + merge `6e5087c35`. Adds an
opt-in `--spec-ensemble` flag that runs all configured `--spec-type`
drafters per inference step and keeps the longest draft (pick-longest),
alongside the unchanged default priority cascade. Useful for exploring
multi-drafter configurations and evaluating different draft strategies
in parallel.

Wired through `common/arg.cpp` (new flag), `common/common.h` (params),
and `common/speculative.cpp` (ensemble selection logic). Feature doc
`docs/features/spec-decode-ensemble.md` describes the flag and use cases.

Testing: CPU build green; smoke on Qwen3.5-9B: 73% draft accept rate,
coherent output, default behavior byte-identical (ensemble disabled).

Changed: `common/arg.cpp`, `common/common.h`, `common/speculative.cpp`,
`docs/features/spec-decode-ensemble.md` (new).

### Fixed — InnerQ: gate Q-rotation on `g_innerq_finalized`, not env variable (TODO 236 + 235) (2026-06-18)

`62790d7b2` + merge `e5c328187`. Fixes a ~2.5× PPL regression where the
`TURBO_INNERQ` environment variable gated the per-channel `scale_inv`
rotation even after InnerQ calibration had been disabled (never
published). Disabling caused the rotation buffer to be zeroed → Q
weights multiplied by 0 → gibberish. Now gates on `g_innerq_finalized`
(set only when InnerQ calibration succeeds and publishes); fallback is
plain WHT (exact baseline). Also closes TODO 235: re-verified K=INNERQ
all-NaN runtime case: PPL 7.1672, fixed by upstream commit `fca7fc93c`
already on main.

Changed: `src/llama-graph.cpp` (+15/−1).

### Added — TURBOQ8_0 8-bit KV cache codec (source: buun) (2026-06-14)

`b22b6492d` + merge `a4a92c62b`. Ports buun's `TURBO8_0` as a new fork KV
type `GGML_TYPE_TURBOQ8_0` (slot 63, CLI string `turboq8`). It is the
high-precision / lowest-compression member of the TurboQuant KV family:
the FWHT rotation followed by a **uniform 256-level grid**
(`centroid[i] = (i − 127.5)/127.5`) with a per-block absmax scale — **no
QJL, no PolarQuant codebook**. Block `block_turboq8_0` is 130 bytes for
128 elements (fp16 absmax + 128×uint8 index), 8.125 bpw, ~1.97× vs fp16
KV. Wired through the CPU `vec_dot` path, CUDA/HIP `fattn-vec`
instances (`turboq8_0`×{`turboq8_0`,`q8_0`,`f16`} K/V combinations),
`kv_cache_type_from_str`, and the layer-adaptive KV machinery. **No
Vulkan kernel yet** (CPU + CUDA/HIP only). Quality benchmarks are pending
(measure-first); see `docs/features/turboquant-kv-base.md` and
`docs/TYPE_ASSIGNMENTS.md`.

Usage: `--cache-type-k turboq8 --cache-type-v turboq8` (or pair as the
high-precision K against a lower-bit V per the asymmetric-KV guideline).

### Fixed — vulkan: `--no-mmap` MoE load deadlock on RADV gfx1150 (2026-06-14)

`9447292d1` + merge `3ff8220f3` (current main HEAD). Loading an MoE model
with `--no-mmap` could deadlock the Vulkan backend on RADV/gfx1150 during
weight upload; adds a sync-upload fallback. Refs upstream #18317 / #18047.

### Added — EAGLE3 `t_layer_inp` per-layer hidden-state capture across 10 archs (TODO 240c) (2026-06-14)

`5bd9b5212` + `5944b2de8` (also `d66c1886e`, `df1d68f69`). Sets
`t_layer_inp` per-layer so EAGLE3 hidden-state capture works across 10
architectures (deepseek2/deepseek32, gemma2/gemma3, llama4, qwen2/qwen2moe,
qwen3next, qwen3vl/qwen3vlmoe, qwen35/qwen35moe). Companion converter work
`e40b79091`/`d487cece8` handles the EAGLE3 `fc_norm` variant and removes a
dead duplicate converter entry. (Attribution: EAGLE3 from carlosfundora
`1-bit-turbo`; runtime GGUF rebuild/validation tracked separately under
TODO 239.)

### Fixed — GGML_OP_COUNT static_assert updated for sync-38 merged op set (2026-06-10)

`2ef6c9d3a`. After the mainline-38 merge the GGML op count grew by one
(upstream added `GGML_OP_COL2IM_1D`). The fork already carries two
fork-specific ops (`GGML_OP_FLASH_ATTN_SPARSE`, `GGML_OP_TURBO_WHT`), so
the total is now 99; the static assert guarding the `ggml_op_name` /
`ggml_op_symbol` arrays was left at the pre-merge value 98. The
`GGML_OP_NAME` / `GGML_OP_SYMBOL` arrays already had 99 entries; only
the assert needed updating. Caught at build verification.

Changed: `ggml/src/ggml.c` (+2/−2).

### Fixed — cuda: guard mul_mat_id fast path on Pascal/SM61 (TODO 221c) (2026-06-10)

`0d08feeaf`. Cherry-pick of turbotan `af0d9d7bc`. The `ggml_cuda_mul_mat_id`
fast path (F32 src1 / F32 dst branch in `ggml-cuda.cu`) was unconditionally
enabled on all NVIDIA devices. On Pascal-class GPUs (compute capability
SM61, e.g. Kaggle P100) this triggers an illegal-memory-access error for
MoE models because the path relies on Volta+ memory-access semantics.
Adds a `cc >= GGML_CUDA_CC_VOLTA` guard so Pascal routes through the
existing conservative fallback; Volta and above are unaffected.

Changed: `ggml/src/ggml-cuda/ggml-cuda.cu` (+6/−1).

### Fixed — graph: extend iswa kq_mask null-buffer guard to fork-local MTP draft path (TODO 221b) (2026-06-10)

`ea5f6c658`. Follow-on to upstream commit `a66d50588` (#24294). The fork
carries a parallel `llm_graph_input_attn_src_kv_iswa` class used for the
MTP / spec-decode shared-cell draft path; it performs the same unguarded
`get_base()->get_swa()->set_input_kq_mask` + `can_reuse_kq_mask` calls
that the upstream commit fixed in the main attention class. A SWA-only
draft head (non-MTP or external assistant GGUF) leaves the base sub-cache
empty, so its `kq_mask` buffer stays null and would assert at graph load.
Applies the identical `mask->buffer` guard. Defensive — no behaviour
change when the base buffer is populated.

Changed: `src/llama-graph.cpp` (+14/−4).

### Fixed — context: output_reorder() use n_embd_out() stride (TODO 221) (2026-06-10)

`abe20ebea`. Hand-port of the `output_reorder()` hunk from turbotan
`b061b5b46`. The embedding-swap loops that reorder multi-token output
slices used `n_embd` as the per-row byte stride. When `n_embd !=
n_embd_out` (e.g. Gemma4 MTP heads, where the assistant output width
differs from the model embedding width) this caused a silent stride
mismatch that corrupted the reordered embeddings. Both `embd` and
`embd_nextn` swap loops are updated to use `n_embd_out()`. The
speculative.cpp and `llama-graph.h` changes from the same upstream commit
do not apply cleanly to the fork's diverged tree and are intentionally
excluded.

Changed: `src/llama-context.cpp` (+7/−6).

### Changed — Merge upstream ggml-org/llama.cpp mainline-38 (b1144) (2026-06-10)

`e65fe2ae6`. True merge of upstream HEAD `e95dae18d` (mainline
`ggml-org/llama.cpp` b1144-equivalent) into fork `main`. The fork was
519 commits ahead and 38 behind; after this merge the "behind" count is
cleared. Conflicts resolved in `ggml-vulkan.cpp` (FWHT-hint enum),
`ggml.c` (op-count), `llama-arch.cpp` and `llama-arch.h` (Gemma4
unified-vision enum), `llama-kv-cache.cpp` (SWA guard). Notable upstream
entries in this wave include Gemma 4 E2B/E4B MTP inference (#24282),
KV-cache improvements (#24267), spec-vocab fix (#24256), and the iSWA
`kq_mask` null-buffer guard (#24294 — `a66d50588`).

### Added — tria-gen: support Qwen3.5/3.6 hybrid-attention architecture (2026-06-09)

`d964e3a2f`. Qwen3.5 and Qwen3.6 use a gated-delta-net +
sparse-full-attention hybrid: only every 4th layer is full softmax
attention. In these models the pre-RoPE query MUL_MAT is named
`Qcur_full` (fused Q + gate, 2×head_dim per head), not `Qcur`. The
existing collector matched only `Qcur-N`, so it captured 0 tokens on
Qwen3.5 / Qwen3.6.

Fixes:
- Match `Qcur_full-N` and de-interleave the Q half (first `head_dim`
  columns of each 2×`head_dim` block; gate half discarded).
- Count tokens per chunk in the decode loop, not on layer 0 (layer 0 is
  a linear layer in hybrid models → token_count stayed 0).
- `write_tria_v4`: take the header scalar from the first captured full-
  attention layer and zero-fill uncaptured linear-attention layers. The
  runtime scorer skips them (their K-capture buffer is null), so they
  are never consulted; the file stays loader-valid. Pure full-attention
  models are unaffected (every layer captured → header == layer 0 as
  before).

Validated: Qwen3.5-0.8B emits a v4 `.tria`; 6/24 layers full-attention
(layers 3, 7, 11, 15, 19, 23) with finite nonzero statistics; 18 linear
layers zeroed; header passes the `triattention.c` loader validation.

Changed: `tools/tria-gen/tria-gen.cpp` (+53/−22).

### Added — WHT3_0/WHT4_0 Vulkan MoE expert dispatch (mul_mat_vec_id) (2026-06-07)

`81ca6b749`. WHT3_0/WHT4_0 weights used as MoE expert tensors aborted on the Vulkan backend:
`ggml_vk_get_dequantize_mul_mat_vec_id()` returned `nullptr` for these types, triggering
`GGML_ASSERT(dmmv != nullptr)` in `ggml_vk_mul_mat_vec_id_q_f16`. No new GLSL shaders are needed
— the generator already emits `mul_mat_vec_id_wht{3,4}_0_f32_f32` and the aggregation arrays from
the existing `mul_mat_vec_wht{3,4}_0.comp` via the `MUL_MAT_ID` define. The fix is purely C++
wiring: register the id-vec pipeline slots for WHT4_0/WHT3_0 and add both types to the id getter
with the correct `DMMV_WG_SIZE_SUBGROUP` butterfly contract. Vulkan-only; CUDA/HIP WHT id path
is separate.

**Verified on gfx1150 (RADV):** test-backend-ops MUL_MAT_ID 764/764 OK (no regression);
Qwen3.6-35B-A3B-WHT4_0 and -WHT3_0 MoE decode at **12.3 / 9.9 t/s** on Vulkan (`-ngl 999`),
coherent output, no abort.

Changed: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (+19/−3).

### Added — CUDA/HIP iq4_nl FlashAttention-vec KV kernel (2026-06-07)

`d8393c386`. Implements `vec_dot_fattn_vec_KQ_iq4_nl`: routes 4-bit indices through the
`kvalues_iq4nl` codebook via `get_int_from_table_16` (no Q8 offset — codebook encodes signed
values), plus V-dequant and dispatch wiring with `f16/iq4_nl` K×V template instances. Allows
`iq4_nl` to be used as a CUDA/HIP KV-cache type without falling back to CPU for FlashAttention.

Background: `iq4_nl` as a KV type previously triggered CPU offload on CUDA/HIP (the cause of a
Kaggle T4 PPL timeout). A temporary `de-advertise iq4_nl` workaround (`479655495`) was reverted
(`e5bfb95d5`) once this kernel landed.

**PPL gate (Qwen3.5-9B, Kaggle T4, wikitext-2, iq4_nl K×V):**
- iq4_nl/iq4_nl: **7.3941** ≈ f16 reference **7.4047** — quality preserved at GPU speed.

Changed: `ggml/src/ggml-cuda/fattn-common.cuh` (+85), `fattn.cu` (+9/−1), 3 template-instance
files (+7 each), `CMakeLists.txt` (+3), `generate_cu_files.py` (+1/−1) (7 files, +118/−2).

### Fixed — Vulkan mul_mat_vec_id: add IQ5_KS/IQ2_KS/IQ1_KT to id-vec getter (TODO 217) (2026-06-06)

`4f39662dd`. `ggml_vk_get_dequantize_mul_mat_vec_id()` was missing three IK row-meta types —
IQ5_KS, IQ2_KS, IQ1_KT — whose id-vec pipelines are already registered (~ggml-vulkan.cpp:4795),
shaders already generated, and types already handled by the non-id getter and `supports_op`. For
an MoE expert tensor of one of these types the id getter returned `nullptr`, triggering
`GGML_ASSERT(dmmv != nullptr)` and aborting model load. This completes the KS/KT/KL set begun by
TODO 194 (`e95ecbecf`). WHT3_0/WHT4_0 (also named in TODO 217) required separate treatment due
to a different shader structure and are handled in `81ca6b749` above.

Changed: `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (+10).

### Chore — purge private/internal info from public fork (TODO 207) (2026-06-06)

`ae6bc152c`. Removes host names, private project references, internal tracker IDs, and internal
matrix/quant tooling scripts from shipped source, comments, and docs so nothing on GitHub
references private infrastructure. Comment/string/docs/script-move only — no logic changes.

### Added — Vulkan TURBOQ{2,3}_INNERQ KV-cache quant support (TODO 212) (2026-06-06)

`031e87b57`. `TURBOQ2_INNERQ` (type 68) and `TURBOQ3_INNERQ` (type 69) are byte-identical to
plain `turboq2_0`/`turboq3_0` — same block layout and dequant. The InnerQ per-channel
pre-scaling lives at the graph/KV-cache layer (the `TURBO_WHT` op), not the backend, so the
Vulkan backend treats them as plain-turbo aliases. No new GLSL shaders.

Changes: accept INNERQ in the three KDD-5 `supports_op` gates (FA/GET_ROWS/SET_ROWS); normalize
INNERQ→plain in `ggml_vk_flash_attn` effective K/V type; alias INNERQ get_rows / set_rows
pipeline slots to the existing plain-turbo handles; route INNERQ through
`quantize_turboq{2,3}_0` in `ggml.c`.

**PPL gate (Vulkan / RADV PHOENIX gfx1103, Qwen3.5-9B-Q4_K_M, wikitext-2, 24 chunks):**
- `turboq2_innerq`: **7.9298** = `turboq2_0` 7.9298 (bit-identical)
- `turboq3_innerq`: **7.6050** = `turboq3_0` 7.6050 (bit-identical)

Also: test-backend-ops FLASH_ATTN_EXT: turboq2_innerq 528/528 OK, turboq3_innerq 528/528 OK.

Changed: `ggml-vulkan.cpp` (+59/−11), `ggml.c` (+3), `tests/test-backend-ops.cpp` (+10/−2).

### Optimized — WHT3_0/WHT4_0: `ne1=1` decode to fused `*_multi<1>` kernel, retire fp32 v12 (2026-06-05)

`6fcd17fce`. Single-token decode (`ne1==1`) for WHT3_0/WHT4_0 was pinned to the slow fp32
`mul_mat_vec_wht*_v12` shared-memory kernel. Routes `ne1==1` through the same fused
`ggml_cuda_mul_mat_tq_multi<1>` path used for `ne1=2..8` (TheTom mmvq-tq dispatch):
dp4a int8 on NVIDIA for WHT4_0; scalar/half on AMD RDNA (dp4a NO-GO on RDNA). Prefill
(`ne1>8` cuBLAS/rocBLAS) path unchanged.

The legacy fp32 v12/v8 decode kernels are retired from the default path but kept reachable
via `GGML_WHT_DECODE_V12=1` for single-binary A/B validation.

**Measured (gfx1103 ROCm, Qwen3.5-9B, scalar/half path, same-binary env-toggle A/B):**
- WHT4_0 tg128: 6.87 → **10.49 t/s** (+52.7%)
- WHT3_0 tg128: 5.70 → **9.42 t/s** (+65.3%)

NVIDIA WHT4_0 dp4a: T4×2 tg128 ~39.88 t/s ≈ IQ4_XS throughput.

Changed: `ggml/src/ggml-cuda/mmvq-tq.cu` (9 lines — retire v12 carve-out, route `ne1==1`
to `*_multi<1>`). test-backend-ops MUL_MAT 158/158 WHT f32 PASS.

### Optimized — WHT3_0/WHT4_0 small-batch throughput: route ne1≤8 to fused TQ multi-token kernel (2026-06-05)

`3abe1c048`. WHT3_0/WHT4_0 small-batch prefill (`ne1=2..8`, i.e. `-ub` values ≤ 8) fell
through to the slow dequant-to-f16 + cuBLAS/rocBLAS path. Routes `ne1 ≤ MMVQ_MAX_BATCH_SIZE`
for contiguous 2D tensors through `ggml_cuda_mul_mat_tq_multi`, reusing each weight block
across all tokens via the fused pre-rotated-activation WHT kernel.

**Measured (gfx1103 ROCm, Qwen3.5-9B, -ub8):**
- WHT3_0 pp512: 7.16 → **27.90 t/s** (+290%)
- WHT4_0 pp512: 52.02 → 52.00 t/s (unchanged — dp4a path is NVIDIA-only; AMD scalar/half ties cuBLAS)

Standard pp512 (`-ub512`) and tg128 are separate dispatch paths and are not affected.
PPL parity verified (WHT4_0 9.4514 vs 9.5759; WHT3_0 9.5442 vs 9.6222 — within error bars).

Changed: `ggml/src/ggml-cuda/mmvq-tq.cu` (+390 LOC: multi-token kernels +
`ggml_cuda_mul_mat_tq_multi` dispatch), `ggml/src/ggml-cuda/ggml-cuda.cu` (route
`is_tq_weight` with `ne[1] ≤ MMVQ_MAX_BATCH_SIZE` + contiguous 2D guard). No CPU or
Vulkan path changes.

### Changed — weight-quant matrix: PPL-reference + bench-only split methodology (2026-06-04)

`a7a2a1d0d`. Refactors the weight-quantization matrix measurement strategy. PPL is
expensive and quality is host-invariant, so a single reference configuration measures
full 50-chunk PPL for every quantization type; every other measurement leg runs
bench-only (pp512 / tg128) plus a short 5-chunk PPL sanity sample on a fixed quant
subset. Changed: `scripts/run-weight-quant-matrix.sh` (−55 / +208 LOC), new
`scripts/weight-matrix-driver.sh` (+79 LOC) for per-cell dispatch.

### Changed — generate-quants.sh: mmproj auto-generation for multimodal models (2026-06-04)

`c5d918feb`. `scripts/generate-quants.sh` now auto-builds the `<model>-mmproj-F16.gguf`
projector for multimodal checkpoints (vision / audio tower). Step 2b runs
`convert_hf_to_gguf.py --mmproj` and emits a companion `<model>-mmproj-f16.gguf`.
Gate: `gq_is_multimodal()` checks `vision_config` / `audio_config` in `config.json`
(`BUILD_MMPROJ=auto|1|0`). Projector is kept at full F16 precision (`MMPROJ_OUTTYPE=f16`)
and is outside the quantization type ladder. New `--mmproj-only` flag builds just the
projector (skips BF16 / imatrix / quant + the `QUANTIZE_BIN` requirement) for targeted
runs. Qwen3.5 / Qwen3.6 and other multimodal checkpoints are handled automatically.

### Changed — WHT3_0/WHT4_0 added to generate-quants quant ladder (non-imatrix) (2026-06-04)

`b57d308f6`. Adds a `WHT_TYPES` family (`WHT3_0 WHT4_0`) to `ALL_QUANT_TYPES` in
`scripts/generate-quants.sh`, propagating to matrix scripts via `source`. WHT types were
held out of `ALL_QUANT_TYPES` pending the imatrix audit that shipped in `a6ccf0bfa`;
they are now included. Deliberately excluded from `IMATRIX_REQUIRED_TYPES` — RHT
rotation misaligns original-basis importance weights (see `a6ccf0bfa` below).

### Fixed — WHT3_0/WHT4_0: imatrix path disabled (rotation misaligns importance weights) (2026-06-04)

`a6ccf0bfa`. Root-cause audit: `WHT3_0` and `WHT4_0` were erroneously marked
imatrix-required (ADR-016 port assumption). The forward RHT mixes all 32 columns of a
block, so post-rotation coefficient `buf[j]` no longer corresponds to original column
`j`; weighting the rotated residual by original-basis importance `iw[j]` misaligns
importance with the rotated coefficient and measurably degrades quality.

Fix: both types quantize unweighted in scale search and WLS refinement, byte-for-byte
matching the `TheTom/llama-cpp-turboquant` upstream reference. `tensor_requires_imatrix()`
returns false for both types — no imatrix is needed or beneficial.

**A/B PPL gate (Qwen3.5-9B WHT3_0, wikitext-2-raw, 30 chunks, ROCm):**
- with-imatrix (defect): PPL 8.6105 ±0.092
- no-imatrix (fixed): PPL **7.2728** ±0.074 — −15.5%; matches and beats the upstream reference (7.6776)

Changed: `ggml/src/ggml-turbo-quant.c`, `src/llama-quant.cpp`, `scripts/generate-quants.sh`
(3 files, +45/−32). See also `docs/features/wht-weight-quants.md` §2 (updated).

### Changed — `scripts/` standardized + internal tooling purged (2026-06-02)

Added `scripts/generate-quants.sh` (the standard download→convert→imatrix→quantize GGUF
pipeline; configured via env / `scripts/matrix-env.sh.example`) so all GGUFs are produced by one
reproducible, apples-to-apples path. Removed 12 internal-only workflow scripts that should never
have shipped (host-/model-specific build + prequant + measurement helpers carrying local infra):
`baseline-matrix*.py`, `build-rocm-gfx110{2,3}.sh`, `ppl-harness.py(+README)`,
`prequant-qwen35-9b-*.sh`, `push-milestone.sh`, `with-bench-mutex.sh`. `scripts/` now contains
only mainline-llama.cpp scripts, ported-fork scripts, and fork-functional tooling.

### Removed — RotorQuant iso/planar KV family (slots 72–75) (2026-06-02)

`55bb0d418`. Removes all four RotorQuant KV types (`RQ_PLANAR3_0`, `RQ_PLANAR4_0`,
`RQ_ISO3_0`, `RQ_ISO4_0`) ported from carlosfundora `1-bit-turbo`. PPL gap is
inherent to the design and not recoverable (ISO3_0: +23.5% vs comparable TurboQ
types); all four are strictly dominated by TurboQ types at identical or lower bpw,
making them zero-rotation scalar duplicates with no advantage. Removals: ggml.h enum
entries (slots 72–75 marked reserved, not renumbered), ggml.c type info table,
ggml-common.h block struct definitions, ggml-quants.h declarations, entire
`ggml-roto-quant.c` (~380 LOC), ggml-cpu.c dispatch entries, ggml-cuda fattn dispatch
(~405 lines). Slots 72–75 are now **reserved**; do not reuse for unrelated types.

### Changed — OScaR INT2 KV: full-dim D=256 WHT + GGML_OP_FWHT removed (2026-06-01)

Replaces the block-wise 128-pt Walsh-Hadamard Transform in OScaR INT2 KV encode/decode with
a single full-dimension D=256 WHT. Both Qwen3.5-0.8B and -9B have `n_embd_head_k=256`; prior
two independent 128-pt transforms were suboptimal. Encode kernel (`set-rows.cu`) now launches D
threads per row in one CUDA block; normalization 1/sqrt(D). Decode (`fattn-vec.cuh`) applies
full D-pt FHT to all D Q elements before dot product; H_D is orthonormal (H^T·H = I).

`GGML_OP_FWHT` standalone op removed (no graph consumers). `fwht.{cu,cuh}` and
`ggml_cuda_op_fwht()` retained for the active `GGML_HINT_SRC0_IS_HADAMARD` mul_mat path.

`attn_rot_k` disabled for `GGML_TYPE_KV_OSCAR_INT2` (`llama-kv-cache.cpp`): the graph-level
H_D Hadamard rotation (via `ggml_mul_mat_aux`) would compose with OScaR's own H_D encode/decode
rotation, giving H_D²=I (identity) — K stored unrotated → poor INT2 quality. With attn_rot
disabled, OScaR's H_256 is sole rotation → 0.8B: -14.3% vs old baseline, 9B: -2.5% improvement.
§-FLAG: K-shift (RoPE update for streaming) behavior with attn_rot=false unverified for live use.

### Added — weight-skip optimization for Q4_K MMVQ — port cenconq25/delta-compress-llm cc47a4a (2026-06-01)

Ports the Q4_K weight-skip optimization from `cenconq25/delta-compress-llm@cc47a4a`
(`ggml/src/ggml-cuda/mmvq.cu`). In the `mul_mat_vec_q` inner loop, a 4-byte scale read
(weight super-block `d` × Q8_1 activation scale) decides whether to skip the ~400-byte
dot product; blocks below `LLAMA_WEIGHT_SKIP_THRESHOLD` are skipped entirely. Upstream
reports ~10–12 % decode speedup on Llama 3.1 70B with zero PPL degradation; 12–47 % of
blocks are skippable per layer. Default-off (env var unset = no skip, bit-identical to
prior behaviour). Scope: `ggml/src/ggml-cuda/mmvq.cu` only, 57 LOC added.

**Gate PASS (ROCm gfx1150, Qwen_Qwen3-8B-Q4_K_M, wikitext-2 32 chunks, -c 512 -ngl 99):**
- **Leg A (default-off, no LLAMA_WEIGHT_SKIP_THRESHOLD):** PPL = 8.8865 RC=0
- **Leg B (LLAMA_WEIGHT_SKIP_THRESHOLD=1e-4):** PPL = 8.8865 RC=0 — bit-identical to Leg A
- Template-signature divergence check: CLEAN (no IQ4_KT/IQ2_KT/delta markers in our mmvq.cu)

### Added — EAGLE3 compact-vocab draft support (SpecForge) — port PR #18039 (2026-05-31)

`b2766ef47`. Ports PR #18039 (SpecForge 32K-draft-vocab EAGLE3 support) from upstream. EAGLE3 draft
GGUFs with a compact vocabulary (e.g. SpecForge 32K-vocab drafters) can now be loaded alongside a
full-vocabulary target (e.g. Qwen3.5-35B-A3B, 248 320-token vocab). The loader derives
`n_draft_vocab` from the width of the `d2t.weight` tensor and builds the output head + draft-to-
target vocabulary remap table at model-load time (`src/models/eagle3.cpp`). Graph-side remap
handles token-space translation during speculative decode. Changes: 8 code files, 144 ins / 11 del
(common/speculative.cpp, common/speculative.h, conversion/eagle3.py,
examples/speculative-simple/speculative-simple.cpp, include/llama.h, src/llama-context.cpp,
src/llama-model.cpp, src/models/eagle3.cpp).

**Gate PASS (ROCm gfx1150, post-reboot, pipefail runner, no SIGSEGV on either run):**
- **35B compact-vocab smoke:** Qwen3.5-35B-A3B-MTP-IQ4_XS target (248 320-token vocab) +
  Qwen3.5-35B-A3B-Eagle3-SpecForge-BF16 drafter (32 000-token vocab + d2t): 130 tok, 33.33 %
  accept at *n_draft*=3, coherent output, RC=0. `n_draft_vocab=32000` confirmed in loader log;
  graph-side d2t remap active.
- **9B full-vocab no-regression:** Qwen3.5-9B-IQ4_XS + eagle3-draft-9b (d2t=none): 33.33 %
  accept at *n_draft*=3, matches prior baseline, RC=0.

**Known limitation (pre-existing, not a regression):** `llama_model_eagle3_get_d2t()` requires
`GGML_TYPE_I32` but SpecForge GGUFs export `d2t.weight` as `GGML_TYPE_I64` → host-side fast
path returns empty (`d2t=none` in log); graph path handles remap correctly (output is coherent).
This guard is from commit `87b5b3d8d` (already on main, ancestor of this commit). Optional
future polish: widen getter to accept I64 or export d2t as I32 in `conversion/eagle3.py`.

### Docs — currency bundle: new pflash.md + oscar-kv.md (2026-05-31)

`fc17aaade`. Adds two new end-user feature docs: `docs/features/pflash.md` (PFlash prompt
compression — CLI flags, how it works, benchmark placeholder) and `docs/features/oscar-kv.md`
(OScaR KV-cache replacement — phase status, design summary). Also updates IK-quant status and
DFlash `--target-model-dir` docs.

### Fixed — IQ3_KT ROCm: add dequantize kernel + GPU dispatch (CLOSED) (2026-05-31)

`c809225f6`. IQ3_KT was silently falling back to CPU on ROCm because the CUDA dequantize kernel
(`dequantize_block_iq3_kt`) was absent and `ggml-cuda.cu:supports_op` never returned true for
IQ3_KT. Fix mirrors IQ4_KT: +73 LOC CUDA dequant (block + matvec), `supports_op` registration,
and two GEMM-dispatch sites. **Gate PASS (gfx1150):** GPU path executes at 99% utilization /
7.66s-per-pass vs CPU-hang; IQ3_KT:IQ4_KT PPL ratio 1.30 ≈ anchor 1.29 (IQ3_KT=9.05/IQ4_KT=6.95).
§-FLAG note: gfx1102 warmup crash is a separate known gfx1102 Tensile confound, not IQ3_KT.

### Added — PFlash scorer generalized to non-Qwen models (2026-05-31)

`500046b0b`. PFlash prompt-compression scorer (`tools/pflash/pflash-scorer.cpp`) was Qwen3.x-only:
hard-coded attn/ffn tensor names broke non-Qwen lookup. Fix: arch table driven by
`llama_model_arch` covering qwen3/qwen35/qwen2/llama/mistral3/mistral4/gemma3/gemma4; NULL-deref
guards on optional fields (G4/G5 models); loud unknown-arch `LLAMA_LOG_ERROR` rejection replaces
silent crash. **Qwen3.x regression: byte-identical.** §-FLAG: Gemma3/Llama live-scorer compile
paths are new but UNVALIDATED (no non-Qwen scorer GGUF on hand at time of ship) — gate on
non-Qwen scorer GGUF when available (filed as follow-up TODO: PFlash non-Qwen live-scorer
validation).

### Docs — EAGLE 3.1 future-watch ledger entry in eagle3.md §5 (2026-05-31)

`9ae70fdc5`. Adds §5 "EAGLE 3.1 — future watch" to `docs/features/eagle3.md`: summarizes the
upstream EAGLE 3.1 paper / implementation status (not yet in mainline or fork); accept-ceiling
architectural note (1/n_draft); monitoring criteria for when to revisit the port decision.

### Docs — InnerQ KV feature doc finalized (2026-05-31)

`6ce2319c3`. Finalizes `docs/features/innerq-kv.md`: backend matrix (CUDA working, HIP working,
Vulkan §-FLAG KDD-5), corrects line-number references that had drifted after the imatrix port.

### Fixed — DFlash converter: --target-model-dir + tokenizer bundling (2026-05-31)

`f86a24a95`. `conversion/dflash_draft.py` lacked the `--target-model-dir` flag needed to bundle
the base-model tokenizer alongside the converted DFlash draft GGUF (required by z-lab DFlash
models that ship without a standalone tokenizer). Fix adds the flag and wires `DFlashDraftModel`
to copy tokenizer files into the output directory. The converter itself already existed
(`ba61a9d39` phase 1 + `a2c9c8c49` follow-on); this closes the remaining tokenizer gap.

### Docs — BACKEND_PARITY IQ2_KT/IQ3_KT Vulkan ported; IQ3_KT ROCm crash flagged (2026-05-31)

`docs/BACKEND_PARITY.md` rows 93 (IQ2_KT) + 97 (IQ3_KT) updated: IQ2_KT now CPU+ROCm+Vulkan (do-not-use flag retained); IQ3_KT now CPU+Vulkan (ROCm crashes in warmup, kernel present, root cause TBD).

### Docs — TriAttention KV-cache eviction feature doc (2026-05-31)

`docs/features/triattention.md`. End-user feature doc for TriAttention trajectory-adaptive
KV-cache pruning (Phase A+B+C all landed). Covers: 5 CLI flags + defaults; `.tria`
calibration file generation via `llama-tria-gen`; CPU + HIP + Vulkan GQA-aware scoring
backends; measured retrieval effectiveness (Qwen3-8B 100% @25% budget, Gemma-4 70%
@25%); honest backend caveat (Gemma-4 hd>128 GPU scoring is a separate perf follow-on).
`docs/features/README.md` updated with new KV Cache Eviction section.

### Docs — IK KT/trellis weight-quant feature doc (2026-05-31)

`docs/features/ik-kt-trellis.md` (`37e755dc6`). End-user feature doc for trellis-coded
quantization family (IQ4_KT, IQ3_KT, IQ2_KT, IQ1_KT). Covers: design rationale (single
codebook + trellis encoding); per-type performance + quality (IQ4_KT −0.5% vs IQ4_K,
IQ3_KT +23.5% vs IQ3_K, IQ2_KT codebook defect flagged do-not-use, IQ1_KT pending);
imatrix quantization requirement (ADR-016); cluster-acceleration tuning (k=60).
**Status: IQ3_KT/IQ4_KT shipped; IQ2_KT RED (§-FLAG: blanket do-not-use, general codebook defect
confirmed at all scales); IQ1_KT pending port from ik_llama.**

### Added — PFlash → CLI prompt-compression wire (2026-05-31)

`tools/cli/cli.cpp` (+20 LOC, `92c37266f`). Wires PFlash prompt compression into the
`llama-cli` prompt path before task submission. Pre-tokenizes and compresses long prompts
using configured scorer + threshold (`--pflash-scorer`, `--pflash-min-tokens`). Logs
`pflash: N -> M tokens (X% kept)` to stderr (always visible at default LOG_LEVEL_ERROR).
Sets `task.cli = false` to prevent double-tokenization. All `--pflash*` CLI flags already
wired in `common/arg.cpp`. **Gate: 3 smokes PASS (passthrough, sub-threshold, 8919→407 token
compression at 4.6% kept).** Mirrors server-side gate (`server-context.cpp:1587`); server
double-compression safety check suppressed for CLI-compressed tokens.

### Verdicts — IQ2_KT Qwen3.5-9B PPL = 33.96 (RED) (CLOSED) (2026-05-31)

**IQ2_KT on Qwen3.5-9B yields PPL = 33.96 ±0.48 (20 chunks, Vulkan gfx1103 b812), confirming general
codebook defect at all scales.** Scale-dependent hypothesis (small-model capacity collapse) rejected:
- 0.8B: 99.58 PPL (broken-but-improvable by scale)
- 9B: 33.96 PPL (catastrophic, 5.2× worse than IQ4_KT at 6.54)
- Ratio 9B/0.8B = 0.34 (3× improvement at larger scale, but both anomalous)

**Verdict: IQ2_KT blanket do-not-use (feature-doc §-FLAG updated). Recommendation: IQ2_KL (26.12 PPL
on 9B) is the viable 2-bit alternative.** Root cause: single per-row float scale (no per-block
adaptation like IQ4_KT), random-hash codebook (not learned VQ), greedy per-group VQ.

### Measurements — IQ3_KT/IQ3_K baseline-matrix Qwen3.5-9B (CLOSED) (2026-05-31)

Vulkan gfx1103 20-chunk baseline confirms both IQ3 types drifted identically (−6.8% binary-refresh drift,
consistent with the Vulkan build rebase from 2026-05-25 mainline sync):
- **IQ3_KT: 8.4299 ±0.107 PPL** (was anchor 9.0493; ratio vs IQ3_K = +23.3%)
- **IQ3_K: 6.8348 ±0.084 PPL** (was anchor 7.3243; same −6.8% drift)

Quality ratio IQ3_KT/IQ3_K = +23.3% (stable; inherent to single-codebook design).
**Action: update baseline-matrix anchors.** §-FLAG: IQ3_KT ROCm backend SEGFAULTs (missing
`mul_mat_vec_iq3_kt` kernel). Vulkan works; CPU fallback unavailable.

### Fixed — EAGLE3 B1+KV: drafter-batch KV-position anchor fix (2026-05-30)

`380c93384`. Combined the three B1 correctness fixes (d2t remap, norm_before_residual gating, rope_factors) with the KV-position fix from DFlash: anchor the draft batch to `llama_memory_seq_pos_max(ctx_dft)+1` instead of cross_len, which grows every iteration as target hiddens are committed. The two coincide only on the first iteration; every draft decode after the first failed the `Y = X + 1` consecutive-position check, blocking checkpoint rollback. **Status: EAGLE3 now functional (0%→33.3% accept on Qwen3.5-9B + eagle3-draft-9b), exceeding DFlash solo (25.1%).** Validation: Qwen3.5-9B + eagle3-draft-9b dual-spec gate PASS (accept 0%→33.3%), single-draft EAGLE3 (not yet measured in matrix).

### Added — TriAttention Phase C Part-2: SWA-layer K/V capture for Gemma-4 hybrid models (2026-05-30)

`086c8508f`. TriAttention was entirely inert on Gemma-4 (hybrid `llama_kv_cache_iswa` bridge returned null, no capture set_rows in SWA `build_attn()`, SWA layers uncaptured). Fix adds `llama_kv_cache_iswa` branch in KV-cache type dispatch, recognizes iswa in bridge, forces `swa_full` when active, captures `kv_swa` per layer. Surgical fix (7 files, no kernels). **Validation: SWA capture ~89% populated; smart retrieval 30% vs random 0%; non-SWA path byte-unchanged (no regression).** TriAttention now functional on Gemma-4. **Followup (not shipped):** per-layer head_dim in scoring runtime to also score full-attn layers (lift >30%).

### Added — MTP C1: eliminate the Qwen catch-up decode + iGPU-default `n_max=1` (2026-05-30)

`54bd1e120`. Driver-only change to `common_speculative_impl_draft_mtp` (`common/speculative.cpp`):
the per-cycle Qwen catch-up `llama_decode(ctx_dft)` is removed; instead `process()` stashes the
verified committed span (tokens + pre-norm h-feeds) and the next `draft()` prepends it KV-only
(logits off) before the lead token, so one decode writes both the catch-up KV and the lead
(**−1 `llama_decode`/cycle**). Plus iGPU auto-clamp of `n_max` to 1 (`common.h` `n_max_set`,
`arg.cpp`). Bug fixed mid-gate: after the KV-only prefix the lead token is off batch index 0, so
chain sampling tracked the lead's true batch index (`lead_ibatch`) to fix a `sampling.cpp`
`GGML_ASSERT(logits)`. **Gate (gfx1150, Qwen3.5-35B-A3B-MTP-IQ4_XS): ON C1 `n_max=1` = 32.4 t/s /
100% accept = 1.16× pure decode (28.0)** — the MTP V-J net-slowdown is resolved for the iGPU-default
config. PPL identical-by-construction (perplexity never invokes the spec path). Option-A iGPU warning
reworded from "net-slowdown, don't use" → tuning guidance. **Caveat:** C1 server-path wiring is
CLI-validated only — validate before enabling C1 in the server.

### Added — TriAttention Phase C: GPU GQA scoring kernel (HIP) (2026-05-30)

`51a64b43c` + `88f94232c`. Activates the dormant `triattention-hip.hip` (never previously compiled
upstream): fixes host-vs-device K pointer (uploads the scored q8_0 K slice H2D internally) and adds
GQA aggregation for `nh != nkv` (mirrors the CPU reference: per-query-head z-normalize → max across
the KV head's query heads → z-normalize + layer-weight → max into `global_scores`), removing the old
`nh==nkv` gate. Compiled as an isolated `tria-hip` HIP static lib linked PRIVATE into `llama` (keeps
`--offload-arch` off the host C++ compiles). Validated: kernel == CPU reference to 1e-5 at Qwen3-8B
scale (keep-set 100%); passkey/needle on GPU = 12/12 (100%) vs random 17%. Activates with
`--cache-type-k q8_0`; falls back to CPU otherwise; skips `nonrot_dim>0` models. **Part 2 (SWA-layer
`kv_swa` capture) is a follow-on.** Known: end-to-end the GPU path (100%) beats the current
CPU-runtime path (50%) on deep needles — a pre-existing CPU-integration quality issue under
investigation, orthogonal to the kernel.

### Fixed — restore #23869 speed-bench files dropped by a stale-based merge (2026-05-30)

`009e4f427`. The C1 feature branch was based on a pre-#23869 `main`, so FF-merging it silently
reverted #23869 (the speed-bench tool): restored `tools/server/bench/speed-bench/*` and the
`requirements-server-bench.txt` / `scripts/server-bench.py` / `docs/speculative.md` edits from
`7a18fcfe4`.

### Fixed — MTP C1 side-bugs (2026-05-30)

`7a9bbf4d5`. Two critical fixes discovered in V-J profiling of the C1 feature branch:
1. **`llama-cli --spec-type draft-mtp -fit on` autofit crash** — when `--fit on` mode triggers autofit with no prior schedule, `ggml_backend_sched_alloc` is passed a null `sched`, causing a null-deref at `ggml-backend.cpp:1945`. **Fix:** check for null sched before `sched_alloc` (1-line guard in `llama-context.cpp:3318`).
2. **MTP infinite rollback in `llama-cli`** — the `llama-cli` main draft loop calls `process()` without respecting rollback return codes, causing stuck-loop hangs on token rejects. **Fix:** route rollback status to the CLI driver loop; speeds up bad-accept recovery 10× (same 10-token rollback now costs 1 draft cycle instead of 10).

Both paths validated (cli/server dispatch wiring checked; C1 option-A server-path validation pending per caveat in C1 entry above).

### Fixed — TriAttention CPU-vs-GPU score divergence (2026-05-30)

`c5f1d135f`. The GPU scoring kernel was run unconditionally on all paths, including the CPU-forced fallback path. When GPU-score upload/download happened on a CPU-only path (e.g., `--no-gpu` or OOM), GPU ops triggered validation failures downstream. **Fix:** guard GPU score upload/download with `TRIA_NO_GPU_SCORE` check; CPU path now runs clean. **Note:** the earlier perception of "CPU-vs-GPU 50%-vs-100% smart" was a false divergence caused by this sentinel artifact, not a real quality gap.

### Added — Vulkan parity: make MTP iGPU auto-tuning backend-agnostic (2026-05-30)

`73dcfce62`. The C1 iGPU auto-tuning (n_max→1 clamp) was CUDA/HIP-only via `mtp_any_igpu()` and `mtp_warn_igpu_once()` in `common/speculative.cpp` (lines 29-61), gated with `#if defined(GGML_USE_CUDA)||defined(GGML_USE_HIP)`. On Vulkan-only builds (no CUDA/HIP), the tuning silently no-ops → Vulkan iGPU never clamps n_max→1, hitting the n_max=3 slowdown. **Fix:** replace CUDA-specific `ggml_backend_cuda_device_is_igpu()` call with generic `ggml_backend_device_type(dev)==GGML_BACKEND_DEVICE_TYPE_IGPU`, which Vulkan backends populate. **Validation:** gfx1150 RADV Vulkan at n_max=1 reaches 32.4 t/s, matching HIP, 100% accept (parity achieved). **Status:** MTP Vulkan parity gap CLOSED.

### Added — TriAttention Phase C: GPU GQA scoring kernel (Vulkan) (2026-05-30)

`0d13ac92b`. Completes GPU-accelerated scoring parity by porting the Phase C HIP kernel to Vulkan. Vulkan compute shader (`src/ggml-vulkan/` entry point behind `g_tria_backend` ABI) mirrors the HIP logic: K slice dequant (q8_0) + upload H2D, GQA-aware z-normalize, max reduction. Compiled as self-contained static lib (`tria-vulkan`), linked PRIVATE into `llama` (keep `--offload-arch` off C++ compiles). **Validation:** kernel == CPU reference 1e-5 (Qwen3.5-9B); passkey 4/4 GPU-smart vs random (100%); no regressions on Vulkan fallback path. Activates with `--cache-type-k q8_0`; falls back to CPU otherwise. **Status:** TriAttention Vulkan parity gap CLOSED. Both HIP and Vulkan now have GPU scoring backends.

**MTP Gemma4 §-FLAG-B** — ✅ LANDED in main @ `d2c332289` (PR #23398 guided port) + `ca62c0756` (§-FLAG-B 0%-accept materialize fix) + `190d83fed` (D1 ASSIST residue retirement). End-to-end external-assistant MTP for Gemma4-26B-A4B is coherent at 47.3% accept (see Fixed entry below).

### Added — Imatrix collection for MTP/NextN draft-head layers (`--imat-mtp`) (2026-05-30)

Port of mainline PR #23476: adds `--imat-mtp` flag to `llama-imatrix`. When set on a model with
bundled NextN layers, creates a second `LLAMA_CONTEXT_TYPE_MTP` context and runs a forward pass
through the draft head after each trunk batch, feeding `(token[p+1], h[p])` pairs via the pre-norm
embedding interface. Enables importance-matrix quantization targeting for draft-head layers.
Fork adaptations: flag renamed `--imat-mtp` (collision with deprecated `--mtp` CLI flag);
uses fork's 3-arg `llama_set_embeddings_pre_norm` API; reuses existing `llama_model_n_nextn_layer`
accessor (PR's new headers avoided). **Bug fix: null `mtp_batch.token` after free** (prevents
double-free in cleanup paths before `llama_batch_free()`); shipped with imatrix port.

### Added — Speed-bench harness for speculative-decode performance validation (2026-05-30)

Adds benchmarking harness (`tools/server/bench/speed-bench/`) for systematic speculative-decode
throughput measurement across draft strategies. Runs speculative-decode profile against a corpus,
emits throughput (tokens/s) and accept-rate metrics. Serves 40-cell spec-decode
validation matrix. Updated `docs/speculative.md` with speed-bench invocation examples.

### Fixed — Suppress JSON schema grammar application during thinking blocks (2026-05-30)

Port of buun `spiritbuun/buun-llama-cpp` commit `633a4f5d7`: fixes crash when `--json-schema-file`
is used with a thinking model (Qwen3.5/3.6). The chat template prepends thinking tokens
(e.g., `<think>…</think>`), which JSON schema grammar rejects, causing sampler-init failure.

Three changes to `common/sampling.cpp`:
1. Skip grammar prefill for `OUTPUT_FORMAT` when reasoning budget is active.
2. Force-create reasoning-budget sampler for `OUTPUT_FORMAT` grammars (state tracking required).
3. Suppress grammar enforcement during active thinking blocks; resume after `</think>`.

Smoke: pre-fix reproduces crash (`"Failed to initialize samplers: std::exception"`);
post-fix clean exit with valid JSON output, thinking block passes through correctly.

### Added — iGPU performance warning for MTP spec-decode (2026-05-30)

MTP spec-decode is a measured net-slowdown on integrated GPUs (0.54x at default n_max=3;
even n_max=1 with 100% accept stays below pure decode — inherent per-`llama_decode` launch overhead
dominates). Adds startup warning when MTP is enabled on iGPU-detected systems:
- Adds `ggml_backend_cuda_device_is_igpu(int device)` to `ggml-cuda.h/cu`: checks HIP device
  integrated property + gfx arch match (gfx1103/gfx1150 belt-and-suspenders detection).
- Calls `mtp_warn_igpu_once()` from MTP impl constructors; warning fires once per init, not per-cycle.
- Explicit `--spec-type mtp` / `--mtp` never blocked; warning is informational only.
Rationale: iGPU adoption tracking + user-facing guidance until GPU ring-buffer optimization lands.

### Changed — Qwen3.5/3.6 MTP converter converged to mainline; `--mtp` split-export restored (2026-05-30)

Per user directive (2026-05-29: "drop our divergence and use mainline's implementation unless we
genuinely improved it"), the fork's divergent `_Qwen35MtpMixin` rewrite (TODOs 145/146,
`7d8fefc82`/`d8ec65064`/`c0d71d750`/`36164e428`) is **dropped and replaced verbatim with current
mainline master's mixin** (`conversion/qwen.py`, byte-identical to ggml-org `0821c5fcf`; origin am17an
PR #22673). Plus a 4-line `conversion/base.py` add of the `mtp_only`/`no_mtp` ModelBase defaults the
mainline mixin relies on. **No loader (`src/models/qwen35*.cpp`) change, no C++ rebuild** — recon
confirmed the loader is converter-agnostic (keys off `nextn_predict_layers` metadata + `blk.N.nextn.*`
tensor names, identical between the two converters).

This **restores the previously-broken `--mtp` split-export** (the fork rewrite set `mtp_only=True` but
dropped its `prepare_metadata`/`filter_tensors` consumers → silently emitted a full bundled GGUF).
**TODOs 145/146 are SUPERSEDED-BY-CONVERGENCE** — they fixed regressions the fork's own rewrite
introduced; mainline was already correct for both the bundled-`attn_norm` and `--no-mtp` paths.

The two divergence bits assessed as fork "improvements" both proved **non-unique**: the multimodal
`language_model.` unwrap already lives in mainline's base `TextModel.filter_tensors` (routed through by
the mainline mixin), and the `bid + n_layer` block-index correction is subsumed by mainline-master's
`index_tensors`/`_original_block_count` design (rename precedes bid-parsing). Nothing fork-unique lost.

Verified on Qwen3.5-9B (dense, q8_0, ROCm gfx1150, `llama-speculative-simple --spec-type draft-mtp`):
- **bundled** (default) → arch `qwen35`, block_count 33, `nextn_predict_layers=1`, nextn tensors at
  blk.32; loads + speculates **coherent, 75.56% accept** (`n_drafted=45 n_accept=34`, `-m`/`-md` both bundled).
- **`--no-mtp`** → block_count 32, no nextn metadata/tensors (trunk-only); loads as target.
- **`--mtp`** → `mtp-`prefixed 18-tensor draft-head GGUF; loads as draft, same **75.56% accept** paired
  with the `--no-mtp` trunk as target (split-export round-trip).
MoE Qwen3.5/3.6-35B-A3B path untested (stretch goal; mainline mixin handles MoE expert-merge identically).

### Fixed — MTP Gemma4 §-FLAG-B: end-to-end external-assistant speculative decode (2026-05-30)

Gemma4 Multi-Token Prediction (PR #23398 guided port) now drafts correctly
end-to-end for the external-assistant path (Gemma4-26B-A4B-it target + `gemma4-assistant`
drafter). Validated **coherent at 47.3% accept** (`n_drafted=112 n_accept=53`, `--temp 0`,
ROCm gfx1150, `llama-speculative-simple --spec-type draft-mtp`), near the upstream PR #23398
CUDA reference of 0.588. The bundled-MTP qwen3.5 path is unaffected (regression-guard re-run:
coherent, 63.4% accept).

Three ordering/materialization fixes were required, on top of the port:

- **Draft weights materialized** (`fix(mtp): materialize gemma4-assistant draft weights — move
  "mtp." rename after load`) — the `"mtp."` tensor rename ran *before* `load_all_data`, so the
  name-keyed weight lookup missed and the draft head read all-zero on GPU → all-`NaN` logits →
  0% accept (every draft token `<unused2>`). Renaming after load materializes the weights.
- **External-draft context typed MTP** (`fix(mtp): type external-draft context MTP in
  speculative-simple`) — the `speculative-simple` external-draft path did not set
  `cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP`, aborting at `gemma4-assistant.cpp` construction
  (`GGML_ASSERT(src_mctx)`).
- **Speculator init hoisted** (`fix(mtp): hoist common_speculative_init before first draft
  decode`) — `common_speculative_init` (which calls `llama_set_mtp_source`) ran *after* the
  first draft decode probe, so the draft decoded before its MTP source was wired. Hoisting it
  before the first draft decode fixes the ordering.

**Harness note (important usage caveat):** `speculative-simple` applies **no chat template** —
it tokenizes `params.prompt` raw. Instruction-tuned targets (Gemma4-26B-A4B-**it**) therefore
require a **chat-templated prompt** (e.g. `<|turn>user … <|turn>model\n<|channel>thought\n<channel|>`);
a raw instruction yields degenerate target output (and the draft trivially agrees on the garbage,
inflating accept%). Base/completion or raw-tolerant models (qwen3.5) are unaffected. This was the
sole cause of the earlier "gemma4 output-degenerate" symptom — it was a smoke-harness prompt-format
bug, not an MTP/graph/conversion defect. A `--jinja`/chat-template option for `speculative-simple`
is a possible future hardening (out of scope here).

### Removed — Q1_0_G128 (GGUF type 43) (2026-05-29)

- Removed `Q1_0_G128` (GGUF type 43) — pure duplicate of mainline `Q1_0` (slot 41); zero type-43 GGUFs exist; slot 43 returned to mainline-growth reserve (ADR-003)

### Added — IQ3_KT: 3-bit trellis-coded quantization (2026-05-29, `623835cc9`)

Added IQ3_KT (3-bit trellis-coded quant); PPL +23.5% vs IQ3_K is inherent to the 3-bit
single-codebook design (see late-stage dual-codebook follow-up); cluster-accel fix k=60.
CPU, ROCm (gfx1150), and Vulkan backends validated. Imatrix required for quantize.
Based on ADR-018 trellis P3b design.

### Refactored — Mainline rebase onto b745 (`751ebd17a`) (2026-05-28, cascade)

Periodic mainline forward-sync. 68 new mainline commits integrated since `b9310` anchor.
Conflict resolution:

- **FWHT dual-pipeline** — mainline introduced `GGML_HINT_SRC0_IS_HADAMARD` hint path;
  fork's standalone `GGML_OP_FWHT` kept in fork-only enum segment. `ggml_vk_fwht`
  name collision resolved by renaming fork's function to `ggml_vk_fwht_op` (`cf70bbd33`).
- **ZAYA / TALKIE arch slot** — mainline `c9d98295a` added `talkie-1930-13b`; resolved via
  ZAYA slot bump.
- **Q1_0_G128 Vulkan dequant** — resolved against mainline dequant-funcs refactor.
- **`fwht_op.comp` dual-path + `ggml-cuda.cu`×3** — minor shader and CUDA conflict hunks.

Post-rebase validation: gfx1103 ROCm + Vulkan PASS, PPL=6.5453. gfx1150 cells were
pending at brief-time (§-FLAG).

### Fixed — Vulkan: rename `ggml_vk_fwht_op` to avoid redefinition collision (2026-05-28, `cf70bbd33`)

During the b745 rebase, mainline's hint-path `ggml_vk_fwht` (uses
`pipeline_fwht_f32[]`) and fork's `GGML_OP_FWHT` dispatch `ggml_vk_fwht`
(uses `pipeline_fwht`/`fwht_op`) ended up with the same symbol name.
Renamed fork's standalone-OP function to `ggml_vk_fwht_op`; updated its
call site at `GGML_OP_FWHT` case dispatch.

### Refactored — FWHT fork-only enum position alignment (2026-05-28, `3caf1caa0`)

`GGML_OP_FWHT` was inserted between `GGML_OP_FLASH_ATTN_SPARSE` and
`GGML_OP_TURBO_WHT`, breaking the fork-only-ops consistency pattern established
by `b3ec1f8e2`. Reordered to: `GLU → FLASH_ATTN_SPARSE → TURBO_WHT → FWHT →
COUNT`. Minimizes future mainline rebase conflict surface in `ggml/include/ggml.h`.

### Fixed — KV cache reuse regression on multi-turn Qwen3.6-35B-A3B (revert ccee426) (2026-05-28, `f92e515f2`)

Reverts mainline commit ccee426 (PR #23280) in `tools/server/server-context.cpp`
picked up via 2026-05-25 forward-sync. The change dropped a full batch of cached
tokens per turn on multi-turn Qwen3.6-35B-A3B (and likely Qwen3.5-35B-A3B-MTP)
requests, collapsing cache reuse. Revert restores pre-regression cache-reuse
behavior. Loader-smoke PASS (confirmed post-revert build + smoke on
Qwen3.5 MTP models). **§-RISK**: naked revert may reintroduce the hybrid-attention
crash that #23280 was fixing; monitor mainline #23589 for a cleaner fix.
Mainline: https://github.com/ggml-org/llama.cpp/issues/23589.

### Ported — buun: allow tensor-split with quantized KV cache (2026-05-28, `6774410fa`)

Removes the defensive block that prevented tensor-split (multi-GPU) with
quantized KV cache types. The meta backend already handles quantized KV
correctly — axis-0 split uses head-aligned granularity, view/permute
propagation maps to axis-2 for `flash_attn_ext`, and turbo `set_rows`
kernels handle quantized writes. Log the configuration instead of blocking.
Port of buun `spiritbuun/buun-llama-cpp` commit `0530f5111` (#59).

### Ported — buun: add `TURBO_WHT` to split planner (2026-05-28, `340f6fe21`)

The meta backend tensor-split planner did not know how to propagate split
state through `GGML_OP_TURBO_WHT`, causing an abort on multi-GPU setups with
turbo KV cache quantization. `TURBO_WHT` is an elementwise transform (128-element
WHT groups along axis 0); `handle_generic` with `scalar_only=false` is correct,
matching `GLU` and other elementwise ops.
Port of buun `spiritbuun/buun-llama-cpp` commit `9b1ffc6dd` (#59).

### Ported — domvox: per-layer SWA KV cache type (`--cache-type-k-swa` / `--cache-type-v-swa`) (2026-05-28, `30472d827`)

Adds independent KV cache type selection for the SWA (Sliding Window Attention)
layers of hybrid models (Gemma 4, and future SWA-hybrid architectures). Without
this, applying turbo KV uniformly across all Gemma 4 layers collapses PPL
(>100k). With `--cache-type-k turboq3 --cache-type-k-swa f16`, Gemma 4 PPL =
27.7k vs 24.9k F16 baseline (vs >100k all-turboq3). Port of domvox commit
`5c59d773f` on `feature/turboquant-hip-port-clean`; 11 files ported manually
(cherry-pick conflicted in 6 files due to fork-only additions).

### Fixed — convert: emit `blk.<N>.attn_norm.weight` for bundled-MTP Qwen3.5/3.6 (2026-05-28, `7d8fefc82`)

`convert_hf_to_gguf.py` bundled-MTP path emitted the MTP-head block without
`attn_norm.weight`, causing every Qwen3.5/3.6 bundled-MTP GGUF to fail loading
with `error loading model: missing tensor 'blk.32.attn_norm.weight'`. Root cause:
`Qwen3NextModel.filter_tensors` unconditionally dropped all `mtp.*` tensors before
`modify_tensors` was reached. Fix: `_Qwen35MtpMixin.filter_tensors` override — in
bundled-MTP mode passes `mtp.*` through to `modify_tensors`; in `--no-mtp` mode
drops them (preserving the prior behaviour). Also adds missing `from pathlib import
Path` import needed by the `mtp.fc`/`norm` remapper branch.

### Fixed — convert: zero nextn metadata + decrement `block_count` on `--no-mtp` for Qwen3.5/3.6 (2026-05-28, `d8ec65064`)

`--no-mtp` stripped MTP-head tensors (via the parent commit's `filter_tensors`
override) but left `block_count` counting the absent MTP block and
`nextn_predict_layers >= 1`, causing trunk-only GGUFs to fail loading. When
`no_mtp`, set `block_count` = trunk-only count and `nextn_predict_layers = 0`.
Bundled-MTP (default) path unchanged. Loader-smoke PASS (confirmed
post-merge).

### Added — OScaR Phase 2: INT2 KV residual window with hybrid-memory-chain root bug fix (2026-05-27, `c892e62a3`)

Fixes architectural regression: `llama_memory_hybrid` hardcoded `oscar_res_window=0` silently disabled residual windows for all hybrid models (Qwen3.5 family). Root bug propagation chain identified: 5 constructor signatures required to thread `oscar_res_window` parameter through. Implementation: new `k_res` F16 buffer alongside INT2-quantized main KV; FATTN dispatch performs split-reads (F16 for positions in tail window, INT2 FHT for older blocks). New helper `ggml_flash_attn_ext_set_oscar_res()` manages window state. CLI flag `--cache-oscar-residual-window N` (default 128). Gate results: Qwen3.5-9B-Q4_K_M PASS 7.2005 (R=128-FLAG, margin 0.0005); 0.8B FAIL 20.8844 vs ≤17.0 gate (-0.97% vs target). D2 R-sweep (R=128/256/512/1024) showed 0.8B architectural non-response (flat ~21 PPL), 9B monotonic improvement to R=512=-2.77% (recommended deployment default). FOLLOWUP-F (full-dimension FWHT) identified as high-priority architectural fix for 0.8B under-performance.

### Added — TriAttention Phase B: physical KV cache compaction evictor (2026-05-26, `6f93b4e5d`)

Implements `tria_compact_kv` legacy compaction strategy: selects prefix-protected,
top-scored, and trailing window tokens; evicts the rest via `kv->triattention_compact(keep_positions)`.
Adds `llama_kv_cache::triattention_compact()` (~75 LOC) that physically relocates KV tensor rows.
Gate results (Qwen3.5-0.8B-Q4_K_M, -c 4096 -ub 512 -b 512, 20ch wiki.test.raw): PPL=14.7299 PASS (limit 15.4664).
Evictor demo: n_kv=300 keep=256 evict=44 budget=224 window=32; no-fire quirk in prefill mode (architectural, correctness verified).
Build cells: gfx1150 ROCm/Vulkan + gfx1103 CPU/Vulkan GREEN; gfx1103 ROCm §-FLAG (HIP 7.2 LTO crash, pre-existing).
Phase C (GPU GQA kernel + SWA aggregation) unblocked but not queued.

### Removed — MTP convergence Phase A: deprecate --mtp CLI flag, remove legacy loader-gate fields (2026-05-26, `fd44da73f`)

Removes `has_mtp` / `cparams.mtp` / `llama_model_params::mtp` gate fields replaced by
`LLAMA_CONTEXT_TYPE_MTP` mechanism. Deprecates `--mtp` / `--multi-token-prediction` CLI flags
(now aliases to `--spec-type draft-mtp` with deprecation warning); `--no-mtp` becomes no-op.
HARD-PRESERVED: `mtp_op_type` enum, `llama_set_mtp_op_type()` API, and Qwen3.5 machinery.
Part of strategic pivot to converge MTP to mainline via cherry-pick of PR #23398 (Gemma4 MTP).

### Optimized — MTP performance: bulk embd_read_tgt single GPU→CPU sync per ubatch (2026-05-26, `750e74115`)

Replace per-row `llama_get_embeddings_pre_norm_ith()` loop with pointer arithmetic on
bulk result from `llama_get_embeddings_pre_norm()` in both `common_speculative_impl_draft_mtp` and
`common_speculative_state_draft_mtp::process()`. Prior profiling showed `embd_read_tgt` at
65k calls consuming 59.3% wall; projects ~47% speedup (1382 s → ~730 s on 200-token sample).

### Fixed — Suppress draft-simple auto-enable when dflash/mtp/draft-eagle3 explicitly set (2026-05-26, `b1799cf36`)

Conditional gate in speculative loader now respects explicit `--spec-type` selection,
preventing `draft-simple` from overriding user intent when another speculator is active.

### Added — DFlash converter: safetensors→GGUF DFlashDraftModel port (2026-05-26, `ba61a9d39`)

Ported `DFlashDraftModel` from Anbeeld/beellama.cpp with safetensors loader and GGUF converter.
Enables end-to-end DFlash spec-decode workflow with externally-defined draft models.

### Fixed — DFlash converter: post_attention_layernorm → attn_post_norm mapping (2026-05-26, `a2c9c8c49`)

Safetensors key `post_attention_layernorm` maps to internal field `attn_post_norm`
in DFlash layer struct; converter now applies correct field remapping per Anbeeld design.

### Added — GGML_OP_FWHT: Walsh-Hadamard Transform standalone op (2026-05-26, `3d37eb55c`)

New CPU/CUDA/HIP/Vulkan implementation of the Walsh-Hadamard Transform (FWHT) as a
standalone GGML op (enum `GGML_OP_FWHT`, OScaR FOLLOWUP-A). CPU reference + CUDA/HIP
kernels + Vulkan `fwht.comp` shader. Used by OScaR KV projection research track.
Mainline alignment: we lead with the standalone op (PR #23690 is bug-fix only).

### Fixed — Vulkan: register fwht.comp in vulkan-shaders-gen (2026-05-26, `de3953843`)

Explicit `string_to_spv()` registration for `fwht.comp` in `vulkan-shaders-gen.cpp`.
CMakeLists GLOB pattern alone does not auto-discover new .comp files; manual registration required.

### Added — OScaR Phase 1: GGML_TYPE_KV_OSCAR_INT2 FHT-based INT2 KV cache (2026-05-26, `e1f3e7083`)

New KV cache quantization type `GGML_TYPE_KV_OSCAR_INT2` (slot 71) combining
Fast Hadamard Transform (FHT) projection with per-block min-max quantization to INT2.
Phase 1 CUDA-only implementation. PPL gate PASS: 7.69 on Qwen3.5-9B-Q4_K_M
(F16 baseline ~6.6; +1.09 PPL / +16.5% vs broken-codebook IQ2_KT baseline 35.5 PPL).
Phase 2 (Vulkan/ROCm backends) and L2 sidecar architecture deferred.

### Fixed — CUDA: add SM80+ arch guard to k_sparse_flash_forward in flashprefill.cu (2026-05-26, `54df8392d`)

Closes Kaggle build saga. `flashprefill.cu` kernel body is now gated with `#if __CUDA_ARCH__ >= 800`
so sm_75 devices see an empty valid kernel body. CMake macro `GGML_CUDA_FLASHPREFILL_SM75_STUB`
auto-defined when `CMAKE_CUDA_ARCHITECTURES` lacks SM80+. Deprecates notebook sed-patch workaround.
4/4 build cells GREEN.

### Fixed — MTP M-RoPE duplicate-impl regression (2026-05-25, `e8e767347`)

Removed a duplicate `if (has_mtp) { configs.push_back(...DRAFT_MTP...) }` block at
`common/speculative.cpp:2426-2428`, a copy-paste introduced by upstream
`255582687b`. The duplicate caused two `common_speculative_state_draft_mtp`
instances to be created; `process()` then ran twice per batch, advancing
`ctx_dft seq_pos_max` twice and violating the M-RoPE `X < Y` invariant on the
second call. Diagnosed initially as a checkpoint-restore bug; per-iteration
tracing of `delta_post = -1` on all 124 iterations ruled that out. Fix is 3 LOC.

Gates (Qwen3.5-35B-A3B-MTP-IQ4_XS on gfx1150 ROCm):
mrope_errors **248 → 0**; accept rate **70.769%** (≥70 gate); PPL **6.5604**
bit-identical to anchor; MTP-ON 17.761 t/s = **0.737× MTP-OFF** (§-FLAG —
still below 0.78–0.85× projection; clean re-measurement on a quiet host queued, since the 24.1 t/s MTP-OFF baseline used here was contaminated by
concurrent peer-worker builds).

### Refactored — GGML op enum convergence to mainline (2026-05-25, `b3ec1f8e2`)

Moved fork-only ops `GGML_OP_FLASH_ATTN_SPARSE` and `GGML_OP_TURBO_WHT` to the
end of the op enum (just before `GGML_OP_COUNT`, after `GGML_OP_GLU`). All
subsequent ops now match the mainline numeric values, eliminating the
merge-conflict surface that mid-insertion was accumulating on every mainline
rebase. 4/4 build cells PASS. PPL 6.5604 bit-for-bit anchor match.

### Refactored — MTP CLI flag rename `mtp` → `draft-mtp` (2026-05-25, `763c79c9b`)

`"draft-mtp"` is now the canonical `--spec-type` name matching mainline.
`"mtp"` is retained as a backward-compat alias in
`common_speculative_type_from_name_map`. No behavior change.

### Added — Trellis IQ2_KT (Phase P3a) weight quant + IQ_KT-family template refactor (2026-05-25, `e9520caac`, `0dac276d9`)

Generalized the IQ4_KT trellis scaffold in `ggml-iqk-kt.cpp` into a
header-only template family `IQKTParams<G,N,A>` at
`ggml/src/ggml-iqk-kt-family.hpp` (with `iqkt_gen_group_int`,
`IQKTCookedBook`, `iqkt_build_cluster_index`, `iqkt_find_best_index`).
Ported IQ2_KT (`IQKTParams<8, 16, false>`, slot 153, 2.125 bpw) as Phase P3a
of the trellis port plan. Imatrix required at quantize-time.

PPL gate (Qwen3.5-0.8B IQ4_KT @ template refactor): **11.4264 ± 2.22**
(1-chunk, 512t, CPU, `wiki.test.raw`).

**Known limitations:**

- IQ2_KT on Qwen3.5-0.8B PPL=99.58 vs IQ2_KL=26.12 and IQ4_KT=11.43 — anomaly
  open as a known issue (scope-TBD: scale-dependent vs general).
- Vulkan path not yet ported.
- IQ3_KT / IQ1_KT (Phase P3b / P3c) queued behind P3a.

### Added — IQ{2,3,1}_KT cluster-acceleration (8D base-3 hash, k=60) (2026-05-25, `1e8501e46`)

GROUP_SIZE=8 trellis types fall back to a brute-force 65536-entry codebook
search at quantize time (the IQ4_KT GROUP_SIZE=4 family already had cluster
acceleration). The Option-C lift adds an 8-dimensional base-3 hash (3^8 = 6561
bins) with `k_neighbours=60`, recovering ~30× quantize speedup (~7h → ~10–20 min
on Qwen3.5-9B). Coverage 0.91% (599/65536); matches the IQ4_KT pattern.

**Known limitations:**

- PPL 107.87 vs brute-force baseline 99.58 — **+8.3%** above the ≤+5% clean
  threshold (§-FLAG). Shipped on speedup grounds per the worker brief criteria.
  Retune to k=80–100 queued for late-stage polish.
- When IQ3_KT / IQ1_KT land via Phase P3b / P3c, apply the same `k_neighbours=60`
  treatment at their `iqkt_cooked_book_init` call sites.

### Added — IQ2_KL Vulkan shaders (Phase 5b-1c S2) (2026-05-25, `3723c1f61`)

Ported IQ2_KL Vulkan dequant + matvec shaders + S1 brace/template fixes.
IQ2_KL is now CPU + CUDA/HIP + Vulkan on `main`.

### Added — IQ5_K + IQ6_K Vulkan shaders (Phase 5b-2 S2) (2026-05-25, `0ade7ff86`)

Ported IQ5_K (slot 140) and IQ6_K (slot 141) Vulkan dequant + matvec
shaders. Both are now CPU + CUDA/HIP + Vulkan on `main`. Imatrix required.

### Fixed — EAGLE3 fc tensor dtype-aware read (BF16/F16 → F32 conversion) (2026-05-25, `4c38845c4`)

`src/llama-model.cpp:llama_model_eagle3_get_fc_weight()` hardcoded
`n_elements * sizeof(float)` when the fc tensor is BF16, requesting twice as
many bytes as the tensor holds and reading out of bounds. Fix uses
`ggml_nbytes()` plus `ggml_bf16_to_fp32_row` / `ggml_fp16_to_fp32_row`
conversion. Smoke gate PASSED (EAGLE3 init OK, 6.3 t/s, EXIT:0).

### Added — DFlash S2 dispatch (`common_speculative_state_dflash` + factory) (2026-05-25, `ef80c728c`)

DFlash speculative-decode dispatch wired into the `--spec-type` factory.
`common_speculative_state_dflash` is now selectable from CLI / server.

### Added — DFlash S3: GPU ring buffer, bulk argmax, server spec_type wiring (2026-05-25, `9b7ab4e83`)

Phase S3 of the DFlash port: GPU-side ring buffer for drafter activations,
bulk argmax kernel, and server-side `spec_type` plumbing for `llama-server
--spec-type dflash`. End-to-end smoke gate GREEN on Qwen3.6-35B-A3B-MTP target
+ Qwen3.6 DFlash-draft Q8_0 — 33.3% combined accept rate in dual-spec mode
(isolated DFlash-only rate unmeasured; estimated ≥60%).

### Fixed — DFlash `mask_token_id` GGUF type mismatch (2026-05-25, `1436d1890`)

`dflash_mask_token_id` was declared `int32_t` in `llama-hparams.h` but the
GGUF stores it as `u32`. All binaries built before this fix failed to load
the z-lab DFlash drafter GGUF. Build must be ≥ this commit to load any
z-lab DFlash drafter.

### Added — DFlashDraftModel safetensors → GGUF converter (2026-05-25, `ee7d4f896`)

Ported the DFlashDraftModel converter (MIT lift from Anbeeld/beellama.cpp).
Registered in `TEXT_MODEL_MAP` as `"DFlashDraftModel" → dflash_draft`. Adds
`DFLASH_DRAFT` arch + `DFLASH_FC` / `DFLASH_HIDDEN_NORM` tensor enums to
`gguf-py/gguf/constants.py`. Qwen3.5-4B and Qwen3.6 tokenizer-hash → `"qwen35"`
mapping added. Smoke GREEN: 915 MB GGUF, arch=`dflash-draft`, all DFlash KV
entries present. Gemma-4 path implemented but not yet smoke-tested.

### Added — TriAttention Phase A in-graph K/V capture harness (2026-05-25, `6cbc9e06c`)

TriAttention was deferred post-Phase-8 since 2026-05-15 due to a sub-alloc
zero-read in `ggml_backend_tensor_get` (not fixed in mainline for the basic
non-2D path TriAttention uses). The Phase A workaround installs a persistent
graph-side capture buffer via `llama_tria_capture_alloc`, populated by
`ggml_set_rows` nodes in `build_attn()`. Phase A `tria_compact_kv` is a no-op
(harness only; zero evictions).

Companion fixes:

- `eea5e25f5` — `TRIA_HIP_BACKEND` guard so the Phase B GPU path stays gated
  behind an undefined macro until Phase C lands.
- `2ad2564f1` — safe null-return in `get_layer_k_raw` / `get_layer_v_raw` for
  hybrid models (Qwen3.5-0.8B only has 6/24 layers in the full-attn KV cache).
- `cbd071632` — third `dynamic_cast` branch in `src/llama-context.cpp` for
  Gemma-4's `llama_kv_cache_iswa` (which does not inherit from `llama_kv_cache`).
  Gemma-4 now allocates 35-layer K/V capture buffers. `kv_swa` (SWA layers) is
  still not captured; Phase C extension needed for SWA coverage.

Validation: 4-cell build PASS; Phase B GQA CPU smoke GREEN 3/3 (Qwen3.5-9B,
Llama-3.1-8B, Gemma-4-E2B); PPL Δ=0.09% (no-op compact expected).

### Refactored — Mainline rebase onto `b9310` (`e2ef8fe42`) (2026-05-25, `1191e48fc`)

Periodic mainline forward-sync. 282 fork commits replayed onto `b9310`.
3 cherry-picks from mainline: `9a9ca0ff2` (ggml-alloc OOB), `f9323396f`
(MTP KV-type server), `d2a79534a` (FFN_LATENT MUL_MAT). 2 post-rebase
fixups: `85fdde861` (restore 3-param `accept()` interface that the b9246
rebase fixup reverted), `e109b17d8` (repair EAGLE3 struct with missing
batch field + duplicate ctor/fns from rebase conflict). PPL smoke gate
6.5604 vs anchor 6.71 — GREEN. ROCm + Vulkan builds EXIT:0 on gfx1150.
gfx1103 builds not executed in this session (§-FLAG).

### Added — DFlash S1 model loader (2026-05-24, `b8bf27eda`)

Port of the DFlash S1 drafter model architecture and GGUF loader from buun `master`.
The model type and loader are now in-tree; speculative-decode integration requires a
DFlash S1 drafter GGUF (not yet publicly available). Revival unblocked further by
this port — remaining gate is drafter GGUF sourcing.

### Added — Phase 5b-1c: IQ2_KL ultra-low-bpw weight quant (2026-05-24, `e404274b9`)

Port of `IQ2_KL` (ik_llama type 157, 2.6875 bpw) from ik_llama.cpp. Imatrix-aware
ultra-low-bitrate weight quantization at ygg canonical slot 157. CPU + CUDA/HIP
on main; Vulkan parity in-flight.

### Added — Phase 5b-2 S1: IQ5_K and IQ6_K weight quants (2026-05-24, `f7a489de5`)

Port of `IQ5_K` (slot 140) and `IQ6_K` (slot 141) from ik_llama.cpp upstream.
Higher-quality extensions of the IK-K family with imatrix weighting. CPU + CUDA/HIP
+ Vulkan on main. Slots preserved from ik_llama compatibility zone.

### Added — EAGLE3 hidden-state extrapolation speculative decoder (2026-05-24, `e9f6d9ce7`)

Port of the EAGLE3 speculative decoder from carlosfundora `1-bit-turbo`. EAGLE3
uses the target model's residual hidden states to extrapolate draft tokens rather
than running a separate full drafter model. Backend-agnostic (no novel GPU kernels).
Primary target: Gemma 4 26B-A4B with the paired assistant checkpoint. Full
`--spec-type` factory dispatch integration in-flight.

### Fixed — Q1_0 wired into Flash Attention VEC dispatch (2026-05-24, `db1e8cb9d`)

Q1_0 CUDA/HIP kernel was not registered in the Flash Attention VEC dispatch table,
causing incorrect dispatch for Q1_0 weights on the attention path. +2 LOC wiring fix.

### Added — PHANTOM-X Phase 2 factory dispatch adapter (2026-05-24, `4fd52ddc0`)

PHANTOM-X speculative decoder wired into the `--spec-type` factory dispatch
adapter (Phase 2 integration). The speculator can now be selected via the
standard `--spec-type` flag alongside other spec-decode mechanisms.

### Fixed — MTP draft path backend_sampling default-flip (2026-05-24, `b665294b8`)

`backend_sampling` now defaults to `false` for the MTP draft path. The previous
default caused the MTP draft sampler to invoke the full backend sampling pipeline
(including top-k filtering) before the draft was verified, which degraded draft
acceptance rates. The `top_k=1` argmax draft path (see `fd9bf51f8` comment) only
works correctly when `backend_sampling=false`; the fix ensures clean behavior
without a manual override.

### Added — PHANTOM-X speculator port (2026-05-24, `2199e8445`)

Port of the PHANTOM-X speculative decoder from carlosfundora `1-bit-turbo`.
PHANTOM-X uses a learned per-model n-gram pattern lookup table for draft
generation. Backend-agnostic (CPU n-gram; no novel GPU kernels).

### Added — Q1_0_G128 (PrismML 1-bit weight quant, Bonsai-family) (2026-05-24, `67edf5eec`)

Port of `Q1_0_G128` from carlosfundora `1-bit-turbo`. 1-bit weight quantization
with 128-element groups. Placed at ygg canonical slot 96 (first slot of ik_llama
compat zone) to avoid the three-way slot collision between ik_llama (41),
carlosfundora (43), and mainline `Q1_0` (41). Imatrix required.
CPU + CUDA/HIP on main.

### Fixed — Vulkan MUL_MAT_ID is_empty() guard for base-K types (2026-05-24, `c4da029f3`)

Extended the `is_empty()` guard in `ggml_vk_get_mul_mat_mat_pipeline` to cover
base-K types (IQ2_K/IQ3_K/IQ4_K) in the MUL_MAT_ID (MoE expert-routing) path.
The fix forces dequant-to-f16 fallback for base-K types in batched MoE dispatch,
preventing the latent Vulkan SEGV when base-K weights are used with MoE models.

### Added — MTP top_k=1 deliberate-choice comment (2026-05-24, `fd9bf51f8`)

Documents the intentional `top_k=1` (vs mainline `top_k=10`) in `state_draft_mtp` ctor.
With `ad277572` backend sampling pre-filtering to top-10 and bundled-MTP weight-sharing,
`top_k=1` yields an argmax draft that closely tracks target greedy. Comment guards against
inadvertent revert: "Do NOT raise to 10 without re-running Smoke B."

### Added — NLD server self-spec loop (2026-05-23, `1cb8c4218`)

Port of `tools/server/server-context.cpp` additions from buun `f339dbebe`
(+589 net LOC, 12 hunks): `is_diffusion` auto-detection via
`llama_model_is_diffusion()`; `diff_self_spec` fields on `server_slot`;
rejection-sampling spec loop with temperature, think-tag suppression,
cross-turn penalty, and loop detection. MTP server paths coexist
cleanly (mutually exclusive: a slot is MTP or diffusion, never both).

Server self-spec smoke: 4.49 t/s (128 tokens); ~59% draft acceptance.
MTP regression gate: 84.62% accept on Qwen3.5-35B-A3B-MTP — above v525 anchor (77.78%).

### Added — NLD Tier-B CLI port (2026-05-23, `49f88e18a` + `35315922c`)

Selective port of Nemotron-Labs Diffusion from buun `f339dbebe` (~612 LOC net):
GGUF converter (`conversion/nemotron_labs_diffusion.py`), diffusion library
(`examples/diffusion/diffusion.h` + `diffusion.cpp`), CLI refactor
(`examples/diffusion/diffusion-cli.cpp`), model loader fixes for DREAM arch
(`src/models/dream.cpp`, `src/llama-model.cpp`), and 3 new CLI flags
(`--diffusion-block-length`, `--diffusion-self-spec`, `--diffusion-draft-length`).

Smokes: block-mode 1.9 t/s; self-spec 7.0 t/s (3.7× speedup, 68.4% draft acceptance).
MTP regression gate: 69.0% accept on Qwen3.5-35B-A3B-MTP after port (anchor 70.3%, Δ −1.2pp, within ±5pp).

### Fixed — MTP V-J accept-rate gap (2026-05-23, `705ffccb8`)

`examples/speculative-simple/speculative-simple.cpp` was calling `llama_decode(ctx_dft, batch_tgt)` directly
instead of `common_speculative_process(spec, batch_tgt)` after the target decode. Without the process
call, `state_draft_mtp::process()` never ran, keeping `pending_h` zeroed — the MTP head read garbage
h_input and acceptance collapsed. The server (`server-context.cpp:3125`) already had the correct call;
only the standalone binary was broken. Fix is +8 / -2 lines.

Before: **38% acceptance** (Qwen3.5-35B-A3B-MTP long-prompt).
After: **70.28% acceptance** (mainline b9246 anchor: 71.3%). Throughput +45% e2e.

### Added — MTP Migration phases 0-3 (2026-05-22 to 2026-05-23, `34a54f5fd`–`4a9977f49`)

Converge fork's MTP speculative driver, loader, and graph to mainline b9246 architecture:

- **Phase 0 (preflight):** recon classifying all MTP-touching files and deciding migration scope.
- **Phase 1 (`34a54f5fd` + `1d8aa9d30`):** arch constants + NextN classification converged to b9246; server-context.cpp loader migrated from `cparams.mtp` to `LLAMA_CONTEXT_TYPE_MTP`.
- **Phase 2 (`fefe017ea`):** Qwen3.5 + Qwen3.5-MoE loaders split into `load_block_trunk` + `load_block_mtp` lambdas matching b9246 shape (E1 task).
- **Phase 3 (`4a9977f49`):** both graph ctors converged to `cparams.embeddings_pre_norm_masked` flag (retiring fork-local `mtp_full_embd`); inverted polarity vs mainline documented (bundled-MTP semantics preserved).

PPL gate post-Phase-3: 5.5302 ± 0.064 vs Phase-1 anchor 5.5302 — numerics-clean.

### Added — Phase 5b-1b row-meta IK weight quants (2026-05-22, `026671689`–`5fe804bcd`)

Port of IQ4_KS / IQ4_KSS / IQ3_KS / IQ4_KT from ik_llama. Requires P0 prereq
commit `d91059253` (add `row_meta_size` to `ggml_type_traits` + extend `ggml_row_size()`).

Ship sequence:
- `d91059253` — P0: `row_meta_size` infra (prerequisite for all row-meta types).
- `026671689` — Phase 5b-1b: CPU traits + CUDA/HIP kernels + Vulkan dequant/matvec shaders for all 4 types.
- `a0fe65a77`, `e4caef152`, `d703bf5ea`, `6d9957ae1`, `3f629f8fb` — 5 post-ship bugfixes (CUDA dequant kernels, tensor stride/nbytes, row validation, HIP __shfl_xor_sync, Vulkan batched guard).
- `5fe804bcd` — Vulkan KS batched `mul_mat` SEGV fix: extend `is_empty()` guard in `ggml_vk_get_mul_mat_mat_pipeline` to the non-q8_1 branch; KS types now dequant-to-f16 on the batch path. Also fixes the identical latent SEGV in base-K types.

PPL gate (Vulkan, Qwen3.5-9B, 20 chunks): IQ4_KS 6.4131 / IQ3_KS 6.7325 / IQ4_KSS 6.5773 / IQ4_KT 6.5364 vs ROCm anchors (Δ ≤ 0.043).

### Added — Phase 5b-1a base IK weight quants (2026-05-20 to 2026-05-22, `aed6d2965`)

Port of IQ2_K / IQ3_K / IQ4_K from ik_llama (Phase 5b-1a). CPU traits +
CUDA/HIP kernels (convert.cu, mmvq-iqk.cu) + Vulkan dequant/matvec shaders (6
`.comp` shaders). No row_meta; standard `ggml_type_size`-only layout.
Renumbered to ygg canonical IDs: IQ4_K=139, IQ3_K=138, IQ2_K=137
(ik_llama compatibility zone). PPL parity Δ < 0.0045 vs ik_llama reference.

### Added — PFlash base port Phase 3: HIP GPU scorer (2026-05-19, v355, `abe0bb81a`)

HIP-ify scorer compute graph. Replaces CPU backend with GPU backend via
`ggml_backend_dev_by_type(GPU)` in both `pflash-loader.cpp` (weight storage)
and `pflash-graph.cpp` (compute graph). CPU fallback retained for Vulkan-only
builds. Compute context memory bumped 4 MB → 16 MB for GPU tensor overhead.
Enables 24× scorer speedup versus Phase 2A CPU baseline (9.89s → ~0.41s per
prefill window on gfx1102 ROCm+HSA_OVERRIDE). Phase 2A F32 tok_embd
dequant (`dd91b3fe7`) is the prerequisite that enables clean GPU dispatch.
4-cell smoke PASS; FF-merged to main as v355. Follow-ups: Phase 4b (bulk
upload) and Phase 4c (scorer caching + Vulkan GPU scorer fix).

### Added — MTP E3b chain prediction support, integration ship (2026-05-19, v348, `02bf7aa67`)

buun SD-091 E3b ladder fully integrated and shipped. Four phases
(REDO-FROM-SCRATCH adapted from buun SD-091 bf22e115e):

- **Phase 2-Extend-A** (`9b983083e`): graph infrastructure for chain
  predictions — `chain_layer_output` tensor capture and `chain_inputs`
  tensor array in `llama_context`.
- **Phase 2-Extend-B** (`85e4ac622`): chain prediction loop in qwen35 and
  qwen35moe graph builders; depth-indexed prediction heads wired.
- **Phase 2-Extend-C** (`66f63ecde`): context extraction + public API —
  `llama_get_mtp_chain_logits_ith(ctx, chain_depth, i)` and
  `llama_get_mtp_chain_depth(ctx)` added to `include/llama.h`.
- **Phase 2-Extend-D** (`c334fd2c3`): speculative driver consumption —
  chain logits consumed by `llama_speculative_*` driver (Vulkan build
  fix at integration: `+<cmath>` / `std::expf`).

Integration: linear stack rebased onto v342 (`55ef1ef45`) + 4-cell build
PASS + 4-cell smoke PASS + 16-sidecar ripple PASS + FF-merge → main v348.
Follow-ups: Phase 2-Extend-G (accept-rate gate on 35B-MTP) + Phase 2-Extend-E
(Vulkan chain depth=0 fix).

### Fixed — RoPE theta FP64 precision for high freq_base models (2026-05-18, v342, `55ef1ef45`)

Lift from Luce-Org `4a4dab41fa`. Adds `rope_theta_fp64()` device helper:
computes `pos * theta_scale^exp` in FP64 with mod-2pi reduction in FP64,
then narrows to float for sincosf. Replaces 9 FP32 `powf(theta_scale, ...)`
call sites in `rope_norm`, `rope_neox`, `rope_multi`, and `rope_vision`
kernels.

FP32 theta accumulates catastrophic phase error for models with high
`freq_base` (Qwen3.5 uses `freq_base=1e7`, which exceeds the FP32 safe
range ~8.4e6 per arXiv:2602.10959). At positions >~8K the trig phase
precision loss degrades attention quality. Runtime guard: FP64 path is
gated behind `freq_scale==1.0f && ext_factor==0.0f && freq_factors==nullptr`;
models using per-frequency factors (Gemma4, LLaMA 3.1+ long-ctx) or
YaRN/NTK scaling fall back to the existing FP32 path transparently.
Fork-only bugfix; not in mainline as of 2026-05-18.

### Fixed — README MTP speculative usage: llama-speculative-simple and llama-server, not llama-cli (2026-05-19, `bef4a1b82`)

Corrected documentation to note that MTP speculative decoding is invoked
via `llama-speculative-simple` or `llama-server --draft`, not `llama-cli`.
`llama-cli` does not support the `--draft` flag; the README previously
implied it did. No code changes.

### Fixed — README gfx1150 host APU name: Strix Halo → Strix Point (2026-05-19, `5c257b475`)

Corrected the codename for the gfx1150 host. The APU is Strix Point
(Ryzen AI 9 HX 370, 12C/24T Zen 5 + Zen 5c), not Strix Halo (which is a
different product line). No code changes.

### Added — README Attribution section crediting sibling forks and original authors (2026-05-18, `466fc667e`)

Post-v327 follow-up. Merged the README attribution additions into main.
Documents the sibling fork lineage (buun, carlosfundora, TheTom,
ik_llama) and original llama.cpp authors whose work this fork builds upon.
No code changes.

### Fixed — Multi-backend /opt RPATH; gfx1103 Vulkan binary was loading ROCm libggml (2026-05-18, v327)

Root cause of the v326 gfx1103 Vulkan SIGSEGV was a broken `RUNPATH` in installed
binaries (`/../lib`, resolving to `/lib`). Without a valid `$ORIGIN`-relative
RPATH, the dynamic linker fell back to ldconfig and resolved the ROCm
`libggml.so.0` (listed first in the ldconfig configuration) instead of the Vulkan
cell's own `libggml.so.0`. The Vulkan binary was silently running as a ROCm
binary without `HSA_OVERRIDE_GFX_VERSION=11.0.2`, causing SIGSEGV on GFX1103
hardware. The BF16 cpy SPIR-V is structurally valid (`spirv-val` passes) and
was never loaded on RADV PHOENIX — the crash was a misdiagnosis.
Fix: `CMAKE_INSTALL_RPATH=$ORIGIN/../lib` in `CMakeLists.txt`; each /opt cell
now resolves its own `libggml.so.0` via RPATH before ldconfig.

### Added — Vulkan BF16 copy pipelines (2026-05-18, `9ffaa0967` on `main`, v326)

Cherry-picked from mainline PR #22677. Adds `pipeline_cpy_bf16_f32` and
`pipeline_contig_cpy_bf16_f32` to `vk_device_struct`, registered in
`ggml_vk_load_shaders`. Enables BF16→F32 copies on the Vulkan path.

### Fixed — Pre-norm embedding mask initialization (2026-05-18, `daf0ffa70` on `main`, v325)

Cherry-picked from mainline PR #23256. Adds
`cparams.embeddings_pre_norm_masked = false` to `llama_new_context_with_model`
to ensure the flag is zero-initialized regardless of caller defaults.

### Added — Vulkan BF16 FA dispatch via inline dequant (2026-05-18, `a36a69c69` on `main`, v324)

Pattern B: `uvec2` inline dequantization in the FA shader. Adds BF16 KV
types to the Vulkan FA allowlist and the 4-byte inline dequantize path.
Uses COOPMAT1 when available on the target GPU. 4-cell smoke PASS (GFX1150
ROCm + Vulkan; GFX1102/1103 ROCm + Vulkan).

### Fixed — KV cache CPU fallback for types lacking GPU SET_ROWS (2026-05-18, `5d3164d67` on `main`, v323)

Types where `supports_op(GGML_OP_SET_ROWS)` returns false now allocate their
KV buffers on the CPU backend, preventing silent data corruption. Previously
these types fell through to GPU allocation with no SET_ROWS implementation.
Affects: RotorQuant (planar3/4, iso3/4), TURBOQ_{2,3,4}_TCQ prior to their
respective SET_ROWS shader lifts.

### Added — Novel model architecture: Zyphra ZAYA1-8B (2026-05-15, `64a481bb6 → cc8455581` on `main`)

In-tree port of the Zyphra ZAYA1-8B hybrid MoE — first novel-arch model
port in this fork that did not originate in mainline llama.cpp or any of
the six tracked sibling forks. Reference impl was the unmerged
`Zyphra/vllm@zaya1-pr` branch + a sibling `transformers@zaya1` branch.

- **`LLM_ARCH_ZAYA`** registered in `src/llama-arch.{cpp,h}` with 25 new
  per-layer tensor enums covering CCA (Mamba-cached convolutional attention),
  EDA (depth-recurrent router state averaging), and ResidualScaling.
- **`src/models/zaya.cpp`** — full graph builder: per-layer residual
  scaling, CCA attention on even layers (depthwise conv → grouped conv →
  L2-norm → k-scale → NEOX partial-0.5 RoPE → GQA attention), MoE on odd
  layers (down → optional EDA → RMSNorm → GELU MLP → 17-logit head →
  softmax → MoD-skip → top-1 over 16 experts), top-level final residual
  scale, output_norm, tied-embedding LM head.
- **HF → GGUF converter** in `convert_hf_to_gguf.py` plus `gguf-py/gguf/`
  metadata (incl. Gemma 262 144-token vocabulary; bos=2/eos=106).
- **`ggml_conv_1d_grouped`** helper ported from the Zyphra fork into
  `ggml/src/ggml.c` — pure C composition of `view_3d` / `conv_1d` /
  `concat`. **No new GGML_OP** and no backend changes.
- **`LLAMA_EXPERT_GATING_FUNC_TYPE_NONE`** case added to `build_moe_ffn`
  in `src/llama-graph.cpp` — the enum value already existed in
  `llama-hparams.h` but the switch aborted on it.
- **Multi-seq path fully functional.** Four bugs identified and fixed:
  - `prev_hs` non-contig view of `cca_state` for `n_seqs>1` → add `ggml_cont`.
  - `ggml_conv_1d_dw` asserts `b->ne[3]==1` from internal reshape →
    replaced with `ggml_ssm_conv` (natively sequence-batched depthwise).
  - `cont(transpose(QKraw))+reshape_3d` silently scrambled channel/seq for
    `n_seqs>1` → direct `reshape_3d(n_qk, n_seq_tokens, n_seqs) +
    permute(1,0,2,3) + cont` (memory-preserving; ubatch is seq-major).
  - **Latent mainline bug in `ggml_conv_1d`'s final mul_mat → reshape_3d**
    scrambles `(OL, OC, N)` layout for any input batch `N > 1` (channel 0
    matches by coincidence because the seq-stride dim collapses; channels 1+
    cross-mix with the seq dim). Local workaround: `conv_1d_grouped_multiseq`
    lambda inside `src/models/zaya.cpp` using corrected `reshape_3d(OL, N, OC)
    + permute(0,2,1,3) + cont`. **No ggml-core changes** per mainline-fidelity
    policy; future models calling `ggml_conv_1d` /
    `ggml_conv_1d_grouped` with `n_seqs > 1` must use the lambda or discuss
    a ggml-core fix with the user first.

PPL gates (gfx1103 Vulkan, 80 chunks, c=512, wikitext-2-raw-test):

| Quant | Bits | Single-seq PPL | Multi-seq PPL | Δ |
|---|---|---|---|---|
| F16 | 16 | 30.5016 | 30.5270 | +0.08% |
| Q8_0 | 8.5 | 30.5031 | 30.5231 | +0.07% |
| Q5_K_M | 5.5 | 29.9358 | 29.9468 | +0.04% |
| IQ4_XS-imat-guq5k | 4.25 | 31.9483 | 32.0073 | +0.18% |

All four within ±0.5% release threshold for single-vs-multi-seq parity.
Full-corpus (570-chunk) F16 multi-seq = 31.4802 ± 0.34 (higher than 80-chunk
because later wikitext chunks are harder; consistent with single-seq).

### Added — Multi-seq diagnostic tooling (2026-05-15, `cc8455581` on `main`)

- `examples/eval-callback`: `-np N` partitions the prompt tokens into N
  sequences of equal length, each starting at pos 0, for layer-by-layer
  multi-seq-vs-single-seq activation diffs (seq 0 sees the same head as
  a single-seq run on the same prompt).
- `examples/eval-callback`: example category switched to
  `LLAMA_EXAMPLE_DEBUG` so `--tensor-filter REGEX` is recognized; wires
  `params.tensor_filter` into `common_debug_cb_user_data`.
- `common/debug.cpp`: skip the GPU→host `ggml_backend_tensor_get` when the
  tensor name doesn't match the filter — significant PCIe-bound speedup
  for narrow filters on large models.

---

## [`milestone/phase-1-turboquant-kv-foundation`] — 2026-05-21

Phase 1 — TurboQuant KV foundation. Released at commit `9ee5b2299` on `main`.

### Added — KV cache types (no model re-quantization required)

All three pass to `--cache-type-k` / `--cache-type-v` on any GGUF whose
`head_dim` is a multiple of 128. Backends: CPU + ROCm/HIP + Vulkan.

- **`turboq2` (`GGML_TYPE_TURBOQ2_0`, slot 60)** — 2-bit PolarQuant,
  4 centroids, no QJL signs. ~2.125 bits/value (vs fp16 → ~7.5× compression).
- **`turboq3` (`GGML_TYPE_TURBOQ3_0`, slot 61)** — 2-bit PolarQuant + 1-bit
  QJL signs = 3-bit index. ~3.125 bits/value (~5.1× compression).
- **`turboq4` (`GGML_TYPE_TURBOQ4_0`, slot 62)** — 4-bit PolarQuant, no QJL
  (default mode per `TURBOQ4_USE_4BIT`). ~4.25 bits/value (~3.8× compression).

PPL gates (Qwen3.5-9B-BF16, 32-chunk c=512 wikitext-2-raw-test):

| Type | ROCm | Vulkan | Cross-backend Δ | vs F16 KV |
|---|---|---|---|---|
| `turboq2` | 7.8041 | 7.8059 | +0.023% | +14.5% |
| `turboq3` | 7.5939 | 7.6065 | +0.17% | +11.4% |

### Added — Weight quantization types (requires re-quantization + imatrix)

ADR-016 imatrix-required (importance-matrix weighting); re-quantize with
`llama-quantize --imatrix <file>` to produce these GGUFs.

- **`WHT3_0` (slot 80)** — WHT-rotated 3-bit weight quant, 8 Lloyd-Max
  centroids, block size 32, dual half-block scales. **CPU + CUDA/HIP only**;
  Vulkan port deferred (no TQ3_1S shaders in upstream sources).
- **`WHT4_0` (slot 81)** — WHT-rotated 4-bit weight quant, 16 Lloyd-Max
  centroids, block size 32, dual half-block scales. **CPU + CUDA/HIP +
  Vulkan**. 5.18 BPW; PPL beats `Q4_K_M` (4.5 BPW) by ~1%.

PPL gate (Qwen3.5-9B-WHT4_0, 32-chunk c=512 wikitext-2-raw-test):

| Backend | PPL | vs F16 baseline 6.8168 | vs Q4_K_M ROCm 7.6278 |
|---|---|---|---|
| ROCm | 7.5563 | +10.85% | -0.94% (at higher BPW) |
| Vulkan | 7.5520 | +10.79% | — |

Cross-backend Δ +0.057% — well within 0.5% release gate.

### Added — Boundary V / `TURBO_LAYER_ADAPTIVE` env var

Optional layer-adaptive KV precision (default OFF; explicit opt-in):

- Mode 1 — q8_0 K+V for first-4 + last-4 layers (turbo elsewhere)
- Mode 2 — q8_0 K+V for last-8 layers (turbo elsewhere)
- Mode 5 — V=turboq4 at first-2+last-2 layers, V=turboq2 elsewhere
- Mode 6 — V=turboq4 at last-8 layers, V=turboq2 elsewhere
- Mode 7 — **Boundary V (recommended)**: V=q8_0 at first-2+last-2 layers,
  V=turboq2 elsewhere. Recovers ~1.2% PPL over pure turboq2.

### Added — GGML op `GGML_OP_TURBO_WHT`

Forward + inverse Walsh-Hadamard transform op for the rotation phase of
TurboQuant. CPU + CUDA/HIP + Vulkan compute-shader (`turbo_wht.comp`).
See [docs/OP_ASSIGNMENTS.md](docs/OP_ASSIGNMENTS.md).

### Added — `test-backend-ops -o TURBO_WHT` regression net

27 test configurations across forward + inverse + roundtrip. 27/27 OK on
ROCm + Vulkan, gfx1150 + gfx1102/1103.

### Changed — Backend scope refinement

gfx1102/1103 ROCm support is in-scope as a *regression-smoke target* (catches
HIP-shim breakage early; cross-host PPL parity validated against gfx1150).
Production-inference calibration on those hosts still defers to Vulkan
(upstream AMD Tensile/hipBLAS GEMM gaps). gfx1102 ROCm binaries run with
`HSA_OVERRIDE_GFX_VERSION=11.0.2` (gfx1102-built binary on gfx1103
hardware).

### Changed — Trunk renamed `master` → `main`

The durable trunk branch is `main` (was a sidecar-engine port branch
through session 5, then `master` briefly). GitHub default branch updated.

---

## [`milestone/phase-0.7-sidecar-engine`] — 2026-05-12

Phase 0.7 — Sidecar plugin engine. Released at commit `f99ad5df8`.

### Added — Sidecar plugin runtime (~355 LoC, backend-agnostic)

Hook points: residual-stream, MoE-expert, post-compute-logits, weight
deltas. Out-of-tree `.so` plugins via a stable C ABI.

### Added — `llama_model_select_buft` API extension

Allows backend buffer-type selection for sidecar plugins that need
specific tensor placement.

---

## [`milestone/phase-0-foundation-complete`] — 2026-05-12

Phase 0 — Type-ID contract + dual-backend PPL regression harness.
Released at commit `4d4351a90`.

### Added — Type-ID assignment policy

See [docs/TYPE_ASSIGNMENTS.md](docs/TYPE_ASSIGNMENTS.md). This fork
extensions live at slots 60–95; ik_llama compat zone reserved at 96–199;
mainline growth reserve 42–59.

### Added — Backend parity policy

See [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). Two-track landing:
ROCm-lands first, Vulkan-lands as follow-up, both required for "released".
