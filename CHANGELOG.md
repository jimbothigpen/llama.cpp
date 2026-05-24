# Changelog

All notable fork-specific changes. Tagged milestones correspond to
phase completions. For mainline llama.cpp changes inherited via upstream
sync, see the [ggml-org/llama.cpp release notes](https://github.com/ggml-org/llama.cpp/releases).

The format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versioning is milestone-driven (one tag per phase completion), not semver.

## [Unreleased]

In-flight: Phase 5b-2 (IQ5_K / IQ6_K recon), EAGLE3 port recon, iso3-K cross-V hang fix (TODO 68).

### Added — MTP top_k=1 deliberate-choice comment (2026-05-24, `fd9bf51f8`)

Documents the intentional `top_k=1` (vs mainline `top_k=10`) in `state_draft_mtp` ctor.
With `ad277572` backend sampling pre-filtering to top-10 and bundled-MTP weight-sharing,
`top_k=1` yields an argmax draft that closely tracks target greedy. Comment guards against
inadvertent revert: "Do NOT raise to 10 without re-running Smoke B."

### Added — NLD server self-spec loop (TODO 86, 2026-05-23, `1cb8c4218`)

Port of `tools/server/server-context.cpp` additions from buun `f339dbebe`
(+589 net LOC, 12 hunks): `is_diffusion` auto-detection via
`llama_model_is_diffusion()`; `diff_self_spec` fields on `server_slot`;
rejection-sampling spec loop with temperature, think-tag suppression,
cross-turn penalty, and loop detection. MTP server paths coexist
cleanly (mutually exclusive: a slot is MTP or diffusion, never both).

Server self-spec smoke: 4.49 t/s (128 tokens); ~59% draft acceptance.
MTP regression gate: 84.62% accept on Qwen3.5-35B-A3B-MTP — above v525 anchor (77.78%).

### Added — NLD Tier-B CLI port (TODO 80, 2026-05-23, `49f88e18a` + `35315922c`)

Selective port of Nemotron-Labs Diffusion from buun `f339dbebe` (~612 LOC net):
GGUF converter (`conversion/nemotron_labs_diffusion.py`), diffusion library
(`examples/diffusion/diffusion.h` + `diffusion.cpp`), CLI refactor
(`examples/diffusion/diffusion-cli.cpp`), model loader fixes for DREAM arch
(`src/models/dream.cpp`, `src/llama-model.cpp`), and 3 new CLI flags
(`--diffusion-block-length`, `--diffusion-self-spec`, `--diffusion-draft-length`).

Smokes: block-mode 1.9 t/s; self-spec 7.0 t/s (3.7× speedup, 68.4% draft acceptance).
MTP regression gate: 69.0% accept on Qwen3.5-35B-A3B-MTP after port (anchor 70.3%, Δ −1.2pp, within ±5pp).

### Fixed — MTP V-J accept-rate gap (TODO 81, 2026-05-23, `705ffccb8`)

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

Port of IQ4_KS / IQ4_KSS / IQ3_KS / IQ4_KT from frankenturbo2. Requires P0 prereq
commit `d91059253` (add `row_meta_size` to `ggml_type_traits` + extend `ggml_row_size()`).

Ship sequence:
- `d91059253` — P0: `row_meta_size` infra (prerequisite for all row-meta types).
- `026671689` — Phase 5b-1b: CPU traits + CUDA/HIP kernels + Vulkan dequant/matvec shaders for all 4 types.
- `a0fe65a77`, `e4caef152`, `d703bf5ea`, `6d9957ae1`, `3f629f8fb` — 5 post-ship bugfixes (CUDA dequant kernels, tensor stride/nbytes, row validation, HIP __shfl_xor_sync, Vulkan batched guard).
- `5fe804bcd` — Vulkan KS batched `mul_mat` SEGV fix: extend `is_empty()` guard in `ggml_vk_get_mul_mat_mat_pipeline` to the non-q8_1 branch; KS types now dequant-to-f16 on the batch path. Also fixes the identical latent SEGV in base-K types.

PPL gate (Vulkan, Qwen3.5-9B, 20 chunks): IQ4_KS 6.4131 / IQ3_KS 6.7325 / IQ4_KSS 6.5773 / IQ4_KT 6.5364 vs ROCm anchors (Δ ≤ 0.043).

### Added — Phase 5b-1a base IK weight quants (2026-05-20 to 2026-05-22, `aed6d2965`)

Port of IQ2_K / IQ3_K / IQ4_K from frankenturbo2 (Phase 5b-1a). CPU traits +
CUDA/HIP kernels (convert.cu, mmvq-iqk.cu) + Vulkan dequant/matvec shaders (6
`.comp` shaders). No row_meta; standard `ggml_type_size`-only layout.
Renumbered to ygg canonical IDs: IQ4_K=139, IQ3_K=138, IQ2_K=137
(ik_llama compatibility zone). PPL parity Δ < 0.0045 vs frankenturbo2 reference.

### Added — PFlash base port Phase 3: HIP GPU scorer (2026-05-19, v355, `abe0bb81a`)

HIP-ify scorer compute graph. Replaces CPU backend with GPU backend via
`ggml_backend_dev_by_type(GPU)` in both `pflash-loader.cpp` (weight storage)
and `pflash-graph.cpp` (compute graph). CPU fallback retained for Vulkan-only
builds. Compute context memory bumped 4 MB → 16 MB for GPU tensor overhead.
Enables 24× scorer speedup versus Phase 2A CPU baseline (9.89s → ~0.41s per
prefill window on ai01 ROCm gfx1102+HSA_OVERRIDE). Phase 2A F32 tok_embd
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

Corrected the codename for the gfx1150 host (ai00). The APU is Strix Point
(Ryzen AI 9 HX 370, 12C/24T Zen 5 + Zen 5c), not Strix Halo (which is a
different product line). No code changes.

### Added — README Attribution section crediting sibling forks and original authors (2026-05-18, `466fc667e`)

Post-v327 follow-up. Merged `feature/readme-attribution-additions-2026-05-18-PM` into main.
Documents the sibling fork lineage (buun, frankenturbo2, carlosfundora, TheTom,
ik_llama) and original llama.cpp authors whose work this fork builds upon.
No code changes.

### Fixed — Multi-backend /opt RPATH; ai01 Vulkan binary was loading ROCm libggml (2026-05-18, v327)

Root cause of the v326 ai01 Vulkan SIGSEGV was a broken `RUNPATH` in installed
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

PPL gates (ai01 Vulkan, 80 chunks, c=512, wikitext-2-raw-test):

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
(upstream AMD Tensile/hipBLAS GEMM gaps). ai01 ROCm binaries run with
`HSA_OVERRIDE_GFX_VERSION=11.0.2` (gfx1102-built binary on gfx1103
hardware).

### Changed — Trunk renamed `master` → `main`

The durable trunk branch is `main` (was `port/frankenturbo2/sidecar-engine`
through session 5, then `master` briefly). GitHub default branch updated.

---

## [`milestone/phase-0.7-sidecar-engine`] — 2026-05-12

Phase 0.7 — Sidecar plugin engine. Released at commit `f99ad5df8`.

### Added — Sidecar plugin runtime (~355 LoC, backend-agnostic)

Hook points: residual-stream, MoE-expert, post-compute-logits, weight
deltas. Out-of-tree `.so` plugins via a stable C ABI. Companion plugin
tools (sidecar-*) tracked separately under `/usr/src/llama-forks/`.

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

### Added — `scripts/ppl-harness.py` (dual-backend PPL regression harness)

Pinned wikitext slice, per-(model, type, backend) baselines, cross-backend
delta tolerance bands per [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md).

### Added — Backend parity policy

See [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). Two-track landing:
ROCm-lands first, Vulkan-lands as follow-up, both required for "released".
