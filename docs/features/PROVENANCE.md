# Provenance map — canonical

This is the single source of truth for **where every ported feature comes from**. It exists so we
can answer one recurring question reliably: *"are there new upstream commits in feature X's source
that we should adopt?"* Each row records the tracked remote and the best-effort synced ref, which is
what makes that drift-check possible.

Three upstreams can differ for one feature, so they get their own columns:

- **Code** — the runtime / kernel source that was ported.
- **Converter** — the GGUF conversion / tooling source (often different from the code).
- **Weights** — where the model weights come from (for drafter/draft-model features).

**Last full re-verification: 2026-06-28.** Verification = `git ls-tree` / `git grep` / `git log`
against the locally-fetched remote refs, cross-checked with the in-tree code, the feature doc's
§Provenance, and the in-source attribution comment. `Status = VERIFIED` means the source was
confirmed present on the named remote; `UNVERIFIED` means it could not be confirmed and needs a
human decision (none currently).

> **Maintenance:** treat this file as standing hygiene. Each sync cycle, re-confirm the synced refs
> and bump `Last-verified`. When a feature is added, removed, or re-sourced, update its row here in
> the same change.

## Remotes referenced

| Remote | URL |
|---|---|
| mainline / upstream | https://github.com/ggml-org/llama.cpp |
| am17an | https://github.com/am17an/llama.cpp |
| thetom | https://github.com/TheTom/llama-cpp-turboquant |
| buun | https://github.com/spiritbuun/buun-llama-cpp |
| carlosfundora | https://github.com/carlosfundora/llama.cpp-1-bit-turbo |
| domvox | https://github.com/domvox/llama.cpp-turboquant-hip |
| ikllama | https://github.com/ikawrakow/ik_llama.cpp (not a git-merge source — see [docs/IK_LLAMA_PORTS.md](../IK_LLAMA_PORTS.md)) |
| turbotan | https://github.com/turbo-tan/llama.cpp-tq3 |
| zyphra | https://github.com/Zyphra/llama.cpp |
| beellama | https://github.com/Anbeeld/beellama.cpp (converter only) |
| lnigam | https://github.com/lnigam/nvidia-diffusion-gemma (mainline DRAFT PR #24427 source) |

## Speculative decode

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| MTP (runtime spec driver) | mainline ggml-org (PR #22673; gemma4-assistant PR #23398/#22738) | n/a (bundled in target GGUF) | n/a (MTP head bundled in target) | mainline | master @ `0821c5fcf` | `common/speculative.cpp`, `src/models/gemma4-assistant.cpp`, `include/llama.h` | [mtp.md](mtp.md) | 2026-06-22 | VERIFIED |
| Qwen3.5/3.6 MTP converter | mainline ggml-org (PR #22673, am17an) | mainline (`_Qwen35MtpMixin`, byte-identical to master `0821c5fcf`) | n/a | am17an / mainline | master @ `0821c5fcf` | `conversion/qwen.py`; loader `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp` (this fork) | [qwen35-mtp-converter.md](qwen35-mtp-converter.md) | 2026-06-22 | VERIFIED |
| EAGLE3 | carlosfundora/llama.cpp-1-bit-turbo (original arch); native-vs-fork reconciled 2026-06-13 — mainline ggml-org PR #18039 (`88a39274e`) adopted wholesale via Path-A merge `9d0602368`, fork's parallel arch retired | mainline (`conversion/llama.py`); fork's `conversion/eagle3.py` is dead code post-merge (see `d487cece8`) | SpecForge drafter checkpoints (external) | carlosfundora + mainline ggml-org | Path-A merge @ `9d0602368` (2026-06-13) | `src/models/eagle3.cpp` (fc_norm + GGUF-key-backcompat additive patches, this fork) | [eagle3.md](eagle3.md) | 2026-06-30 | VERIFIED |
| DFlash | **buun** (runtime: spec loop, cross-attn ring, dispatch); native-vs-buun reconciled 2026-07-01 KEEP-DIVERGED — mainline's own native DFlash (`LLM_ARCH_DFLASH`, KV-injection mechanism, PR #22105/#25110) is a distinct technique, zero identifier collision, not an ancestor of this fork's code (ik_llama.cpp carries the same buun-family arch, corroborating no mainline collision) | **Anbeeld/beellama.cpp** (`conversion/dflash_draft.py`, MIT) | **z-lab** (DFlash drafter family) | buun (code) + beellama (converter) | buun master @ `87c351d28` | `src/models/dflash_draft.cpp`, `conversion/dflash_draft.py` | [dflash.md](dflash.md) | 2026-07-01 | VERIFIED |
| PHANTOM-X | carlosfundora/llama.cpp-1-bit-turbo (algorithm ref: carlosfundora/**sglang**-1-bit-turbo `phantom_worker.py`) | n/a (self-speculative; no draft model) | n/a | carlosfundora | integration @ `acb6be6b3` | `common/phantom.h` (`--spec-type` factory wiring = this fork) | [phantom-x.md](phantom-x.md) | 2026-06-22 | VERIFIED |
| NLD diffusion self-spec | **buun** (self-spec layer); base Dream/LLaDA/RND1 arch = mainline (PRs #14644/#14771/#16003/#17433) | **buun** (`conversion/nemotron_labs_diffusion.py`) | NVIDIA Nemotron-Labs-Diffusion / DreamLM (external) | buun | buun master @ `87c351d28` | `examples/diffusion/diffusion.cpp`, `conversion/nemotron_labs_diffusion.py` | [nld-diffusion-self-spec.md](nld-diffusion-self-spec.md) | 2026-06-22 | VERIFIED |
| Speculative-decode cascade ensemble | **this fork** (cascade dispatch); arms = mainline `ngram-simple` + the MTP feature above | n/a | n/a | this fork | n/a | `common/speculative.cpp` (cascade dispatch) | [spec-decode-ensemble.md](spec-decode-ensemble.md) | 2026-06-22 | VERIFIED |

## KV-cache quantization & compression

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| TurboQuant KV base (`turboq2/3/4`, slots 60–62) | **thetom** `feature/turboquant-kv-cache` (arXiv 2504.19874 + PolarQuant 2502.02617) | n/a (runtime KV type, never serialized) | n/a | thetom | feature/turboquant-kv-cache @ `4595fff0b` | `ggml/src/ggml-cuda/turbo-quant.cuh`, `ggml/include/ggml.h:433-435` | [turboquant-kv-base.md](turboquant-kv-base.md) | 2026-06-22 | VERIFIED |
| TurboQuant 8-bit KV (`turboq8`, slot 63) | **buun** (`TURBO8_0`; FWHT uniform 256-level grid) | n/a | n/a | buun | buun master @ `87c351d28` | `ggml/include/ggml.h:436` | (covered in [turboquant-kv-base.md](turboquant-kv-base.md)) | 2026-06-22 | VERIFIED |
| TCQ KV (`turboq2/3_tcq`, slots 66–67) | **buun** (`TURBO2/3_TCQ`) | n/a | n/a | buun | buun master @ `87c351d28` | `ggml/src/ggml-cuda/turbo-quant.cuh`, `ggml/include/ggml.h:438-439` | [tcq-kv.md](tcq-kv.md) | 2026-06-22 | VERIFIED |
| InnerQ calibrated KV (`turboq2/3_innerq`, slots 68–69) | **thetom** `feature/turboquant-kv-cache` (calibration engine) | n/a (online calibration; nothing GGUF-shipped) | n/a | thetom | feature/turboquant-kv-cache @ `4595fff0b` | ~~`ggml/src/ggml-cuda/turbo-innerq.cu`~~ (RETIRED — purged `7082ea4ed` 2026-06-20; slots 68–69 returned to reserve) | [innerq-kv.md](innerq-kv.md) | 2026-06-28 | **RETIRED** — subsystem purged; InnerQ disproven (< 0.25% PPL within σ) |
| OScaR INT2 K-cache (`kv_oscar_int2`, slot 71) | **this fork** (original impl; algorithm ref arXiv:2605.19660) — no upstream fork has it | n/a | n/a | this fork | n/a | `ggml/src/ggml-cuda/*` (oscar paths), `ggml/include/ggml.h:445` | [oscar-kv.md](oscar-kv.md) | 2026-06-22 | VERIFIED |
| Asymmetric / alpha-scaled KV | **thetom** `feature/alpha-scaling` + `experiment/asymmetric-kv` | n/a | n/a | thetom | feature/alpha-scaling | `ggml/src/ggml-cuda/turbo-quant.cuh` (alpha defaults) | [concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md) | 2026-06-22 | VERIFIED |
| Per-layer-class KV type (SWA) | mainline ISWA machinery + **domvox** per-layer type knob | n/a | n/a | mainline + domvox | domvox feature/triattention-scoring @ `f9a308d0a` | `include/llama.h:394-395`, `common/arg.cpp` | [swa-per-layer-kv.md](swa-per-layer-kv.md) | 2026-06-22 | VERIFIED |
| TriAttention KV eviction | **domvox** `feature/triattention-scoring` | n/a (per-model `.tria` generated by `llama-tria-gen`, this fork) | n/a | domvox | feature/triattention-scoring @ `f9a308d0a` | `src/triattention*.c`, `src/triattention-hip.hip` | [triattention.md](triattention.md) | 2026-06-22 | VERIFIED |
| EpiCache prefill bounding (`#ifdef LLAMA_EPICACHE`) | **this fork** (within TriAttention runtime; algorithm ref arXiv 2509.17396) | n/a | n/a | this fork | n/a | `src/triattention-runtime.h/.c` | (covered in [triattention.md](triattention.md)) | 2026-06-22 | VERIFIED |
| PFlash prompt compression | **buun** `experiment/SD-089-pflash` | n/a (scorer is a standard small model dir) | scorer model dir (external, e.g. Qwen3.5-0.8B) | buun | SD-089-pflash @ `2aeee7d3f` | `common/pflash*.cpp/.h` | [pflash.md](pflash.md) | 2026-06-22 | VERIFIED |
| TurboKV prefill optimization (turboq2/turboq4, D=128+256) | **this fork** (tiled cooperative flash-attn decode for turboq2/4; `cols_per_block` dispatch; D=128 landed `0f577724a`, D=256 extension landed `35c06110e` — engages on hd=256 production models, e.g. Qwen3.5/3.6) | n/a | n/a | this fork | n/a | `ggml/src/ggml-cuda/fattn-common.cuh`, `ggml/src/ggml-cuda/fattn-vec.cuh` | — | 2026-06-28 | VERIFIED |

## Weight quantization

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| IK base-K (IQ2_K/IQ3_K/IQ4_K) | **ikllama** | mainline `llama-quantize` + imatrix | n/a | ikllama | (subsystem port, not a git merge — see IK_LLAMA_PORTS.md) | `ggml/src/ggml-quants.c`, `ggml/src/ggml-cuda/*`, renumbered slots 137–139 | [ik-base-k.md](ik-base-k.md) | 2026-06-22 | VERIFIED |
| IK high-bit-K (IQ5_K/IQ6_K) | **ikllama** | mainline `llama-quantize` + imatrix | n/a | ikllama | subsystem port | renumbered slots 140–141 | [ik-high-bit-k.md](ik-high-bit-k.md) | 2026-06-22 | VERIFIED |
| IK KS row-meta (IQ4_KS/IQ3_KS/IQ4_KSS/IQ2_KL/IQ2_KS/IQ5_KS) | **ikllama** | mainline `llama-quantize` + imatrix (mandatory) | n/a | ikllama | subsystem port (IQ2_KS from `67817fb5b`, IQ5_KS from `90e53a0b8`) | renumbered slots within 96–199 (IQ2_KL=157, IQ2_KS=145, IQ5_KS=152) | [ik-ks-row-meta.md](ik-ks-row-meta.md) | 2026-06-28 | VERIFIED |
| IK KT trellis (IQ4_KT/IQ3_KT/IQ2_KT/IQ1_KT) | **ikllama** (`andrew_trellis` branch) | mainline `llama-quantize` + imatrix (mandatory) | n/a | ikllama | andrew_trellis | slots 153–155, 158 | [ik-kt-trellis.md](ik-kt-trellis.md) | 2026-06-22 | VERIFIED |
| WHT weight quants (WHT3_0/WHT4_0, slots 80–81) | **thetom** `feature/turboquant-kv-cache` (upstream names `TQ3_1S`/`TQ4_1S`) | mainline `llama-quantize` (unweighted — imatrix path removed, ADR-016) | n/a | thetom | feature/turboquant-kv-cache @ `4595fff0b` | `ggml/src/ggml-cuda/turbo-wht.cu`, `ggml/include/ggml.h:448-449` | [wht-weight-quants.md](wht-weight-quants.md) | 2026-06-22 | VERIFIED |
| WHT5/6/8 weight quants (WHT5_0/WHT6_0/WHT8_0, slots 82–84) | **this fork** (extends **thetom** WHT3/4 Lloyd-Max centroid design to 5/6/8-bit; same FWHT-rotation + centroid decode; fused mmvq decode path for single-token TPS) | mainline `llama-quantize` | n/a | thetom (design base) | feature/turboquant-kv-cache @ `4595fff0b` (WHT3/4 base) | `ggml/src/ggml-cuda/turbo-wht.cu`, `ggml/include/ggml.h:450-452` (branch `feature/wht568-perf-mmvq-2026-06-22`) | [wht-weight-quants.md](wht-weight-quants.md) | 2026-06-23 | VERIFIED |
| WQ3 TCQ weight quant (slot 92) | **buun** `feat/tcq-wq3-ffn-fusion` (TCQ 3-bit weight + FWHT rotation; k=3, L=10, 1024-state Viterbi; FFN fusion) | mainline `llama-quantize` + imatrix | n/a | buun | buun master @ `87c351d28` | `ggml/src/ggml-cuda/wq3-tcq.cu`, `ggml/include/ggml.h:452` (branch chain `feature/wq3-tcq-ph4-vulkan-2026-06-23`; CUDA+HIP+Vulkan backends) | [wq3-tcq.md](wq3-tcq.md) | 2026-06-23 | VERIFIED — **NOT COMPETITIVE**: measured PPL 7.51 at 9B (worst-in-class vs IQ3_KS 6.74 / WHT3_0 7.33 / Q3_K_M 6.85 / Q4_K_M 6.44); **implemented-not-recommended** |

## Model architectures

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| ZAYA1-8B | **this fork** (first ggml/llama.cpp backend); references Zyphra/vllm@`zaya1-pr`, Zyphra/transformers@`zaya1`, Zyphra/llama.cpp@`CCA` (`src/models/zaya.cpp`) | this fork (`ZayaModel` converter) | Zyphra/ZAYA1-8B (HF) | zyphra | CCA (reference only) | `src/models/zaya.cpp` | [zaya1.md](zaya1.md) | 2026-06-22 | VERIFIED |
| DiffusionGemma (block-diffusion) | **lnigam/nvidia-diffusion-gemma** (mainline DRAFT PR #24427, baseline head `201052a16a`); minimal generic port — dense self-conditioning, no CUDA fast-sampling/sparse top-k/fused embed/device denoise loop | this fork (DiffusionGemmaForBlockDiffusion in `conversion/gemma.py`) | unsloth/diffusiongemma-26B-A4B-it (external) | lnigam (code) | mainline DRAFT PR #24427 @ `201052a16a` | `src/models/diffusion-gemma.cpp`, `examples/diffusion-gemma/diffusion-gemma-cli.cpp`, `conversion/gemma.py` | — | 2026-06-28 | VERIFIED |

## Evaluation tooling

Not a ported model feature — original tooling in this fork. External dataset provenance documented here per maintenance policy.

| Tool | Source | External datasets | In-repo paths | Commit | Last-verified | Status |
|---|---|---|---|---|---|---|
| conv-QA eval harness | **this fork** (original — not a ported model feature); benchmarks EpiCache KV-cache quality via multi-session QA tasks (EpiCache ref: arXiv 2509.17396) | **LoCoMo** (`snap-research/locomo`, HuggingFace — paper-sourced multi-session conversations with multi-hop/temporal/adversarial QA pairs; fetched via `fetch_data.sh`, Apache 2.0); **LongMemEval** adapter included in `prepare_subset.py --format longmemeval` (paper-sourced multi-session QA benchmark; external fetch, not bundled) | `tools/epicache-convqa/` | `5de8b2705` (2026-06-21) | 2026-06-22 | VERIFIED |
| turboq2 test-quantize-fns buffer fix | **this fork** (test-harness fix — not a model feature); corrects KV-cache type heap overflow in `dot_product_error()` via `ggml_row_size(vec_dot_type, …)` sizing + F32 guard skipping KV types (discriminator: `vec_dot_type == GGML_TYPE_F32`) from the `vec_dot` path | n/a | `tests/test-quantize-fns.cpp` (branch `feature/turboq2-quantize-buffer-fix-2026-06-23`) | `a31ce489b` (2026-06-23) | 2026-06-23 | VERIFIED |

## GPU compute optimizations

Platform-specific kernel tuning ported from external contributors. No converter or weights component.

| Feature | Upstream source | Tracked ref | In-repo paths | In-tree since | Last-verified | Status |
|---|---|---|---|---|---|---|
| RDNA3.5 MMQ + FATTN tile tuning (gfx1150/gfx1151/gfx1152/gfx1153) | **justinappler** (commits `5dc18d7f4`+`3511e7d1c`; net: MMQ y=64, nwarps=4, dense-aware x_max=48 for MoE layers; FATTN tile config D=256 override) | upstream justinappler `3511e7d1c` (net of both) | `ggml/src/ggml-cuda/mmq.cuh`, `ggml/src/ggml-cuda/fattn-tile.cuh` | commits `006252e63`+`3c3838ff2` (2026-05-31) | 2026-06-23 | VERIFIED — §-FLAG-PENDING-BENCH: FATTN tile override (`3c3838ff2`) unbenched on gfx1150; MMQ tuning (`006252e63`) confirmed present |
| Asymmetric KV-quant FA matrix (K-bpw ≥ V-bpw) | **this fork** (original impl; wired all 57 missing flash-attn vec K×V pairs where bpw(K) ≥ bpw(V) across 17 KV-capable types — 174 total K≥V pairs; fixes `fattn.cu` vec-dispatch `GGML_ABORT` for unwired combos, e.g. turboq6×turboq3 SIGABRT rc=134 — NOT OOM) | n/a | `ggml/src/ggml-cuda/fattn.cu`, `ggml/src/ggml-cuda/fattn-vec-f16.cu` (FA_ALL_QUANTS dispatch) | `393307e58` (2026-06-26) | 2026-06-26 | CUDA/HIP only; Vulkan FA path separate/unaffected |

## Tracked but NOT currently in-tree (drift watch only)

These appear in the contributing-forks table / roadmap but are **not** live ported features on `main`.
Listed so the drift-check does not mistake their absence for an oversight.

| Item | Source | State |
|---|---|---|
| RotorQuant KV (`RQ_*` / iso / planar, slots 72–75) | carlosfundora | **Removed** — zero-rotation scalar duplicate, strictly dominated; slots returned to reserve (`ggml/include/ggml.h:446`) |
| RaBitQ TQ3 (`RBQ3_*`) | turbotan | **Not ported** — roadmap layer 6, pending imatrix retrofit |
| WHT3_4S (slot 82) | ft2 / turbotan (TQ3_4S) | **Not ported** — 4-scale variant evaluated NO-GO; slot 82 re-assigned to WHT5_0 on `feature/wht568-perf-mmvq-2026-06-22` |
| WHT_MIX mixed-precision imatrix steering (TODO 252) | this fork (row/tensor-selective mixed WHT4_0/WHT5_0 precision, imatrix-driven) | **CLOSED-NEGATIVE** — PPL 7.0914±0.08652 @ ~4.38 BPW vs plain WHT4_0's 6.5525±0.07919 @ ~4.00 BPW and Q4_K_M's 6.4550±0.07777 @ ~4.51 BPW (20 chunks); mixed-precision selection strictly worse on both PPL and BPW axes; archived `archive/wht-mixed-precision-252-NEGATIVE-2026-07-01`, not merged |
| WHT rotated-block-covariance imatrix steering (TODO 253) | this fork (Hadamard-rotated per-block covariance used to recover imatrix signal for WHT3_0/WHT4_0) | **CLOSED-NEGATIVE** — WHT3_0 rotated-cov 7.6377±0.0606 vs no-imatrix baseline 7.6587±0.0607 (−0.27%, within 1σ, not significant; gate required <7.2728); WHT4_0 rotated-cov 6.9862±0.0545 vs baseline 6.9605±0.0541 (+0.37%, regression); commit `5730dea33` on `feature/wht-rotated-cov-importance-253-2026-06-30`, archived `archive/wht-rotated-cov-importance-253-NEGATIVE-2026-06-30`, not merged |
| TurboQuant 5/6-bit KV (turboq5/turboq6, slots 64–65) | this fork (FWHT uniform-grid, TODO 250) | **IN-TREE but DORMANT** — measured NO-WIN (PPL ties mainline KV types within σ; PP throughput −31%/−26%); merged from `feature/turboq56-kv-implement-2026-06-22`; feature doc [turboquant-hibit-kv.md](turboquant-hibit-kv.md); FA matrix wired via `393307e58` |
| Delta-KV | — | No in-tree type |
| modelai graph-exec KV consumption | — | Experimental branch only; not on `main` |
