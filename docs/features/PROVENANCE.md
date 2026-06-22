# Provenance map — canonical

This is the single source of truth for **where every ported feature comes from**. It exists so we
can answer one recurring question reliably: *"are there new upstream commits in feature X's source
that we should adopt?"* Each row records the tracked remote and the best-effort synced ref, which is
what makes that drift-check possible.

Three upstreams can differ for one feature, so they get their own columns:

- **Code** — the runtime / kernel source that was ported.
- **Converter** — the GGUF conversion / tooling source (often different from the code).
- **Weights** — where the model weights come from (for drafter/draft-model features).

**Last full re-verification: 2026-06-22.** Verification = `git ls-tree` / `git grep` / `git log`
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

## Speculative decode

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| MTP (runtime spec driver) | mainline ggml-org (PR #22673; gemma4-assistant PR #23398/#22738) | n/a (bundled in target GGUF) | n/a (MTP head bundled in target) | mainline | master @ `0821c5fcf` | `common/speculative.cpp`, `src/models/gemma4-assistant.cpp`, `include/llama.h` | [mtp.md](mtp.md) | 2026-06-22 | VERIFIED |
| Qwen3.5/3.6 MTP converter | mainline ggml-org (PR #22673, am17an) | mainline (`_Qwen35MtpMixin`, byte-identical to master `0821c5fcf`) | n/a | am17an / mainline | master @ `0821c5fcf` | `conversion/qwen.py`; loader `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp` (this fork) | [qwen35-mtp-converter.md](qwen35-mtp-converter.md) | 2026-06-22 | VERIFIED |
| EAGLE3 | carlosfundora/llama.cpp-1-bit-turbo | this fork (`conversion/eagle3.py`, SpecForge format) | SpecForge drafter checkpoints (external) | carlosfundora | integration @ `acb6be6b3` | `src/models/eagle3.cpp`, `conversion/eagle3.py` | [eagle3.md](eagle3.md) | 2026-06-22 | VERIFIED |
| DFlash | **buun** (runtime: spec loop, cross-attn ring, dispatch) | **Anbeeld/beellama.cpp** (`conversion/dflash_draft.py`, MIT) | **z-lab** (DFlash drafter family) | buun (code) + beellama (converter) | buun master @ `87c351d28` | `src/models/dflash_draft.cpp`, `conversion/dflash_draft.py` | [dflash.md](dflash.md) | 2026-06-22 | VERIFIED |
| PHANTOM-X | carlosfundora/llama.cpp-1-bit-turbo (algorithm ref: carlosfundora/**sglang**-1-bit-turbo `phantom_worker.py`) | n/a (self-speculative; no draft model) | n/a | carlosfundora | integration @ `acb6be6b3` | `common/phantom.h` (`--spec-type` factory wiring = this fork) | [phantom-x.md](phantom-x.md) | 2026-06-22 | VERIFIED |
| NLD diffusion self-spec | **buun** (self-spec layer); base Dream/LLaDA/RND1 arch = mainline (PRs #14644/#14771/#16003/#17433) | **buun** (`conversion/nemotron_labs_diffusion.py`) | NVIDIA Nemotron-Labs-Diffusion / DreamLM (external) | buun | buun master @ `87c351d28` | `examples/diffusion/diffusion.cpp`, `conversion/nemotron_labs_diffusion.py` | [nld-diffusion-self-spec.md](nld-diffusion-self-spec.md) | 2026-06-22 | VERIFIED |
| Speculative-decode cascade ensemble | **this fork** (cascade dispatch); arms = mainline `ngram-simple` + the MTP feature above | n/a | n/a | this fork | n/a | `common/speculative.cpp` (cascade dispatch) | [spec-decode-ensemble.md](spec-decode-ensemble.md) | 2026-06-22 | VERIFIED |

## KV-cache quantization & compression

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| TurboQuant KV base (`turboq2/3/4`, slots 60–62) | **thetom** `feature/turboquant-kv-cache` (arXiv 2504.19874 + PolarQuant 2502.02617) | n/a (runtime KV type, never serialized) | n/a | thetom | feature/turboquant-kv-cache @ `4595fff0b` | `ggml/src/ggml-cuda/turbo-quant.cuh`, `ggml/include/ggml.h:433-435` | [turboquant-kv-base.md](turboquant-kv-base.md) | 2026-06-22 | VERIFIED |
| TurboQuant 8-bit KV (`turboq8`, slot 63) | **buun** (`TURBO8_0`; FWHT uniform 256-level grid) | n/a | n/a | buun | buun master @ `87c351d28` | `ggml/include/ggml.h:436` | (covered in [turboquant-kv-base.md](turboquant-kv-base.md)) | 2026-06-22 | VERIFIED |
| TCQ KV (`turboq2/3_tcq`, slots 66–67) | **buun** (`TURBO2/3_TCQ`) | n/a | n/a | buun | buun master @ `87c351d28` | `ggml/src/ggml-cuda/turbo-quant.cuh`, `ggml/include/ggml.h:438-439` | [tcq-kv.md](tcq-kv.md) | 2026-06-22 | VERIFIED |
| InnerQ calibrated KV (`turboq2/3_innerq`, slots 68–69) | **thetom** `feature/turboquant-kv-cache` (calibration engine) | n/a (online calibration; nothing GGUF-shipped) | n/a | thetom | feature/turboquant-kv-cache @ `4595fff0b` | `ggml/src/ggml-cuda/turbo-innerq.cu`, `ggml/include/ggml.h:439-440` | [innerq-kv.md](innerq-kv.md) | 2026-06-22 | VERIFIED |
| OScaR INT2 K-cache (`kv_oscar_int2`, slot 71) | **this fork** (original impl; algorithm ref arXiv:2605.19660) — no upstream fork has it | n/a | n/a | this fork | n/a | `ggml/src/ggml-cuda/*` (oscar paths), `ggml/include/ggml.h:445` | [oscar-kv.md](oscar-kv.md) | 2026-06-22 | VERIFIED |
| Asymmetric / alpha-scaled KV | **thetom** `feature/alpha-scaling` + `experiment/asymmetric-kv` | n/a | n/a | thetom | feature/alpha-scaling | `ggml/src/ggml-cuda/turbo-quant.cuh` (alpha defaults) | [concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md) | 2026-06-22 | VERIFIED |
| Per-layer-class KV type (SWA) | mainline ISWA machinery + **domvox** per-layer type knob | n/a | n/a | mainline + domvox | domvox feature/triattention-scoring @ `f9a308d0a` | `include/llama.h:394-395`, `common/arg.cpp` | [swa-per-layer-kv.md](swa-per-layer-kv.md) | 2026-06-22 | VERIFIED |
| TriAttention KV eviction | **domvox** `feature/triattention-scoring` | n/a (per-model `.tria` generated by `llama-tria-gen`, this fork) | n/a | domvox | feature/triattention-scoring @ `f9a308d0a` | `src/triattention*.c`, `src/triattention-hip.hip` | [triattention.md](triattention.md) | 2026-06-22 | VERIFIED |
| EpiCache prefill bounding (`#ifdef LLAMA_EPICACHE`) | **this fork** (within TriAttention runtime; algorithm ref arXiv 2509.17396) | n/a | n/a | this fork | n/a | `src/triattention-runtime.h/.c` | (covered in [triattention.md](triattention.md)) | 2026-06-22 | VERIFIED |
| PFlash prompt compression | **buun** `experiment/SD-089-pflash` | n/a (scorer is a standard small model dir) | scorer model dir (external, e.g. Qwen3.5-0.8B) | buun | SD-089-pflash @ `2aeee7d3f` | `common/pflash*.cpp/.h` | [pflash.md](pflash.md) | 2026-06-22 | VERIFIED |

## Weight quantization

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| IK base-K (IQ2_K/IQ3_K/IQ4_K) | **ikllama** | mainline `llama-quantize` + imatrix | n/a | ikllama | (subsystem port, not a git merge — see IK_LLAMA_PORTS.md) | `ggml/src/ggml-quants.c`, `ggml/src/ggml-cuda/*`, renumbered slots 137–139 | [ik-base-k.md](ik-base-k.md) | 2026-06-22 | VERIFIED |
| IK high-bit-K (IQ5_K/IQ6_K) | **ikllama** | mainline `llama-quantize` + imatrix | n/a | ikllama | subsystem port | renumbered slots 140–141 | [ik-high-bit-k.md](ik-high-bit-k.md) | 2026-06-22 | VERIFIED |
| IK KS row-meta (IQ4_KS/IQ3_KS/IQ4_KSS/IQ2_KL) | **ikllama** | mainline `llama-quantize` + imatrix (mandatory) | n/a | ikllama | subsystem port | renumbered slots within 96–199 (IQ2_KL=157) | [ik-ks-row-meta.md](ik-ks-row-meta.md) | 2026-06-22 | VERIFIED |
| IK KT trellis (IQ4_KT/IQ3_KT/IQ2_KT/IQ1_KT) | **ikllama** (`andrew_trellis` branch) | mainline `llama-quantize` + imatrix (mandatory) | n/a | ikllama | andrew_trellis | slots 153–155, 158 | [ik-kt-trellis.md](ik-kt-trellis.md) | 2026-06-22 | VERIFIED |
| WHT weight quants (WHT3_0/WHT4_0, slots 80–81) | **thetom** `feature/turboquant-kv-cache` (upstream names `TQ3_1S`/`TQ4_1S`) | mainline `llama-quantize` (unweighted — imatrix path removed, ADR-016) | n/a | thetom | feature/turboquant-kv-cache @ `4595fff0b` | `ggml/src/ggml-cuda/turbo-wht.cu`, `ggml/include/ggml.h:448-449` | [wht-weight-quants.md](wht-weight-quants.md) | 2026-06-22 | VERIFIED |

## Model architectures

| Feature | Upstream (code) | Upstream (converter) | Upstream (weights) | Tracked remote | Synced ref (best-effort) | In-repo paths | §Provenance doc | Last-verified | Status |
|---|---|---|---|---|---|---|---|---|---|
| ZAYA1-8B | **this fork** (first ggml/llama.cpp backend); references Zyphra/vllm@`zaya1-pr`, Zyphra/transformers@`zaya1`, Zyphra/llama.cpp@`CCA` (`src/models/zaya.cpp`) | this fork (`ZayaModel` converter) | Zyphra/ZAYA1-8B (HF) | zyphra | CCA (reference only) | `src/models/zaya.cpp` | [zaya1.md](zaya1.md) | 2026-06-22 | VERIFIED |

## Tracked but NOT currently in-tree (drift watch only)

These appear in the contributing-forks table / roadmap but are **not** live ported features on `main`.
Listed so the drift-check does not mistake their absence for an oversight.

| Item | Source | State |
|---|---|---|
| RotorQuant KV (`RQ_*` / iso / planar, slots 72–75) | carlosfundora | **Removed** — zero-rotation scalar duplicate, strictly dominated; slots returned to reserve (`ggml/include/ggml.h:446`) |
| RaBitQ TQ3 (`RBQ3_*`) | turbotan | **Not ported** — roadmap layer 6, pending imatrix retrofit |
| WHT3_4S (slot 82) | ft2 / turbotan (TQ3_4S) | **Not ported** — 4-scale variant evaluated NO-GO |
| Delta-KV | — | No in-tree type |
| modelai graph-exec KV consumption | — | Experimental branch only; not on `main` |
