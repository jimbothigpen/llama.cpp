# Changelog

All notable yggdrasil-specific changes. Tagged milestones correspond to
phase completions. For mainline llama.cpp changes inherited via upstream
sync, see the [ggml-org/llama.cpp release notes](https://github.com/ggml-org/llama.cpp/releases).

The format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versioning is milestone-driven (one tag per phase completion), not semver.

## [Unreleased]

Phase 2 (MTP spec-decode spine) in design.

### Added — Novel model architecture: Zyphra ZAYA1-8B (2026-05-15, `64a481bb6 → cc8455581` on `main`)

In-tree port of the Zyphra ZAYA1-8B hybrid MoE — first novel-arch model
port in yggdrasil that did not originate in mainline llama.cpp or any of
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
    policy; future yggdrasil models calling `ggml_conv_1d` /
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

See [docs/TYPE_ASSIGNMENTS.md](docs/TYPE_ASSIGNMENTS.md). Yggdrasil
extensions live at slots 60–95; ik_llama compat zone reserved at 96–199;
mainline growth reserve 42–59.

### Added — `scripts/ppl-harness.py` (dual-backend PPL regression harness)

Pinned wikitext slice, per-(model, type, backend) baselines, cross-backend
delta tolerance bands per [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md).

### Added — Backend parity policy

See [docs/BACKEND_PARITY.md](docs/BACKEND_PARITY.md). Two-track landing:
ROCm-lands first, Vulkan-lands as follow-up, both required for "released".
