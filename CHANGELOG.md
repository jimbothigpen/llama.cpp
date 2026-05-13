# Changelog

All notable yggdrasil-specific changes. Tagged milestones correspond to
phase completions. For mainline llama.cpp changes inherited via upstream
sync, see the [ggml-org/llama.cpp release notes](https://github.com/ggml-org/llama.cpp/releases).

The format loosely follows [Keep a Changelog](https://keepachangelog.com/);
versioning is milestone-driven (one tag per phase completion), not semver.

## [Unreleased]

Phase 2 (MTP spec-decode spine) in design.

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
