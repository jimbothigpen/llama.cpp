# PFlash Prompt Compression

> **§-FLAG (PFL-1): Non-Qwen scorer paths COMPILE but are UNVALIDATED.**
> Qwen3.x is the only validated scorer architecture. Gemma3, Llama, Qwen2,
> and Mistral code paths were generalized in `500046b0b` but have not had a
> live end-to-end PPL or acceptance-rate gate run. Do not deploy non-Qwen
> PFlash scorers in production; treat them as experimental pending validation
> sub-2 validation.

---

## At a glance

| | Value |
|---|---|
| **What it is** | Prompt compression via a small scorer model; reduces the token budget fed to the target model's prefill |
| **Upstream** | buun `master` (SD-089-pflash) |
| **Flag** | `--pflash-scorer <scorer-model-dir>` (minimum required flag) |
| **Validated arch** | Qwen3.x scorer (Qwen3.5-0.8B recommended) |
| **Key tunable** | `--pflash-keep-ratio 0.05` (default: keep 5% of tokens above `--pflash-min-tokens`) |
| **Backend** | CPU (scorer); CUDA/HIP GPU via `pflash-graph.cpp` (`abe0bb81a`) — Vulkan builds fall back to CPU |
| **CLI commits** | `92c37266f` (CLI wire), `076f8c069` (server path) |
| **Scorer generalization** | `500046b0b` (non-Qwen arch support — §-FLAG unvalidated) |

---

## How PFlash works

PFlash runs a small scorer model over the prompt tokens before the main decode.
The scorer assigns an importance weight to each token; a top-K selection retains
the most important tokens up to `keep_ratio × prompt_length` (floored at
`--pflash-min-tokens`). The compressed token list replaces the original prompt
for the target model's prefill. The scorer is a standard causal LM loaded via
the fork's loader; only its logit scores (not sampled tokens) are used.

The scorer is unrelated to speculative decode — PFlash runs during prefill,
before any draft model is involved.

---

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--pflash-scorer PATH` | — | Path to the scorer model (GGUF or directory). **Required to enable PFlash.** |
| `--pflash-keep-ratio F` | `0.05` | Fraction of tokens to retain above the `--pflash-min-tokens` floor |
| `--pflash-alpha F` | `0.12` | Scorer weighting exponent; controls sharpness of importance distribution |
| `--pflash-min-tokens N` | `8192` | PFlash only activates if the prompt exceeds this length (no-op on short prompts) |
| `--pflash-scorer-cache-size N` | `64` | LRU cache entries for scorer state (0 = disabled) |
| `--pflash-cache-dir PATH` | — | On-disk scorer cache directory; cache is auto-invalidated when scorer path or mtime changes |

**Minimal invocation (CLI):**

```bash
llama-speculative-simple \
    -m target.gguf \
    --pflash-scorer scorer-qwen3.5-0.8b-q8_0.gguf \
    --pflash-keep-ratio 0.05 \
    --pflash-min-tokens 8192 \
    -fa on -ngl 999 --no-mmap \
    -p "$(cat long-prompt.txt)"
```

**Minimal invocation (server):**

```bash
llama-server \
    -m target.gguf \
    --pflash-scorer scorer-qwen3.5-0.8b-q8_0.gguf \
    --pflash-keep-ratio 0.05 \
    -fa on -ngl 999 --no-mmap
```

PFlash is active only when `--pflash-scorer` is set and the prompt exceeds
`--pflash-min-tokens`. Below that threshold the path is a no-op; no scorer
inference is done.

---

## Phase status

| Phase | Commits | What it added | Status |
|---|---|---|---|
| **Phase 1 — base port** | buun `master` | Scorer loader (`pflash-loader.cpp`), graph builder (`pflash-graph.cpp`), CPU scorer path | ✅ Ported |
| **Phase 2A — CPU baseline** | — | Scorer model weight storage on CPU; 9.89s → ~0.41s per scorer pass | ✅ Shipped |
| **Phase 3 — HIP GPU scorer** | `abe0bb81a` | GPU scorer compute via `ggml_backend_dev_by_type(GPU)` in `pflash-loader.cpp` + `pflash-graph.cpp`; CPU fallback retained for Vulkan-only builds | ✅ Shipped 2026-05-19 |
| **CLI wire** | `92c37266f` | `--pflash-*` flags wired in `tools/cli/cli.cpp`; compression logged as `pflash: N -> M tokens (X% kept)` | ✅ Shipped 2026-05-31 |
| **Server path** | `076f8c069` | PFlash wired into `llama-server` prefill path | ✅ Shipped |
| **Scorer generalization** | `500046b0b` | Non-Qwen arch support (`llama_model_arch` switch covering qwen3/qwen35/qwen2/llama/mistral3/mistral4/gemma3/gemma4); NULL-deref guards | ✅ Shipped 2026-05-31 — **§-FLAG: unvalidated (see top of doc)** |
| **Phase 4b / 4c** | — | Follow-up scorer-path refinements | 🔄 Deferred |

---

## §-FLAG detail (PFL-1)

The scorer generalization in `500046b0b` adds dispatch paths for
non-Qwen architectures by extending the `llama_model_arch` switch in
`tools/pflash/pflash-scorer.cpp`. The Qwen3.x regression test is byte-identical
after the change. However, **no live scorer gate** (PPL measurement or
compression quality check) has been run for Gemma3, Llama, Qwen2, or Mistral
scorer models.

Known risks:
- G4/G5 architecture field lookups have NULL-deref guards added but are not
  live-tested.
- Token-importance calibration constants are tuned on Qwen3.5; non-Qwen models
  may require different `--pflash-alpha` values.

Tracking: PFlash non-Qwen live-scorer validation.

---

## Backend notes

- **CUDA/HIP:** Full GPU scorer path (`abe0bb81a`). Recommended for performance.
- **Vulkan-only builds:** CPU scorer fallback. Scorer throughput is lower;
  `--pflash-scorer-cache-size` helps amortize cost across requests.
- **ROCm:** Tested on gfx1150 (Strix Point); no known issues.

---

## Further reading

- **Upstream source:** buun `master` (`spiritbuun/buun-llama-cpp`)
- **Scorer sources:** `common/pflash-loader.cpp`, `common/pflash-graph.cpp`,
  `common/pflash-score.cpp`, `tools/pflash/pflash-scorer.cpp`
- **Feature index:** [docs/features/README.md](README.md)
- **Related docs (this repo):**
  - [DFlash speculative decode](dflash.md) — buun-sourced spec-decode companion
  - [InnerQ KV cache](innerq-kv.md) — complementary KV compression
