# TriAttention KV-Cache Eviction (`--triattention`)

> **Status: Experimental — requires a `.tria` calibration file; retains retrieval quality at
> aggressive KV budgets for long-context workloads.**
>
> TriAttention scores cached K/V entries by attention-trajectory importance and evicts
> low-scoring entries to keep KV memory within a user-set budget. It is complementary to
> KV quantization — both can be applied simultaneously (GPU scoring additionally requires
> `--cache-type-k q8_0`).
>
> **Not a zero-config feature.** You must generate a per-model `.tria` calibration file with
> `llama-tria-gen` before enabling scoring. CPU scoring works for any architecture; GPU
> scoring (HIP + Vulkan) is active for models with **uniform** `head_dim ≤ 256` (Qwen3.x,
> Qwen3.5/3.6, Llama-3.1, most GQA models). Gemma-4 (`head_dim` varies per layer: 256/512)
> scores on CPU — GPU scoring requires uniform head_dim (see §6).

---

## At a glance

| | Value |
|---|---|
| **What it is** | Trajectory-adaptive KV-cache eviction: score tokens by calibrated RoPE-direction importance, evict low-value entries beyond a budget |
| **Calibration** | Per-model `.tria` file required; generate with `llama-tria-gen` |
| **Core flags** | `--triattention <.tria>`, `--tri-budget <pct>`, `--tri-window <N>`, `--tri-interval <N>`, `--tri-sink <N>` |
| **Backend: scoring** | CPU (all GQA architectures) + HIP/Vulkan GPU (GQA, hd ≤ 256, uniform head_dim, `--cache-type-k q8_0`) |
| **GPU models** | Qwen3-8B/9B/35B (hd 128) and Qwen3.5/3.6 (uniform hd 256), Llama-3.1, MistralNeMo, most NEOX-RoPE GQA; **not** Gemma-4 (hybrid hd 256/512 — CPU only) |
| **Retrieval quality (Qwen3-8B, passkey)** | **100 %** at budgets 25 / 50 / 75 % vs random 15 / 45 % — +55–85 pp delta |
| **Retrieval quality (Gemma-4, passkey)** | **70 %** @25 % budget (CPU scoring, all layers including SWA) |
| **Eviction fires** | Decode mode only, every `--tri-interval` steps; PPL runs do not fire the evictor |
| **Composes with** | KV quantization (`--cache-type-k`, `--cache-type-v`); speculative decode |
| **Phase A capture** | `6cbc9e06c` (in-graph CPU capture sidesteps the ROCm sub-alloc zero-read bug) |
| **Phase B evictor** | `6f93b4e5d` (score-sorted compaction; prefix-protect + window-protect) |
| **Phase C GPU kernel** | HIP `51a64b43c`+`88f94232c`, Vulkan `0d13ac92b` (GQA-aware, hd ≤ 256 uniform supported) |
| **SWA capture** | `086c8508f` (Gemma-4 / hybrid iswa layers now captured and scored) |
| **Calibration tool** | `d6ecb3245`+`53eb84dd9`+`60ece65ca` (`llama-tria-gen`, tria-gen v1; v4 `.tria` format) |

> **Long-context only.** The evictor fires during decode — on a chat-length (< 4 K token)
> prompt you will not observe KV savings. TriAttention is designed for sessions where KV
> memory is the bottleneck: multi-document QA, long code generation, RAG with long retrieved
> passages.

---

## Provenance

TriAttention is ported from the **domvox** fork
([domvox/llama.cpp-turboquant-hip](https://github.com/domvox/llama.cpp-turboquant-hip),
remote `domvox`), branch **`feature/triattention-scoring`** — verified
2026-06-22 (`src/triattention*.c`, `src/triattention-hip.hip` present at synced
ref `f9a308d0a`). The per-model `.tria` calibration file is generated locally by
`llama-tria-gen`; there is no upstream weight or converter dependency.

This fork's additions on top of domvox's CPU scorer: the GPU GQA scoring kernel
(HIP + Vulkan, `head_dim ≤ 256`), the per-layer head_dim handling (`.tria` v4),
the Gemma-4 ISWA SWA-layer capture, and the CPU-vs-GPU divergence fix.

The **EpiCache** prefill-bounding path (compiled in under `LLAMA_EPICACHE`, in
`src/triattention-runtime.*`) is this fork's own implementation, with algorithm
reference **arXiv:2509.17396**; it is not part of domvox's TriAttention. See the
canonical [PROVENANCE.md](PROVENANCE.md).

## Quick start

```bash
# 1 — generate a calibration file for your model (once per model)
llama-tria-gen \
    -m  Qwen3-8B.gguf \
    -f  calibration-corpus.txt \
    -o  qwen3-8b.tria

# 2 — run inference with KV eviction (50 % budget example)
llama-cli \
    -m  Qwen3-8B.gguf \
    --triattention qwen3-8b.tria \
    --tri-budget   50 \
    --cache-type-k q8_0 \
    -fa on -ngl 999 --no-mmap \
    -p "..." -n 2048
```

For Gemma-4 (CPU scoring — drop `--cache-type-k q8_0` or keep it for other reasons; GPU
scoring requires uniform head_dim and Gemma-4's hybrid hd remains CPU-only):

```bash
llama-tria-gen -m Gemma4-27B.gguf -f corpus.txt -o gemma4-27b.tria

llama-cli \
    -m  Gemma4-27B.gguf \
    --triattention gemma4-27b.tria \
    --tri-budget   50 \
    -fa on -ngl 999 --no-mmap \
    -p "..." -n 2048
```

---

## §1 CLI flags

| Flag | Arg | Default | Description |
|---|---|---|---|
| `--triattention` | `PATH` | disabled | Path to the `.tria` calibration file. **Must be set to enable eviction** — without it TriAttention is entirely inactive regardless of the other flags. |
| `--tri-budget` | `N` (1–100) | `0` (off) | Retain **N %** of `n_ctx` tokens after each eviction pass. E.g. `50` keeps the 50 % of cached tokens with the highest trajectory score plus the protected prefix and window. Set this flag to activate eviction. |
| `--tri-window` | `N` | `512` | Always keep the **most recent N tokens** regardless of score. Protects against evicting recently-seen context that is still causally relevant. |
| `--tri-interval` | `N` | `128` | Run the evictor every **N decode steps**. Lower values reduce peak KV usage but add more scoring overhead; higher values amortize overhead over more steps. |
| `--tri-sink` | `N` | `0` | Always keep the **first N tokens** as attention sinks. Set to 1–4 if your model shows quality degradation at very aggressive budgets — some models route disproportionate attention through the BOS token. |

Both `--triattention` and `--tri-budget > 0` must be set for eviction to fire. `--triattention`
alone loads the calibration stats but does not evict; `--tri-budget > 0` alone has no effect
without a stats file.

**Verified locations:** `common/arg.cpp:2152–2184`; defaults `common/common.h:596–600`.

---

## §2 Calibration file (`.tria`)

TriAttention scores tokens using **per-layer, per-head RoPE-direction statistics** collected
from a calibration corpus. These statistics (mean query directions + per-frequency importance
weights) cannot be derived at inference time — they must be pre-computed once per model and
stored in a `.tria` file.

### Generating a `.tria` file

```bash
llama-tria-gen \
    -m  <model.gguf> \
    -f  <corpus.txt> \
    -o  <model.tria> \
    [-c <context_length>]   # default 512; longer = more stable statistics
```

- `-f` accepts any plain-text corpus (e.g. `wiki.test.raw`, a project's source files, or a
  sample of your actual workload). 512–2 K tokens per calibration sample is typical.
- The tool hooks `Qcur-{layer}` tensors during forward passes to accumulate pre-RoPE Q
  statistics; it does not modify the model or require GPU compute for the stat accumulation.
- Output is a `.tria` v4 binary (one stats block per layer, per-layer `head_dim` table for
  hybrid models like Gemma-4 that mix SWA and full-attention layers).

A `.tria` file is model-architecture-specific and not portable across model families.
Re-generate when switching between Qwen3-8B and Qwen3-35B, or between Llama and Gemma families.

**Source:** `tools/tria-gen/tria-gen.cpp`.

---

## §3 Effectiveness — measured

Effectiveness numbers are from the passkey / needle-in-a-haystack retrieval benchmark
(long-context decode workload, not `llama-perplexity` — see §5 for the PPL caveat).

### Qwen3-8B — GPU scoring (HIP, gfx1150)

| KV budget | Smart eviction | Random eviction | Delta |
|---|---|---|---|
| 75 % | **100 %** | ~45 % | +55 pp |
| 50 % | **100 %** | ~30 % | +70 pp |
| 25 % | **100 %** | ~15 % | +85 pp |

Retaining 25 % of the KV cache preserves **100 % retrieval accuracy**, while random
eviction at the same budget drops to ~15 %. GPU scoring (HIP) and CPU scoring produce
identical keep-sets (GPU == CPU parity gate PASS; prune count 251 136 identical).

### Gemma-4 — CPU scoring (all layers incl. SWA, per-layer head_dim)

| KV budget | Smart eviction |
|---|---|
| 25 % | **70 %** |

CPU-only scoring with SWA-layer capture (`086c8508f`) and per-layer `head_dim` support
(`41151f8db`) lifts Gemma-4 retrieval from a ~30 % ceiling (SWA layers unscored) to 70 %
at 25 % budget. GPU scoring for Gemma-4 (`head_dim` varies: 256/512 across layers) remains CPU-only because
GPU scoring requires uniform head_dim — not an hd > 128 constraint (hd ≤ 256 uniform is
already GPU-eligible). A per-layer hd table for hybrid models is a separate follow-on.

**Frame:** TriAttention retains retrieval quality at aggressive KV budgets. A 50 % budget
on Qwen3-8B is effectively lossless for needle-retrieval tasks while halving KV memory.

---

## §4 Backend and model support

### GPU scoring requirements

| Condition | Explanation |
|---|---|
| `--cache-type-k q8_0` | GPU kernel operates on Q8_0 K slices; without it scoring falls back to CPU |
| `nh % nkv == 0` | GQA ratio must be an integer (holds for all standard GQA models) |
| `head_dim ≤ 256, head_dim % 32 == 0` | GPU kernel constraint; hd=64/96/128/256 all eligible (uniform hd only — hybrid hd models fall back to CPU) |
| Non-rotary dim = 0 | Models with a non-rotary (content) term fall back to CPU — GPU kernel implements the RoPE-direction term only |

Gate log printed once at startup:
```
tria: gpu gate — is_q8_0=1 nh=32 nkv=8 hd=128 nonrot=0 -> eligible
```
**Verified location:** `src/triattention-runtime.c:376–397`.

### Model compatibility table

| Model family | GPU scoring | Notes |
|---|---|---|
| Qwen3-8B / 9B / 35B | ✅ HIP + Vulkan | `hd=128`, GQA 32/8; GPU == CPU parity confirmed |
| Llama-3.1 8B / 70B | ✅ HIP + Vulkan | `hd=128`, GQA |
| Gemma-4 27B / 82B | CPU only | `hd=256/512` (full-attn), `hd=64` (SWA) — hybrid hd across layers forces CPU-only scoring (GPU kernel requires uniform head_dim; see §6) |
| Hybrid SSM + attention (ZAYA1-8B) | ✅ CPU (attn layers only) | SSM layers have no KV; attention layers scored normally |
| MLA (DeepSeek) | Not supported | MLA lacks a standalone `Qcur` tensor; `tria-gen` cannot hook it |

---

## §5 How it works

### Phase A — in-graph capture

During each forward pass, `build_attn()` inserts `ggml_set_rows` copy nodes that write
K and V from the KV cache into dedicated **CPU-side capture buffers**
(`src/llama-graph.cpp:2365–2387`, allocated in `llama_tria_capture_alloc`,
`src/llama-triattention.cpp:28–94`). Capture covers both standard (full-attention) layers
and SWA layers (`kv_swa`) for hybrid models.

This in-graph approach sidesteps a ROCm sub-alloc zero-read bug: `ggml_backend_tensor_get`
reads zeros for tensors past a ~272 MiB sub-allocation boundary on ROCm, making
post-decode readback unreliable. The `ggml_set_rows` copy nodes run during the forward
pass before the sub-alloc boundary is an issue.

### Phase B — score-sorted eviction

Every `--tri-interval` decode steps, `tria_compact_kv` (`src/triattention-bridge.cpp`) calls
`triattention_compact()` (`src/llama-kv-cache.cpp`):

1. **Score** each cached token using the `.tria` calibration stats (CPU or GPU path — §4).
2. **Protect** the first `--tri-sink` tokens (attention sinks) and the last `--tri-window`
   tokens (recent context).
3. **Sort** the remaining tokens by score; **evict** the lowest-scoring entries down to
   `--tri-budget %` of `n_ctx`.
4. **Compact** the KV cache in-place (physical compaction; no padding gaps left).

The evictor fires only in **decode mode** (single-token steps). It does not fire during
prefill or during `llama-perplexity` runs (which are pure prefill). PPL figures therefore
reflect baseline quality with the evictor inactive — the PPL gate is an upper bound, not
evidence of eviction quality.

### Phase C — GPU GQA scoring (HIP + Vulkan)

The GPU scoring kernel (`triattention-hip.hip`, `triattention-vulkan.cpp`) implements
per-query-head z-normalization with aggregation across the GQA query-head group:

- Uploaded once per scoring call: K slice (Q8\_0, H2D), calibration omega + per-head q\_mean
  / q\_abs arrays covering **all** query heads (layout `[n_layers × n_heads × freq_count]`).
- Per KV head: z-normalize each query head's score independently, then reduce to the KV
  head's score by taking the **max across query heads** — matching the CPU reference
  aggregation.
- Vulkan port ships 3 GLSL compute shaders (`tria_raw_score.comp`, `tria_znorm_agg.comp`,
  `tria_znorm_global.comp`); unit-test max\_abs\_err ≤ 1e-5 vs CPU reference.

---

## §6 Limitations

- **Not zero-config.** A `.tria` calibration file is required; generating one takes a few
  minutes on a representative corpus. The right corpus matters — calibrate on text similar
  to your inference workload.
- **Long-context only.** The evictor fires every `--tri-interval` decode steps. On short
  sequences (chat, short completions) it may never fire or evict a negligible number of
  tokens. Peak KV savings only appear when the sequence is long enough to fill the budget.
- **GPU scoring requires `--cache-type-k q8_0`.** Without it the scorer falls back to CPU,
  which is functionally identical but slower on long sequences.
- **Large heads (hd ≤ 256) are GPU-eligible for *uniform* head_dim models.** The GPU kernel
  now supports head_dim up to 256 (freq_count ≤ 128) — e.g. Qwen3.5/3.6 (uniform hd=256, iSWA;
  the global-attention layers score on the GPU). The capture-layer gate samples the first
  *captured* layer (not hardcoded layer 0), so iSWA models whose layer 0 is a sliding-window
  (uncaptured) layer are correctly recognized as Q8_0-eligible.
- **Hybrid head_dim models are CPU-only (follow-on).** Gemma-4 mixes hd=256 (SWA) and hd=512
  (full-attention) layers. The GPU omega/q_mean stats buffers use a uniform-`fc` layout
  (offset `li·nh·fc`), so a model whose per-layer head_dim varies must score on CPU. A
  per-layer stats offset table (to GPU-accelerate the eligible hd≤256 layers of hybrid models)
  is a tracked perf follow-on, not a correctness blocker.
- **Non-rotary models.** Models with a non-rotary (content) attention term (`nonrot_dim > 0`)
  fall back to CPU scoring — the GPU kernel implements only the RoPE-direction term.
- **MLA (DeepSeek) not supported.** Multi-head Latent Attention fuses Q/K/V projections;
  `tria-gen` cannot hook a standalone `Qcur` tensor for those architectures.
- **Eviction vs. quantization tradeoff.** TriAttention drops whole tokens; KV quantization
  shrinks each token's bytes but keeps all tokens. At aggressive budgets (< 25 %) eviction
  tends to preserve retrieval quality better than quantization. In the middle range (50–75 %)
  combining both (quantize K to Q8_0 + evict to budget) gives the best memory/quality ratio.

---

## §7 Further reading

- **Runtime:** `src/triattention-runtime.c`, `src/triattention.c` — scoring math
- **Bridge/evictor:** `src/triattention-bridge.cpp`, `src/llama-kv-cache.cpp` — `triattention_compact`
- **Capture injection:** `src/llama-graph.cpp:2365–2387` — `build_attn()` set_rows nodes
- **GPU kernel (HIP):** `src/triattention-hip.hip`, `src/triattention-hip.h`
- **GPU kernel (Vulkan):** `src/triattention-vulkan.cpp`, shaders `triattention/tria_*.comp`
- **Calibration tool:** `tools/tria-gen/tria-gen.cpp`
- **Feature index:** [docs/features/README.md](README.md)
- **Related docs (this repo):**
  - [SWA per-layer KV types](swa-per-layer-kv.md) — assign different KV quant types to SWA vs global layers
  - [TurboQuant KV base](turboquant-kv-base.md) — KV quantization (composes with TriAttention)
