# SWA Per-Layer KV Cache Types (`--cache-type-k-swa` / `--cache-type-v-swa`)

> **Status: Stable** — known limitation: applies only to hybrid sliding-window models
> (Gemma 4 / Gemma 2/3, Llama 4, MiMo2); the benefit is avoiding catastrophic degradation
> under aggressive KV quant, not free quality.

---

## At a glance

| Flag | Short | Default | Effect |
|---|---|---|---|
| `--cache-type-k-swa` | `-ctks` | (unset → uses `--cache-type-k`) | K type for SWA / sliding-window layers |
| `--cache-type-v-swa` | `-ctvs` | (unset → uses `--cache-type-v`) | V type for SWA / sliding-window layers |

**TL;DR.** On hybrid sliding-window models, the KV cache is split into two sub-caches: one
for global full-attention layers and one for SWA sliding-window layers. These flags let each
sub-cache carry a different KV quantization type, adding a second asymmetry axis —
**global vs SWA** — on top of the existing K-vs-V axis. The primary use case is avoiding
the catastrophic-degradation failure mode of uniform aggressive KV quant on hybrid models
by keeping one layer class at a safer precision.

- Unset → `GGML_TYPE_COUNT` sentinel → SWA layers fall back to `--cache-type-k` / `--cache-type-v`
  (`llama-context.cpp:303-304`). Purely opt-in.
- Two independent axes: **(K vs V) × (global vs SWA)** — four independently tunable slots.
- On non-hybrid-SWA architectures the flags are silently ignored.
- Env: `LLAMA_ARG_CACHE_TYPE_K_SWA` / `LLAMA_ARG_CACHE_TYPE_V_SWA`.

**Quick start (Gemma 4 — keep global layers precise, mild-quant SWA layers):**

```bash
llama-server \
    -m gemma4.gguf \
    --cache-type-k f16 --cache-type-v f16 \
    --cache-type-k-swa q8_0 --cache-type-v-swa q8_0 \
    -fa on --no-mmap -c 4096 -ngl 99
```

---

## §1 Provenance

### The ISWA two-sub-cache machinery (mainline)

The split between a global full-attention sub-cache (`kv_base`) and a sliding-window
sub-cache (`kv_swa`) — implemented in `llama_kv_cache_iswa` — is **mainline llama.cpp**.
It was introduced upstream and first appeared in this fork at the mainline merge `fc2b0053f`.
The SWA sub-cache is window-sized and padded to 256 cells for performance (upstream ref:
[ggml-org/llama.cpp#17037](https://github.com/ggml-org/llama.cpp/issues/17037)).

### The per-layer-class type knob (this fork — domvox)

The ability to assign a *different* KV type to each sub-cache — the
`--cache-type-k-swa` / `--cache-type-v-swa` CLI flags and the `type_k_swa` / `type_v_swa`
fields in `llama_context_params` (`include/llama.h:394-395`) — is a contribution of
**domvox**, ported to this fork in commit `30472d827`.

### Differences from upstream

Upstream llama.cpp uses a single KV type pair (`type_k` / `type_v`) for both the global
and SWA sub-caches; the SWA sub-cache inherits the same type as the global sub-cache.
This fork adds a per-layer-class selector so the two sub-caches can carry independent types.

**No new type-ID slot.** The flags select from the existing pool of KV types (F16, Q8_0,
`turboq2`, `turboq3`, etc.) — no new enum value or block struct is introduced.

---

## §2 Use in production

### Requirements

1. **Flash attention** — pass `-fa on` (`--flash-attn on`). Any quantized KV type requires
   the fused flash-attention kernel.
2. **`--no-mmap` recommended** — pass `--no-mmap` when using quantized KV types alongside
   aggressive cache configurations to avoid mapping-related issues.
3. **Hybrid SWA model required** — the flags only take effect on architectures that split
   their KV cache into global and SWA sub-caches
   (`hparams.swa_type != LLAMA_SWA_TYPE_NONE`, `llama-model.cpp:2119`). On all other
   models the flags are silently ignored and the main `--cache-type-k` / `--cache-type-v`
   types apply uniformly.

### Applicable model families

| Architecture | SWA condition |
|---|---|
| Gemma 2 | always (`src/models/gemma2.cpp:4`) |
| Gemma 3 / Gemma 4 | when `n_swa > 0` in the model config — all current Gemma 3/4 checkpoints (`src/models/gemma3.cpp:5-13`) |
| Llama 4 | chunked-SWA by default (`src/models/llama4.cpp:8-25`) |
| MiMo2 | always (`src/models/mimo2.cpp:6`) |
| AFMoE | architecture-conditional (`src/models/afmoe.cpp:16,25`) |

On any other architecture (Llama 3, Qwen3, Mistral, …) both flags are inert.

### Flags reference

| Flag | Short | Env var | Default |
|---|---|---|---|
| `--cache-type-k-swa TYPE` | `-ctks TYPE` | `LLAMA_ARG_CACHE_TYPE_K_SWA` | (unset → falls back to `--cache-type-k`) |
| `--cache-type-v-swa TYPE` | `-ctvs TYPE` | `LLAMA_ARG_CACHE_TYPE_V_SWA` | (unset → falls back to `--cache-type-v`) |

Resolved by `kv_cache_type_from_str` in `common/arg.cpp:2118-2138`. Both flags accept the
same type strings as `--cache-type-k` / `--cache-type-v`.

### Two independent asymmetry axes

These flags introduce a **global-vs-SWA** split on top of the existing K-vs-V asymmetry,
giving four independently tunable slots:

```
global K:  --cache-type-k      (existing flag)
global V:  --cache-type-v      (existing flag)
SWA    K:  --cache-type-k-swa  (this feature)
SWA    V:  --cache-type-v-swa  (this feature)
```

All four can be set independently. Common split patterns:

```bash
# Precise global, mild-quant SWA (conservative starting point)
--cache-type-k f16 --cache-type-v f16 --cache-type-k-swa q8_0 --cache-type-v-swa q8_0

# Precise global, aggressive SWA
--cache-type-k f16 --cache-type-v f16 --cache-type-k-swa turboq3 --cache-type-v-swa turboq3

# Aggressive global, precise SWA (reverse split — tests which class is more sensitive)
--cache-type-k turboq3 --cache-type-v turboq3 --cache-type-k-swa f16 --cache-type-v-swa f16
```

Which direction best preserves quality is model-dependent; the benchmark matrix (§3) is
designed to answer this. See [concepts/asymmetric-kv-cache.md](concepts/asymmetric-kv-cache.md)
for background on the K-vs-V axis.

### `--swa-full` interaction

`--swa-full` makes the SWA sub-cache full-sized (equal to the global cache size) rather
than window-sized. The per-layer-class type selection still applies; only the memory
savings from the SWA cache being smaller are lost.

---

## §3 Benefits & potential drawbacks

### Benefits

- **Catastrophe-avoidance on hybrid SWA models.** Uniform aggressive KV quant on Gemma 4
  collapses perplexity to >100 000 (vs ~25 F16 baseline). Keeping at least one layer class
  at a higher precision dramatically reduces the damage. Reference stress-test data point
  (commit `30472d827`):
  — uniform `turboq3` → PPL **>100 000**;
  — `turboq3` global + `f16` SWA → PPL **~28 000**;
  — `f16` baseline → PPL **~25 000**.
- **SWA sub-cache is window-sized.** The SWA cache is sized to the attention window
  (`n_swa` tokens), not the full context, so it is already smaller than `kv_base`.
  Quantizing it adds memory savings at lower cost than quantizing the full cache.
- **Runtime-only.** No offline quantization, no model re-download, no GGUF change.

### Important caveats

- **Catastrophe-avoidance, not free quality.** Even the "good" split configurations in the
  stress test are far above a usable perplexity baseline. Aggressive KV quant on Gemma-4-class
  models is rough regardless; this feature keeps a bad situation from becoming catastrophic.
- **Which layer class is the more sensitive one is not yet benchmarked.** The available
  data point measured only one direction (precise SWA, aggressive global). Benchmark matrix
  row 7 (reverse split) will answer whether the SWA or global layers carry the quality.
  Do not assume one direction is always better.
- **Inert on non-hybrid architectures.** The flags silently fall back to the main types
  on Llama 3, Qwen3, Mistral, and any model without a hybrid SWA split.

### Benchmark matrix

> Numbers to be filled in after benchmarking. See caption for configuration.
>
> **Configuration:** a Gemma-4-class hybrid SWA model · context length TBD · backend TBD ·
> GPU class TBD. Rows 1–3 establish baselines; rows 4–6 show split configurations;
> row 7 (reverse split) isolates which layer class is more sensitive to quantization.

*(TBD — pending benchmark. See TODO 154.)*

| # | Global K/V | SWA K/V | Category | PPL | TG (t/s) | PP (t/s) | KV mem vs F16 |
|---|---|---|---|---|---|---|---|
| 1 | F16 / F16 | F16 / F16 | Baseline — full precision | TBD | TBD | TBD | 1.0× (reference) |
| 2 | Q8_0 / Q8_0 | Q8_0 / Q8_0 | Uniform mid-precision (no split) | TBD | TBD | TBD | TBD |
| 3 | turboq3 / turboq3 | turboq3 / turboq3 | Uniform aggressive — **catastrophe case** | TBD | TBD | TBD | TBD |
| 4 | F16 / F16 | Q8_0 / Q8_0 | Split: precise global, mild SWA | TBD | TBD | TBD | TBD |
| 5 | F16 / F16 | turboq3 / turboq3 | Split: precise global, aggressive SWA | TBD | TBD | TBD | TBD |
| 6 | Q8_0 / Q8_0 | turboq3 / turboq3 | Split: mild global, aggressive SWA | TBD | TBD | TBD | TBD |
| 7 | turboq3 / turboq3 | F16 / F16 | Reverse split — which class is sensitive? | TBD | TBD | TBD | TBD |

---

## §4 How it works under the hood

### ISWA split and layer routing

When a hybrid SWA model is loaded (`hparams.swa_type != LLAMA_SWA_TYPE_NONE`,
`src/llama-model.cpp:2119`), `llama_kv_cache_iswa` is created in place of a plain
`llama_kv_cache`. The ISWA object owns two independent caches
(`src/llama-kv-cache-iswa.cpp:65-75`):

```cpp
kv_base = std::make_unique<llama_kv_cache>(
        model, type_k, type_v, ...
        0, LLAMA_SWA_TYPE_NONE, ..., filter_base, reuse);  // global layers

kv_swa  = std::make_unique<llama_kv_cache>(
        model, type_k_swa, type_v_swa, ...
        hparams.n_swa, hparams.swa_type, ..., filter_swa, reuse);  // SWA layers
```

Layer routing uses `model.hparams.is_swa(il)`:

- **`filter_base`** (`llama-kv-cache-iswa.cpp:33-38`): admits layer `il` when
  `!hparams.is_swa(il)` — the global / full-attention layers.
- **`filter_swa`** (`llama-kv-cache-iswa.cpp:41-47`): admits layer `il` when
  `hparams.is_swa(il)` — the sliding-window layers.

Each layer's attention is dispatched to only the sub-cache matching its class; the two
caches never share cells.

### SWA sub-cache sizing

```cpp
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
```
(`llama-kv-cache-iswa.cpp:53`)

The SWA cache is padded to 256 cells but otherwise bounded by the attention window
(`n_swa` tokens), not the full context. This makes it structurally smaller than `kv_base`,
which limits how much memory quantizing it can save compared with quantizing the global
sub-cache. Pass `--swa-full` to set `size_swa = size_base` (disables the size advantage).

### Default / fallback

`llama_context_params` defaults both fields to `GGML_TYPE_COUNT`
(`llama-context.cpp:3466-3467`). Before constructing the ISWA cache, the context resolves
the sentinel to the corresponding main type (`llama-context.cpp:303-304`):

```cpp
/*.type_k_swa =*/ params.type_k_swa == GGML_TYPE_COUNT ? params.type_k : params.type_k_swa,
/*.type_v_swa =*/ params.type_v_swa == GGML_TYPE_COUNT ? params.type_v : params.type_v_swa,
```

When neither `-ctks` nor `-ctvs` is set, both sub-caches use the same types — matching
vanilla mainline behavior.

### Non-SWA models

When `hparams.swa_type == LLAMA_SWA_TYPE_NONE` (`llama-model.cpp:2139`), a plain
`llama_kv_cache` is instantiated with only `type_k` / `type_v`. The `type_k_swa` /
`type_v_swa` values resolved by the context are never passed to anything; the flags are
fully inert.

### API (`include/llama.h:394-395`)

```cpp
enum ggml_type type_k_swa; // data type for K cache of SWA layers (GGML_TYPE_COUNT = use type_k)
enum ggml_type type_v_swa; // data type for V cache of SWA layers (GGML_TYPE_COUNT = use type_v)
```

---

## §5 Further reading

- **Related docs (this repo):**
  - [docs/features/README.md](README.md) — index of all feature docs
- **Concept primers:**
  - [Asymmetric KV cache primer](concepts/asymmetric-kv-cache.md) — the K-vs-V axis this
    feature composes with; background on why K and V tolerate quantization differently
  - [Feature maturity levels & backend support](concepts/feature-maturity-levels.md)
- See TriAttention docs when available — the SWA sub-cache (`kv_swa`) is the same
  sub-cache object that TriAttention operates on.
