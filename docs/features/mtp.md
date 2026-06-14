# MTP Speculative Decode

> **Status: Stable** — two delivery paths, both verified working. On integrated GPUs the runtime defaults to `n_max=1`; published 1.5–2.5× speedups are the discrete-GPU regime (see §2).

---

## At a glance

MTP (Multi-Token Prediction) speculative decode lets a model predict multiple future tokens cheaply from an auxiliary head, then verify them in a single forward pass of the full model. Accepted drafts increase the number of tokens generated per step without changing output quality.

**Two delivery mechanisms, both Stable:**

| | Internal NextN-tail | External assistant |
|---|---|---|
| Models | Qwen3.5 / Qwen3.6 (dense + MoE) | Gemma 4 26B-A4B |
| Draft source | MTP head bundled inside the same GGUF | Separate assistant GGUF (`-md`) |
| Accept rate (measured) | 75.56% (Qwen3.5/3.6 9B bundled) | 47.3% |
| Key CLI flag | `--spec-type draft-mtp` | `-md assistant.gguf --spec-type draft-mtp --spec-draft-n-max 4` |
| Backends | ROCm, Vulkan | ROCm verified |

**Quick start — internal NextN-tail (Qwen3.5/3.6):**

```bash
llama-speculative-simple \
    -m qwen3.5-9b-mtp.gguf \
    --spec-type draft-mtp
```

**Quick start — external assistant (Gemma 4):**

```bash
llama-speculative-simple \
    -m gemma4-27b.gguf \
    -md gemma4-assistant.gguf \
    --spec-type draft-mtp \
    --spec-draft-n-max 4
```

**Server quick start — Qwen3.5/3.6 (read the gotchas in §2 first):**

```bash
llama-server \
    -m qwen3.5-9b-mtp.gguf \
    --spec-type draft-mtp \
    --parallel 1 \
    --reasoning-budget 0
```

---

## §1 Provenance

The **internal NextN-tail** path is mainline-aligned. The MTP head is bundled as extra layers in the target GGUF and loaded as a dedicated draft context (`LLAMA_CONTEXT_TYPE_MTP`, `include/llama.h:228`). The speculative loop runs through the shared driver in `common/speculative.cpp`.

The **external-assistant** path for Gemma 4 is a guided port of mainline PR #23398. The assistant model uses `LLM_ARCH_GEMMA4_ASSISTANT` and runs through the NEXTN_PRE/POST projection path defined in `src/models/gemma4-assistant.cpp` (208 LOC). The older D1/ASSIST_ code path has been retired.

The Qwen3.5/3.6 *converter* that creates or splits MTP-bundled GGUFs at conversion time is a separate feature — see [`qwen35-mtp-converter.md`](qwen35-mtp-converter.md). This doc covers runtime speculative decode only.

---

## §2 Use in production

### CLI surface

The canonical flag is **`--spec-type draft-mtp`**. The string `mtp` is retained as an alias (`common/speculative.cpp:67`). The older `--mtp` / `--multi-token-prediction` flags are **deprecated** (`common/arg.cpp:1387–1389`) — they still function but log a warning; update scripts to use `--spec-type draft-mtp`.

| Flag | Description |
|---|---|
| `--spec-type draft-mtp` | Enable MTP speculative decode |
| `--spec-draft-n-max N` | Maximum draft depth per step (`common/arg.cpp:3701`). Default 3; Gemma 4 **requires `--spec-draft-n-max 4`** to draft at all. On iGPUs the runtime auto-clamps to 1 unless you set this explicitly. |
| `--reasoning-budget 0` | Forces immediate `</think>` on Qwen3.5/3.6 thinking models (see gotcha below) |

### Which binary actually speculates

`llama-cli --spec-type draft-mtp` loads the NextN tail but runs **plain autoregressive decode** — it does not speculate. Speculative decode is triggered by:

- **`llama-speculative-simple --spec-type draft-mtp`** — recommended lightweight CLI speculator
- **`llama-server --spec-type draft-mtp`** — server path (see gotchas below)

### Production gotchas

**`llama-server` requires `--parallel 1`** — the MTP path is single-sequence only. The server logs a reminder at startup (`common/arg.cpp:3777`).

**Qwen3.5/3.6 thinking-mode trap** — per-request `"thinking":{"type":"disabled"}` is insufficient to suppress thinking traces. Pass `--reasoning-budget 0` at server startup to force immediate `</think>` termination. Without it, the server emits 10–20 minute thinking traces and subsequent requests time out.

**iGPU throughput** — on integrated GPUs, MTP is tuned to `n_max=1` by default after the C1 catch-up-batching optimization. At that depth it reaches ~1.16× pure decode throughput (gfx1150 ROCm and Vulkan; `CHANGELOG.md:33–34` and `:74`). Setting `n_max≥2` on an iGPU remains a net slowdown due to unified-memory sync overhead — the default is correct and should not be overridden on iGPU hardware. The runtime logs a one-time notice explaining this when an integrated GPU is detected.

**PPL is unchanged by construction** — perplexity evaluation only exercises the prefill pass; the speculative path never fires. MTP is a decode-driver feature only.

---

## §3 Benefits & potential drawbacks

### Benefits

- **Real throughput gain on discrete GPUs** — 1.5–2.5× token generation throughput at high accept rates in discrete-GPU deployments.
- **Modest iGPU win** — C1 catch-up batching delivers ~1.16× at `n_max=1` even on integrated GPUs (Vulkan parity confirmed).
- **No model-quality cost** — PPL is identical-by-construction; speculation is transparent to the output distribution.
- **High accept rates** — 75.56% measured (Qwen3.5/3.6 9B bundled); 47.3% (Gemma 4 external, near the CUDA reference at 0.588).
- **Cross-backend parity (Vulkan)** — gfx1150 RADV Vulkan reaches the same 32.4 t/s as ROCm at `n_max=1`.

### Potential drawbacks

- **iGPU net-negative at `n_max≥2`** — unified-memory sync cost dominates any draft-accept gain; stay at the default.
- **`--parallel 1` restriction** — MTP cannot serve multiple simultaneous sequences in `llama-server`.
- **Thinking-mode footgun** — Qwen3.5/3.6 requires a server-level `--reasoning-budget 0`; per-request thinking disable is not enough.
- **Gemma 4 needs a separate assistant GGUF** — the external assistant path requires downloading and providing the matching assistant model file.

### Benchmark matrix

*TBD (pending benchmark)*

**Metric:** accept-rate (%) + tokens/s vs MTP-OFF baseline at the same prompt and `n_predict`. Perplexity is not the relevant metric here (PPL is identical-by-construction).

| Config | MTP-OFF baseline (t/s) | MTP accept-% | MTP t/s | Speedup |
|---|---|---|---|---|
| Qwen3.5/3.6 internal — iGPU, n_max=1 (measured anchor) | 28.0 | 100% | 32.4 | 1.16× |
| Qwen3.5/3.6 internal — iGPU, n_max≥2 | — | — | TBD | net slowdown |
| Qwen3.5/3.6 internal — dGPU | TBD | TBD | TBD | TBD |
| Gemma 4 external assistant — dGPU | TBD | 47.3% (anchor) | TBD | TBD |

---

## §4 How it works under the hood

### Speculative type and dispatch

MTP is registered as `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` (`common/common.h:168`). The factory at `common/speculative.cpp:2424` dispatches to a unified implementation class `common_speculative_impl_draft_mtp` (`:610`), which holds the draft-context state in `common_speculative_state_draft_mtp` (`:1058`).

### iGPU detection and n_max clamp

`mtp_any_igpu()` (`speculative.cpp:52–60`) queries every registered backend device for `GGML_BACKEND_DEVICE_TYPE_IGPU` — the check is backend-agnostic and covers both CUDA/HIP and Vulkan builds. When an iGPU is found and the user has not explicitly set `--spec-draft-n-max`, the constructor clamps `n_max` to 1 at `speculative.cpp:681`.

### MTP driver loop

Two hooks manage KV cache state across draft and verify steps:

- **`mtp_update_kv_cache()`** (`speculative.cpp:2719`) — appends draft tokens to the target context's KV cache after each draft step
- **`mtp_accept_tokens()`** (`speculative.cpp:2753`) — commits accepted tokens and rolls back rejected ones

### Context wiring

The target context is connected to the MTP draft context through three API calls:

- **`llama_set_mtp_op_type()`** (`llama-context.cpp:4033`) — marks the context's MTP operation role
- **`llama_set_embeddings_nextn()`** (`llama-context.cpp:4041`; internal declaration `llama-context.h:143`) — activates the masked next-token (nextn) embedding path that feeds the MTP head
- **`llama_set_mtp_source()`** (`llama-context.cpp:4037`) — registers the target context as the source for MTP draft generation

### Gemma 4 external-assistant path

The assistant model reads `LLM_KV_NEXTN_PREDICT_LAYERS` at load time to determine how many draft steps to generate (`src/models/gemma4-assistant.cpp:13`). Draft logits are computed via learned projection tensors `LLM_TENSOR_NEXTN_PRE_PROJ` / `LLM_TENSOR_NEXTN_POST_PROJ` (`:44–45`), allowing the assistant to produce drafts entirely from the target backbone's hidden states (Q-only drafter; borrows target KV).

---

## §5 Further reading

- **Related speculative decode docs:** [NLD diffusion self-spec](nld-diffusion-self-spec.md) · [DFlash drafter spec-decode](dflash.md) · [PHANTOM-X n-gram drafter](phantom-x.md)
- **Converter (separate feature):** [Qwen3.5/3.6 MTP converter](qwen35-mtp-converter.md) — bundling or splitting the MTP draft head at GGUF-creation time
- **Upstream references:** mainline MTP spine (`LLAMA_CONTEXT_TYPE_MTP`) · PR #23398 (Gemma 4 external assistant port) · PR #22673 (Qwen3.5/3.6 converter)
- **Concept primer:** [Feature maturity levels & backend support](concepts/feature-maturity-levels.md)
