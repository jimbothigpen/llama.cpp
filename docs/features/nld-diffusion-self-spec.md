# NLD — Diffusion-LM Self-Speculative Decode

> **Status: Stable** — CLI and server; Vulkan-validated; no `--spec-type` flag required.

---

## At a glance

| | Value |
|---|---|
| **What it is** | Self-speculative decode for diffusion-LM models — bidirectional draft, causal verify, shared KV |
| **Supported arches** | Dream, LLaDA, LLaDA-MoE, RND1 (`llm_arch_is_diffusion`, `llama-arch.cpp:960`) |
| **Headline model family** | NVIDIA Nemotron-Labs-Diffusion 3B / 8B / 14B (via Dream arch); also Dream 7B / LLaDA / LLaDA-MoE / RND1 |
| **Binary** | `llama-diffusion-cli` (`examples/diffusion/diffusion-cli.cpp`) |
| **CLI self-spec flags** | `--diffusion-self-spec` + `--diffusion-draft-length N` (`arg.cpp:4089-4096`) |
| **Block-mode (default)** | No `--diffusion-self-spec` flag — standard masked-diffusion decode |
| **Server** | Auto-detects diffusion models via `llama_model_is_diffusion()` and enables self-spec — no flag needed |
| **NOT `--spec-type`** | NLD is a separate path from MTP / EAGLE3 / DFlash; there is no `--spec-type nld` |
| **Perf (reported)** | CLI self-spec 7.0 t/s, 3.7× over block-mode, ~68% draft accept; server 4.49 t/s (128 tokens) |

**TL;DR.** Run any Dream / LLaDA / RND1 GGUF through `llama-diffusion-cli` with `--diffusion-self-spec` to get 3–4× throughput over standard block-mode diffusion decode. The model serves as its own drafter — no second GGUF, no separate drafter download.

**Quick start:**

```bash
# Block-mode (default — no self-spec)
llama-diffusion-cli -m nemotron-diffusion-14b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Write a Python function for Fibonacci." \
    -n 256 --diffusion-block-length 32 --diffusion-steps 32

# Self-speculative decoding (~3.7× speedup, ~68% draft accept)
llama-diffusion-cli -m nemotron-diffusion-14b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Write a Python function for Fibonacci." \
    -n 128 --diffusion-self-spec --diffusion-draft-length 8
```

---

## §1 Provenance

### Upstream (mainline llama.cpp)

Base diffusion-LM model support and block-mode generation are upstream contributions:

| PR | What it added |
|---|---|
| [#14644](https://github.com/ggml-org/llama.cpp/pull/14644) | Dream 7B arch (`LLM_ARCH_DREAM`) |
| [#14771](https://github.com/ggml-org/llama.cpp/pull/14771) | LLaDA support |
| [#16003](https://github.com/ggml-org/llama.cpp/pull/16003) | LLaDA-MoE support |
| [#17433](https://github.com/ggml-org/llama.cpp/pull/17433) | RND1 support |
| [#22590](https://github.com/ggml-org/llama.cpp/pull/22590) | `llama-diffusion-cli` refactor |

### This fork (lifted from buun)

Two additions not in mainline:

1. **Self-speculation layer** — the `diffusion_self_spec_generate()` function (`examples/diffusion/diffusion.cpp:611-807`): bidirectional-draft-over-shared-causal-KV cycle with greedy-argmax acceptance.
2. **Nemotron-Labs Diffusion converter** — `conversion/nemotron_labs_diffusion.py`, class `NemotronLabsDiffusionModel` (`model_arch = MODEL_ARCH.DREAM`). Supports NVIDIA Nemotron-Labs-Diffusion 3B / 8B / 14B; emits arch `dream` with the `diffusion.shift_logits` flag. Differs from Dream 7B only in shift-logits convention and tokenizer.

### Model identity note

`LLM_ARCH_DREAM` hosts **two families**: the original Dream 7B (DreamLM, arXiv 2508.15487) and NVIDIA Nemotron-Labs-Diffusion 3B/8B/14B. The converter maps Nemotron weights onto the Dream arch at conversion time; at runtime there is one arch and one self-spec path.

**Open question:** whether the Nemotron 8B-VLM (vision) variant is supported is unverified — do not assume multimodal support.

---

## §2 Use in production

### Requirements

1. **A diffusion-LM GGUF** — Dream (Dream 7B or Nemotron-Labs-Diffusion 3B/8B/14B), LLaDA, LLaDA-MoE, or RND1. Standard causal-LM GGUFs are not compatible.
2. **Flash attention:** `-fa on` (or `--flash-attn on`). The self-spec path uses `llama_set_causal_attn()` to toggle between bidirectional draft and causal verify passes on a shared KV cache; flash-attention is required for this to function.
3. **`--no-mmap`**: recommended for diffusion models; the upstream README examples use it.
4. **Self-spec is greedy/argmax accept** — sampling temperature does not affect draft acceptance. Generation quality is deterministic given the prompt.

### CLI flags (self-speculative mode)

| Flag | Default | Description |
|---|---|---|
| `--diffusion-self-spec` | off | Enable bidirectional draft + causal verify loop |
| `--diffusion-draft-length N` | 8 | Tokens drafted per cycle |

These flags are additive to the standard diffusion parameters (`--diffusion-steps`, `--diffusion-block-length`, `--diffusion-algorithm`, etc.). Without `--diffusion-self-spec`, the CLI uses standard block-mode diffusion decode.

### CLI examples

**Dream 7B — self-spec:**
```bash
llama-diffusion-cli -m dream-7b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Explain the Pythagorean theorem." \
    -n 256 --diffusion-self-spec --diffusion-draft-length 8
```

**Nemotron-Labs Diffusion — self-spec:**
```bash
llama-diffusion-cli -m nemotron-diffusion-14b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Write a Python function for Fibonacci." \
    -n 128 --diffusion-self-spec --diffusion-draft-length 8
```

**LLaDA — block-mode only (self-spec is supported but block-mode is the upstream default for LLaDA):**
```bash
llama-diffusion-cli -m llada-8b.gguf \
    -ngl 99 --no-mmap -fa on \
    -p "Write code to train MNIST in PyTorch." \
    -ub 512 --diffusion-block-length 32 --diffusion-steps 256
```

### Server (auto-detection, no flag)

The server auto-detects diffusion models at load time via `llama_model_is_diffusion()` and enables self-spec automatically — no `--spec-type` or `--diffusion-self-spec` flag is needed (`tools/server/server-context.cpp:928-930, :1130-1133`). Per-slot fields `diff_self_spec` / `diff_draft_length` / `diff_mask_token_id` hold the state (`:76-78`).

**This is NOT `--spec-type`.** MTP, EAGLE3, DFlash, and PHANTOM use `common_speculative_type` and are selected via `--spec-type`. NLD is a separate diffusion-model-specific path; there is no `--spec-type nld` value.

```bash
llama-server -m nemotron-diffusion-14b-Q8_0.gguf \
    -ngl 99 --no-mmap -fa on
# Server auto-enables self-spec; no additional flags required.
```

---

## §3 Benefits and potential drawbacks

### Benefits

- **Large throughput gain over block-mode** — reported 3.7× speedup (CLI) at ~68% draft acceptance. Server measured at 4.49 t/s vs CLI 7.0 t/s (128-token generation).
- **No separate drafter** — the diffusion model generates its own draft tokens in a bidirectional pass; no second GGUF download, no vocab remap, no separate model slot.
- **Server zero-config** — auto-detection means existing server deployments gain self-spec without flag changes after upgrading.
- **Greedy accept is lossless in the diffusion sense** — the acceptance criterion (argmax match between draft and verify logits) is exact; no probability redistribution is needed.

### Potential drawbacks

- **Diffusion-LM only** — NLD is only applicable to Dream / LLaDA / LLaDA-MoE / RND1 arches. It does not apply to causal LLMs (Llama, Qwen, Gemma, etc.).
- **Flash attention required** — same as the base diffusion path.
- **Draft length tuning** — `--diffusion-draft-length 8` is the README default; shorter drafts (4–6) may perform better on shorter prompts or lower-accept-rate models.
- **Greedy-only accept** — self-spec uses argmax at both draft and verify stages; stochastic sampling is not currently applied to draft token acceptance.

### Benchmark matrix

*TBD (pending benchmark)*

**Configuration:** model=TBD, context=TBD tokens, backend=TBD, GPU class=TBD.

| Configuration | Tokens/s | Accept rate | Notes |
|---|---|---|---|
| **Block-mode baseline** (no self-spec) | TBD | — | Standard masked diffusion |
| **Self-spec** `--diffusion-draft-length 4` | TBD | TBD | |
| **Self-spec** `--diffusion-draft-length 8` | TBD | TBD | README-reported: 7.0 t/s CLI, ~68% accept |
| **Self-spec** `--diffusion-draft-length 16` | TBD | TBD | |
| **Server self-spec** (auto) | TBD | TBD | README-reported: 4.49 t/s, 128 tokens |

---

## §4 How it works under the hood

The self-spec loop runs inside `diffusion_self_spec_generate()` (`examples/diffusion/diffusion.cpp:611-807`). Each cycle:

### Phase 1 — Causal prefill
`llama_set_causal_attn(ctx, true)` → standard causal forward pass over the input prompt. Populates the KV cache and yields logits at the last committed position (`prev_logits`). (`diffusion.cpp:629`)

### Phase 2 — Bidirectional draft
`llama_set_causal_attn(ctx, false)` → a single non-causal forward pass decodes all `draft_length` masked positions simultaneously. Each masked slot sees the full context (bidirectional attention). (`diffusion.cpp:672`)

### Phase 3 — Greedy argmax sample
Draft tokens are chosen via `argmax` over the bidirectional logits — one token per position, no temperature. (`diffusion.cpp:702`)

### Phase 4 — Clear bidirectional KV
`llama_memory_seq_rm(mem, ...)` evicts the bidirectional draft KV entries. Those entries are not valid for the upcoming causal verify pass and must be removed to keep the shared cache consistent. (`diffusion.cpp:705`)

### Phase 5 — Causal verify
`llama_set_causal_attn(ctx, true)` → a single causal forward pass over the draft tokens, producing `verify_logits`. (`diffusion.cpp:727`)

### Phase 6 — Accept longest prefix
Compare `prev_logits` → `draft[0]` and `verify_logits[j]` → `draft[j+1]` for each position. Accept the longest agreeing prefix (`n_accept`). (`diffusion.cpp:744`)

### Phase 7 — Bonus token
Sample a bonus token from `verify_logits[n_accept-1]` (greedy). If zero tokens were accepted, sample from `prev_logits`. The bonus token is always committed. (`diffusion.cpp:776`)

### Phase 8 — Commit and trim
Evict rejected KV positions (`llama_memory_seq_rm` for positions `committed+n_accept` to `committed+draft_len`), decode the bonus token to update KV and advance `prev_logits`, then loop. (`diffusion.cpp:776`)

### Key toggle

`llama_set_causal_attn()` (`llama-context.cpp:3756`) is the shared-KV toggle. Phases 1/5/8 run causal; Phase 2 runs bidirectional. Both share the same KV cache — Phase 4 is the cleanup step between the two attention modes.

---

## §5 Further reading

- **Upstream PRs:** [#14644](https://github.com/ggml-org/llama.cpp/pull/14644) (Dream) · [#14771](https://github.com/ggml-org/llama.cpp/pull/14771) (LLaDA) · [#16003](https://github.com/ggml-org/llama.cpp/pull/16003) (LLaDA-MoE) · [#17433](https://github.com/ggml-org/llama.cpp/pull/17433) (RND1) · [#22590](https://github.com/ggml-org/llama.cpp/pull/22590) (diffusion-cli refactor)
- **Dream 7B paper:** arXiv 2508.15487 — DreamLM
- **LLaDA paper:** arXiv 2502.09992 — LLaDA: Large Language Diffusion with mAsking
- **NVIDIA Nemotron-Labs-Diffusion:** via `conversion/nemotron_labs_diffusion.py` (this fork)
- **diffusion-cli README:** `examples/diffusion/README.md` — full flag reference
- **Related docs (this repo):**
  - [docs/features/README.md](README.md) — index of all feature docs
  - [docs/features/concepts/feature-maturity-levels.md](concepts/feature-maturity-levels.md) — what Stable means; backend support notation
