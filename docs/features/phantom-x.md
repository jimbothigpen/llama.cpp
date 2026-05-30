# PHANTOM-X — Self-Speculative N-Gram Drafter

> **Status: Stable** — backend-agnostic; no separate draft model required; `--spec-type phantom`.

---

## At a glance

| | Value |
|---|---|
| **What it is** | Self-speculative n-gram drafter — bloom-filtered pattern tables built from context; no separate draft model |
| **Trigger** | `--spec-type phantom` via `llama-speculative-simple` |
| **Draft mechanism** | Per-token bloom-filtered n-gram transition tables in ghost-buffer ring slots; adaptive eviction prioritizing high-frequency transitions |
| **Backend** | CPU n-gram lookup; backend-agnostic; no novel GPU kernels |
| **Key flags** | `--phantom-buffers N` (ring size), `--phantom-bloom-bits N` (bits per bloom filter) |
| **Best workloads** | Code-heavy / context-repetitive generation — **+34% measured** on repetitive C code |
| **Avoid** | General-chat / creative-writing — near-zero net benefit due to low n-gram repetition |
| **Provenance** | Ported from carlosfundora `1-bit-turbo` (`2199e8445`); `--spec-type` factory wiring added in this fork (`4fd52ddc0`) |

**TL;DR.** Point `--spec-type phantom` at any GGUF and PHANTOM-X builds a live n-gram table from the running context — no second model download. On repetitive code it reaches 86.6% accept rate and +34% throughput over greedy baseline. On novel prose the win is near-zero; workload selection is the key decision.

---

## Quick start

```bash
llama-speculative-simple \
    --spec-type phantom \
    -m <model.gguf> \
    --phantom-buffers 2 \
    --phantom-bloom-bits 16384 \
    -ngl 99 --no-mmap -fa on \
    -p "<your prompt here>" \
    -n 256
```

> **Current limitation — draft-model path workaround:** `speculative-simple.cpp` requires a `-md <path>` argument even for self-speculative types (phantom, ngram-cache) due to a code-path quirk where the `else` branch unconditionally calls `llama_model_load_from_file`. The workaround is to pass any small model as a dummy `-md`; the multi-speculator loop runs PHANTOM-X first and the dummy draft-model speculator only fires as a fallback on the rare token where PHANTOM-X produces zero candidates. A proper self-speculative fast-path (skip draft-model load when `--spec-type phantom` or `--spec-type ngram-*`) is a future cleanup item. The benchmark below used this workaround; the Arm A vs Arm C comparison remains valid because the same workaround applies to both arms equally.

---

## §1 Provenance

PHANTOM-X originates from **carlosfundora / 1-bit-turbo** (`2199e8445`). It is a complete self-speculative n-gram drafter — not an adapter layer over another mechanism.

This fork made two contributions:

1. **Port** (`2199e8445`) — brought the PHANTOM-X drafter implementation in-tree.
2. **`--spec-type` factory wiring** (`4fd52ddc0`) — hooked PHANTOM-X into the `common_speculative_type` dispatch so it can be selected alongside MTP, EAGLE3, DFlash, and `ngram-cache` using the standard `--spec-type` flag. The factory wiring was this fork's addition; PHANTOM-X itself is carlosfundora's.

---

## §2 Use in production

### Requirements

1. **Any causal-LM GGUF** — PHANTOM-X is model-architecture-agnostic. It builds its n-gram table from the live token stream, not from model weights.
2. **`llama-speculative-simple`** with `--spec-type phantom`.
3. **`-fa on`** (flash attention) — recommended for all speculative-decode paths.
4. **`--no-mmap`** — recommended; avoids page-fault latency during generation.
5. **Dummy `-md` path** — see the current-limitation note above.

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--spec-type phantom` | — | Selects PHANTOM-X as the primary speculator |
| `--phantom-buffers N` | 2 | Number of ghost-buffer ring slots for n-gram pattern tables |
| `--phantom-bloom-bits N` | 16384 | Bits allocated per bloom filter (larger = fewer false positives; more memory) |

The defaults (`--phantom-buffers 2 --phantom-bloom-bits 16384`) are suitable for most workloads. Increase `--phantom-buffers` if the context has long-range repetition patterns across many token windows.

### Example — code generation

```bash
llama-speculative-simple \
    --spec-type phantom \
    -m Qwen3.5-9B-Q4_K_M.gguf \
    -md <any-small-model.gguf> \
    --phantom-buffers 2 --phantom-bloom-bits 16384 \
    -ngl 99 --no-mmap -fa on \
    --temp 0 -n 256 \
    -p "Implement an LRU cache in C."
```

### When to use PHANTOM-X vs ngram-cache

Both `--spec-type phantom` and `--spec-type ngram-cache` are n-gram drafters. On the same repetitive-code prompt (Arm A vs Arm C below), phantom reaches 86.6% accept vs ngram-cache's 63.3%, delivering +34% vs +1% throughput. Phantom's bloom-filtered adaptive-eviction machinery is the differentiator — it substantially outperforms plain ngram-cache on code tasks. For novel prose neither mechanism has meaningful n-gram history to exploit, so both converge to near-baseline performance.

---

## §3 Measured performance

**Test configuration:** Qwen3.5-9B-Q4_K_M, ROCm gfx1150, `--temp 0 --ignore-eos -n 256 -ngl 99 --no-mmap -fa on`. Baseline (no speculation): **13.6 t/s**.

| Arm | `--spec-type` | Prompt domain | Accept | Tok/s | vs baseline |
|-----|---------------|---------------|--------|-------|-------------|
| A | `phantom` | Code — repetitive C LRU implementation | **86.6%** | **18.2** | **+34%** |
| B | `phantom` | Novel prose (creative writing) | 71.1% | 13.9 | +3% (flat) |
| C | `ngram-cache` | Code — same prompt as Arm A | 63.3% | 13.7 | +1% (flat) |
| — | none | Baseline | — | 13.6 | 1.00× |

### Interpretation

**Arm A (+34%):** Phantom's bloom-filtered tables correctly cache repetitive C patterns — struct field names, pointer dereferences, common identifiers — and predict them with 86.6% accuracy. This is well into high-value territory for speculative decoding.

**Arm B (+3%, flat):** Even at 71% accept rate the speculation overhead nearly cancels the benefit on novel text. The loop drafts ~246 tokens to accept ~175; the verification cycle overhead erodes the gain at this hardware speed. The result is neutral-to-marginal — expected for low-repetition workloads.

**Arm A vs Arm C (phantom vs ngram-cache on code):** Phantom's bloom+adaptive-eviction machinery delivers 86.6% vs 63.3% on identical input, translating to +34% vs +1% throughput. This validates phantom's reason to exist — the overhead of the bloom machinery is more than paid back on code tasks.

**Hardware scaling note:** On faster hardware (higher baseline t/s), the speedup ratio should improve further — the verification batch cost amortizes better as the baseline accelerates.

### Workload guidance

| Workload | Expected benefit | Notes |
|---|---|---|
| Code completion / refactoring | High (+20–35%) | High n-gram density; field names, keywords, patterns repeat frequently |
| Template / boilerplate generation | High | Highly structured repetitive output |
| Document continuation (same style) | Moderate | Depends on how much phrasing repeats |
| Novel prose / creative writing | Near zero | Low n-gram repetition; overhead erodes gain |
| General chat / Q&A | Near zero | Conversational text has low structural repetition |

---

## §4 How it works

PHANTOM-X maintains a set of **bloom-filtered n-gram transition tables** in a ghost-buffer ring. For each position in the token stream it records n-gram patterns (sequences of preceding tokens) and maps them to predicted continuations. The bloom filter is used to test candidate n-grams quickly — a false-positive rate is acceptable because the verify pass catches mispredictions. The ring structure provides a bounded-memory sliding window over recent context; an adaptive eviction policy retains high-frequency transition entries over low-frequency ones.

At draft time, PHANTOM-X looks up the current context suffix in its pattern tables and proposes candidate token(s). The standard speculative-decode verify pass then accepts or rejects each draft token via greedy or sampled comparison with the target model's logits. The CPU n-gram lookup is fast relative to the GPU inference cost, so the overhead of the lookup is negligible compared to the amortized draft-token savings.

**Key property:** Because PHANTOM-X builds its tables entirely from the running token stream, it requires no prior training, no model-specific weights, and no architecture knowledge. It works with any causal-LM GGUF.

---

## §5 Further reading

- **Upstream source:** carlosfundora / 1-bit-turbo (`2199e8445`) — original PHANTOM-X implementation
- **Speculative decode binary:** `examples/speculative-simple/` — `llama-speculative-simple`
- **Related spec-type docs:**
  - [Qwen3.5/3.6 MTP converter](qwen35-mtp-converter.md) — `--spec-type draft-mtp`; MTP head bundled in GGUF
  - [NLD diffusion self-spec](nld-diffusion-self-spec.md) — diffusion-model-specific path; separate from `--spec-type`
- **Feature index:** [docs/features/README.md](README.md)
