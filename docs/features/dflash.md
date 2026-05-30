# DFlash Speculative Decode (`--spec-type dflash`)

> **Status: Preview — functionally correct, net slowdown until S3 GPU ring lands**
>
> Solo DFlash is verified end-to-end with 25.1 % draft accept and coherent output.
> **It is not yet a performance win.** The S2 CPU cross-attention path is roughly
> 2.5× slower than plain no-spec decode on the same hardware. A speedup requires
> the S3 GPU ring buffer, which is still in progress.

---

## At a glance

| | Value |
|---|---|
| **What it is** | Drafter-model speculative decode via a cross-attention ring; drafter sees target hidden states |
| **Flag** | `--spec-type dflash` (with `-md <dflash-draft.gguf>`) |
| **Phase shipped** | S1 model loader (`b6a75e524`) + S2 dispatch (`ef80c728c`) |
| **Phase in progress** | S3 GPU ring buffer + bulk argmax (Phase 7a — required for a speedup) |
| **Solo accept rate** | **25.1 %** (`n_drafted=195`, `n_accept=49`; ROCm gfx1150, `--temp 0`) |
| **Throughput vs no-spec** | **≈0.4× (net slowdown)** — S2 CPU path ≈10.7 tok/s vs ≈26.7 tok/s baseline |
| **Runtime upstream** | buun `master` |
| **Converter upstream** | Anbeeld / beellama.cpp (MIT) |
| **Drafter weights** | z-lab DFlash drafter family |
| **Minimum build** | `2726a56c0` (`mask_token_id` type fix required to load z-lab drafters) |

> **No speedup yet.** Do not deploy DFlash expecting faster decode. Use it for
> correctness validation and accept-rate measurement while S3 is in progress.

**Quick start (CLI):**

```bash
# Convert the z-lab DFlash drafter (Qwen3.6 family shown)
python3 conversion/dflash_converter.py <dflash-drafter-dir> \
    --outfile dflash-draft.gguf

# Run speculative decode
llama-speculative-simple \
    --spec-type dflash \
    -m  target.gguf \
    -md dflash-draft.gguf \
    -fa on -ngl 999 --no-mmap \
    --temp 0 --ignore-eos
```

**Quick start (server):**

```bash
llama-server \
    --spec-type dflash \
    -m  target.gguf \
    -md dflash-draft.gguf \
    -fa on -ngl 999 --no-mmap
```

---

## §1 Provenance

| Component | Source |
|---|---|
| Runtime (speculative loop, cross-attention ring, dispatch) | **buun `master`** |
| GGUF converter (`conversion/dflash_converter.py`) | **Anbeeld / beellama.cpp** (MIT) |
| Drafter weights (z-lab DFlash drafter family) | **z-lab** |

DFlash is a novel speculative-decode mechanism in which a small drafter model
attends to **hidden states** from the target model (via a learned cross-attention
ring) rather than to tokens alone. The drafter does not share the target's
vocabulary projection; it produces a single drafted token per step.

---

## §2 Phase status

| Phase | Commits | What it added | Status |
|---|---|---|---|
| **S1 — model loader** | `b6a75e524` | Drafter model architecture + GGUF loader in-tree | ✅ Shipped |
| **S2 — dispatch** | `ef80c728c` | `common_speculative_state_dflash` + factory dispatch; `--spec-type dflash` wired | ✅ Shipped |
| **`mask_token_id` type fix** | `1436d1890` | `int32_t → uint32_t` to match `llama-hparams.h` (required to load any z-lab drafter) | ✅ Shipped |
| **Converter** | `ee7d4f896` | safetensors → GGUF conversion for the z-lab DFlash drafter family | ✅ Shipped |
| **KV-position correctness fix** | `003ecc2d1` | Anchors drafter batch to drafter KV pos (was: cross-attn ring length — see §4) | ✅ Shipped 2026-05-30 |
| **S3 — GPU ring buffer + bulk argmax + server `spec_type` wiring** | — | Eliminates per-iteration CPU cross-attention; required for a net speedup | 🔄 In progress |

**Known open items:**

- Gemma-4 DFlash converter path exists but is not yet smoke-tested (z-lab
  Gemma-4 drafter directory was missing tokenizer files at last check).
- Server multi-batch prompt accumulation into the cross-attention ring is not
  yet implemented; `llama-server` with a long cached prompt may produce a
  ring-length mismatch. CLI (single-batch prompt) is unaffected and is what
  the gate exercised.

---

## §3 Use in production

### Requirements

1. **A DFlash drafter GGUF.** Convert a z-lab DFlash drafter (Qwen3.6 or
   Gemma-4) with `conversion/dflash_converter.py`. The drafter GGUF is not a
   standard causal LM — it will error or SIGSEGV if loaded as `-m`.
2. **Build ≥ `2726a56c0`** — the `mask_token_id` type fix (`1436d1890`) is
   required to load any z-lab DFlash drafter. Earlier builds will fail at
   model-load time.
3. **Flash attention:** `-fa on`. The cross-attention ring uses flash attention.
4. **`--no-mmap`**: recommended; consistent with upstream DFlash usage.

### CLI flags

| Flag | Required | Description |
|---|---|---|
| `--spec-type dflash` | Yes | Selects the DFlash speculative path |
| `-md <dflash-draft.gguf>` | Yes | DFlash drafter GGUF |
| `-m <target.gguf>` | Yes | Target causal LM (any arch) |
| `-fa on` | Yes | Flash attention (required) |
| `--ignore-eos` | Recommended for measurement | Prevents early termination on Qwen3.6 greedy prompts; omit for normal chat use |
| `--temp 0` | Optional | Deterministic greedy; used for accept-rate measurement |

### Accept rate and throughput

**Measured configuration:** Qwen3.6-35B-A3B-MTP-IQ4_XS (target) + Qwen3.6
DFlash-draft Q8_0, ROCm gfx1150, `--temp 0`, chat-templated, `--ignore-eos`.

| Metric | Value |
|---|---|
| Accept rate | **25.1 %** (`n_drafted=195`, `n_accept=49`) |
| Throughput (DFlash S2 CPU) | ≈10.7 tok/s |
| Throughput (no-spec baseline) | ≈26.7 tok/s |
| Speed ratio | **≈0.4× — net slowdown** |

The 25.1 % accept figure supersedes the earlier 33.3 % figure, which was
measured under dual-spec (DFlash + draft-simple combined) and is no longer
applicable. Solo DFlash has run since `06d570ab5` suppressed the dual-spec
auto-enable.

---

## §4 The KV-position correctness fix (2026-05-30)

Before commit `003ecc2d1`, solo DFlash produced near-zero useful drafts after
the first decode iteration. The cause was a drafter-batch position bug in
`common_speculative_impl_dflash::draft()` (`common/speculative.cpp`):

**The bug.** The drafter batch was positioned at `cross_len` — the
cross-attention ring length returned by `build_cross_data()`. That length grows
by `n_accepted + 1` every iteration as target hiddens are committed into the
ring. The drafter context, however, is stateless per iteration: it is trimmed
back to the prompt checkpoint each round, so its last KV position stays pinned
at the prompt end.

`cross_len` and the drafter KV position coincide only on the first iteration.
Thereafter `cross_len > kv_pos + 1`, so `llama_batch_allocr::init()` rejects
the batch (the `Y = X + 1` consecutive-position check fails), and every DFlash
draft decode after the first silently drops. During the dual-spec era this was
masked by the `draft-simple` fallback; suppressing dual-spec exposed it.

**The fix.** Anchor the drafter batch to the drafter context's own next KV
slot:

```cpp
// common/speculative.cpp — draft()
const llama_pos dft_pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), sid) + 1;
common_batch_add(batch_dft, id_last,       dft_pos0,     { sid }, true);
for (int i = 1; i < batch_len; ++i)
    common_batch_add(batch_dft, mask_token_id, dft_pos0 + i, { sid }, true);
```

`build_cross_data()` is still called — it injects the target hidden ring into
the drafter via `llama_set_cross_data`; only its return value is no longer used
for positioning. The fix is localized to the DFlash impl; MTP is a separate
impl and is byte-for-byte untouched.

---

## §5 Benefits and limitations

### Benefits

- **Architecture-agnostic target.** DFlash drafts against target hidden states,
  not tokens — the target model can be any architecture the fork supports.
- **Functionally correct.** Solo DFlash produces coherent, position-error-free
  output with 25.1 % draft acceptance at `--temp 0`.
- **Converter in-tree.** The z-lab DFlash drafter family can be converted
  locally without external tooling.

### Limitations

- **Net slowdown until S3.** The S2 CPU cross-attention path is the dominant
  cost: per-iteration hidden-state assembly + drafter decode + rejected-draft
  target verification add up to roughly 2.5× the baseline decode time per
  token. No speedup is possible until the S3 GPU ring buffer lands.
- **Not a standard causal drafter.** The DFlash drafter GGUF is not
  interchangeable with a causal LM draft model. It must be paired with a
  matching target architecture and ring dimension.
- **Gemma-4 converter path untested.** Qwen3.6 family is smoke-tested GREEN;
  Gemma-4 is present in the converter but not yet smoke-tested.
- **Server multi-batch prompt limitation.** CLI is the validated path. Server
  support for long cached prompts requires per-batch ring accumulation (deferred
  to a follow-up).

---

## §6 Further reading

- **buun `master`** — runtime upstream (speculative loop + cross-attention ring)
- **Anbeeld / beellama.cpp** (MIT) — converter upstream
- **z-lab** — DFlash drafter weights (Qwen3.6 family)
- **Converter source:** `conversion/dflash_converter.py`
- **Runtime source:** `common/speculative.cpp` — `common_speculative_impl_dflash`
- **Feature index:** [docs/features/README.md](README.md)
- **Related docs (this repo):**
  - [Qwen3.5/3.6 MTP Converter](qwen35-mtp-converter.md) — MTP speculative decode for Qwen3.5/3.6 (75.6 % accept; `--spec-type draft-mtp`)
  - [NLD diffusion self-spec](nld-diffusion-self-spec.md) — self-speculative decode for diffusion-LM models
