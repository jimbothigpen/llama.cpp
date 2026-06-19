# EAGLE3 Speculative Decode (`--spec-type draft-eagle3`)

> **Status: Preview — functional; architectural accept ceiling at 1/*n_draft* for *n_draft* ≥ 2.**
>
> At `--spec-draft-n-max 1` EAGLE3 accepts every drafted token (100 %). This is the **recommended
> operating mode**. For *n_draft* ≥ 2 the EAGLE3-3.0 autoregressive recurrence imposes a hard
> ceiling of 1/*n_draft* regardless of draft model quality — see §2 for the full explanation.
> The upstream fix (EAGLE 3.1, not yet in this fork) is tracked in §5 (Future).

---

## At a glance

| | Value |
|---|---|
| **What it is** | Separate-model speculative decode; the drafter consumes auxiliary hidden states extracted from 3 target layers |
| **Flag** | `--spec-type draft-eagle3` + `-md <eagle3-draft.gguf>` |
| **Recommended depth** | `--spec-draft-n-max 1` (100 % accept; 1 free token per cycle) |
| **Ceiling at n_draft=3** | **33.333 %** (65/195 drafted — 1/*n_draft* architectural ceiling; see §2) |
| **Server support** | Single-slot only — `--parallel 1` required |
| **Correctness baseline** | `380c93384` — 3 B1 correctness fixes + KV-position anchor fix |
| **Measurement hardware** | AMD gfx1150 Vulkan |

> **Architectural limit.** At *n_draft* ≥ 2 the 1/*n_draft* ceiling is **not** a draft model
> quality issue — a better drafter will not lift it. Use `--spec-draft-n-max 1` for practical
> deployment. See §2.

**Quick start (CLI):**

```bash
llama-speculative-simple \
    -m  target.gguf \
    -md eagle3-draft.gguf \
    --spec-type draft-eagle3 \
    --spec-draft-n-max 1 \
    -fa on -ngl 99 --no-mmap
```

**Quick start (server — read the `--parallel 1` requirement in §2 first):**

```bash
llama-server \
    -m  target.gguf \
    -md eagle3-draft.gguf \
    --spec-type draft-eagle3 \
    --spec-draft-n-max 1 \
    --parallel 1 \
    -fa on -ngl 99 --no-mmap
```

---

## §1 Provenance

EAGLE3 is a speculative-decode architecture from the carlosfundora / 1-bit-turbo upstream. The
drafter is a small separate transformer that consumes **hidden-state features extracted from 3
auxiliary target layers** (`eagle3.extract_layers` GGUF key, typically layers 1, 15, 28 for a
28-layer target) rather than tokens alone.

Three correctness bugs were fixed in `380c93384` (2026-05-30) before the current accept numbers
were reached:

| Fix | Commit component | Effect |
|---|---|---|
| d2t token remap | B1 — `87b5b3d8d` (bundle) | Draft tokens are post-sampling mapped from draft-vocab to target-vocab via a `d2t.weight` I32 tensor; fix is dormant when the draft uses the full target vocabulary (no `d2t.weight` present) |
| `norm_before_residual` gating | B1 | `eagle3.eagle3.norm_before_residual` GGUF key now gates whether `g_embd_norm` or raw `g_embd` feeds the residual connection; previously the key had no effect |
| `rope_factors` propagation | B1 | Per-layer RoPE scaling factors (`rope_freqs` tensors) were loaded but passed as `nullptr` to `ggml_rope_ext`; now forwarded correctly |
| Draft-batch KV-position anchor | `380c93384` | Positions the draft batch from the drafter's own `llama_memory_seq_pos_max(ctx_dft)+1` instead of the advancing target `dp.n_past`; mirrors DFlash fix `003ecc2d1`; unblocked accept from 0 % to the measured rates |

Prior to `380c93384`, accept was **0 %** on every run: the draft batch was anchored to the target
position counter (`dp.n_past`), which advances every accepted cycle while the drafter KV stays
pinned at the prompt-end position. From cycle 2 on, the `Y = X + 1` consecutive-position check in
`llama_batch_allocr::init()` rejected every draft decode silently ("eagle3 decoder failed").

---

## §2 Use in production

### CLI flags

| Flag | Required | Description |
|---|---|---|
| `--spec-type draft-eagle3` | Yes | Selects the EAGLE3 speculative path |
| `-md <eagle3-draft.gguf>` | Yes | EAGLE3 drafter GGUF (separate file from the target) |
| `-m <target.gguf>` | Yes | Target causal LM |
| `--spec-draft-n-max N` | Recommended | Draft depth per cycle. **Default 3; set to 1 for production** (see ceiling below) |
| `-fa on` | Recommended | Flash attention |
| `--no-mmap` | Recommended | Consistent with validated configurations |

### Accept-rate table

Measured configuration: Qwen3.5-9B target + matching eagle3-draft-9b, AMD gfx1150 Vulkan,
`--temp 0`.

| *n_draft* | Drafted | Accepted | Accept % | Notes |
|---|---|---|---|---|
| 1 | 101 | 101 | **100.000 %** | Every d0 accepted; 1 free token per cycle — **recommended** |
| 2 | 202 | 101 | **50.000 %** | 1/*n_draft* ceiling (d1 always rejected) |
| 3 | 195 | 65 | **33.333 %** | 1/*n_draft* ceiling (d1 and d2 always rejected) |

### The architectural ceiling

At *n_draft* ≥ 2, **accept cannot exceed 1/*n_draft* under EAGLE3-3.0 regardless of draft model
quality.** The mechanism:

- **d0** is seeded with `g_embd = FC(target_hidden)` — an affine projection of the target's
  last-layer hidden state. The drafter attends the correct context, so d0 is reliably accepted.
- **d1, d2, …** are seeded with `g_embd = llama_get_embeddings_ith(ctx_dft, -1)` — the
  drafter's **own prenorm output** from the previous step. This diverges immediately from the
  target distribution: the drafter has no access to the true target hidden states at d1+ positions,
  so every token after d0 is effectively random with respect to the target. Measured consequence:
  d1 and d2 are accepted 0 % of the time in the current gate models.
- A better-trained or larger drafter does not help — the recurrence itself feeds the wrong signal.
  This is an EAGLE3-3.0 architectural property, not a model-quality gap.

**Recommendation:** deploy at `--spec-draft-n-max 1`. Every d0 is accepted (100 %), delivering
1 free token per draft cycle with no rejected-draft overhead.

### Production gotchas

**`llama-server` requires `--parallel 1`** — EAGLE3 must run with a single active generating slot.
The drafter indexes target features by the global last token in the verify batch with no per-sequence
mapping; with `n_parallel > 1`, multiple slots' tokens share one target ubatch and all slots are
seeded from the same (wrong) token → cross-sequence contamination and incorrect drafts.

**Drafter GGUF is not a standard causal LM.** The EAGLE3 draft GGUF requires `eagle3.extract_layers`
and related hparams. It cannot be loaded as `-m` (it will fail the architecture check or produce
garbage). Always pass it via `-md`.

**PPL is unchanged by construction** — perplexity evaluation exercises only the prefill pass; the
speculative path does not fire during PPL measurement.

### Compact-vocab draft (SpecForge)

**Landed in `b2766ef47` (2026-05-31) — port of PR #18039.**

SpecForge 32K-vocab EAGLE3 drafters carry a compact output vocabulary (`output.weight` shaped
`[n_embd, 32000]`) plus a `d2t.weight[32000]` draft-to-target token map, rather than the full
target vocabulary. The compact-vocab path is transparent at the CLI level — use the same flags
as a full-vocab drafter:

```bash
llama-speculative-simple \
    -m target-248320-vocab.gguf \
    -md eagle3-specforge-32000-vocab.gguf \
    --spec-type draft-eagle3 \
    --spec-draft-n-max 1 \
    -fa on -ngl 99 --no-mmap
```

**How it works:** `llama_model_load_internal` detects `d2t.weight` and derives `n_draft_vocab`
from its tensor width. The loader (`src/models/eagle3.cpp`) builds the output head at draft-vocab
width and registers the d2t remap table. Graph-side scatter remap translates compact-vocab logits
to target-vocab positions during speculative decode; no changes are needed to the CLI or the
speculative driver.

**Smoke gate (ROCm gfx1150, 2026-05-31, pipefail runner, RC=0 on both runs):**

| Config | Accept % | Notes |
|---|---|---|
| Qwen3.5-35B-A3B-MTP-IQ4_XS (248 320-vocab) + SpecForge-BF16 drafter (32 000-vocab + d2t), *n_draft*=3 | **33.333 %** | compact-vocab path; d2t graph-remap active; coherent output |
| Qwen3.5-9B-IQ4_XS + eagle3-draft-9b (d2t=none), *n_draft*=3 | **33.333 %** | full-vocab; no-regression vs. prior baseline |

**Known limitation (pre-existing, harmless):** The SpecForge GGUF exports `d2t.weight` as
`GGML_TYPE_I64`, but `llama_model_eagle3_get_d2t()` (commit `87b5b3d8d`, already on main)
requires `GGML_TYPE_I32` → the host-side fast path returns empty and the log prints `d2t=none`.
The graph-side remap path is active and correct; output is coherent. Optional future fix: widen
the getter to accept I64, or export d2t as I32 in `conversion/eagle3.py`.

---

## §3 Benefits & limitations

### Benefits

- **100 % accept at *n_draft*=1** — every draft cycle produces one free token at zero reject
  overhead. This is the effective config: no draft rejection means no wasted target verification.
- **Higher accept than DFlash solo** — 33.333 % at *n_draft*=3 vs. DFlash solo 25.1 %, despite
  sharing the same "stateless drafter KV" architecture.
- **Server path works** — `llama-server` is fully wired end-to-end via the generic
  `common_speculative_{init,begin,draft,process,accept}` spine; no server-specific code is needed.
  Feature extraction is passive and automatic per ubatch.
- **Backend-agnostic** — EAGLE3 does not require a specific backend; the drafter runs on whatever
  backend the draft context is assigned to.

### Limitations

- **Architectural ceiling at *n_draft* ≥ 2** — the 1/*n_draft* hard limit means *n_draft*=3 cannot
  exceed 33 % accept, and *n_draft*=2 cannot exceed 50 %. The ceiling is inherent to EAGLE3-3.0's
  autoregressive recurrence (see §2). EAGLE 3.1 is the upstream fix path (see §5).
- **`--parallel 1` only on `llama-server`** — multi-slot server support would require per-sequence
  feature-index plumbing in the drafter; this is not yet implemented.
- **Separate drafter GGUF required** — unlike MTP (which bundles the draft head inside the target
  GGUF), EAGLE3 requires a separately distributed drafter file matched to the target architecture.
- **Drafter KV is stateless per iteration** — accepted tokens are not decoded into the draft KV.
  The drafter attends from a frozen positional frame (prompt-end RoPE positions), which is the same
  tradeoff DFlash makes and is why the ceiling persists even at *n_draft*=1 as sequence length
  grows. A future MTP-style catch-up path (decode accepted tokens into the draft KV) would address
  this but is not yet implemented.

### Benchmark matrix

| Config | Accept % | Notes |
|---|---|---|
| Qwen3.5-9B + eagle3-draft-9b, gfx1150 Vulkan, *n_draft*=1 | **100.000 %** | 101/101; recommended config |
| Qwen3.5-9B + eagle3-draft-9b, gfx1150 Vulkan, *n_draft*=2 | **50.000 %** | 101/202; 1/*n_draft* ceiling |
| Qwen3.5-9B + eagle3-draft-9b, gfx1150 Vulkan, *n_draft*=3 | **33.333 %** | 65/195; 1/*n_draft* ceiling |

Throughput (tokens/s vs. no-spec baseline) is TBD — the speedup depends on draft overhead relative
to the target decode; at *n_draft*=1 the overhead is minimal. Benchmark pending.

---

## §4 How it works under the hood

### Feature extraction

When `llama_set_eagle3(ctx_tgt, model_eagle3)` is called during speculative-init, it sets
`cparams.eagle3_extract_enabled = true` on the target context (`src/llama-context.cpp:1320–1324`).
After every target decode, `llama_context::process_ubatch` copies the hidden-state tensors from the
3 configured auxiliary layers into a host buffer `eagle3_state.target_features` laid out
`[n_aux][n_embd][n_tok]` (`:1540–1554`). Extraction is passive — no extra flags or logit overrides
are needed.

### Draft generation

`common_speculative_impl_draft_eagle3::draft()` (`common/speculative.cpp:501–599`):

1. Reads `llama_get_eagle3_target_features(ctx_tgt)` — this call internally calls
   `ctx_tgt->synchronize()` before returning the host pointer, draining any in-flight GPU compute
   before features are read (`:3800–3804`).
2. CPU-side FC-projects the target features to produce `g_embd` — one vector per token position,
   one per auxiliary layer.
3. Seeds the draft decode via `llama_set_eagle3_g_embeddings` and runs d0 autoregressively.
4. d1+ steps reuse the drafter's own prenorm output as the next `g_embd` — the architectural
   ceiling source (§2).

The draft batch is anchored to `llama_memory_seq_pos_max(ctx_dft, seq_id) + 1` (the drafter's own
next-KV slot), not to the target position counter `dp.n_past` — this is the fix in `380c93384`
that mirrors DFlash commit `003ecc2d1`.

### Verify and accept

After drafting, the shared `common_speculative_process` / `common_speculative_accept` driver runs
the target model over the full draft + verify pass and commits accepted tokens. EAGLE3's `process()`
and `accept()` are no-ops — rollback and KV management are handled by the driver's checkpoint/restore
machinery (`ckpt.update_dft` / `ckpt.load_dft` / `llama_memory_seq_rm`).

---

## §5 Further reading

### Related speculative-decode docs (this fork)

- [MTP speculative decode](mtp.md) — 75.56 % accept (Qwen3.5/3.6 9B internal NextN-tail); head
  bundled inside the target GGUF; `--spec-type draft-mtp`
- [DFlash drafter spec-decode](dflash.md) — 25.1 % accept; cross-attention ring drafter; `--spec-type dflash`
- [PHANTOM-X self-speculative n-gram drafter](phantom-x.md) — no separate draft model; `--spec-type phantom`
- [NLD diffusion self-spec](nld-diffusion-self-spec.md) — bidirectional draft for diffusion LMs

### Future: EAGLE 3.1

> **Future-watch — not yet in this fork.**

EAGLE 3.1 was released 2026-05-26 (vLLM v0.22.0; Kimi K2.6 checkpoint is the first compatible
weight set). The 3.1 architecture changes the d1+ `g_embd` recurrence — specifically, a post-norm
FC path that feeds the draft prenorm with a closer approximation of `FC(target_hidden)` at each
autoregressive step rather than the draft's own prenorm output.

If the 3.1 fix lands as described, it would dissolve the 1/*n_draft* ceiling: d1+ tokens would
receive a higher-quality `g_embd` signal, and accept could in principle approach the d0 rate at all
depths. This is the upstream fix path for the ceiling documented in §2.

**Current status in this fork:** EAGLE 3.1 is not in tree. There are no 3.1 checkpoint weights
available for the current validated target (Qwen3.5-9B). The compact-vocab (32K-vocab + d2t)
loader prerequisite (formerly tracked) is now resolved — `b2766ef47` ports PR #18039
and enables loading SpecForge-style compact-vocab EAGLE3 drafters. The remaining blocker is the
absence of 3.1 checkpoint weights. Monitor upstream vLLM and EAGLE3 repositories for releases.

### Ledger entry: EAGLE 3.1 future-watch (2026-05-31)

| Item | Value |
|---|---|
| **Status** | FUTURE-WATCH — awaiting 3.1 checkpoint + compatible loader |
| **Canonical upstream** | SafeAILab EAGLE + vLLM v0.22.0 (released 2026-05-26) |
| **Architecture change** | FC-norm (prenorm) + post-norm dual-path `g_embd` at d1+ steps — signals closer approximation of `FC(target_hidden)` rather than drafter prenorm output |
| **Impact** | Dissolves 1/*n_draft* ceiling; d1+ accept could approach d0 rate (near 100 %) at all depths |
| **Re-check trigger** | When EAGLE 3.1 checkpoint weights exist for Qwen3.5-9B (or current validated target) — compact-vocab loader (formerly tracked) is now RESOLVED (`b2766ef47`) |
| **Cross-reference** | PR #18039 LANDED `b2766ef47` (compact-vocab, 2026-05-31); see §2 for ceiling explanation |
| **Next step** | Monitor SafeAILab EAGLE and vLLM repositories for weight releases; import when available + validate accept on Qwen3.5-9B |

---

## §6 Model store — drafter GGUFs (TODO 239, 2026-06-19)

All BF16 GGUFs rebuilt 2026-06-14 from new-schema converter (PR #18039 path A — keys
`eagle3.target_layers` / `eagle3.target_hidden_size` / `eagle3.norm_before_residual`, no double
prefix). Quant ladders rebuilt 2026-06-19 from those new-schema BF16s. Build: `819 (3ff8220f3)` on
ai01 (gfx1103 ROCm). IQ3_M produced without imatrix — architecturally impossible for standalone
eagle3 GGUFs (no `token_embd.weight` → llama-imatrix fails at tensor count check).

### H1 — Qwen3.6-35B-A3B (EAGLE3.1 head, n_embd_tgt=2048)

Path: `/mnt/cephfs/0/Container/models/Qwen/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-eagle3-<q>.gguf`

| Quant   | Size | BPW   | Magic      | Built      |
|---------|------|-------|------------|------------|
| BF16    | 408M | 16.00 | 47475546   | 2026-06-14 |
| Q8_0    | 222M |  8.51 | 47475546   | 2026-06-19 |
| Q4_K_M  | 148M |  3.67 | 47475546   | 2026-06-19 |
| Q3_K_M  | 127M |  3.12 | 47475546   | 2026-06-19 |
| IQ3_M   | 122M |  3.01 | 47475546   | 2026-06-19 |
| Q2_K    | 112M |  2.76 | 47475546   | 2026-06-19 |

Has `fc_norm.weight` tensor (EAGLE3.1 FC normalization). 15 tensors total.

### H2 — Qwen3.5-9B (EAGLE3 head, n_embd_tgt=4096, compact-vocab d2t)

Path: `/mnt/cephfs/0/Container/models/Qwen/Qwen3.5-9B/Qwen3.5-9B-eagle3-<q>.gguf`

| Quant   | Size | BPW   | Magic      | Built      |
|---------|------|-------|------------|------------|
| BF16    | 773M | 16.00 | 47475546   | 2026-06-14 |
| Q8_0    | 416M |  8.51 | 47475546   | 2026-06-19 |
| Q4_K_M  | 272M |  5.58 | 47475546   | 2026-06-19 |
| Q3_K_M  | 234M |  4.69 | 47475546   | 2026-06-19 |
| IQ3_M   | 227M |  4.54 | 47475546   | 2026-06-19 |
| Q2_K    | 206M |  4.10 | 47475546   | 2026-06-19 |

Has `d2t` tensor (compact-vocab 32K→248320 mapping). 14 tensors total.

### H3 — Qwen3.5-35B-A3B (EAGLE3 head, n_embd_tgt=2048)

Path: `/mnt/cephfs/0/Container/models/Qwen/Qwen3.5-35B-A3B/Qwen3.5-35B-A3B-eagle3-<q>.gguf`

| Quant  | Size | BPW   | Magic    | Built      |
|--------|------|-------|----------|------------|
| BF16   | 464M | 16.00 | 47475546 | 2026-06-14 |

No quant ladder yet — target Qwen3.5-35B-A3B is not in active use; build on demand.

### PRISM — Qwen3.6-27B (Ex0bit PRISM drafter, n_embd_tgt=2048, compact-vocab d2t)

Path: `/mnt/cephfs/0/Container/models/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3/Qwen3.6-27B-eagle3-<q>.gguf`

| Quant  | Size | BPW   | Magic    | Built      |
|--------|------|-------|----------|------------|
| BF16   | 1.1G | 16.00 | 47475546 | 2026-06-14 |
| Q8_0   | 569M |  8.51 | 47475546 | 2026-06-19 |
| Q4_K_M | 368M |  5.51 | 47475546 | 2026-06-19 |
| IQ3_M  | 303M |  4.54 | 47475546 | 2026-06-19 |

Has `d2t` tensor (compact-vocab). 14 tensors total. Only 3-quant ladder (no Q3_K_M/Q2_K) — lower
priority drafter.

### Load-test results (2026-06-19, build 819)

All 4 BF16 heads loaded successfully under `llama-gguf r` + `llama-quantize --dry-run`. New-schema
keys confirmed present (`eagle3.target_layers`, `eagle3.target_hidden_size`,
`eagle3.norm_before_residual`). No old double-prefix keys (`eagle3.eagle3.*`) detected.
