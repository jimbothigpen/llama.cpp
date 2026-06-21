# EpiCache P2 + P3 implementation plan (code-grounded, current main)

**Date:** 2026-06-20 · Worker: epicache-convqa-harness-baseline-plan ·
**Base:** main `0ae8c2bb46` (epicache P1 merged) ·
**Source:** EpiCache arXiv 2509.17396 (no reference code; from-paper).
**Companion:** conv-QA accuracy harness in `tools/epicache-convqa/` (this branch).

> All file:line below were **re-verified on `0ae8c2bb46`** (the feasibility recon
> mapped against the older `89e2a491c3`; lines have moved — these are current).

---

## 0. Where we are

| stage | paper | status on main |
|---|---|---|
| P1 block-wise prefill bounding | Alg.1 stage 3a | **MERGED** (`0ae8c2bb46`), FA-safe, reuses TriAttn proxy scorer |
| P2 layer-wise budget (Eq.10) | stage 2 | **not built** — `.tria` already carries `layer_budget_scales[]`; Eq.10 is an alternate populator |
| P3 episodic multi-cache + routing | stages 1 + 4 | **not built** — needs embed model + K-Means + per-episode cache + router |
| Faithful Eq.3 scorer (non-FA) | stage 3 (Eq.3) | **not built** — now in scope per user directive 2026-06-20 16:00 |

The merged P1 substitutes EpiCache's Eq.3 attention-score scorer with
TriAttention's FA-safe z-score proxy (`tria_score_kv_head`). P3 is the part that
actually delivers the conv-QA *accuracy* win (episodic routing); P2 is a cheap
accuracy refinement; the faithful Eq.3 scorer is the from-paper fidelity option.

---

## 1. Current-main code map (re-verified)

| concern | file:line | note |
|---|---|---|
| EpiCache prefill hook | `src/llama-context.cpp:4470-4484` | calls `tria_maybe_score` (decode) then `epicache_prefill_evict` when `epicache_prefill_enabled(g_tria_rt)` and batch is multi-token (prefill) |
| EpiCache prefill evict | `src/triattention-runtime.c:1039-1060` (`epicache_prefill_evict`) | sets `epicache_prefill_active`, calls `tria_maybe_score`, logs proof line |
| prefill budget override | `src/triattention-runtime.c:247-255` | `if (epicache_prefill_active && prefill_evict_budget>0) budget = prefill_evict_budget` |
| interval-gate bypass in prefill | `src/triattention-runtime.c:197-202` | scores every block during prefill |
| proxy scorer | `src/triattention-runtime.c` `tria_score_kv_head` / `tria_maybe_score` (~185-820) | z-score proxy, GQA-aggregated, FA-safe |
| physical compaction | `src/triattention-runtime.c` `tria_compact_kv` (decl `.h:117`) + bridge `src/triattention-bridge.cpp` | position-preserving |
| **layer_budget_scales** (P2 target) | struct `src/triattention.h:53`; loaded `src/triattention.c:75-93`; **consumed** `src/triattention-runtime.c:395, 605, 759` | `layer_weight = layer_budget_scales[li] / layer_weight_mean` |
| **Eq.3 materialized softmax** (non-FA) | `src/llama-graph.cpp:2294` (`kq = ggml_mul_mat(k,q)`) → `:2328` (`kq = ggml_soft_max_ext(...)`) | only when `use_flash_attn == false` (`:2237-2238`) |
| FA path (no scores materialized) | `src/llama-graph.cpp:2237-2263` (`ggml_flash_attn_ext`) | online softmax; per-key scores never formed |
| **server prompt cache** (P3 E-blob target) | `struct server_prompt` + `struct server_prompt_cache` `tools/server/server-task.h:~580-647`; impl `tools/server/server-task.cpp:1598-1780` | `server_prompt = {tokens, data (KV state blob), checkpoints}`; `_cache.states` is a `std::list<server_prompt>` |
| prompt save/load glue | `tools/server/server-context.cpp:143-176` (`prompt_save`/`prompt_load`), alloc at `:1346` | |
| KV state serialize/restore | `llama_state_seq_get_data` `src/llama-context.cpp:4403`; `..._set_data` `:4407`; ext variants `:986/:993` (llama.h) | per-seq KV → byte blob and back (episodic swap primitive) |
| seq remove | `llama_memory_seq_rm` `src/llama-context.cpp:4242` (decl `include/llama.h:810`) | |
| embeddings | `--embedding` / `--pooling {none,mean,cls,last}` `common/arg.cpp:1954-1960`; `llama_get_embeddings*` | for the Qwen3-0.6B router embedder |

---

## 2. P2 — layer-wise budget via Eq.10

**Goal:** allocate per-layer KV budget by layer sensitivity instead of a uniform
scale. EpiCache Eq.10 sets layer ℓ's budget ∝ its profiled attention-mass
concentration (sensitive layers keep more). Our scorer **already multiplies the
per-token score by a per-layer weight** at `triattention-runtime.c:605` and `:759`
(`layer_budget_scales[li] / layer_weight_mean`). So P2 = compute Eq.10 weights and
write them into `rt->stats->layer_budget_scales[]`.

**Two ways to populate (pick A; B is the from-paper variant):**

- **A (offline, recommended first):** extend the `.tria` calibration generator to
  emit Eq.10 weights into the existing `layer_budget_scales[]` slot (already loaded
  at `triattention.c:75-93`). **Zero runtime change** — the consumer at `:605/:759`
  already applies them. New work lives in the tria-gen tooling, not llama core.
- **B (online, faithful Eq.10):** during the first prefill block, profile each
  layer's attention-mass dispersion and overwrite `layer_budget_scales[]` in
  `tria_maybe_score` before the budget split at `:240-256`. Requires materialized
  per-layer attention stats → only available on the non-FA path (§4) or via a cheap
  separate reduction. Defer until Eq.3 path exists.

**Insertion point (A):** none in llama core. (B): new helper called once from
`triattention-runtime.c` ~`:235` guarded by a new `epicache_layer_budget` flag.

**New flag:** `LLAMA_EPICACHE_LAYER_BUDGET=1` (env, runtime) for variant B; variant
A needs only a `.tria` rebuilt with Eq.10 weights (no flag).

**Validation gate (harness):** `p1` arm with Eq.10 `.tria` vs `p1` with uniform
`.tria` at the same budget M — F1 should be **≥** uniform (non-regression), ideally
+1–3 pts on multi-hop (category 1). Run via `run_baselines.sh` swapping `TRIA=`.

---

## 3. P3 — episodic multi-cache + semantic routing

This is the accuracy payload. Four sub-pieces.

### 3a. Embedder (stages 1 & 4 input)
- **Model:** Qwen3-0.6B (paper's `f_embed`; we have `models/Qwen3-0.6B/Qwen3-0.6B-F16.gguf`).
- **How loaded:** a **second `llama_context`** created with `embeddings = true`,
  `pooling_type = MEAN` (or `LAST`) — config surface already exists at
  `common/arg.cpp:1954-1960`. In the server, add a `ctx_embed` alongside `ctx_tgt`
  (mirror the existing `ctx_dft` draft-context plumbing in `server-context.cpp`).
  Reuse `llama_get_embeddings_seq()` after a decode of the segment text.
- **Why 2nd context, not the embedding endpoint:** routing must run *in-process* per
  query with no HTTP round-trip; a resident `ctx_embed` is the low-overhead path
  ("<5%" budget in the paper).

### 3b. Offline episodic clustering (stage 1)
- Segment history into `w_embed = 4`-utterance chunks; embed each chunk.
- **K-Means (k-means++), E = 4 episodes.** Implement as a **tiny standalone C++
  preproc** (no new llama-core dep) — ~150 LOC, input = chunk embeddings (floats),
  output = per-chunk episode id + E centroids + medoid chunk per episode. Runs once
  per conversation ("<1 min"). Suggested home: `tools/epicache-cluster/` (sibling to
  the harness), invoked by the server at conversation-ingest time or precomputed.
- Output artifact: `episodes.json` = `{centroids:[E][d], assignments:[n_chunks],
  medoid_chunk:[E]}`.

### 3c. Per-episode compressed cache (stage 3, per episode)
- For each episode e: prefill the **episode's segments** (or the whole history with
  the episode's medoid as the eviction query) under P1 block-wise bounding to budget
  M, then **snapshot** the resulting KV via `llama_state_seq_get_data`
  (`llama-context.cpp:4403`) into a blob.
- **Storage = extend `server_prompt_cache`** (`server-task.h:624`). Today
  `states` is a `std::list<server_prompt>` keyed by token-prefix LCP. Add an
  **episodic map** `std::vector<server_episode>` where
  `server_episode = { std::vector<float> centroid; server_prompt cache; }` — i.e.
  reuse `server_prompt{tokens,data,checkpoints}` as the per-episode blob, plus its
  centroid. New methods `alloc_episode()/load_episode(e)` mirror the existing
  `alloc()/load()` at `server-task.cpp:1620/1675`.
- Single-seq-per-slot is sufficient (recon §5): one active episode blob restored
  into the slot's seq at a time via `llama_state_seq_set_data` (`:4407`); no
  multi-seq-per-slot refactor.

### 3d. Episode-matched retrieval / router (stage 4, Eq.7)
- Per query q: embed q (3a) → `e* = argmax_e cos(q, centroid_e)` (Eq.7) → if `e* !=`
  current resident episode, `seq_rm` (`:4242`) the slot + `load_episode(e*)`
  (`llama_state_seq_set_data`). If same as last turn, no reload.
- **Insertion:** in `server-context.cpp` request handling, before prompt
  decode — compute route, swap blob, then proceed with the normal decode path.

**New flags:** `--epicache-episodes E` (default 4), `--epicache-embed-model PATH`
(Qwen3-0.6B), `--epicache-wembed N` (default 4), `--epicache-budget M`.

**Validation gates (harness):**
1. **Router correctness:** standalone unit — feed known-episode queries, assert
   argmax-cos picks the right episode (no model needed).
2. **Accuracy:** `eq3`/`p3` arm vs `full` and vs `p1` on `run_baselines.sh`.
   Target: P3 **recovers most of the P1→full F1 gap** (paper: near-full at 4–6×;
   ≥ +N pts over `plain-evict`). The exact gap to close is the
   **P1-vs-full number this harness measures** (see baseline table in brief).
3. **Overhead:** routing + swap < ~5% added latency per query (paper claim).

---

## 4. Faithful Eq.3 importance scorer (non-FA) — per user directive 2026-06-20

**Directive (supersedes recon "FA-safe proxy only"):** implement the faithful
Eq.3 scorer; running it **without `-fa`** is acceptable (note + accept the perf
cost). Eq.3 = `score(x_i) = max_t softmax(QK^T)[t, i]` over patched-prompt query
tokens t (max attention any query token pays to key i).

**Where the data exists:** the materialized `kq = softmax(QK^T)` is built **only on
the non-FA path** at `src/llama-graph.cpp:2328` (guarded by `use_flash_attn` at
`:2237-2238`). The FA path (`:2254 ggml_flash_attn_ext`) uses online softmax and
**never forms per-key scores** — hence the incompatibility.

**Implementation:**
1. Add `cparams.flash_attn=false` requirement for this arm (server launches with
   `-fa off`). Document the throughput cost.
2. After `:2328`, when an `epicache_eq3` flag is set, **reduce `kq` over the query
   dim with max** → a `[n_kv]` per-key importance vector per (layer, head); aggregate
   across heads (mean or max) → feed into the same eviction selection that
   `tria_compact_kv` consumes (replace the proxy `global_scores` with these). This
   keeps compaction/budget logic unchanged; only the **score source** swaps.
3. Gather scores via a small `ggml` reduction node (cheap relative to the QK^T it
   reuses) tagged for readback, or via the existing capture-buffer accessors
   (`tria_get_*_capture`, `.h:79-80`).

**New flag:** `LLAMA_EPICACHE_EQ3=1` (forces non-FA; swaps scorer source).

### 4b. Can Eq.3 be made FA-compatible? (investigation — recommend)
Three options, in increasing fidelity loss / decreasing perf cost:

- **(i) Separate cheap scoring kernel** *(recommended)*: don't ask FA to emit
  scores; run a **small dedicated `max_t softmax(QK^T)` reduction** over just the
  patched-prompt query tokens (few medoid tokens, not the whole prefill) against all
  keys, as a side computation while the main attention stays FA. Cost is
  `O(n_medoid · n_kv)` per layer — tiny vs full prefill. **Keeps `-fa on` for the
  model; computes faithful Eq.3 only for scoring.** This is the best of both worlds
  and avoids the global non-FA penalty.
- **(ii) FA kernel exposing per-key stats:** patch `ggml_flash_attn_ext` to
  optionally accumulate per-key max-attention into an aux output (like the existing
  `oscar_res` hook at `:2263`). High effort (CUDA/HIP/Vulkan/CPU variants); only if
  (i) proves insufficient.
- **(iii) FA-available approximation** `max_t Attn(x_t→x_i)` from rowmax/logsumexp
  surrogates: cheap but lossy; this is essentially what the TriAttn proxy already
  approximates, so it adds little over P1.

**Recommendation:** ship the non-FA faithful Eq.3 (§4) first to *validate accuracy
ceiling*; then implement **(i)** as the production path (faithful scores, FA-on
model), measuring both against `full`/`p1` on the harness. Only pursue (ii) if (i)'s
medoid-only scoring underperforms full-prefill Eq.3.

---

## 5. Phasing, flags, gates (summary)

| phase | new code (est.) | new flags | harness gate |
|---|---|---|---|
| P2 (Eq.10, variant A) | ~120 (tria-gen) | none (rebuilt `.tria`) | `p1`(Eq.10 tria) F1 ≥ `p1`(uniform) |
| Eq.3 non-FA (§4) | ~150 (graph + runtime) | `LLAMA_EPICACHE_EQ3` | `eq3` arm F1 vs `full`/`p1` |
| Eq.3 FA-compat (i) (§4b) | ~250 (scoring kernel) | `LLAMA_EPICACHE_EQ3_FA` | `eq3-fa` F1 ≈ `eq3`, `-fa on` TPS |
| P3a clustering | ~150 (`tools/epicache-cluster/`) | — | router unit test |
| P3b embedder | ~120 (server `ctx_embed`) | `--epicache-embed-model` | embeddings sane |
| P3c per-episode cache | ~300 (`server_prompt_cache` ext) | `--epicache-episodes` | snapshot/restore round-trip |
| P3d router | ~120 (server route+swap) | `--epicache-budget` | **accuracy: recover P1→full gap** |

Total ≈ **1,200–1,500 LOC** + Qwen3-0.6B embedder dependency. Order:
**P2 → Eq.3(non-FA) → P3a/b/c/d → Eq.3 FA-compat**. Each lands behind its flag with
`LLAMA_EPICACHE` default OFF (byte-unaffected OFF build, as P1).

---

## 6. Risks / caveats

- **TODO 242 (seq_rm crash):** P3's per-turn `seq_rm` + state restore exercises the
  exact path that segfaulted under agentic load (recon §B3). **P3 multi-cache swap
  must be gated behind / land after TODO 242 is fixed**, or be proven safe in the
  single-seq-per-slot conv-QA path first.
- **agentic ≠ conv-QA (recon §B6):** EpiCache's episodic-routing win is largest with
  *many distinct topics per conversation* (LoCoMo/LongMemEval). Single-task agentic
  loops have fewer episodes → smaller routing win. The block-prefill memory bound
  (P1) transfers regardless; set expectations for routing (P3) accordingly.
- **Eq.3 perf:** the non-FA faithful path (§4) costs the fused-attention speedup
  globally; only acceptable as a validation/ceiling arm. The production path is the
  FA-compatible medoid-only scoring kernel §4b(i).
- **No reference code / no oracle:** from-paper port; gate **only** on conv-QA
  accuracy (this harness) + PPL non-regression, not on diff-against-upstream.
