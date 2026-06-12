# TODO 204 — Block-level GET_ROWS decode inverse for InnerQ×TCQ — Design & Findings

**Date:** 2026-06-12 · **Worker:** opus (204-innerq-blockdecode-2026-06-12) · **Branch:** `feature/204-innerq-blockdecode-2026-06-12` (base `318a54cf0`)

## TL;DR — disposition: ESCALATE (block-decode op is redundant, not implemented)

The block-level decode op TODO 204 asks for is **mathematically redundant** with the
Q-side `scale_inv` compensation that the fork already ships (and which **236-L2**,
merged into `main` earlier today as `a87187c4d` / `f27e8d9cb`, made robust). It would
add per-decode compute and a new graph op for **zero** correctness or accuracy gain.
Moreover, the non-FA TCQ-K path the op was meant to repair **is not runtime-reachable
today**, so there is no live mis-decode to fix. Recommend the orchestrator either
**close 204 as superseded by 236-L2**, or **re-scope** it (see §5) if the goal is to
*enable* a non-FA TCQ path for FA-less backends — that work needs K→F32 materialization,
not a block-decode, and would still reuse the existing Q-side correction.

---

## §1 The math the task is built on

InnerQ applies a per-channel diagonal scale in the original basis **before** the rotation,
then TCQ-quantizes in the rotated domain (`ggml/src/ggml-cuda/set-rows.cu`,
`ggml/src/ggml-cuda/turbo-quant.cuh::turbo_rotate_forward`):

```
rotate_forward(x) = SIGNS2 ⊙ FWHT( SIGNS1 ⊙ x )            # orthonormal
stored            = trellis( rotate_forward( scale ⊙ K ) )  # InnerQ active
decode ≈ R        = rotate_forward( scale ⊙ K )             # any decode path
```

`rotate_forward` is orthonormal (±1 sign diagonals + normalized Hadamard), so it
**preserves dot products**: `⟨rotate_forward(a), rotate_forward(b)⟩ = ⟨a, b⟩`.

The escalation that spawned 204 (2026-06-06, §-FLAG-A) correctly rejected a *per-bin
scalar inverse inside the dequant*: a diagonal does **not** commute with FWHT
(`S·F ≠ F·S'`), so no `c[t]` recovers `⟨Q,K⟩` from a single rotated bin `t`. That
rejection is sound — but it does not imply a block-decode is *needed*, because the
correction is already done on the **Q side**.

### Two ways to recover `⟨Q,K⟩` from `R`

- **A) Q-side compensation — WHAT THE FORK SHIPS** (`src/llama-graph.cpp:2544-2546`,
  `ggml_turbo_wht_innerq`):
  `Qrot = rotate_forward(scale_inv ⊙ Q)`,  score `= ⟨Qrot, R⟩`
  `= ⟨scale_inv ⊙ Q, scale ⊙ K⟩ = ⟨Q, K⟩` (since `scale_inv ⊙ scale = 1`). ✅

- **B) Block-level decode — WHAT TODO 204 PROPOSES:**
  `Korig = scale_inv ⊙ rotate_inverse(R) = K`, re-rotate, dot with plain `rotate_forward(Q)`
  `= ⟨Q, K⟩`. ✅ — *identical result, more compute, and requires removing A to avoid double-correction.*

## §2 Synthetic numerical proof (the validation the starter mandated)

`tests/innerq_blockdecode_math_proof.py` models the exact rotation with **injected
non-unit per-channel scale** (`scale ∈ ~0.37..2.7`), 2000 random trials:

```
=== EXACT (no quantization) — pure inverse-math check ===
  A) Q-side compensation    max rel err vs dot(Q,K): 2.473e-12   -> RECOVERS
  B) block-level decode     max rel err vs dot(Q,K): 1.869e-12   -> RECOVERS
  C) per-bin scalar inverse max rel err vs dot(Q,K): 2.668e+03   -> UNSOUND (as expected)

=== WITH 3-bit quantization in the rotated domain (TCQ-like) ===
  A) Q-side comp   rms=4.3805
  B) block-decode  rms=4.3805
  ratio B/A rms = 1.0000  (≈1 -> no quantization-accuracy advantage)
```

Both A and B recover `⟨Q,K⟩` to fp tolerance and carry **statistically identical**
quantization error. The per-bin inverse (which §-FLAG-A rejected) fails catastrophically,
confirming the original analysis — but A already solves the problem B was meant to solve.

## §3 The non-FA TCQ-K path is not runtime-reachable today

The task says "wire it where the InnerQ-on + non-FA GET_ROWS path currently mis-decodes."
There is no such live path:

1. **Q-side correction already covers non-FA.** `ggml_turbo_wht_innerq` is applied in
   `build_attn` (`llama-graph.cpp:2544-2548`) **before** `build_attn_mha`, which is where
   the FA / non-FA branch happens. So both paths get `rotate_forward(scale_inv ⊙ Q)`.
2. **Symmetric TCQ (K=V=TURBOQ3_TCQ, the 236-L2 config) is FA-forced.**
   `src/llama-context.cpp:3764` returns `nullptr` ("V cache quantization requires
   flash_attn") whenever `type_v` is quantized + FA disabled. So the canonical InnerQ×TCQ
   config cannot run non-FA at all.
3. **K=TCQ + V=F16 + non-FA would abort in mul_mat.** `get_k()`
   (`llama-kv-cache.cpp:1696`) returns a **raw TCQ view**; `build_attn_mha` then does
   `ggml_mul_mat(k_tcq, q)`. TURBOQ?_TCQ has **no MMVQ case** (`mmvq.cu`), **no MMQ case**,
   and **no `to_fp16`/`to_fp32`** (`convert.cu`) — so the dispatch falls to cuBLAS-dequant
   and hits the missing-converter assert. There is no functional non-FA decode to "fix."
   (TCQ `vec_dot` exists **only** for fattn-vec — the FA path — `fattn-common.cuh:931+`.)

## §4 Why 236-L2 is the actual closure of §-FLAG-A's intent

236-L2 (`a87187c4d`) found the real InnerQ×TCQ ON-path defect: the Q-side `scale_inv`
tensor was being re-zeroed by `ggml_backend_buffer_clear` between PPL chunks (one-shot
sync), so `ggml_turbo_wht_innerq` multiplied Q by 0 → attention collapse → PPL 17.34.
The fix re-syncs `scale_inv` every decode and scales **K only** (V un-rotation has no
scale_inv). Result: `turboq3_tcq×turboq3_tcq, TURBO_INNERQ=256: 17.34 → 6.90`. That fix
makes the **Q-side correction (approach A) robust** — i.e. it closes the InnerQ×TCQ
correctness gap that §-FLAG-A was a facet of, for the path that actually runs (FA).

## §5 Recommendation (orchestrator decision)

- **Option 1 — CLOSE 204 as superseded by 236-L2 (recommended).** The dot-product
  correction is mathematically complete via Q-side `scale_inv`; the block-decode adds cost
  for no gain. Keep this branch's findings doc + proof as the record.
- **Option 2 — RE-SCOPE to "enable non-FA TCQ K".** If the orchestrator wants TCQ KV to
  work on FA-less backends (old CUDA arch, certain Vulkan), the missing piece is a
  **K→F32 get_rows materialization** in the non-FA graph (TCQ get_rows *is* already
  `supports_op`-true, `ggml-cuda.cu:5411`), after which the **existing** Q-side correction
  makes it correct — still **no block-decode needed**. This is a different, larger TODO
  with its own perf/gate story; not a correctness completion.

## §6 What this branch contains

- `tests/innerq_blockdecode_math_proof.py` — runnable, no GPU, ~1s. Reproduces §2.
- `docs/design/todo-204-innerq-blockdecode-findings.md` — this doc.
- **No source change** to any kernel/graph file. Default + FA + non-FA paths are
  byte-identical to `318a54cf0` (nothing touched).
