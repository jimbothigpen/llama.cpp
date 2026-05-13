# llama-yggdrasil GGUF type-ID contract — v1

Authoritative assignments for `enum ggml_type` and `enum llama_ftype` in
llama-yggdrasil. Every cherry-pick from a contributing fork MUST renumber to
match this table before landing on a yggdrasil branch.

This document is normative. If reality disagrees with this document, fix the
code, not the document — unless the contract itself is being revised, in
which case bump the version at the top and call out the change.

## Why this exists

Five contributing forks have independently extended `ggml_type` in mutually
incompatible ways. Concrete known collisions:

| Slot | Mainline | TheTom (TQ-KV HEAD) | TheTom (alpha-scaling, stale) | Buun (master) | Carlosfundora (1-bit-turbo) | Turbo-tan (main) | ik_llama (main) |
|---|---|---|---|---|---|---|---|
| 41 | `Q1_0` | `Q1_0` (mainline-aligned) | `TURBO3_0` | `Q1_0` | (gap) | `Q1_0` | `Q1_0_G128` |
| 42 | — | `TURBO2_0` | `TURBO4_0` | `TURBO3_0` | `Q1_0` | — | — |
| 43 | — | `TURBO3_0` | `TURBO2_0` | `TURBO4_0` | `Q1_0_g128` | — | — |
| 44 | — | `TURBO4_0` | `TQ3_1S` | `TURBO2_0` | `PLANAR3_0` | **`TQ3_1S`** (different layout from TheTom's) | — |
| 45 | — | `TQ3_1S` | `TQ4_1S` | `TURBO3_TCQ` | `PLANAR4_0` | — | — |
| 46 | — | `TQ4_1S` | — | `TURBO2_TCQ` | `ISO3_0` | `TQ3_4S` | — |
| 47 | — | — | — | — | `ISO4_0` | — | — |

**Note (2026-05-12, recon/06):** TheTom HEAD branch is `feature/turboquant-kv-cache`; alpha-scaling is now stale and superseded. Between alpha-scaling and TQ-KV, the TURBO_*_0 trio was reordered (TURBO2/3/4 = 42/43/44 instead of 43/41/42) and the TQ_1S types shifted up by one slot. This affects the cherry-pick recipe's "FROM" mapping but not yggdrasil's own slot assignments (60–95 zone unchanged).

GGUF files produced by any one fork are silently misread by any other.
Cherry-picking without renumbering would propagate this hazard into yggdrasil.

## Partitioning policy

ggml_type is a `uint32_t`-valued enum. We partition the address space into
fixed-purpose zones:

| Range | Purpose | Owner |
|---|---|---|
| 0–41 | Mainline core types | upstream ggml-org/llama.cpp |
| 42–59 | **Mainline growth reserve** — DO NOT USE | upstream (future) |
| 60–95 | **Yggdrasil extensions** — new types from contributing forks | this project |
| 96–199 | ik_llama compatibility zone | preserve ik_llama assignments |
| 200–255 | Row-interleaved / packed variants | preserve ik_llama R-suffix layout |

**Why a mainline growth reserve.** Mainline added `Q1_0` at slot 41 after
several forks had already placed their own types at 41. Mainline will keep
adding types in this range. Yggdrasil refuses to play tug-of-war for these
slots. We accept whatever mainline assigns; we never compete.

**Why a high-number zone (60–95).** Same reason ik_llama did it: collisions
with mainline's next 10 additions become impossible.

**Why preserve ik_llama's 96+ assignments.** Pragmatic: ik_llama GGUFs are
the most numerous fork-quantized files in the wild. Renumbering them would
require either (a) a loader compatibility shim or (b) forcing users to
re-quantize. Preserving the IDs lets us read existing ik_llama GGUFs
without modification.

## Yggdrasil extension zone (60–95) — canonical assignments

### 60–65: TurboQuant KV family (source: TheTom `feature/turboquant-kv-cache`)

Source-fork canonical branch confirmed by recon `recon/06-thetom-branches.md` (2026-05-12). Earlier drafts of this document named `feature/alpha-scaling` as the source; alpha-scaling has since been superseded by TQ-KV (1 substantive unique commit, the optional `TURBO_ALPHA` env var knob). All "TheTom name (renamed)" slot numbers below reflect TQ-KV HEAD as of `5aeb2fdbe`.

| Slot | Yggdrasil name | TheTom name (renamed) | Description |
|---|---|---|---|
| 60 | `GGML_TYPE_TURBOQ2_0` | `TURBO2_0` (42) | 2-bit PolarQuant, no QJL |
| 61 | `GGML_TYPE_TURBOQ3_0` | `TURBO3_0` (43) | 2-bit PolarQuant + 1-bit QJL |
| 62 | `GGML_TYPE_TURBOQ4_0` | `TURBO4_0` (44) | 3-bit PolarQuant + 1-bit QJL |
| 63–64 | reserved | | future TurboQuant variants |
| 65 | `GGML_TYPE_TURBOQ3_NATIVE` | turbo-tan `TQ3_0` (200) | 3-bit native KV (turbo-tan); see "Row-interleaved / packed variants" |

Symbol prefix: `turboq_` (kernels), `TURBOQ_` (constants). The `Q` suffix
disambiguates from the `TURBO*_0` collisions in contributing forks.

### 66–71: TCQ KV family (source: buun `master`)

| Slot | Yggdrasil name | Buun name (renamed) | Description |
|---|---|---|---|
| 66 | `GGML_TYPE_TURBOQ2_TCQ` | `TURBO2_TCQ` (46) | TCQ k=2, L=8, 256 states |
| 67 | `GGML_TYPE_TURBOQ3_TCQ` | `TURBO3_TCQ` (45) | TCQ k=3, Viterbi-decoded |
| 68–71 | reserved | | future TCQ variants |

Symbol prefix: `turboq_tcq_`. TCQ extends the TurboQuant family
conceptually but uses Viterbi-coded trellises instead of scalar codebooks.

### 72–79: RotorQuant KV family (source: carlosfundora `1-bit-turbo`)

| Slot | Yggdrasil name | Carlosfundora name (renamed) | Description |
|---|---|---|---|
| 72 | `GGML_TYPE_RQ_PLANAR3_0` | `PLANAR3_0` (44) | Givens-rotation, 8 centroids, sign-mag split |
| 73 | `GGML_TYPE_RQ_PLANAR4_0` | `PLANAR4_0` (45) | Givens-rotation, 16 centroids |
| 74 | `GGML_TYPE_RQ_ISO3_0` | `ISO3_0` (46) | Hadamard, 8 centroids, sign-mag split |
| 75 | `GGML_TYPE_RQ_ISO4_0` | `ISO4_0` (47) | Hadamard, 16 centroids |
| 76–79 | reserved | | future RotorQuant variants |

Symbol prefix: `rq_` (kernels), `RQ_` (constants). Disambiguates from buun's
TCQ and TheTom's TurboQuant.

### 80–85: WHT weight family (source: TheTom `feature/turboquant-kv-cache`)

Originally drafted against `pr/tq4-weight-compression`; that branch is fully subsumed by `feature/turboquant-kv-cache` (zero unique commits by subject — see `recon/06-thetom-branches.md`). All slot numbers below reflect TQ-KV HEAD as of `5aeb2fdbe`.

| Slot | Yggdrasil name | TheTom name (renamed) | Description |
|---|---|---|---|
| 80 | `GGML_TYPE_WHT3_0` | `TQ3_1S` (45) | WHT-rotated 8-level Lloyd-Max, block_size=32 |
| 81 | `GGML_TYPE_WHT4_0` | `TQ4_1S` (46) | WHT-rotated 16-level Lloyd-Max, block_size=32 |
| 82–85 | reserved | | future WHT variants |

Symbol prefix: `wht_`. The `TQ` prefix in TheTom's naming collided with
turbo-tan's RaBitQ TQ3 family; renaming to `WHT` reflects the actual
transform (Walsh-Hadamard) and breaks the collision.

### 86–91: RaBitQ weight family (source: turbo-tan `main`)

| Slot | Yggdrasil name | Turbo-tan name (renamed) | Description |
|---|---|---|---|
| 86 | `GGML_TYPE_RBQ3_1S` | `TQ3_1S` (44) | RaBitQ 3-bit, two half-block scales |
| 87 | `GGML_TYPE_RBQ3_4S` | `TQ3_4S` (46) | RaBitQ 3-bit, four u8 per-8 scales (4.0 bpw) |
| 88–91 | reserved | | future RaBitQ variants |

Symbol prefix: `rbq_`. Disambiguates from TheTom's WHT family.

### 92–95: reserved for unanticipated weight quants

Held open. New contributing forks would land here.

## ik_llama compatibility zone (96–199) — preserved IDs

ik_llama's chosen IDs are preserved verbatim. Renumbering would break
existing ik_llama-quantized GGUFs. The full list of preserved assignments:

| Slot | Name | Source |
|---|---|---|
| 97 | `Q8_0_X4` | interleaved 8-bit, ×4 packing |
| 98 | `Q8_1_X4` | interleaved 8-bit (signed-bias), ×4 packing |
| 99 | `Q8_2_X4` | interleaved 8-bit (variant), ×4 packing |
| 133 | `Q6_0` | revived legacy format |
| 134 | `IQ1_BN` | BitNet 1-bit |
| 135 | `IQ2_BN` | BitNet 2-bit |
| 136 | `Q8_K64` | K-quant with 64-element blocks |
| 137–141 | `IQ2_K`, `IQ3_K`, `IQ4_K`, `IQ5_K`, `IQ6_K` | "K" family |
| 144 | `IQ4_KS` | IK-quant small |
| 145 | `IQ2_KS` | |
| 146 | `IQ4_KSS` | IK-quant small-small |
| 147–151 | `Q8_K16`, `Q8_K32`, `Q8_KR8`, `Q8_K128`, `Q8_KV` | Q8 K-block variants |
| 152 | `IQ5_KS` | |
| 153–155 | `IQ2_KT`, `IQ3_KT`, `IQ4_KT` | trellis family (dormant; preserve IDs anyway) |
| 156 | `IQ3_KS` | |
| 157 | `IQ2_KL` | |
| 158 | `IQ1_KT` | trellis 1-bit (dormant) |

### Special case: `Q1_0_G128`

Three forks place "Q1_0 with 128-element groups" at three different slots:

- ik_llama: `Q1_0_G128 = 41` (collides with mainline `Q1_0`)
- Carlosfundora: `Q1_0_g128 = 43` (lowercase `g`)
- Mainline: no equivalent

**Resolution:** Place `GGML_TYPE_Q1_0_G128 = 96` in yggdrasil (first slot of
ik_llama compat zone). This means:

- Existing ik_llama GGUFs marked with type 41 must be re-quantized OR loaded
  through a fork-detection shim (see "Reader compatibility" below).
- Existing carlosfundora GGUFs marked with type 43 must be re-quantized OR
  loaded through a fork-detection shim.

Slot 96 is the canonical yggdrasil ID for this type going forward.

Symbol: `q1_0_g128_` (kernels), `Q1_0_G128` (constant). Uppercase `G` follows
ik_llama's convention.

## Row-interleaved / packed variants (200–255)

Preserve ik_llama's R-suffix layout verbatim:

| Slot | Name |
|---|---|
| 202 | `Q4_0_R8` |
| 206 | `Q5_0_R4` |
| 208 | `Q8_0_R8` |
| 210–214 | `Q2_K_R4`, `Q3_K_R4`, `Q4_K_R4`, `Q5_K_R4`, `Q6_K_R4` |
| 216–223 | `IQ2_XXS_R4`, `IQ2_XS_R4`, `IQ3_XXS_R4`, `IQ1_S_R4`, `IQ4_NL_R4`, `IQ3_S_R4`, `IQ2_S_R4`, `IQ4_XS_R8` |
| 229 | `IQ1_M_R4` |
| 230 | `BF16_R16` |

Slots 200–201, 203–205, 207, 209, 215, 224–228, 231–255 are reserved for
future packed-variant additions.

**Turbo-tan's `TQ3_0 = 200`** (KV-cache only) is NOT preserved at 200 —
slot 200 is in our packed-variant zone, and the type name suggests it's
yet another KV variant best placed adjacent to the TurboQuant KV family.
**Assignment:** `GGML_TYPE_TURBOQ3_NATIVE = 65` (last slot of TurboQuant
zone), renaming to disambiguate from the WHT/TCQ families. If turbo-tan's
TQ3_0 turns out to not be a TurboQuant variant at all, revisit.

## llama_ftype assignments

`enum llama_ftype` values for the `MOSTLY_*` variants are derived from
ggml_type via:

```
LLAMA_FTYPE_MOSTLY_<NAME> = <next-available-slot>
```

Assignment order follows ggml_type numeric order, starting at the first
unused mainline ftype slot (currently 41 after mainline's `Q1_0=40`).

llama_ftype assignments are mechanical; this document does not enumerate
them. They are settled at the moment each ggml_type lands. The
**implementation** that adds a new ggml_type MUST also add the
corresponding `LLAMA_FTYPE_MOSTLY_*` in the same commit.

## Reader compatibility for legacy fork GGUFs

We will NOT silently re-interpret legacy fork-specific GGUFs. If a user
brings a GGUF quantized with (e.g.) buun's `TURBO3_0=42`, the yggdrasil
loader will fail with an explicit error:

```
unrecognized ggml_type 42 in <file.gguf>. This appears to be a buun-fork
GGUF; yggdrasil places TurboQuant3 at type 61. Re-quantize with
`llama-quantize <model> turboq3_0`.
```

A future optional loader flag (`--legacy-fork-ids=<fork-name>`) MAY
implement on-the-fly remapping. This is not part of the v1 contract.

## Policy for adding new types

When yggdrasil grows a new quant type (whether ported from a fork or
invented):

1. Allocate the lowest-numbered available slot in the appropriate family
   zone (60–95). If the family zone is full, expand into 92–95 reserves
   before considering 96+ (which is owned by ik_llama compat).
2. The naming MUST follow the family's symbol prefix.
3. The `ggml_type`, `llama_ftype`, type-traits row, and CPU vecdot must
   land in the same commit. Partial landings are rejected.
4. Update this document in the same PR.
5. The PPL regression harness must include the new type before the PR can
   merge. No exceptions.

## Open issues

- **TheTom's `TURBO3_0` shipped existing GGUFs at slot 41.** Production
  TheTom-quantized models exist with this ID. The yggdrasil v1 reader
  rejects them. A `--legacy-fork-ids=thetom` flag is a likely v2 addition.

- **Mainline may at some point claim slots 42–59 with types whose names
  collide with our renames.** E.g., mainline could add a future
  `GGML_TYPE_PLANAR3_0` unrelated to RotorQuant. Our policy is: rename
  ours, never theirs. The `RQ_` and `TURBOQ` prefixes are already there
  to absorb this.

- **GGUF metadata format** (the part outside the type-ID enum) may also
  need fork-specific keys (e.g., turbo-tan's WHT rotation tables, buun's
  TCQ codebook indices). Out of scope for this document — to be addressed
  in a separate GGUF_METADATA_KEYS.md.

## Version log

- **v1** (2026-05-12) — initial contract. Authored before any cherry-picks
  land. Authoritative for Phase 0+.
