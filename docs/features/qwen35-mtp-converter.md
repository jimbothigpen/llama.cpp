# Qwen3.5 / Qwen3.6 MTP Converter

> **Status: Stable** — dense (`qwen35`) and MoE (`qwen35moe`) variants; all three
> converter modes verified on Qwen3.5-9B Q8\_0.

---

## At a glance

| | Value |
|---|---|
| **What it is** | GGUF converter for Qwen3.5/3.6 models with an internal Multi-Token-Prediction (NextN) tail |
| **Supported arches** | `qwen35` (dense), `qwen35moe` (MoE — Qwen3.5-35B-A3B family) |
| **Converter entry** | `python convert_hf_to_gguf.py <hf-model-dir> [--mtp\|--no-mtp]` |
| **Mixin** | `_Qwen35MtpMixin` in `conversion/qwen.py` — shared by dense and MoE classes |
| **Default mode** | Bundled MTP (no flag) — single GGUF includes trunk + MTP head |
| **`--no-mtp`** | Trunk-only GGUF; MTP tensors dropped, `block_count` decremented |
| **`--mtp`** | Standalone draft-head GGUF with `mtp-` prefix; 18 tensors (Qwen3.5-9B) |
| **Speculation binary** | `llama-speculative-simple --spec-type draft-mtp` |
| **Verified accept rate** | **75.6%** — both bundled and split-export modes, Qwen3.5-9B Q8\_0 |
| **MoE** | Qwen3.5/3.6-35B-A3B: expected-OK (mainline mixin design handles MoE expert-merge correctly), not yet independently verified |
| **Upstream credit** | Mainline ggml-org **PR #22673** — am17an (Aman Gupta), "llama + spec: MTP Support"; this fork's mixin is byte-identical to mainline master |

**TL;DR.** Convert any Qwen3.5 or Qwen3.6 HuggingFace checkpoint to GGUF with
built-in Multi-Token-Prediction support. Use the bundled default for single-file
deployment, or `--mtp` to export a compact standalone draft head (2.4 GB for
Qwen3.5-9B Q8\_0) that pairs with a separate trunk for tiered serving.

**Quick start — dense Qwen3.5-9B:**

```bash
# Download HF weights
hf download Qwen/Qwen3.5-9B --local-dir Qwen3.5-9B

# Bundled (default): one GGUF with trunk + MTP head
python convert_hf_to_gguf.py Qwen3.5-9B \
    --outfile Qwen3.5-9B-Q8_0.gguf --outtype q8_0

# Run speculative decode (self-speculative with the bundled GGUF)
llama-speculative-simple \
    -m Qwen3.5-9B-Q8_0.gguf -md Qwen3.5-9B-Q8_0.gguf \
    --spec-type draft-mtp -ngl 99 -ngld 99 -fa on --temp 0 -n 128
```

---

## §1 Provenance

The converter — `_Qwen35MtpMixin` in `conversion/qwen.py` — is mainline
ggml-org, contributed via **PR #22673 "llama + spec: MTP Support"** by
[am17an (Aman Gupta)](https://github.com/am17an). The mixin in this fork is
byte-identical to mainline master (`0821c5fcf`).

**What this fork adds:** the loader side (`src/models/qwen35.cpp`,
`src/models/qwen35moe.cpp`) — the C++ inference backend that reads `nextn_predict_layers`
from the GGUF metadata and routes MTP tensors into the speculative decode path.
The loader is converter-agnostic: a GGUF produced by either the mainline
converter or this fork's converter loads identically.

**Related upstream PRs:** #23398 *(Gemma 4 MTP, WIP)*, #23274 *(StepFun 3.5
MTP)*. Earlier Qwen3.5-MTP converter attempts: #20700, #19937, #20533 (all
closed; superseded by #22673).

---

## §2 Use in production

### Convert

Three CLI modes; flags are mutually exclusive:

| Mode | Flag | Description |
|---|---|---|
| **Bundled** (default) | *(no flag)* | Single GGUF: trunk + MTP head together; `block_count` = trunk layers + MTP layers; `nextn_predict_layers` = N |
| **Trunk-only** | `--no-mtp` | MTP tensors dropped; `block_count` = trunk layers only; no `nextn_predict_layers` key |
| **Draft head only** | `--mtp` | Standalone MTP draft head; compact GGUF with `mtp-` prefix; 18 tensors for Qwen3.5-9B |

```bash
# Bundled (default)
python convert_hf_to_gguf.py Qwen3.5-9B \
    --outfile Qwen3.5-9B-Q8_0.gguf --outtype q8_0

# Trunk only (use as the target model in split-export)
python convert_hf_to_gguf.py Qwen3.5-9B \
    --no-mtp --outfile Qwen3.5-9B-nomtp-Q8_0.gguf --outtype q8_0

# Draft head only (use as -md in split-export)
python convert_hf_to_gguf.py Qwen3.5-9B \
    --mtp --outfile ./  # mtp- prefix added automatically; pass a dir
```

The `--mtp` flag adds the `mtp-` filename prefix automatically when the output
path is a directory. Pass the directory containing the HF weights as
`<hf-model-dir>`.

**Verified output (Qwen3.5-9B, q8\_0):**

| Mode | `block_count` | `nextn_predict_layers` | Tensors | Size |
|---|---|---|---|---|
| Bundled | 33 | 1 | 442 | 9.8 GB |
| `--no-mtp` | 32 | absent | 427 | 9.5 GB |
| `--mtp` draft head | 33 | 1 | 18 | 2.4 GB |

The `--mtp` draft head (18 tensors for Qwen3.5-9B) contains: the full MTP
transformer block at `blk.32` (attention q/k/v/o, q/k norms, attn\_norm,
post\_attention\_norm, FFN gate/up/down), the nextn projection tensors
(`blk.32.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`), and the shared
head (`token_embd`, `output`, `output_norm`).

### Speculative decode

Both bundled and split-export modes use `--spec-type draft-mtp` with
`llama-speculative-simple`:

```bash
# Bundled: pass the same GGUF as both -m and -md
llama-speculative-simple \
    -m Qwen3.5-9B-Q8_0.gguf \
    -md Qwen3.5-9B-Q8_0.gguf \
    --spec-type draft-mtp -ngl 99 -ngld 99 -fa on --temp 0 -n 128 \
    -p "Explain Rayleigh scattering."
```

```bash
# Split-export: trunk as -m, draft head as -md
llama-speculative-simple \
    -m Qwen3.5-9B-nomtp-Q8_0.gguf \
    -md mtp-Qwen3.5-9B-Q8_0.gguf \
    --spec-type draft-mtp -ngl 99 -ngld 99 -fa on --temp 0 -n 128 \
    -p "Explain Rayleigh scattering."
```

Both modes yield identical accept rates. Use chat template flags
(`-cnv`, `--chat-template`, or `--in-prefix`/`--in-suffix`) for chat models.
The `--temp 0` is not required but enables deterministic comparison; the
speculative path operates correctly at any temperature.

**Verified accept rate: 75.6%** (`n_drafted=45`, `n_accept=34`, `n=49`),
Qwen3.5-9B dense Q8\_0, `--spec-type draft-mtp -ngl 99 -ngld 99 -fa on
--temp 0 -n 48`, chat-templated prompt.

### MoE (Qwen3.5/3.6-35B-A3B)

The same three flags apply to `Qwen3_5MoeForCausalLM` /
`Qwen3_5MoeForConditionalGeneration` checkpoints. The mixin's
`_original_block_count` design handles expert-merge block indexing correctly
for MoE variants (expert tensors are remapped before MTP block indexing is
applied). **This path is expected-OK but has not been independently verified
in this fork** — test on a Q4 or Q8 MoE checkpoint before relying on it in
production.

---

## §3 Benefits and potential drawbacks

### Benefits

- **Single-file deployment with no configuration.** The bundled default produces
  one GGUF that works as both the target and draft model — no separate download
  or `-md` management.
- **Compact split-export.** The `--mtp` draft head is ~2.4 GB (Qwen3.5-9B
  Q8\_0) vs ~9.8 GB for the full bundled GGUF — useful when tiered serving or
  VRAM allocation calls for a separate draft model slot.
- **Loader compatibility.** The GGUF format emitted here is byte-level compatible
  with mainline ggml-org builds that have MTP loader support — no fork-specific
  format divergence.
- **75.6% draft accept** on Qwen3.5-9B with `--temp 0`; throughput gain depends
  on hardware and draft depth (`-nd`).

### Potential drawbacks

- **MoE unverified.** Qwen3.5-35B-A3B conversion is expected to work but has
  not been smoke-tested in this fork. Verify before production use.
- **`--mtp` requires a directory output.** Pass a directory path (not a `.gguf`
  filename) to let the converter apply the `mtp-` prefix automatically. Passing
  a file path will produce a file with that exact name — the `mtp-` prefix will
  not be added.
- **Speculative decode requires `--spec-type draft-mtp`.** Do not omit this flag;
  without it the `-md` draft model is treated as a standard causal draft and the
  MTP nextn tensors are unused.
- **MTP head is not a standalone causal LM.** The `--mtp` GGUF will SIGSEGV or
  error if loaded as a full model (`-m`). It is valid only as a draft head
  (`-md`).

---

## §4 How it works under the hood

### `_Qwen35MtpMixin` — three-method design

The mixin overrides three methods on the base `TextModel`:

**`__init__`** — extends `block_count` to include MTP layers
(`hparams["num_hidden_layers"] + hparams["mtp_num_hidden_layers"]`) and
rebuilds the tensor name map for the extended block range. Skipped when
`--no-mtp` (`cls.no_mtp = True`).

**`index_tensors`** — captures `_original_block_count` (the trunk-only layer
count) before delegating to the base class. This value is used during tensor
remapping to compute the correct MTP block index (`trunk_layers + mtp_idx`).

**`filter_tensors`** — the main remap logic, called per-tensor:

1. Delegates to `TextModel.filter_tensors` for the standard language\_model prefix
   unwrap (handles `model.language_model.` and `language_model.` prefixes for VL
   variants with a shared backbone).
2. Strips `model.mtp.` → `mtp.` nesting if present.
3. For `mtp.` tensors: remaps to `model.layers.{trunk_count + mtp_idx}.*` so
   the standard tensor\_map emits the correct `blk.N.nextn.*` GGUF names that
   the C++ loader expects:

   | HF source name | GGUF target name (Qwen3.5-9B, trunk=32) |
   |---|---|
   | `mtp.fc.weight` | `blk.32.nextn.eh_proj.weight` |
   | `mtp.pre_fc_norm_embedding.weight` | `blk.32.nextn.enorm.weight` |
   | `mtp.pre_fc_norm_hidden.weight` | `blk.32.nextn.hnorm.weight` |
   | `mtp.norm.weight` | `blk.32.nextn.shared_head.norm.weight` |
   | `mtp.layers.0.*` | `blk.32.*` (transformer block tensors) |

4. For non-MTP tensors when `--mtp` (`cls.mtp_only = True`): drops all tensors
   except the shared head (`embed_tokens`, `norm`, `lm_head`) — producing the
   compact draft-head-only GGUF.
5. Drops all MTP tensors when `--no-mtp` (`cls.no_mtp = True`).

**`set_gguf_parameters`** — emits `{arch}.nextn_predict_layers = N` into the
GGUF metadata unless `--no-mtp`. This is the key the C++ loader
(`src/models/qwen35.cpp:16`) reads to activate the MTP speculative path.

**`prepare_metadata`** — when `--mtp` and the output path is a directory,
rewrites `fname_out` to prepend `mtp-` to the filename.

### Loader side (C++)

`src/models/qwen35.cpp` and `src/models/qwen35moe.cpp` are converter-agnostic.
The loader keys off:

- `LLM_KV_NEXTN_PREDICT_LAYERS` → `qwen35.nextn_predict_layers` metadata
- Base arch tag `qwen35` / `qwen35moe` (not a separate `qwen35_mtp` arch)
- Tensor names `blk.{N}.nextn.{eh_proj,enorm,hnorm,shared_head_norm}`

A GGUF produced by either this fork's converter or the upstream mainline
converter loads identically — the loader cannot distinguish them.

### Class hierarchy

```
Qwen3_5TextModel(_Qwen35MtpMixin, _Qwen35MRopeMixin, _LinearAttentionVReorderBase)
Qwen3_5MoeTextModel(_Qwen35MtpMixin, _Qwen35MRopeMixin, _LinearAttentionVReorderBase)
    _LinearAttentionVReorderBase(Qwen3NextModel)
        Qwen3NextModel(Qwen2MoeModel)
```

Both dense and MoE classes share the same `_Qwen35MtpMixin`; MoE expert
tensors pass through before MTP block indexing is applied.

---

## §5 Further reading

- **Upstream PR:** [ggml-org/llama.cpp #22673](https://github.com/ggml-org/llama.cpp/pull/22673) — am17an "llama + spec: MTP Support" (source of `_Qwen35MtpMixin`)
- **Related upstream PRs:** [#23398](https://github.com/ggml-org/llama.cpp/pull/23398) (Gemma 4 MTP, WIP) · [#23274](https://github.com/ggml-org/llama.cpp/pull/23274) (StepFun 3.5 MTP)
- **Converter source:** `conversion/qwen.py` — `_Qwen35MtpMixin` (line 537)
- **C++ loaders:** `src/models/qwen35.cpp`, `src/models/qwen35moe.cpp`
- **GGUF metadata constants:** `gguf-py/gguf/constants.py` (`nextn_predict_layers`, `NEXTN_*` tensor names)
- **Speculative decode:** `examples/speculative-simple/` — `llama-speculative-simple` binary
- **Feature index:** [docs/features/README.md](README.md)
