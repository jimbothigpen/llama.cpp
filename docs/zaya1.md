# Zyphra ZAYA1-8B (`LLM_ARCH_ZAYA`)

This page documents the in-tree port of Zyphra's ZAYA1-8B hybrid MoE
architecture into llama-yggdrasil. ZAYA1 is the first novel model port
that did not originate in mainline llama.cpp or any of the six tracked
sibling forks; the reference implementations were the unmerged
`Zyphra/vllm@zaya1-pr` and `transformers@zaya1` branches.

- Hugging Face model card: <https://huggingface.co/Zyphra/ZAYA1-8B>
- vLLM reference PR: `Zyphra/vllm@zaya1-pr` —
  `vllm/model_executor/models/zaya.py` (~750 LoC) +
  `vllm/model_executor/layers/mamba/cca.py` (~530 LoC)
- transformers reference branch: `Zyphra/transformers@zaya1`

## What it is

8.4 B-parameter (~760 M active) hybrid MoE. 80 layers alternating two
substantially different blocks:

- **Even layers (40 of 80) — CCA attention.** "Cached Convolutional Attention":
  a Mamba-cached conv preprocessor wrapped around a small GQA attention head.
  - Depthwise 1-D conv (kernel width 2) on a `(2 × n_qk)` channel pool fed
    by `[Q || K]` concatenation, then a grouped 1-D conv (same kernel),
    then L2-norm + per-head k-scale, then NEOX partial-RoPE (`partial_rot=0.5`),
    then GQA attention with `n_head=8 / n_head_kv=2` and `head_dim=128`.
  - Two-stream state: a `conv_state` (kernel-width history) plus a
    `prev_hs` (last hidden-state vector). Both live in the recurrent
    S-stream of the hybrid cache; the R-stream is unused but allocated.
  - Value path is `concat(V1(x), V2(prev_hs))`, where `V1` reads the
    current token and `V2` reads the one-step-delayed hidden state
    (during prefill, this is a 1-token shift; during decode, it comes
    from the recurrent state).
- **Odd layers (40 of 80) — MoE.** 16 experts top-1, with a deep router:
  `down → optional EDA → RMSNorm → GELU MLP (mlp0/mlp2/mlp4) → 17-logit head →
  softmax → drop MoD-skip → top-1 over 16 experts`. The 17th logit gates the
  MoD (mixture-of-depths) "skip the whole expert call" path.
- **EDA (Expert Depth Averaging) second hidden stream.** Routers carry a
  D-wide state across layers — a second residual alongside the main
  hidden-stream, fed back into the next layer's router input.
- **Per-layer learned ResidualScaling.** Each layer has `res_scale_hs`
  and `res_scale_res` scale+bias pairs that rescale the residual stream
  before/after the sub-layer.
- **Tokenizer.** Gemma family (262 144 tokens; `bos=2`, `eos=106`).
  Context length 131 072. Tied LM head (`output.weight` is aliased to
  `tok_embd.weight` at load).

Stock-and-already-supported components reused as-is: RMSNorm, SwiGLU,
NEOX partial-RoPE, GQA, fused gate_up MoE expert pattern, the Gemma
tokenizer (262 144 vocab).

## Status of this implementation

| Capability | Status |
|---|---|
| Convert Hugging Face checkpoint → GGUF | ✅ |
| Model load (`llama_model_load_from_file`) | ✅ |
| Single-sequence forward / `llama_decode` | ✅ — coherent generation |
| Multi-sequence (`n_seq_max > 1`) forward — default `llama-perplexity`, `llama-server` | ✅ |
| Quant matrix (F16, Q8_0, Q5_K_M, IQ4_XS-imat-guq5k) — single-seq | ✅ within ±0.5% F16 |
| Quant matrix — multi-seq | ✅ within ±0.5% of single-seq for all 4 quants |
| HF parity (single-seq vs Zyphra `transformers@zaya1`) | top-1 identical; KL 0.007 bits, cosine 0.986 vs BF16 HF |
| ROCm runtime (gfx1150) | ✅ |
| Vulkan runtime (RDNA3 / RDNA3.5) | ✅ |
| ROCm runtime (gfx1102 / gfx1103) | ⛔ compiles but dead per Tensile/hipBLAS GEMM gaps |

## Conversion

```bash
# Download the Zyphra HF release:
hf download Zyphra/ZAYA1-8B --local-dir ZAYA1-8B

# Convert (the converter is registered against the Zaya1ForCausalLM
# architecture and mirrors the Gemma tokenizer at 262 144 tokens):
python3 convert_hf_to_gguf.py ZAYA1-8B \
    --outfile zaya1-8B-F16.gguf --outtype f16
```

GGUF metadata keys (selected; full schema in `gguf-py/gguf/constants.py`):

- `zaya.ssm.conv_kernel` — 2
- `zaya.attention.partial_rotary_factor` — 0.5
- `zaya.attention.head_count` / `.head_count_kv` — 8 / 2
- `zaya.attention.head_dim` — 128
- `zaya.expert_count` / `.expert_used_count` — 16 / 1
- `zaya.expert_gating_func` — `NONE` (router outputs are already softmax probabilities)
- `zaya.rope.freq_base` — 5e6 (NEOX scheme)

## Tensor schema (per-block tensors for `blk.0..79`)

CCA layers (even `il`):

| GGUF tensor | Shape | Source name |
|---|---|---|
| `blk.{i}.attn_q.weight` | `[hidden, n_head*head_dim]` | `layers.{i}.self_attn.q_proj.weight` |
| `blk.{i}.attn_k.weight` | `[hidden, n_head_kv*head_dim]` | `layers.{i}.self_attn.k_proj.weight` |
| `blk.{i}.cca_val_proj1.weight` | `[hidden, head_dim]` | `layers.{i}.self_attn.cca.val_proj1.weight` |
| `blk.{i}.cca_val_proj2.weight` | `[hidden, head_dim]` | `layers.{i}.self_attn.cca.val_proj2.weight` |
| `blk.{i}.attn_output.weight` | `[n_head*head_dim, hidden]` | `layers.{i}.self_attn.o_proj.weight` |
| `blk.{i}.cca_conv_dw.weight` | `[d_conv=2, n_qk]` | `layers.{i}.self_attn.cca.conv1d_dw.weight` |
| `blk.{i}.cca_conv_dw.bias` (optional) | `[n_qk]` | `layers.{i}.self_attn.cca.conv1d_dw.bias` |
| `blk.{i}.cca_conv_grp.weight` | `[d_conv=2, n_qk/n_groups, n_qk]` | `layers.{i}.self_attn.cca.conv1d_grp.weight` |
| `blk.{i}.cca_conv_grp.bias` | `[n_qk]` | `layers.{i}.self_attn.cca.conv1d_grp.bias` |
| `blk.{i}.cca_k_scale.weight` | `[n_head_kv]` | `layers.{i}.self_attn.cca.k_scale` |
| `blk.{i}.attn_norm.weight` | `[hidden]` | `layers.{i}.input_layernorm.weight` |
| `blk.{i}.res_scale_hs.{weight,bias}` | `[hidden]` | `layers.{i}.res_scale_hs.{weight,bias}` |
| `blk.{i}.res_scale_res.{weight,bias}` | `[hidden]` | `layers.{i}.res_scale_res.{weight,bias}` |

MoE layers (odd `il`):

| GGUF tensor | Shape | Source name |
|---|---|---|
| `blk.{i}.zaya_router_down.{weight,bias}` | `[hidden, router_hidden]` | `layers.{i}.router.down_proj.{weight,bias}` |
| `blk.{i}.zaya_router_mlp{0,2,4}.weight` | router MLP | `layers.{i}.router.mlp.{0,2,4}.weight` |
| `blk.{i}.zaya_router_norm.weight` | `[router_hidden]` | `layers.{i}.router.norm.weight` |
| `blk.{i}.ffn_gate_up_exps.weight` | `[hidden, 2*intermediate, n_expert]` | fused gate+up across 16 experts |
| `blk.{i}.ffn_down_exps.weight` | `[intermediate, hidden, n_expert]` | `layers.{i}.experts.{e}.down_proj.weight` |

Plus the standard top-level `token_embd.weight`, `output_norm.weight`, and
top-level `top_res_scale_{hs,res}` final scale pair. `output.weight` is **not
present** in the GGUF — the loader aliases it to `tok_embd.weight` (tied LM
head); quantizing `tok_embd` therefore directly hits LM-head precision.

## Quantization

ZAYA1 has four load-bearing fragility points that interact with
`llama-quantize`'s default precision picker:

1. **`cca_conv_dw` kernel** — depthwise conv kernel width is 2; per-channel
   weight count is small enough that any k-quant on these tensors collapses
   the conv response.
2. **`cca_conv_grp` kernel** — same constraint.
3. **Tied LM head** — `tok_embd` doubles as the output head; precision loss
   propagates to logits.
4. **Deep router MLP** — `zaya_router_mlp{0,2,4}` is a small MLP whose output
   drives top-1 expert selection; k-quant noise on this destabilizes routing.

Recommended override-tensor list (P5 ship recipe):

```
cca_conv_dw\.weight=f16
cca_conv_grp\.weight=f16
zaya_router_down\.weight=q8_0
zaya_router_mlp0\.weight=q8_0
zaya_router_mlp2\.weight=q8_0
zaya_router_mlp4\.weight=q8_0
token_embd\.weight=q8_0
ffn_gate_up_exps\.weight=q5_K
```

The `ffn_gate_up_exps=q5_K` override (NOT `ffn_down_exps`) is what makes
the difference between `IQ4_XS-imat` (PPL 32.86) and `IQ4_XS-imat-guq5k`
(PPL 31.95) — IQ-quants beat K-quants on this MoE except on the fused
gate+up where the IQ-quant degrades router-relevant activations. Apply via
`llama-quantize --override-tensor`:

```bash
OVERRIDES="cca_conv_dw\.weight=f16,cca_conv_grp\.weight=f16,zaya_router_down\.weight=q8_0,zaya_router_mlp0\.weight=q8_0,zaya_router_mlp2\.weight=q8_0,zaya_router_mlp4\.weight=q8_0,token_embd\.weight=q8_0,ffn_gate_up_exps\.weight=q5_K"

llama-quantize --imatrix imatrix.dat --override-tensor "$OVERRIDES" \
    zaya1-8B-F16.gguf zaya1-8B-IQ4_XS-imat-guq5k.gguf IQ4_XS
```

## Variants shipped

PPL gates measured on ai01 Vulkan, 80 chunks, c=512, wikitext-2-raw-test,
default multi-seq (`-np 4`, n_seq_max=4):

| Quant | Size | Multi-seq PPL | vs F16 30.5270 |
|---|---|---|---|
| F16 | 17 GB | 30.5270 | — |
| Q8_0 | 9.4 GB | 30.5231 | -0.01% |
| Q5_K_M | 6.5 GB | 29.9468 | -1.9% (in-noise; PPL improves slightly) |
| IQ4_XS-imat-guq5k | 5.1 GB | 32.0073 | +4.9% |

Single-seq vs multi-seq parity is within ±0.5% for all four quants (i.e. the
`n_seq_max>1` path is bit-equivalent within F16 chunk-noise to single-seq).

## Multi-seq fix history (P5 → P6 → P7)

The initial port (P3, `f274fe3fc` on the original `feature/zaya1-port`) ran
end-to-end for `n_seqs=1` only; default-flag `llama-perplexity`
(`n_parallel=4`) crashed at `sched_reserve` on `ggml_reshape_3d`. P6
identified three bugs by inspection; with all three fixed, multi-seq PPL was
still ~466K vs the expected ~30, indicating at least one more bug not
findable by inspection. P7 added activation-dump instrumentation
(`examples/eval-callback` was extended with `-np N` multi-seq batch split and
`--tensor-filter REGEX` to dump named tensors at layer 0; see
[CHANGELOG.md](../CHANGELOG.md) for details) and located bug #4 by diffing
seq-0 of multi-seq vs single-seq.

The four bugs and their fixes:

| # | Site | Bug | Fix |
|---|---|---|---|
| 1 | `src/models/zaya.cpp` `prev_hs` reshape | strided `view_2d` of `cca_state` is non-contig for `n_seqs>1`, trips `ggml_reshape_3d` assertion | `ggml_cont(prev_hs)` before `ggml_reshape_3d` |
| 2 | `src/models/zaya.cpp` cca depthwise conv | `ggml_conv_1d_dw` internally `reshape_4d`s b to `(L,1,IC,N)` which sets `b->ne[3]=n_seqs`, tripping `im2col`'s `b->ne[3]==1` assertion | replace with `ggml_ssm_conv` (natively sequence-batched k=2 s=1 p=0 depthwise; `cca_conv_dw` GGUF layout already matches `c=(d_conv, d_inner)`) |
| 3 | `src/models/zaya.cpp` `QKraw_t` build | `cont(transpose(QKraw))+reshape_3d(., n_seq_tokens, n_qk, n_seqs)` silently scrambles channel/seq for `n_seqs>1` (mapping coincidentally correct for `n_seqs=1`); **same bug present in Zyphra's reference fork** | reshape `QKraw` directly to `(n_qk, n_seq_tokens, n_seqs)` (memory-preserving because ubatch is seq-major per `llama_batch_allocr::split_equal`), then `permute(1,0,2,3) + cont` |
| 4 | `ggml/src/ggml.c` `ggml_conv_1d` (latent mainline bug) | final `mul_mat(im2col_2d, a_2d) → reshape_3d(., OL, OC, N)` reinterprets the `(N*OL, OC)` mul_mat output as `(OL, OC, N)`, but flat layout only matches semantically when `N=1` (seq-stride collapse) or `OC=1`; for `N>1` and `OC>1` channels and seqs get cross-mixed | local `conv_1d_grouped_multiseq` lambda in `src/models/zaya.cpp` with corrected `reshape_3d(OL, N, OC) + permute(0,2,1,3) + cont`; **no ggml-core changes** |

Future yggdrasil models that call `ggml_conv_1d` / `ggml_conv_1d_grouped`
with `n_seqs > 1` must either copy the lambda or discuss a ggml-core fix
with the user first (per the mainline-fidelity policy).

## Why this port is in-tree (not future-watch)

ZAYA1 was originally cataloged in `yggdrasil-context/future-watch.md` as a
HARD-BUT-DOABLE 4–8-week port, deferred because none of the planned
yggdrasil phases share kernel surface with CCA / EDA and the vLLM reference
PR was unmerged. Trigger #4 of that ledger entry — user production need —
fired off-tracker, and the port was built end-to-end in seven phases
(P1 converter, P2 arch + stub, P3 graph, P4 HF parity, P5 single-seq quant
matrix, P6 multi-seq investigation, P7 multi-seq fix + ship). Original
scope estimate of 4–8 weeks was approximately right.
