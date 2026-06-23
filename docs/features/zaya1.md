# Zyphra ZAYA1-8B (`LLM_ARCH_ZAYA`)

> **Status: Stable** — CPU, ROCm gfx1150, and Vulkan RDNA3/RDNA3.5 backends.
>
> **Note:** ROCm on gfx1102/gfx1103 compiles but does not run correctly
> (Tensile/hipBLAS GEMM gap); use CPU or Vulkan on those GPUs instead.

---

## At a glance

| | Value |
|---|---|
| Architecture | `LLM_ARCH_ZAYA` ("zaya") |
| Parameters | 8.4B total · ~760M active |
| Layer count | 80 (even = CCA, odd = MoE) |
| Context length | 131 072 tokens |
| Tokenizer | Gemma family, 262 144 vocab |
| Active experts | top-1 of 16 |
| Backends | CPU, ROCm gfx1150, Vulkan RDNA3/RDNA3.5 |
| Mainline llama.cpp | **Not present** — fork-only port |

**TL;DR.** ZAYA1-8B is an 8.4B-parameter hybrid MoE from Zyphra. 80 layers
alternate CCA (Mamba-cached convolutional attention) on even layers and
16-expert top-1 MoE on odd layers. Quantization requires a small
`--override-tensor` list to protect the CCA convolution kernels and the deep
expert router.

**Quick start:**

```bash
# Download
hf download Zyphra/ZAYA1-8B --local-dir ZAYA1-8B

# Convert
python3 convert_hf_to_gguf.py ZAYA1-8B \
    --outfile zaya1-8B-F16.gguf --outtype f16

# Quantize (see §2 for the full override list and rationale)
OVERRIDES="cca_conv_dw\.weight=f16,cca_conv_grp\.weight=f16,\
zaya_router_down\.weight=q8_0,zaya_router_mlp0\.weight=q8_0,\
zaya_router_mlp2\.weight=q8_0,zaya_router_mlp4\.weight=q8_0,\
token_embd\.weight=q8_0,ffn_gate_up_exps\.weight=q5_K"

llama-quantize --imatrix imatrix.dat --override-tensor "$OVERRIDES" \
    zaya1-8B-F16.gguf zaya1-8B-IQ4_XS-imat-guq5k.gguf IQ4_XS

# Run
llama-server -m zaya1-8B-IQ4_XS-imat-guq5k.gguf -ngl 99 --no-mmap
```

---

## §1 Provenance

ZAYA1-8B is a novel model architecture from
[Zyphra](https://huggingface.co/Zyphra/ZAYA1-8B). This fork provides the
first production-ready ggml/llama.cpp inference backend; the architecture is
not present in mainline ggml-org llama.cpp.

The port was built against the Zyphra reference implementations:
`Zyphra/vllm@zaya1-pr`
(`vllm/model_executor/models/zaya.py`, ~750 LoC;
`vllm/model_executor/layers/mamba/cca.py`, ~530 LoC) and
`Zyphra/transformers@zaya1`. Zyphra also maintains a llama.cpp branch
(`Zyphra/llama.cpp@CCA`, remote `zyphra`, `src/models/zaya.cpp`) used as an
additional cross-reference; this fork's backend is independently implemented and
is the production path. None of these references is in mainline ggml-org
llama.cpp. See the canonical [PROVENANCE.md](PROVENANCE.md).

### Differences from the Zyphra reference

- **Multi-sequence forward.** The Zyphra reference supports single-sequence
  inference only. This port validates both single-seq and multi-seq
  (`n_seq_max > 1`) paths, as required by default-flag `llama-perplexity` and
  `llama-server`. CCA recurrent state handling and the CCA convolution path were
  reworked for sequence-batched execution. A latent multi-sequence reshape bug
  in `ggml_conv_1d` (present in both the Zyphra reference and ggml-core) is
  worked around locally in `src/models/zaya.cpp` without ggml-core changes.
- **Backend coverage.** CPU, ROCm gfx1150, and Vulkan RDNA3/RDNA3.5 are
  validated. The Zyphra reference targets PyTorch/CUDA only.
- **Quantization recipe.** A tested override-tensor list and perplexity gate
  results are provided (see §2).
- **Tied LM head.** `output.weight` is not stored in the GGUF; the loader
  aliases it to `tok_embd.weight` at load time, matching the reference weight
  layout.

---

## §2 Use in production

### Convert

```bash
hf download Zyphra/ZAYA1-8B --local-dir ZAYA1-8B

python3 convert_hf_to_gguf.py ZAYA1-8B \
    --outfile zaya1-8B-F16.gguf --outtype f16
```

The converter is registered for the `Zaya1ForCausalLM` architecture and uses
the Gemma tokenizer at 262 144 tokens.

### Quantize — override-tensor list

Default blanket low-bpw quantization breaks coherence on ZAYA1. The following
override list is mandatory:

| Tensor pattern | Override | Reason |
|---|---|---|
| `cca_conv_dw\.weight` | `f16` | Depthwise conv kernel width is 2; k-quant block alignment collapses the conv response at this size |
| `cca_conv_grp\.weight` | `f16` | Same constraint |
| `zaya_router_down\.weight` | `q8_0` | Deep top-1 router MLP — routing is noise-sensitive |
| `zaya_router_mlp{0,2,4}\.weight` | `q8_0` | Deep top-1 router MLP — routing is noise-sensitive |
| `token_embd\.weight` | `q8_0` | Tied LM head — precision loss propagates directly to output logits |

`ffn_gate_up_exps` (NOT `ffn_down_exps`) is the primary load-bearing low-bpw
lever. Adjust this tensor's quant type to target a given file size; leave
`ffn_down_exps` at the base quantization level.

```bash
OVERRIDES="cca_conv_dw\.weight=f16,cca_conv_grp\.weight=f16,\
zaya_router_down\.weight=q8_0,zaya_router_mlp0\.weight=q8_0,\
zaya_router_mlp2\.weight=q8_0,zaya_router_mlp4\.weight=q8_0,\
token_embd\.weight=q8_0,ffn_gate_up_exps\.weight=q5_K"

llama-quantize --imatrix imatrix.dat --override-tensor "$OVERRIDES" \
    zaya1-8B-F16.gguf zaya1-8B-IQ4_XS-imat-guq5k.gguf IQ4_XS
```

### Perplexity gates

Wikitext-2-raw-test, 80 chunks, c=512, multi-seq (`-np 4`).
Single-seq vs multi-seq parity is within ±0.5% for all quants.

| Quant | Size | PPL | vs F16 |
|---|---|---|---|
| F16 | 17 GB | 30.5270 | — |
| Q8_0 | 9.4 GB | 30.5231 | −0.01% |
| Q5_K_M | 6.5 GB | 29.9468 | −1.9% (in-noise; sweet spot) |
| IQ4_XS-imat-guq5k | 5.1 GB | 32.0073 | +4.9% |

The Q5_K_M result (−1.9%) is within the noise floor of wikitext-2 chunk
variance and represents no meaningful quality loss.

### Backend support

| Backend | Status |
|---|---|
| CPU | ✅ validated |
| ROCm gfx1150 | ✅ validated |
| Vulkan RDNA3 / RDNA3.5 | ✅ validated |
| ROCm gfx1102 / gfx1103 | ⛔ compiles but non-functional (Tensile/hipBLAS GEMM gap) |

---

## §3 Benefits & potential drawbacks

### Benefits

- **Novel architecture in production.** ZAYA1's hybrid CCA+MoE design is not
  available in any other GGUF-based inference stack.
- **Efficient active parameter count.** ~760M active out of 8.4B total; the
  MoD skip logit can bypass entire expert calls.
- **Full multi-sequence support.** Both single-seq and multi-seq paths are
  parity-validated (within ±0.5% PPL).
- **Strong HF parity.** Top-1 token match vs. the Zyphra reference; KL
  0.007 bits, cosine similarity 0.986 vs. BF16.

### Potential drawbacks

- **Override-tensor list required for quantization.** Blanket low-bpw
  quantization silently degrades coherence; the override list in §2 is not
  optional.
- **Not loadable in mainline llama.cpp.** `LLM_ARCH_ZAYA` is not registered
  in ggml-org mainline. GGUF files produced here will not load in an
  unmodified mainline build.
- **ROCm gfx1102/gfx1103 not functional.** See backend table above.

### Benchmark matrix

*TBD (pending benchmark)*

| # | Quant | Size | PPL | TG (t/s) | PP (t/s) | Backend |
|---|---|---|---|---|---|---|
| 1 | F16 | 17 GB | 30.5270 | TBD | TBD | TBD |
| 2 | Q8_0 | 9.4 GB | 30.5231 | TBD | TBD | TBD |
| 3 | Q5_K_M | 6.5 GB | 29.9468 | TBD | TBD | TBD |
| 4 | IQ4_XS-imat-guq5k | 5.1 GB | 32.0073 | TBD | TBD | TBD |

---

## §4 How it works under the hood

### Layer structure

80 layers alternating two block types (`src/models/zaya.cpp:29-31`):

- **Even layers (0, 2, … 78) — CCA.** Mamba-cached convolutional attention.
- **Odd layers (1, 3, … 79) — MoE.** 16-expert top-1 with deep router.

Graph dispatch branches at `zaya.cpp:244` (`if (il%2==0)` → CCA; else →
MoE at `:400+`). Architecture registered as `LLM_ARCH_ZAYA "zaya"` in
`llama-arch.cpp:138`; model implementation is `src/models/zaya.cpp` (487 LoC,
`llama_model_zaya`); dispatch entry in `llama-model.cpp:291`.

### CCA (Cached Convolutional Attention) — even layers

CCA wraps a small GQA attention head with a Mamba-style conv preprocessor.

**Projection.** Full `attn_q` / `attn_k` projections (`zaya.cpp:82-83`) —
standard full-rank, not MLA low-rank.

**Conv path.** Depthwise 1-D conv (kernel width 2) on the `[Q ∥ K]`
channel pool via `ggml_ssm_conv` (`:353`), then a grouped 1-D conv
(`cca_conv_grp`), then L2-norm + per-head `cca_k_scale`.

**RoPE + GQA.** NEOX partial-RoPE (`partial_rot=0.5`); GQA
`n_head=8 / n_head_kv=2`, `head_dim=128`.

**Dual-stream value.**
`V = concat(V1(x), V2(prev_hs))` — `V1` reads the current token, `V2` reads
the one-step-delayed hidden state (recurrent state during decode; 1-token
shift during prefill).

**Recurrent state.** The conv state (`conv_state`, kernel-width history) and
`prev_hs` (delayed hidden state) live in the S-stream of the hybrid cache
(`get_s_l(il)`). The R-stream is allocated but unused. ZAYA1 is NOT listed
in `is_recurrent` — the hybrid memory path is the correct one.

### MoE — odd layers

16 experts, top-1 routing with a 17th MoD-skip logit:

```
zaya_router_down → optional EDA residual → RMSNorm →
GELU MLP (mlp0 / mlp2 / mlp4) → 17-logit head →
softmax → drop MoD-skip → top-1 over 16 experts
```

The 17th logit gates Mixture-of-Depths skip routing (bypass the entire expert
call). Deep router tensors: `zaya_router_{down,mlp0,mlp2,mlp4}` +
`zaya_router_eda_scale`. Expert weights are fused: `ffn_gate_up_exps`
(gate+up across all 16 experts) and `ffn_down_exps`.

### EDA (Expert Depth Averaging)

A D-wide second residual stream (`zaya_router_eda_scale`) carries router state
across layers, feeding back into the next layer's router input as an auxiliary
hidden stream alongside the main residual.

### Hybrid memory (`llama_memory_hybrid`)

ZAYA1 uses `llama_memory_hybrid` (`llama-model.cpp:2046+`);
`LLM_ARCH_ZAYA` is registered in `llm_arch_is_hybrid()`. CCA recurrent state
lives in the S-stream; attention KV lives in the standard KV cache.

### Tied LM head

`output.weight` is registered `TENSOR_NOT_REQUIRED` and aliases
`tok_embd.weight` if absent (`zaya.cpp:46-50`). Quantizing `tok_embd`
therefore directly sets LM-head precision — do not drop it below Q8_0.

### Per-layer residual scaling

Each layer carries `res_scale_hs` and `res_scale_res` scale+bias pairs that
rescale the residual stream before/after the sub-layer.

---

## §5 Further reading

- **Zyphra model card:** [Zyphra/ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B)
- **Zyphra reference:**
  `Zyphra/vllm@zaya1-pr` · `Zyphra/transformers@zaya1`
- **GGUF metadata constants:** `gguf-py/gguf/constants.py` (`zaya.*` keys)
- **Feature index:** [docs/features/README.md](README.md)
