#include "models.h"

// DiffusionGemma (mainline DRAFT PR #24427, lnigam/nvidia-diffusion-gemma).
//
// A gemma4 MoE checkpoint run as a bidirectional block-diffusion denoiser. The transformer body
// is identical to gemma4 (dense shared MLP + routed MoE, dual head dims, iSWA, qk-norm, scale-less
// v-norm, final-logit softcap, per-layer layer_scalar). The only architectural addition is a
// self-conditioning gated MLP applied to the input embedding during the DECODER (denoise) phase:
//
//   soft = (probs @ token_embd^T) * sqrt(n_embd)   // probs = previous step's softmax (0 on step 1)
//   inpL = rms_norm(scaled_embed + sc_mlp(rms_norm(soft)))
//
// Phase is derived at graph-build time from cparams.causal_attn (the runner toggles it via
// llama_set_causal_attn): encoder phase = causal prefill / canvas commit (plain gemma4, no self-
// conditioning); decoder phase = bidirectional canvas denoise (self-conditioned). The unified
// sliding-window KV cache holds the prompt / committed-canvas prefix; the decoder reads it and
// the runner rolls back the in-flight canvas K/V after each step.
//
// This is the minimal generic (backend-agnostic) port: it uses the DENSE self-conditioning path
// (full-vocab probs, transposed embedding computed in-graph). The CUDA fast-sampling kernels,
// sparse top-k gather, fused self-cond embedding, device denoise loop, separate encoder/decoder
// graphs and multimodal prefill from the PR are intentionally omitted (not needed for a coherent
// ROCm/Vulkan/CPU generation).

void llama_model_diffusion_gemma::load_arch_hparams(llama_model_loader & ml) {
    // reuse gemma4 hparam loading (sliding-window pattern, dual head dims, MoE, rope, softcap, ...)
    llama_model_gemma4::load_arch_hparams(ml);

    // the diffusion decoder attends bidirectionally; the runner flips this to causal for the
    // encoder (prefill / commit) phase via llama_set_causal_attn().
    hparams.causal_attn = false;
}

void llama_model_diffusion_gemma::load_arch_tensors(llama_model_loader & ml) {
    // load the shared gemma4 tensors (token embd, attention, dense + MoE FFN, norms,
    // per-layer layer_scalar, output)
    llama_model_gemma4::load_arch_tensors(ml);

    const int64_t n_embd  = hparams.n_embd;
    const int64_t n_ff    = hparams.n_ff();
    const int64_t n_layer = hparams.n_layer();

    // self-conditioning gated MLP: hidden_size -> intermediate_size -> hidden_size
    self_cond_norm = create_tensor(tn(LLM_TENSOR_SC_PRE_NORM, "weight"), { n_embd          }, 0);
    self_cond_gate = create_tensor(tn(LLM_TENSOR_SC_GATE,     "weight"), { n_embd, n_ff     }, 0);
    self_cond_up   = create_tensor(tn(LLM_TENSOR_SC_UP,       "weight"), { n_embd, n_ff     }, 0);
    self_cond_down = create_tensor(tn(LLM_TENSOR_SC_DOWN,     "weight"), { n_ff,   n_embd    }, 0);

    // per-layer encoder output scale: present in the checkpoint (the text encoder shares the
    // decoder weights and differs only by this per-layer scalar). The single-stack decoder graph
    // does not use it, but we create it so the loader consumes the tensor.
    for (int i = 0; i < n_layer; ++i) {
        create_tensor(tn(LLM_TENSOR_ENC_LAYER_OUT_SCALE, "weight", i), { 1 }, 0);
    }
}

// Dense self-conditioning graph input: the previous denoising step's per-position softmax over the
// whole vocabulary, {n_vocab, n_tokens} column-major. Refreshed each decode from the context's
// llama_diffusion_cond (set via llama_set_diffusion_self_cond). Topology is fixed across decoder
// steps, so the graph is reused and only this copy re-runs.
namespace {
class llm_graph_input_diffusion_self_cond : public llm_graph_input_i {
public:
    llm_graph_input_diffusion_self_cond(const llama_diffusion_cond * cond, int64_t n_vocab, int64_t n_tokens)
        : cond(cond), n_vocab(n_vocab), n_tokens(n_tokens) {}

    void set_input(const llama_ubatch * /*ubatch*/) override {
        if (!probs) {
            return;
        }
        const int64_t n = n_vocab * n_tokens;
        if (cond && cond->enabled && (int64_t) cond->probs.size() == n) {
            ggml_backend_tensor_set(probs, cond->probs.data(), 0, (size_t) n * sizeof(float));
        } else {
            // zero self-conditioning (first denoising step / no cond set)
            ggml_backend_tensor_memset(probs, 0, 0, (size_t) n * sizeof(float));
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override {
        return true;
    }

    const llama_diffusion_cond * cond = nullptr;
    int64_t n_vocab  = 0;
    int64_t n_tokens = 0;
    ggml_tensor * probs = nullptr; // F32 [n_vocab, n_tokens]
};
} // namespace

ggml_tensor * llama_model_diffusion_gemma::graph::build_input(bool is_decoder) {
    const auto & dmodel = static_cast<const llama_model_diffusion_gemma &>(model);

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    // important: do not normalize weights for raw embeddings input (e.g. encoded image embeddings)
    inpL = ggml_scale(ctx0, inpL, ubatch.token ? sqrtf(n_embd) : 1.0f);
    cb(inpL, "inp_scaled", -1);

    if (is_decoder && dmodel.self_cond_gate) {
        const int64_t n_vocab = model.tok_embd->ne[1];

        auto inp = std::make_unique<llm_graph_input_diffusion_self_cond>(dcond, n_vocab, n_tokens);
        inp->probs = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_vocab, n_tokens);
        ggml_set_input(inp->probs);
        ggml_tensor * probs = inp->probs;
        res->add_input(std::move(inp));

        // soft = (probs @ token_embd^T) * sqrt(n_embd). mul_mat contracts ne[0], so the embedding
        // is transposed to {n_vocab, n_embd}; dequantize+transpose in-graph (the generic path).
        ggml_tensor * embed_t = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_cast(ctx0, model.tok_embd, GGML_TYPE_F32)));
        ggml_tensor * soft    = ggml_mul_mat(ctx0, embed_t, probs); // {n_embd, n_tokens}
        soft = ggml_scale(ctx0, soft, sqrtf((float) n_embd));
        cb(soft, "self_cond_soft_embd", -1);

        ggml_tensor * scn = build_norm(soft, dmodel.self_cond_norm, nullptr, LLM_NORM_RMS, -1);
        ggml_tensor * sc  = build_ffn(scn,
                dmodel.self_cond_up,   nullptr, nullptr,
                dmodel.self_cond_gate, nullptr, nullptr,
                dmodel.self_cond_down, nullptr, nullptr,
                nullptr, LLM_FFN_GELU, LLM_FFN_PAR, -1);

        // scale-less post-norm of the self-conditioned input embedding
        inpL = ggml_rms_norm(ctx0, ggml_add(ctx0, inpL, sc), hparams.f_norm_rms_eps);
        cb(inpL, "self_cond_input", -1);
    }

    return inpL;
}

llama_model_diffusion_gemma::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        dcond(params.diffusion) {
    ggml_tensor * cur;

    // encoder phase (causal prefill / canvas commit) vs decoder phase (bidirectional self-
    // conditioned denoise). The runner toggles cparams.causal_attn in lockstep; graph reuse
    // already keys on causal_attn so the two phases get distinct graphs.
    const bool is_decoder = !cparams.causal_attn;

    ggml_tensor * inpL = build_input(is_decoder);

    // inp_pos - contains the positions
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv_iswa();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // per-layer token embeddings (gemma3n-style) are not supported in the diffusion port; the
    // diffusiongemma-26B-A4B checkpoint does not use them.
    if (model.per_layer_tok_embd) {
        GGML_ABORT("diffusion-gemma: per-layer token embeddings are not supported");
    }

    for (int il = 0; il < n_layer; ++il) {
        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        GGML_ASSERT(n_embd_head == hparams.n_embd_head_v(il));

        const int64_t n_head    = hparams.n_head(il);
        const int64_t n_head_kv = hparams.n_head_kv(il);

        const float freq_base_l  = model.get_rope_freq_base(cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);
        const int   n_rot_l      = hparams.n_rot(il);

        res->t_layer_inp[il] = inpL;

        // norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_tensor * freq_factors = nullptr;
        if (!hparams.is_swa(il)) {
            // full_attention layers use rope_freqs for proportional rope
            freq_factors = model.layers[il].rope_freqs;
        }

        // Q projection (shared for both non-KV and KV layers), mirrors Gemma4Attention
        ggml_tensor * Qcur;
        {
            Qcur = build_lora_mm(model.layers[il].wq, cur, model.layers[il].wq_s);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);

            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_pos", il);
        }

        // self-attention
        if (hparams.has_kv(il)) {
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur, model.layers[il].wk_s);
            cb(Kcur, "Kcur", il);

            ggml_tensor * Vcur = model.layers[il].wv
                                    ? build_lora_mm(model.layers[il].wv, cur, model.layers[il].wv_s)
                                    : Kcur; // if v_proj is not present, use Kcur as Vcur
            cb(Vcur, "Vcur", il);

            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Vcur = ggml_rms_norm(ctx0, Vcur, hparams.f_norm_rms_eps);

            cb(Kcur, "Kcur_normed", il);
            cb(Vcur, "Vcur_normed", il);

            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, freq_factors, n_rot_l, rope_type, n_ctx_orig, freq_base_l, freq_scale_l,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur, "Kcur_pos", il);

            cur = build_attn(inp_attn, model.layers[il].wo,
                    nullptr, model.layers[il].wo_s, Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    hparams.f_attention_scale, il);
        } else {
            // reuse KV cache of earlier layers
            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr, hparams.f_attention_scale, il);
        }

        if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
            cur  = ggml_get_rows(ctx0,  cur, inp_out_ids);
            inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);
        }
        cur = build_norm(cur,
                model.layers[il].attn_post_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);
        cb(attn_out, "attn_out", il);

        // feed-forward network: dense MLP (shared expert) + routed MoE, summed (mirrors gemma4)
        const bool is_moe_layer = model.layers[il].ffn_gate_inp != nullptr;
        if (is_moe_layer) {
            // MLP (shared exp)
            ggml_tensor * cur_mlp = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_norm_1", il);

            cur_mlp = build_ffn(cur_mlp,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cur_mlp = build_norm(cur_mlp,
                    model.layers[il].ffn_post_norm_1, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_mlp, "ffn_mlp", il);

            // Expert FFN
            ggml_tensor * cur_moe = build_norm(attn_out,
                    model.layers[il].ffn_pre_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_norm_2", il);

            // custom MoE logits calculation (router operates on attn_out, not cur)
            ggml_tensor * tmp = ggml_rms_norm(ctx0, attn_out, hparams.f_norm_rms_eps);
            tmp = ggml_scale(ctx0, tmp, 1.0f / sqrtf((float) n_embd));
            tmp = ggml_mul(ctx0, tmp, model.layers[il].ffn_gate_inp_s);
            ggml_tensor * logits = build_lora_mm(model.layers[il].ffn_gate_inp, tmp); // [n_expert, n_tokens]
            cb(logits, "ffn_moe_logits", il);

            cur_moe = build_moe_ffn(cur_moe,
                    nullptr, // gate_inp
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    nullptr, // exp_probs_b (not used for gemma4)
                    n_expert, n_expert_used,
                    LLM_FFN_GELU, true,
                    1.0f,
                    LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX,
                    il, logits,
                    model.layers[il].ffn_gate_up_exps,
                    model.layers[il].ffn_up_exps_s,
                    model.layers[il].ffn_gate_exps_s,
                    model.layers[il].ffn_down_exps_s);
            cur_moe = build_norm(cur_moe,
                    model.layers[il].ffn_post_norm_2, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur_moe, "ffn_moe", il);

            cur = ggml_add(ctx0, cur_mlp, cur_moe);
            cb(cur, "ffn_moe_combined", il);
        } else {
            cur = build_norm(attn_out,
                    model.layers[il].ffn_norm, nullptr,
                    LLM_NORM_RMS, il);
            cb(cur, "ffn_norm", il);

            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   nullptr, model.layers[il].ffn_up_s,
                    model.layers[il].ffn_gate, nullptr, model.layers[il].ffn_gate_s,
                    model.layers[il].ffn_down, nullptr, model.layers[il].ffn_down_s,
                    nullptr,
                    LLM_FFN_GELU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        }
        cur = build_norm(cur,
                model.layers[il].ffn_post_norm, nullptr,
                LLM_NORM_RMS, -1);
        cb(cur, "ffn_post_norm", il);

        // residual connection
        cur = ggml_add(ctx0, cur, attn_out);

        // layer_scalar
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cur = build_sidecar(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }
    cur = inpL;

    cur = build_norm(cur,
            model.output_norm, nullptr,
            LLM_NORM_RMS, -1);

    if (!cparams.embeddings_nextn_masked && inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);

    if (hparams.f_final_logit_softcapping) {
        cur = ggml_scale(ctx0, cur, 1.0f / hparams.f_final_logit_softcapping);
        cur = ggml_tanh(ctx0, cur);
        cur = ggml_scale(ctx0, cur, hparams.f_final_logit_softcapping);
    }

    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

std::unique_ptr<llm_graph_context> llama_model_diffusion_gemma::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
