// DFlash drafter model loader (hidden-state cross-attention speculative decode).
// Provenance: runtime ported from buun (remote `buun`,
//   github.com/spiritbuun/buun-llama-cpp). GGUF converter
//   (conversion/dflash_draft.py) is from Anbeeld/beellama.cpp (MIT); drafter
//   weights are the z-lab DFlash family. See docs/features/dflash.md.
#include "models.h"

#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// llama_model_dflash_draft — model class methods
// ---------------------------------------------------------------------------

void llama_model_dflash_draft::load_arch_hparams(llama_model_loader & ml) {
    auto & hparams = this->hparams;

    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE,           hparams.dflash_block_size,        false);
    ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID,        hparams.dflash_mask_token_id,     false);
    ml.get_key(LLM_KV_DFLASH_N_TARGET_FEATURES,    hparams.dflash_n_target_features, false);

    {
        const std::string key = ml.llm_kv(LLM_KV_DFLASH_TARGET_LAYER_IDS);
        const int64_t kid = gguf_find_key(ml.metadata, key.c_str());
        if (kid >= 0) {
            const size_t n = gguf_get_arr_n(ml.metadata, kid);
            hparams.dflash_n_target_layers = std::min((uint32_t) n, (uint32_t) 8);
            const void * data = gguf_get_arr_data(ml.metadata, kid);
            for (uint32_t i = 0; i < hparams.dflash_n_target_layers; ++i) {
                hparams.dflash_target_layer_ids[i] = ((const uint32_t *) data)[i];
            }
        }
    }

    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa, false);
    if (hparams.n_swa > 0) {
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
        ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer(), false);
    }
}

void llama_model_dflash_draft::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},          0);

    dflash_fc          = create_tensor(tn(LLM_TENSOR_DFLASH_FC,          "weight"), {(int64_t) hparams.dflash_n_target_features, n_embd}, 0);
    dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, 0);

    for (int i = 0; i < (int) hparams.n_layer(); ++i) {
        auto & layer = layers[i];

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd, n_embd_k_gqa},           0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd, n_embd_v_gqa},           0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd},  0);

        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash_draft::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<build_graph>(*this, params);
}

// ---------------------------------------------------------------------------
// DFlash drafter custom graph input
// Holds the target hidden states, context positions, and asymmetric attention mask
// ---------------------------------------------------------------------------

// Default cross-attention context length. Env GGML_DFLASH_MAX_CTX overrides.
static int64_t dflash_max_cross_ctx() {
    static const int64_t val = [] {
        const char * e = getenv("GGML_DFLASH_MAX_CTX");
        return e ? (int64_t) atoi(e) : (int64_t) 4096;
    }();
    return val;
}

class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_cross * cross, int64_t ctx_len, int64_t n_block, uint32_t n_swa)
        : cross(cross), ctx_len(ctx_len), n_block(n_block), n_swa(n_swa) {}

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * target_hidden   = nullptr; // [n_target_features, ctx_len]
    ggml_tensor * pos_ctx         = nullptr; // [ctx_len]
    ggml_tensor * kq_mask         = nullptr; // [ctx_len + n_block, n_block, 1, 1]
    ggml_tensor * kq_mask_cnv     = nullptr;
    // Only allocated when hparams.is_swa_any()
    ggml_tensor * kq_mask_swa     = nullptr;
    ggml_tensor * kq_mask_swa_cnv = nullptr;

    const llama_cross * cross;
    int64_t ctx_len;
    int64_t n_block;
    uint32_t n_swa;
};

void llm_graph_input_dflash::set_input(const llama_ubatch * ubatch) {
    // ygg llama_cross only has a single flat v_embd (no per-seq map, no GPU tape)
    const float * src_data  = nullptr;
    int64_t       src_n_enc = 0;

    if (cross && !cross->v_embd.empty()) {
        src_data  = cross->v_embd.data();
        src_n_enc = cross->n_enc;
    }

    // Sliding window: if source has more tokens than ctx_len, take the most recent
    const int64_t n_real  = src_n_enc > 0 ? src_n_enc : 0;
    const int64_t n_copy  = std::min(n_real, ctx_len);
    const int64_t win_off = (n_real > ctx_len) ? (n_real - ctx_len) : 0;

    if (target_hidden && src_data && n_copy > 0) {
        const int64_t n_feat       = cross ? cross->n_embd : 0;
        const size_t copy_bytes    = (size_t) n_feat * (size_t) n_copy * sizeof(float);
        const size_t tensor_bytes  = ggml_nbytes(target_hidden);
        const size_t actual_bytes  = std::min(copy_bytes, tensor_bytes);

        ggml_backend_tensor_set(target_hidden, src_data + win_off * n_feat, 0, actual_bytes);
        if (copy_bytes < tensor_bytes) {
            ggml_backend_tensor_memset(target_hidden, 0, copy_bytes, tensor_bytes - copy_bytes);
        }
    } else if (target_hidden) {
        ggml_backend_tensor_memset(target_hidden, 0, 0, ggml_nbytes(target_hidden));
    }

    if (pos_ctx && pos_ctx->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(pos_ctx->buffer));
        int32_t * data = (int32_t *) pos_ctx->data;
        for (int64_t i = 0; i < ctx_len; ++i) {
            data[i] = (i < n_copy) ? (int32_t) (win_off + i) : 0;
        }
    }

    if (kq_mask && kq_mask->buffer) {
        GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask->buffer));
        float * data = (float *) kq_mask->data;
        const int64_t n_kv = ctx_len + n_block;
        for (int64_t q = 0; q < n_block; ++q) {
            for (int64_t k = 0; k < n_kv; ++k) {
                // mask padding slots (past real data, before noise block)
                if (k >= n_copy && k < ctx_len) {
                    data[q * n_kv + k] = -INFINITY;
                } else {
                    data[q * n_kv + k] = 0.0f;
                }
            }
        }
    }

    if (kq_mask_swa && kq_mask_swa->buffer && n_swa > 0) {
        GGML_ASSERT(ggml_backend_buffer_is_host(kq_mask_swa->buffer));
        float * data = (float *) kq_mask_swa->data;
        const int64_t n_kv   = ctx_len + n_block;
        const int32_t window = (int32_t) n_swa;
        const bool have_pos  = (ubatch != nullptr) && (ubatch->pos != nullptr)
                             && ((int64_t) ubatch->n_tokens >= n_block);
        for (int64_t q = 0; q < n_block; ++q) {
            const int32_t q_pos = have_pos ? ubatch->pos[q] : (int32_t) (n_real + q);
            for (int64_t k = 0; k < n_kv; ++k) {
                float v = 0.0f;
                if (k < n_copy) {
                    if (q_pos - (int32_t) k > window) { v = -INFINITY; }
                } else if (k < ctx_len) {
                    v = -INFINITY;
                } else {
                    const int64_t b_k = k - ctx_len;
                    if (b_k > q) { v = -INFINITY; }
                }
                data[q * n_kv + k] = v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// DFlash drafter graph builder
// ---------------------------------------------------------------------------

llama_model_dflash_draft::build_graph::build_graph(
        const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {

    const int64_t n_embd_head       = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    const int64_t n_target_features = hparams.dflash_n_target_features;

    // ctx_len: number of target hidden-state tokens to attend over
    int64_t ctx_len = (cross && cross->n_enc > 0) ? cross->n_enc : (int64_t) 4096;
    const int64_t max_ctx = dflash_max_cross_ctx();
    if (max_ctx > 0 && ctx_len > max_ctx) {
        ctx_len = max_ctx;
    }

    const int64_t n_kv_total = ctx_len + n_tokens;
    const bool have_swa = hparams.is_swa_any();

    auto inp_dflash = std::make_unique<llm_graph_input_dflash>(cross, ctx_len, n_tokens, hparams.n_swa);

    // concatenated target hidden states [n_target_features, ctx_len]
    inp_dflash->target_hidden = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_target_features, ctx_len);
    ggml_set_input(inp_dflash->target_hidden);
    cb(inp_dflash->target_hidden, "dflash_target_hidden", -1);

    // context positions for K RoPE [ctx_len]
    inp_dflash->pos_ctx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, ctx_len);
    ggml_set_input(inp_dflash->pos_ctx);
    cb(inp_dflash->pos_ctx, "dflash_pos_ctx", -1);

    // asymmetric non-causal mask [n_kv_total, n_tokens, 1, 1]
    inp_dflash->kq_mask = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
    ggml_set_input(inp_dflash->kq_mask);
    inp_dflash->kq_mask_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_dflash->kq_mask, GGML_TYPE_F16)
        : inp_dflash->kq_mask;

    if (have_swa) {
        inp_dflash->kq_mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, n_kv_total, n_tokens, 1, 1);
        ggml_set_input(inp_dflash->kq_mask_swa);
        cb(inp_dflash->kq_mask_swa, "dflash_kq_mask_swa", -1);
        inp_dflash->kq_mask_swa_cnv = cparams.flash_attn
            ? ggml_cast(ctx0, inp_dflash->kq_mask_swa, GGML_TYPE_F16)
            : inp_dflash->kq_mask_swa;
    }

    ggml_tensor * kq_mask_full = inp_dflash->kq_mask_cnv;
    ggml_tensor * kq_mask_swa  = inp_dflash->kq_mask_swa_cnv;
    ggml_tensor * pos_ctx      = inp_dflash->pos_ctx;
    ggml_tensor * target_hidden = inp_dflash->target_hidden;

    res->add_input(std::move(inp_dflash));

    // Embedding — may be nullptr during graph reservation (shared from target at runtime)
    // Use a Q4_0 placeholder to avoid a multi-GB F32 allocation during reservation
    ggml_tensor * tok_embd_use = model.tok_embd;
    if (!tok_embd_use) {
        tok_embd_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    ggml_tensor * inpL   = build_inp_embd(tok_embd_use);
    ggml_tensor * inp_pos = build_inp_pos();

    // Fusion layer: project target hidden states into drafter embedding space
    ggml_tensor * fused_target = build_lora_mm(model.dflash_fc, target_hidden);
    fused_target = build_norm(fused_target, model.dflash_hidden_norm, nullptr, LLM_NORM_RMS, -1);
    cb(fused_target, "fused_target", -1);

    // Transformer layers
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * kq_mask = (hparams.is_swa(il) && kq_mask_swa) ? kq_mask_swa : kq_mask_full;

        ggml_tensor * cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // KV-injection attention: Q from drafter, K/V from [target context ++ noise tokens]
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                                 n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // K from drafter noise tokens
            ggml_tensor * Kcur_noise = build_lora_mm(model.layers[il].wk, cur);
            Kcur_noise = ggml_reshape_3d(ctx0, Kcur_noise, n_embd_head, n_head_kv, n_tokens);
            Kcur_noise = build_norm(Kcur_noise, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_noise = ggml_rope_ext(ctx0, Kcur_noise, inp_pos, nullptr,
                                       n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                       ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_noise, "Kcur_noise", il);

            // K from target context features
            ggml_tensor * Kcur_ctx = build_lora_mm(model.layers[il].wk, fused_target);
            Kcur_ctx = ggml_reshape_3d(ctx0, Kcur_ctx, n_embd_head, n_head_kv, ctx_len);
            Kcur_ctx = build_norm(Kcur_ctx, model.layers[il].attn_k_norm, nullptr, LLM_NORM_RMS, il);
            Kcur_ctx = ggml_rope_ext(ctx0, Kcur_ctx, pos_ctx, nullptr,
                                     n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                     ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_ctx, "Kcur_ctx", il);

            // V from drafter noise tokens
            ggml_tensor * Vcur_noise = build_lora_mm(model.layers[il].wv, cur);
            Vcur_noise = ggml_reshape_3d(ctx0, Vcur_noise, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_noise, "Vcur_noise", il);

            // V from target context features
            ggml_tensor * Vcur_ctx = build_lora_mm(model.layers[il].wv, fused_target);
            Vcur_ctx = ggml_reshape_3d(ctx0, Vcur_ctx, n_embd_head, n_head_kv, ctx_len);
            cb(Vcur_ctx, "Vcur_ctx", il);

            // Concatenate K, V: [ctx, noise] along sequence dim
            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_ctx, Kcur_noise, 2);
            cb(Kcur, "Kcur", il);
            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_ctx, Vcur_noise, 2);
            cb(Vcur, "Vcur", il);

            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // Asymmetric attention: mask covers [ctx_len + n_tokens]
            cur = build_attn_mha(Qcur, Kcur, Vcur, nullptr, kq_mask, nullptr, nullptr,
                                 1.0f / sqrtf(float(n_embd_head)), il);
            cb(cur, "kqv_out", il);

            cur = build_lora_mm(model.layers[il].wo, cur);
        }

        // Residual
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_residual", il);

        ggml_tensor * ffn_residual = cur;

        // Post-attention RMSNorm
        cur = build_norm(cur, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);

        // SwiGLU FFN
        cur = build_ffn(cur,
            model.layers[il].ffn_up,   nullptr, nullptr,
            model.layers[il].ffn_gate, nullptr, nullptr,
            model.layers[il].ffn_down, nullptr, nullptr,
            nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // Final RMSNorm
    ggml_tensor * cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head — may be nullptr during reservation (shared from target at runtime)
    ggml_tensor * output_use = model.output;
    if (!output_use) {
        output_use = ggml_new_tensor_2d(ctx0, GGML_TYPE_Q4_0, n_embd, model.vocab.n_tokens());
    }
    cur = build_lora_mm(output_use, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
