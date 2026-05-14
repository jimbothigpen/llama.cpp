#include "models.h"

void llama_model_gemma4_mtp::load_arch_hparams(llama_model_loader & ml) {
    uint32_t n_embd_assist = 0;
    ml.get_key(LLM_KV_EMBEDDING_LENGTH, n_embd_assist);

    uint32_t n_embd_backbone = n_embd_assist;
    ml.get_key(LLM_KV_BACKBONE_EMBEDDING_LENGTH, n_embd_backbone, false);

    hparams.n_embd = n_embd_backbone;
    hparams.n_layer_kv_from_start = -1;
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = false;
    }

    // SWA setup: MTP has mixed SWA (layers 0..n-2) and global (last layer) attention.
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    // Gemma4 uses a fixed attention scale of 1.0 (no pre-attention sqrt scaling).
    hparams.f_attention_scale = 1.0f;

    // RoPE SWA base frequency (global is loaded by generic loader via LLM_KV_ROPE_FREQ_BASE).
    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA, hparams.rope_freq_base_train_swa, false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_gemma4_mtp::load_arch_tensors(llama_model_loader & ml) {
    uint32_t n_embd_assist = 0;
    ml.get_key(LLM_KV_EMBEDDING_LENGTH, n_embd_assist);

    const int64_t n_assist  = n_embd_assist;
    const int64_t n_backbone = hparams.n_embd;
    const int64_t n_ff      = hparams.n_ff();
    const int64_t n_vocab   = vocab.n_tokens();

    tok_embd     = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,    "weight"), { n_assist, n_vocab },          0);
    output_norm  = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM,   "weight"), { n_assist },                   0);
    mtp_pre_proj = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,  "weight"), { 2 * n_backbone, n_assist },  0);
    mtp_post_proj = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ, "weight"), { n_assist, n_backbone },      0);

    for (int i = 0; i < (int) hparams.n_layer; ++i) {
        auto & layer = layers[i];

        // Per-layer attention output size: depends on SWA vs global head dimension.
        const int64_t n_embd_head = hparams.n_embd_head_k(i);
        const int64_t n_head      = hparams.n_head(i);
        const int64_t n_attn_out  = n_embd_head * n_head;

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), { n_assist },          0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), { n_assist },          0);
        layer.ffn_norm       = create_tensor(tn(LLM_TENSOR_FFN_NORM,       "weight", i), { n_assist },          0);
        layer.ffn_post_norm  = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), { n_assist },          0);
        layer.out_scale      = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), { 1 },                0);

        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), { n_embd_head },       0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), { n_assist, n_attn_out }, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_attn_out, n_assist }, 0);

        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), { n_assist, n_ff }, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), { n_ff, n_assist }, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), { n_assist, n_ff }, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_mtp::graph::graph(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_backbone = hparams.n_embd;

    auto inp = std::make_unique<llm_graph_input_embd>(n_backbone);

    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);

    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_backbone, n_tokens);
    ggml_set_input(inp->embd);
    ggml_set_name(inp->embd, "gemma4_mtp_h_input");

    ggml_tensor * h_input  = inp->embd;
    ggml_tensor * tok_embd = ggml_get_rows(ctx0, model.tok_embd, inp->tokens);
    cb(tok_embd, "gemma4_mtp_tok_embd", -1);

    res->add_input(std::move(inp));

    // Positions for RoPE.
    ggml_tensor * inp_pos = build_inp_pos();

    // KV attention inputs for mixed SWA + global attention.
    auto * inp_attn = build_attn_inp_kv_iswa();

    // Pre-projection: concat backbone hidden with itself (approx. of [h_backbone, context]),
    // then project into assistant embedding space and add the next-token embedding.
    ggml_tensor * concat = ggml_concat(ctx0, h_input, h_input, /*dim=*/ 0);
    cb(concat, "gemma4_mtp_backbone_concat", -1);

    ggml_tensor * cur = build_lora_mm(model.mtp_pre_proj, concat);
    cb(cur, "gemma4_mtp_pre_projection", -1);

    cur = ggml_add(ctx0, cur, tok_embd);
    cb(cur, "gemma4_mtp_input", -1);

    for (int il = 0; il < (int) hparams.n_layer; ++il) {
        const auto & layer = model.layers[il];

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);
        const int64_t n_head_kv   = hparams.n_head_kv(il);
        const int     n_rot       = hparams.n_rot(il);

        const float freq_base  = model.get_rope_freq_base(cparams, il);
        const float freq_scale = model.get_rope_freq_scale(cparams, il);

        // kq_scale = 1.0 (Gemma4 uses fixed scaling, set in hparams.f_attention_scale).
        const float kq_scale = hparams.f_attention_scale == 0.0f
                ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

        // --- Attention block ---
        ggml_tensor * residual = cur;
        cur = build_norm(cur, layer.attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_attn_norm", il);

        // Q projection: [n_assist, n_tokens] → [n_embd_head * n_head, n_tokens] → reshape.
        ggml_tensor * Qcur = build_lora_mm(layer.wq, cur, layer.wq_s);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        cb(Qcur, "gemma4_mtp_Qcur", il);

        // Q normalization (attn_q_norm applied per-head).
        Qcur = build_norm(Qcur, layer.attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "gemma4_mtp_Qcur_normed", il);

        // RoPE on Q.
        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
        cb(Qcur, "gemma4_mtp_Qcur_pos", il);

        // K = first n_head_kv heads of Q (tied weights — use_alternative_attention / attention_k_eq_v).
        // ggml_cont ensures contiguous memory layout before KV cache write.
        ggml_tensor * Kcur = ggml_cont(ctx0, ggml_view_3d(ctx0, Qcur,
                n_embd_head, n_head_kv, n_tokens,
                Qcur->nb[1], Qcur->nb[2], 0));
        cb(Kcur, "gemma4_mtp_Kcur", il);

        // V = K (attention_k_eq_v: K and V share the same values).
        ggml_tensor * Vcur = ggml_cont(ctx0, ggml_view_3d(ctx0, Qcur,
                n_embd_head, n_head_kv, n_tokens,
                Qcur->nb[1], Qcur->nb[2], 0));
        cb(Vcur, "gemma4_mtp_Vcur", il);

        cur = build_attn(inp_attn, layer.wo, nullptr, layer.wo_s,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);
        cb(cur, "gemma4_mtp_attn_out", il);

        // Post-attention norm then residual (Gemma4 "sandwich norm" pattern).
        cur = build_norm(cur, layer.attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_attn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "gemma4_mtp_attn_residual", il);

        // --- FFN block ---
        residual = cur;
        cur = build_norm(cur, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_ffn_norm", il);

        cur = build_ffn(cur,
                layer.ffn_up,   nullptr, layer.ffn_up_s,
                layer.ffn_gate, nullptr, layer.ffn_gate_s,
                layer.ffn_down, nullptr, layer.ffn_down_s,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "gemma4_mtp_ffn_out", il);

        // Post-FFN norm then residual.
        cur = build_norm(cur, layer.ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "gemma4_mtp_ffn_post_norm", il);

        cur = ggml_add(ctx0, cur, residual);
        cb(cur, "gemma4_mtp_ffn_residual", il);
    }

    // Post-projection produces the next backbone hidden state for chained MTP steps.
    ggml_tensor * mtp_out = build_lora_mm(model.mtp_post_proj, cur);
    cb(mtp_out, "gemma4_mtp_post_projection", -1);
    res->t_mtp_out = mtp_out;

    // Output head: norm + tied embedding projection.
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "gemma4_mtp_output_norm", -1);

    cur = build_lora_mm(model.tok_embd, cur);
    cb(cur, "result_output", -1);

    res->t_logits = cur;
    ggml_build_forward_expand(gf, cur);
    ggml_build_forward_expand(gf, mtp_out);
}
