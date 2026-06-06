#include "models.h"

#include "llama-impl.h"

void llama_model_eagle3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    ml.get_key(LLM_KV_EAGLE3_TARGET_HIDDEN_SIZE,   hparams.eagle3_target_hidden_size,   false);
    ml.get_key(LLM_KV_EAGLE3_NORM_BEFORE_RESIDUAL, hparams.eagle3_norm_before_residual,  false);
    ml.get_key_or_arr(LLM_KV_EAGLE3_EXTRACT_LAYERS, hparams.eagle3_extract_layers, 3,   false);

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_eagle3::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_embd_fc_in = 3 * n_embd;
    const int64_t n_embd_2x    = 2 * n_embd;

    // Compact-vocab EAGLE3 (SpecForge): the output head + d2t use a smaller draft vocab than the
    // full target token space. Input embeddings (tok_embd) stay full-vocab and are inherited from
    // the target model at runtime when absent from the draft. Derive draft vocab from d2t width.
    int64_t n_draft_vocab = n_vocab;
    const struct ggml_tensor * d2t_meta = ml.get_tensor_meta(tn(LLM_TENSOR_EAGLE3_D2T, "weight").str().c_str());
    if (d2t_meta) {
        n_draft_vocab = d2t_meta->ne[0];
    }

    // tok_embd is full target vocab; optional so compact drafts can inherit it from the target.
    // EAGLE3 32K-vocab + d2t loader: a compact-vocab draft (d2t present) may ship its
    // own draft-space token_embd ({n_embd, n_draft_vocab}), but the EAGLE3 driver feeds the draft
    // *target-space* token ids (last accepted target token + sampled ids already scattered to the
    // target vocab in llama-context.cpp), which a draft-space table cannot index. Skip the draft's
    // token_embd so the target's full-vocab tok_embd is inherited at
    // common_speculative_setup_draft_model(); loading the compact one would index out of range at
    // decode. (TENSOR_SKIP also bypasses the strict vocab-dim check that otherwise rejects the load.)
    const int tok_embd_flags = TENSOR_NOT_REQUIRED | (d2t_meta ? TENSOR_SKIP : 0);
    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab},       tok_embd_flags);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd},                0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_draft_vocab}, TENSOR_NOT_REQUIRED);
    if (!output && tok_embd) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // encoder: fc projection [n_embd_fc_in → n_embd]
    fc  = create_tensor(tn(LLM_TENSOR_EAGLE3_FC,  "weight"), {n_embd_fc_in, n_embd}, 0);

    // draft-to-target vocab mapping (compact-vocab only; absent for full-vocab drafts)
    d2t = create_tensor(tn(LLM_TENSOR_EAGLE3_D2T, "weight"), {n_draft_vocab}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.eagle3_hidden_norm = create_tensor(tn(LLM_TENSOR_EAGLE3_HIDDEN_NORM, "weight", i), {n_embd}, 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q,   "weight", i), {n_embd_2x, n_embd_head_k * n_head},  0);
        layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K,   "weight", i), {n_embd_2x, n_embd_k_gqa},            0);
        layer.wv = create_tensor(tn(LLM_TENSOR_ATTN_V,   "weight", i), {n_embd_2x, n_embd_v_gqa},            0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd},      0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd},                  0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {hparams.n_ff(i), n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, hparams.n_ff(i)}, 0);

        layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_rot/2},
                TENSOR_NOT_REQUIRED | (i != 0 ? TENSOR_DUPLICATED : 0));
    }
}

std::unique_ptr<llm_graph_context> llama_model_eagle3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph_decode>(*this, params);
}

// ---------------------------------------------------------------------------
// graph_encode: fc projection only — builds g_embeddings from target features
// Input: target_features [fc_input_size, n_tokens] (concatenated hidden states)
// Output: t_embd = g_embeddings [n_embd, n_tokens]
// ---------------------------------------------------------------------------
llama_model_eagle3::graph_encode::graph_encode(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    int64_t n_extract = 0;
    for (int i = 0; i < 3; i++) {
        if (model.hparams.eagle3_extract_layers[i] >= 0) n_extract++;
    }
    if (n_extract == 0) n_extract = 3;
    const int64_t fc_input_size = n_extract * n_embd;

    ggml_tensor * inp = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, fc_input_size, n_tokens);
    ggml_set_input(inp);
    ggml_set_name(inp, "inp_eagle3_features");

    ggml_tensor * cur = ggml_mul_mat(ctx0, model.fc, inp);
    cb(cur, "eagle3_fc_out", -1);

    res->t_embd = cur;
    ggml_build_forward_expand(gf, cur);
}

// ---------------------------------------------------------------------------
// graph_decode: 1-layer transformer taking g_embd + tok_embd → draft logits
// Residual carried on g_embeddings, prenorm stored as t_embd for autoregressive loop
// ---------------------------------------------------------------------------
llama_model_eagle3::graph_decode::graph_decode(const llama_model & model, const llm_graph_params & params)
    : llm_graph_context(params) {

    const int64_t n_embd_head = hparams.n_embd_head_v();
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());

    ggml_tensor * cur;

    ggml_tensor * inpL      = build_inp_embd(model.tok_embd);
    ggml_tensor * inp_pos   = build_inp_pos();
    auto        * inp_attn  = build_attn_inp_kv();
    ggml_tensor * inp_out_ids = build_inp_out_ids();
    ggml_tensor * g_embd    = build_inp_eagle3_g_embd();

    for (int il = 0; il < n_layer; ++il) {
        // 1. norm token embedding with input_layernorm
        ggml_tensor * embd_norm = build_norm(inpL,
                model.layers[il].attn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(embd_norm, "embd_norm", il);

        // 2. norm g_embeddings with hidden_norm
        ggml_tensor * g_embd_norm = build_norm(g_embd,
                model.layers[il].eagle3_hidden_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(g_embd_norm, "g_embd_norm", il);

        // 3. concat → [2*n_embd, n_tokens]
        ggml_tensor * attn_input = ggml_concat(ctx0, embd_norm, g_embd_norm, 0);
        cb(attn_input, "attn_input_concat", il);

        // 4. QKV projection + RoPE + attention
        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, attn_input);
            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, attn_input);
            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, attn_input);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            cur = build_attn(inp_attn,
                    model.layers[il].wo, nullptr, nullptr,
                    Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                    1.0f / sqrtf(float(n_embd_head)), il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur    = ggml_get_rows(ctx0, cur,    inp_out_ids);
            g_embd = ggml_get_rows(ctx0, g_embd, inp_out_ids);
            if (hparams.eagle3_norm_before_residual) {
                g_embd_norm = ggml_get_rows(ctx0, g_embd_norm, inp_out_ids);
            }
        }

        // 5. residual on g_embeddings; norm_before_residual uses g_embd_norm as the skip connection
        ggml_tensor * residual = hparams.eagle3_norm_before_residual ? g_embd_norm : g_embd;
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, residual);
        cb(ffn_inp, "ffn_inp", il);

        // 6. FFN with pre-norm
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, nullptr,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        g_embd = cur;
        inpL   = cur;
    }

    cur = inpL;

    // Prenorm output stored as t_embd for autoregressive g_embeddings recurrence.
    // The speculative loop reads this via llama_get_embeddings_ith() and feeds it
    // back as g_embeddings for the next decode step.
    ggml_set_output(cur);
    res->t_embd = cur;

    // output_norm + lm_head → draft logits
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
