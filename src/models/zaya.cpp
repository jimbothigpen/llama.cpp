#include "models.h"

#include "ggml.h"
#include "llama-impl.h"

// Zyphra ZAYA1-8B port — Phase 2.
//
// This file currently provides only the hparam reader and the tensor-binding
// pass. The graph builder is a stub: it aborts before any decode work happens
// so we can verify that arch detection, hparam reads, and the 1283-tensor
// bind all succeed cleanly.
//
// Phase 3 will port the full CCA attention + MoE-router graph from the Zyphra
// reference fork at ~/kernel-work/zaya1-zyphra-fork/src/models/zaya.cpp.

void llama_model_zaya::load_arch_hparams(llama_model_loader & ml) {
    // Accept either the RMS-name or the plain-name eps key: ZAYA uses RMSNorm
    // internally but ZAYA1-8B's HF config field is named `layer_norm_eps`, which
    // round-trips through the converter as `*.attention.layer_norm_epsilon`.
    // Future converter writes will also emit the RMS-named key (see
    // ZayaModel.set_gguf_parameters); accepting both keeps existing GGUFs loadable.
    if (!ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps, false)) {
        ml.get_key(LLM_KV_ATTENTION_LAYERNORM_EPS, hparams.f_norm_rms_eps);
    }
    ml.get_key(LLM_KV_SSM_CONV_KERNEL, hparams.ssm_d_conv);

    // CCA conv state holds {Q, K} (each n_head * head_dim and n_head_kv * head_dim)
    // plus a delayed hidden-state stream of width n_embd. Sized via the SSM
    // "inner" hparam so the existing hybrid-recurrent plumbing routes the right
    // per-cell footprint.
    const uint32_t n_qk = (hparams.n_head() + hparams.n_head_kv()) * hparams.n_embd_head_k();
    hparams.ssm_d_inner = 2*n_qk + hparams.n_embd;
    hparams.ssm_d_state = 1;
    hparams.ssm_n_group = 0;

    // Even layers carry CCA recurrent state; odd layers are pure MoE FFN.
    for (uint32_t i = 0; i < hparams.n_layer; ++i) {
        hparams.recurrent_layer_arr[i] = (i % 2) == 0;
    }

    switch (hparams.n_layer) {
        case 80: type = LLM_TYPE_8B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_zaya::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd    = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, 0);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Tied embeddings: ZAYA1-8B has no separate output.weight tensor in the GGUF.
    // Fall back to the input embedding (graph builder relies on this alias).
    output = create_tensor(tn(LLM_TENSOR_OUTPUT, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = tok_embd;
    }

    // Top-level final residual-scale tensors (applied after the layer stack).
    zaya_res_scale_hs    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_FINAL,    "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_hs_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_B_FINAL,  "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_FINAL,   "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    zaya_res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_B_FINAL, "bias"),   {n_embd}, TENSOR_NOT_REQUIRED);

    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t d_conv      = hparams.ssm_d_conv;
    // Router MLP hidden size — ZAYA1-8B's zaya_mlp_expansion. Hardcoded here
    // because the converter does not store it as a separate GGUF KV (matches
    // Zyphra's reference loader at ~/kernel-work/zaya1-zyphra-fork/src/models/zaya.cpp:49).
    const int64_t n_ff_exp    = 256;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head    = hparams.n_head(i);
        const int64_t n_head_kv = hparams.n_head_kv(i);
        const int64_t n_embd_q  = n_head    * n_embd_head;
        const int64_t n_embd_k  = n_head_kv * n_embd_head;
        const int64_t n_qk      = n_embd_q + n_embd_k;
        const int64_t n_groups  = n_head + n_head_kv;
        const int64_t n_ff      = hparams.n_ff(i);
        const int64_t n_expert  = hparams.n_expert;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // CCA attention (even layers only)
        if (i % 2 == 0) {
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_q}, 0);
            layer.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, n_embd_k}, 0);

            layer.cca_val_proj1 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ1, "weight", i),
                {n_embd, n_embd_head}, 0);
            layer.cca_val_proj2 = create_tensor(tn(LLM_TENSOR_CCA_VAL_PROJ2, "weight", i),
                {n_embd, n_embd_head}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_q, n_embd}, 0);

            layer.cca_conv_dw   = create_tensor(tn(LLM_TENSOR_CCA_CONV_DW,   "weight", i), {d_conv, n_qk}, 0);
            layer.cca_conv_dw_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_DW_B, "bias",   i), {n_qk}, TENSOR_NOT_REQUIRED);

            layer.cca_conv_grp   = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "weight", i),
                {d_conv, n_qk / n_groups, n_qk}, 0);
            layer.cca_conv_grp_b = create_tensor(tn(LLM_TENSOR_CCA_CONV_GRP, "bias",   i), {n_qk}, 0);

            layer.cca_k_scale = create_tensor(tn(LLM_TENSOR_CCA_K_SCALE, "weight", i), {n_head_kv}, 0);
        }

        // Per-layer residual scaling — layer 0 lacks res_scale_res[+b] (no prior residual).
        layer.res_scale_hs    = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS,    "weight", i), {n_embd}, 0);
        layer.res_scale_hs_b  = create_tensor(tn(LLM_TENSOR_RES_SCALE_HS_B,  "bias",   i), {n_embd}, 0);
        layer.res_scale_res   = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES,   "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.res_scale_res_b = create_tensor(tn(LLM_TENSOR_RES_SCALE_RES_B, "bias",   i), {n_embd}, TENSOR_NOT_REQUIRED);

        // MoE layers (odd indices). zaya_router_eda is missing on the last odd layer (79).
        if (i % 2 == 1) {
            layer.zaya_router_down   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_DOWN,   "weight", i),
                {n_embd, n_ff_exp}, 0);
            layer.zaya_router_down_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_DOWN_B, "bias",   i),
                {n_ff_exp}, 0);
            layer.zaya_router_norm   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_NORM,   "weight", i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp0   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP0,   "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp0_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP0_B, "bias",   i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp2   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2,   "weight", i),
                {n_ff_exp, n_ff_exp}, 0);
            layer.zaya_router_mlp2_b = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP2_B, "bias",   i),
                {n_ff_exp}, 0);
            layer.zaya_router_mlp4   = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_MLP4,   "weight", i),
                {n_ff_exp, n_expert + 1}, 0);
            layer.zaya_router_biases = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_BIASES, "weight", i),
                {n_expert + 1}, TENSOR_NOT_REQUIRED);
            layer.zaya_router_eda_scale = create_tensor(tn(LLM_TENSOR_ZAYA_ROUTER_EDA_SCALE, "weight", i),
                {n_ff_exp}, TENSOR_NOT_REQUIRED);

            // MoE experts (fused gate_up + separate down)
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff, n_expert, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i),
                {n_ff, n_embd, n_expert}, 0);
        }
    }
}

// P2 stub. The full graph builder lands in P3 — see
// ~/kernel-work/zaya1-zyphra-fork/src/models/zaya.cpp for the reference
// implementation (CCA attention with depthwise + grouped 1D conv; deep router
// MLP with optional EDA cross-layer feedback; top-1 MoE over 16 experts).
llama_model_zaya::graph::graph(const llama_model & /*model*/, const llm_graph_params & params)
    : llm_graph_context(params) {
    GGML_ABORT("ZAYA graph builder not yet implemented (P3 work)");
}

std::unique_ptr<llm_graph_context> llama_model_zaya::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}
