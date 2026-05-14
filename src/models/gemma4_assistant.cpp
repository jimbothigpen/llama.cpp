#include "models.h"

#include "../llama-context.h"
#include "../llama-kv-cache-iswa.h"

#include <algorithm>

// Gemma 4 Multi-Token Prediction (MTP) drafter ("assistant") model.
//
// Released: https://blog.google/innovation-and-ai/technology/developers-tools/multi-token-prediction-gemma-4/
// Reference impl: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gemma4_assistant
// Model card:    https://huggingface.co/google/gemma-4-E2B-it-assistant
//                https://huggingface.co/google/gemma-4-E4B-it-assistant
//
// Architectural notes (vs. the Gemma 4 backbone):
// - 4 transformer layers, all of which read K/V from the target backbone's last
//   layer of the matching attention type (full vs. sliding). The drafter has
//   only Q/Q-norm/O projections — no K/V projections.
// - `pre_projection`: [2*backbone_hidden -> hidden] reduces a concat of
//   (target final hidden state, embedding of next-step token in backbone-dim).
// - `post_projection`: [hidden -> backbone_hidden] feeds the next chained MTP step.
// - Optional centroid masked-embedding head: a top-K of `num_centroids` clusters
//   selects a small slice of the 262144-row LM head; logits outside the slice
//   are masked to a small constant.
//
// IMPORTANT: this assistant cannot run with `llama_decode` alone — it borrows
// K/V from a separate target-context decode step. The graph builder reads the
// target backbone through `llm_graph_context::mtp_target_ctx` (wired by the
// speculative driver via `llama_set_mtp_target_context`). When no target ctx
// is attached — e.g. during graph reservation at load time — the builder
// takes a degenerate fast-path that keeps the graph valid without touching a
// foreign cache.

void llama_model_gemma4_assistant::load_arch_hparams(llama_model_loader & ml) {
    // Most Gemma 4 hparams apply: SWA pattern, dual head dim (full vs SWA),
    // shared-KV layout, etc. We deliberately reuse the same metadata keys.
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.swa_layers, hparams.n_layer);

    uint32_t n_kv_shared_layers = 0;
    ml.get_key(LLM_KV_ATTENTION_SHARED_KV_LAYERS, n_kv_shared_layers, false);
    hparams.n_layer_kv_from_start = hparams.n_layer - (int32_t) n_kv_shared_layers;
    hparams.f_attention_scale     = 1.0f;

    ml.get_key(LLM_KV_ROPE_FREQ_BASE_SWA,          hparams.rope_freq_base_train_swa, false);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW,    hparams.n_swa);
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_SWA,    hparams.n_embd_head_k_swa);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_SWA,  hparams.n_embd_head_v_swa);

    // assistant-specific
    uint32_t backbone_hidden = 0;
    ml.get_key(LLM_KV_BACKBONE_HIDDEN_SIZE,             backbone_hidden);
    hparams.n_embd_backbone = backbone_hidden;

    uint32_t num_centroids = 0;
    uint32_t centroid_top_k = 0;
    bool     use_ordered = false;
    ml.get_key(LLM_KV_ASSISTANT_NUM_CENTROIDS,           num_centroids,  false);
    ml.get_key(LLM_KV_ASSISTANT_CENTROID_TOP_K,          centroid_top_k, false);
    ml.get_key(LLM_KV_ASSISTANT_USE_ORDERED_EMBEDDINGS,  use_ordered,    false);
    hparams.n_assist_centroids      = num_centroids;
    hparams.n_assist_centroid_top_k = centroid_top_k;
    hparams.use_ordered_embeddings  = use_ordered;

    // E2B / E4B assistants both have 4 layers; mirror Gemma 4 backbone naming so
    // metadata reads stay consistent.
    switch (hparams.n_layer) {
        case 4:
            // Disambiguate by the backbone hidden size — the only shape that
            // varies between the two officially-released drafters.
            type = (backbone_hidden >= 2560) ? LLM_TYPE_E4B : LLM_TYPE_E2B;
            break;
        default:
            type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_gemma4_assistant::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    if (hparams.n_embd_backbone == 0) {
        throw std::runtime_error("gemma4-assistant: missing backbone_hidden_size in GGUF metadata");
    }
    if (hparams.n_embd_head_k_full != hparams.n_embd_head_v_full) {
        throw std::runtime_error("gemma4-assistant requires n_embd_head_k == n_embd_head_v");
    }
    if (hparams.n_embd_head_k_swa != hparams.n_embd_head_v_swa) {
        throw std::runtime_error("gemma4-assistant requires n_embd_head_k_swa == n_embd_head_v_swa");
    }

    const int64_t n_embd_backbone = (int64_t) hparams.n_embd_backbone;
    const int64_t n_centroids     = (int64_t) hparams.n_assist_centroids;

    // Token embedding is tied with `lm_head` per the released checkpoints
    // (`tie_word_embeddings: true`); `output` is therefore a duplicate.
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);
    output   = create_tensor(tn(LLM_TENSOR_OUTPUT,     "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);

    // Drafter <-> backbone projections.
    assist_pre_proj  = create_tensor(tn(LLM_TENSOR_ASSIST_PRE_PROJ,  "weight"), {2 * n_embd_backbone, n_embd}, 0);
    assist_post_proj = create_tensor(tn(LLM_TENSOR_ASSIST_POST_PROJ, "weight"), {n_embd, n_embd_backbone}, 0);

    // Optional centroid head — only present when `use_ordered_embeddings` was
    // baked into the checkpoint (the released E2B/E4B drafters both have it).
    if (hparams.use_ordered_embeddings) {
        if (n_centroids == 0) {
            throw std::runtime_error("gemma4-assistant: use_ordered_embeddings is set but num_centroids is 0");
        }
        assist_embed_centroids = create_tensor(tn(LLM_TENSOR_ASSIST_EMBED_CENTROIDS, "weight"), {n_embd, n_centroids}, 0);
        assist_token_ordering  = create_tensor(tn(LLM_TENSOR_ASSIST_TOKEN_ORDERING,  "weight"), {n_vocab}, 0);
    }

    int rope_freqs_flag = 0;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        const int64_t n_head_l    = hparams.n_head(i);
        const int64_t n_embd_head = hparams.n_embd_head_k(i);

        layer.attn_norm      = create_tensor(tn(LLM_TENSOR_ATTN_NORM,      "weight", i), {n_embd}, 0);
        // Q-only attention: K/V are borrowed from the target backbone, so the
        // drafter checkpoint has no `wk` / `wv` weights at all.
        layer.wq             = create_tensor(tn(LLM_TENSOR_ATTN_Q,         "weight", i), {n_embd, n_embd_head * n_head_l}, 0);
        layer.wo             = create_tensor(tn(LLM_TENSOR_ATTN_OUT,       "weight", i), {n_embd_head * n_head_l, n_embd}, 0);
        layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);
        layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        if (!hparams.is_swa(i)) {
            // Full-attention layer reuses the Gemma 4 partial-RoPE shim.
            layer.rope_freqs = create_tensor(tn(LLM_TENSOR_ROPE_FREQS, "weight", i), {n_embd_head/2}, rope_freqs_flag);
            rope_freqs_flag = TENSOR_DUPLICATED;
        }

        layer.ffn_norm      = create_tensor(tn(LLM_TENSOR_FFN_NORM,      "weight", i), {n_embd}, 0);
        layer.ffn_post_norm = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate      = create_tensor(tn(LLM_TENSOR_FFN_GATE,      "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        layer.ffn_up        = create_tensor(tn(LLM_TENSOR_FFN_UP,        "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        layer.ffn_down      = create_tensor(tn(LLM_TENSOR_FFN_DOWN,      "weight", i), {hparams.n_ff(i), n_embd}, 0);

        // Per-layer scalar gating (Gemma 4 layer_scalar): a single bf16 weight.
        layer.out_scale     = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1u}, 0);
    }
}

// Map an assistant layer to the target backbone layer it borrows K/V from.
// The assistant's per-layer SWA flag picks whether we want a sliding or a
// full-attention target layer; we then walk the target's KV-bearing layers
// from the tail and return the first whose attention type matches ("last
// layer of each attention type", per mainline #22738). Honors the target's
// `n_layer_kv_from_start` (some Gemma 4 backbones only keep KV for a prefix).
static int32_t gemma4_assistant_target_kv_layer(
        const llama_hparams & assistant_hparams,
        const llama_hparams & target_hparams,
        int32_t il) {
    GGML_ASSERT(il >= 0 && il < (int32_t) assistant_hparams.n_layer);
    const bool is_sliding = assistant_hparams.is_swa(il);
    const int target_n_kv_layer = target_hparams.n_layer_kv_from_start > 0
        ? std::min<int>((int) target_hparams.n_layer, target_hparams.n_layer_kv_from_start)
        : (int) target_hparams.n_layer;
    int target_il = target_n_kv_layer - 1;
    for (; target_il >= 0; --target_il) {
        if (target_hparams.is_swa(target_il) == is_sliding) {
            break;
        }
    }
    GGML_ASSERT(target_il >= 0 && "gemma4-assistant: no matching target KV layer");
    return target_il;
}

namespace {

// Foreign-KV attention mask input. Sized to the target backbone's KV-cache
// capacity (one mask for full-attn layers, one for SWA), shaped
// [target_n_kv, n_tokens, 1, 1] for build_attn_mha. The populator does NOT
// build a mask itself — it delegates to the target cache's own
// `set_input_kq_mask`, reusing the fork's causal + SWA + seq + empty-cell
// logic against the target's `v_cells[]`.
class llm_graph_input_kq_mask_mtp_target : public llm_graph_input_i {
public:
    llm_graph_input_kq_mask_mtp_target(
            const llama_kv_cache * target_kv_full,
            const llama_kv_cache * target_kv_swa,
            llama_seq_id target_seq_id)
        : target_kv_full(target_kv_full),
          target_kv_swa(target_kv_swa),
          target_seq_id(target_seq_id) {}
    ~llm_graph_input_kq_mask_mtp_target() override = default;

    void set_input(const llama_ubatch * ubatch) override {
        // The backbone writes its KV cells with seq_id = slot.id, which does
        // not generally equal the assistant draft ubatch's hardcoded seq_id=0.
        // If a target_seq_id was supplied (server-context via
        // llama_set_mtp_target_seq_id), build a synthetic ubatch with the
        // seq_id replaced so set_input_kq_mask's `cells.seq_has()` test hits
        // the right backbone cells. target_seq_id < 0 → legacy fallback (only
        // correct when slot.id == 0).
        std::vector<llama_seq_id>   seq_buf;
        std::vector<llama_seq_id *> seq_ptrs;
        std::vector<int32_t>        n_seq_id_buf;
        llama_ubatch local_ub;
        const llama_ubatch * eff_ub = ubatch;
        if (target_seq_id >= 0 && ubatch && ubatch->n_tokens > 0) {
            local_ub = *ubatch;
            seq_buf.assign(ubatch->n_tokens, target_seq_id);
            seq_ptrs.resize(ubatch->n_tokens);
            n_seq_id_buf.assign(ubatch->n_tokens, 1);
            for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
                seq_ptrs[i] = &seq_buf[i];
            }
            local_ub.seq_id   = seq_ptrs.data();
            local_ub.n_seq_id = n_seq_id_buf.data();
            eff_ub = &local_ub;
        }
        if (mask_full && target_kv_full) {
            target_kv_full->set_input_kq_mask(mask_full, eff_ub, /*causal_attn=*/true);
        }
        if (mask_swa && target_kv_swa) {
            target_kv_swa->set_input_kq_mask(mask_swa, eff_ub, /*causal_attn=*/true);
        }
    }

    bool can_reuse(const llm_graph_params & /*params*/) override {
        // Mask shape depends only on the attached target capacity, which is
        // fixed across graph rebuilds for a given assistant↔target pairing.
        return true;
    }

    const llama_kv_cache * target_kv_full = nullptr;
    const llama_kv_cache * target_kv_swa  = nullptr;
    llama_seq_id target_seq_id = -1;       // backbone slot.id (override for mask seq match)
    ggml_tensor * mask_full     = nullptr; // F32 [target_n_kv_full, n_tokens, 1, 1]
    ggml_tensor * mask_full_cnv = nullptr; // == mask_full or F16 cast
    ggml_tensor * mask_swa      = nullptr; // F32 [target_n_kv_swa,  n_tokens, 1, 1]
    ggml_tensor * mask_swa_cnv  = nullptr; // == mask_swa or F16 cast
};

// Populator for the assistant's inp_tokens. Mirrors llm_graph_input_embd's
// token path, but the embeddings are gathered from the TARGET model's
// tok_embd (the assistant has no full-vocab embedding of its own), so the
// shared build_inp_embd helper — which is bound to the assistant's own
// n_embd — cannot be reused. Without this populator the tensor is left
// uninitialised and ggml_get_rows reads random vocab rows.
class llm_graph_input_mtp_tokens : public llm_graph_input_i {
public:
    llm_graph_input_mtp_tokens() = default;
    ~llm_graph_input_mtp_tokens() override = default;

    void set_input(const llama_ubatch * ubatch) override {
        if (tokens && ubatch->token) {
            ggml_backend_tensor_set(tokens, ubatch->token, 0,
                                    ubatch->n_tokens * ggml_element_size(tokens));
        }
    }

    bool can_reuse(const llm_graph_params & params) override {
        return tokens && tokens->ne[0] == params.ubatch.n_tokens;
    }

    ggml_tensor * tokens = nullptr;
};

} // namespace

std::unique_ptr<llm_graph_context> llama_model_gemma4_assistant::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Gemma 4 MTP external-assistant graph builder.
//
// Topology: pre_projection [concat(scale(get_rows(target.tok_embd, ids),
// sqrt(n_backbone)), hidden_state) -> n_embd] → 4 Q-only attention layers
// against the backbone's borrowed K/V → post_projection [n_embd ->
// n_backbone] (chained to the next draft step via res->t_embd) → final norm
// → output head.
//
// Q uses the assistant's own wq / attn_q_norm, but rotates with the TARGET's
// per-layer RoPE params so it lands in the backbone's positional frame; K/V
// are inline ggml_view_3d views over the target cache's raw layer tensors,
// fed straight to build_attn_mha (the class-based build_attn API cannot
// express foreign K/V). wo is applied manually afterwards.
llama_model_gemma4_assistant::graph::graph(const llama_model & model, const llm_graph_params & params) :
        llm_graph_context(params),
        model(model),
        n_embd_backbone(model.hparams.n_embd_backbone),
        n_assist_centroids(model.hparams.n_assist_centroids),
        n_assist_centroid_top_k(model.hparams.n_assist_centroid_top_k),
        use_ordered_embeddings(model.hparams.use_ordered_embeddings) {
    GGML_ASSERT(n_embd_backbone > 0);
    GGML_ASSERT(n_embd_backbone >= n_embd);

    // Backbone hidden state for each batch token, fed in via res->t_mtp_states.
    // Sized to the backbone width (wider than the assistant's own n_embd) —
    // pre_projection concats token-embd ⊕ hidden_state and projects
    // [2*n_backbone -> n_embd]. The decode loop fills it through
    // llama_context::prepare_mtp_graph_inputs before graph_compute.
    ggml_tensor * hidden_state = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd_backbone, n_tokens);
    ggml_set_input(hidden_state);
    ggml_set_name(hidden_state, "mtp_states");
    res->t_mtp_states = hidden_state;
    cb(hidden_state, "mtp_states", -1);

    const bool has_target_ctx = (mtp_target_ctx != nullptr);
    const bool has_tokens     = (ubatch.token != nullptr);

    if (!has_target_ctx || !has_tokens) {
        // Degenerate fast-path: no foreign cache to attend. Taken during graph
        // reservation at load time (before llama_set_mtp_target_context) and
        // for token-less ubatches. Keeps the graph valid — view the leading
        // n_embd columns of the backbone hidden state as the assistant's
        // residual stream and run only the output head. allow_reuse keys on
        // mtp_target_ctx, so attaching a target later forces a rebuild onto
        // the full path below.
        ggml_tensor * cur = ggml_view_2d(ctx0, hidden_state, n_embd, n_tokens,
                hidden_state->nb[1], 0);
        cb(cur, "mtp_init_hidden_view", -1);

        // Chained-hidden output placeholder, backbone-width to match the full
        // path's post_projection result.
        res->t_embd = ggml_dup(ctx0, hidden_state);
        cb(res->t_embd, "result_mtp_embd", -1);
        ggml_build_forward_expand(gf, res->t_embd);

        cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
        cb(cur, "result_norm", -1);

        cur = build_lora_mm(model.output, cur);
        cb(cur, "result_output", -1);
        res->t_logits = cur;
        ggml_build_forward_expand(gf, cur);
        return;
    }

    // ---- Full foreign-KV transformer path ----

    llama_context      * target_ctx     = mtp_target_ctx;
    const llama_model   & target_model   = target_ctx->get_model();
    const llama_hparams & target_hparams = target_model.hparams;
    const llama_cparams & target_cparams = target_ctx->get_cparams();

    // Yarn-family RoPE params come from the TARGET's cparams: the assistant's
    // Q must rotate in the same positional frame the backbone's K was written
    // under. (Gemma 4 typically runs yarn-disabled, so these rarely move
    // numbers, but this is the safety-correct sourcing.)
    const int32_t n_ctx_orig_t  = (int32_t) target_cparams.n_ctx_orig_yarn;
    const float   ext_factor_t  = target_cparams.yarn_ext_factor;
    const float   attn_factor_t = target_cparams.yarn_attn_factor;
    const float   beta_fast_t   = target_cparams.yarn_beta_fast;
    const float   beta_slow_t   = target_cparams.yarn_beta_slow;

    // Gemma 4 backbone is ISWA (interleaved SWA + full-attn). Cast the target
    // memory to the iswa wrapper so we can pick per-layer between the base
    // (full) and SWA caches. dynamic_cast fails loudly if a non-ISWA backbone
    // ever attaches.
    const auto * target_iswa = dynamic_cast<const llama_kv_cache_iswa *>(target_ctx->get_memory());
    GGML_ASSERT(target_iswa && "gemma4-assistant: target context must use an ISWA KV cache");

    const llama_kv_cache * target_kv_full = target_iswa->get_base();
    const llama_kv_cache * target_kv_swa  = target_iswa->get_swa();
    GGML_ASSERT(target_kv_full && target_kv_swa);

    // Foreign-KV mask inputs, sized to the target's cache capacity so the
    // graph shape stays constant across decodes; the per-batch populator
    // emits a real causal+SWA mask against the assistant's ubatch positions,
    // with empty target cells collapsing to -INFINITY.
    auto inp_mask_owned = std::make_unique<llm_graph_input_kq_mask_mtp_target>(
        target_kv_full, target_kv_swa, mtp_target_seq_id);
    auto * inp_mask = inp_mask_owned.get();

    const uint32_t target_n_kv_full = target_kv_full->get_size();
    const uint32_t target_n_kv_swa  = target_kv_swa ->get_size();

    inp_mask->mask_full = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, target_n_kv_full, n_tokens, 1, 1);
    ggml_set_name(inp_mask->mask_full, "mtp_kq_mask_full");
    ggml_set_input(inp_mask->mask_full);
    inp_mask->mask_full_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_mask->mask_full, GGML_TYPE_F16)
        : inp_mask->mask_full;

    inp_mask->mask_swa = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, target_n_kv_swa, n_tokens, 1, 1);
    ggml_set_name(inp_mask->mask_swa, "mtp_kq_mask_swa");
    ggml_set_input(inp_mask->mask_swa);
    inp_mask->mask_swa_cnv = cparams.flash_attn
        ? ggml_cast(ctx0, inp_mask->mask_swa, GGML_TYPE_F16)
        : inp_mask->mask_swa;

    res->add_input(std::move(inp_mask_owned));

    // Input token IDs — gather backbone-vocab embeddings. Needs its own
    // populator (see llm_graph_input_mtp_tokens) since the embeddings come
    // from the target model, not the assistant.
    auto inp_tok_owned = std::make_unique<llm_graph_input_mtp_tokens>();
    inp_tok_owned->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tok_owned->tokens, "inp_tokens");
    ggml_set_input(inp_tok_owned->tokens);
    cb(inp_tok_owned->tokens, "inp_tokens", -1);
    ggml_tensor * inp_tokens = inp_tok_owned->tokens;
    res->add_input(std::move(inp_tok_owned));

    ggml_tensor * inp_pos = build_inp_pos();

    // pre_projection: concat(scale(embed(next_tok), sqrt(n_backbone)),
    // hidden_state) along the feature axis → [2*n_backbone, n_tokens] →
    // assist_pre_proj → [n_embd, n_tokens]. Embeddings come from the target
    // model's tied input/output embedding (the assistant shares it).
    ggml_tensor * token_embd = ggml_get_rows(ctx0, target_model.tok_embd, inp_tokens);
    cb(token_embd, "inp_embd_target", -1);
    token_embd = ggml_scale(ctx0, token_embd, sqrtf((float) n_embd_backbone));
    cb(token_embd, "inp_embd_scaled", -1);

    ggml_tensor * cur = ggml_concat(ctx0, token_embd, hidden_state, 0);
    cb(cur, "mtp_pre_proj_in", -1);
    cur = build_lora_mm(model.assist_pre_proj, cur);
    cb(cur, "assist_pre_proj", -1);

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpL = cur;

        const int64_t n_embd_head = hparams.n_embd_head_k(il);
        const int64_t n_head      = hparams.n_head(il);
        const bool    is_sliding  = hparams.is_swa(il);

        // attn_norm
        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // Q projection — assistant's own wq + attn_q_norm.
        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
        cb(Qcur, "Qcur", il);
        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
        Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, nullptr, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);

        // RoPE in the TARGET's positional frame: freq base/scale, n_rot and
        // proportional-RoPE freq factors all sourced from the matched target
        // layer so Q lines up with the backbone's stored K.
        const int32_t target_il = gemma4_assistant_target_kv_layer(hparams, target_hparams, il);

        const float freq_base_l  = target_model.get_rope_freq_base (target_cparams, target_il);
        const float freq_scale_l = target_model.get_rope_freq_scale(target_cparams, target_il);
        const int   n_rot_l      = (int) target_hparams.n_rot(target_il);
        ggml_tensor * freq_factors = target_hparams.is_swa(target_il)
            ? nullptr
            : target_model.layers[target_il].rope_freqs;

        Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, freq_factors,
                             n_rot_l, rope_type, n_ctx_orig_t,
                             freq_base_l, freq_scale_l,
                             ext_factor_t, attn_factor_t, beta_fast_t, beta_slow_t);
        cb(Qcur, "Qcur_rope", il);

        // Foreign K/V views over the target cache's raw layer tensors. Sized
        // to the target's capacity (get_size()); empty cells collapse via the
        // mask. Stride patterns mirror llama_kv_cache::get_k / get_v.
        const llama_kv_cache * target_kv = is_sliding ? target_kv_swa : target_kv_full;
        const uint32_t target_n_kv = target_kv->get_size();

        ggml_tensor * k_raw  = target_kv->get_layer_k_raw(target_il);
        ggml_tensor * v_raw  = target_kv->get_layer_v_raw(target_il);
        const bool    v_trans = target_kv->get_v_trans();

        const int64_t head_k_t     = target_hparams.n_embd_head_k(target_il);
        const int64_t head_v_t     = target_hparams.n_embd_head_v(target_il);
        const int64_t n_head_kv_t  = target_hparams.n_head_kv(target_il);
        const int64_t n_embd_k_gqa = k_raw->ne[0];
        const int64_t n_embd_v_gqa = v_raw->ne[0];

        // K view: [head_k, n_head_kv, target_n_kv], single-stream, offset 0.
        ggml_tensor * k_view = ggml_view_3d(ctx0, k_raw,
                head_k_t, n_head_kv_t, target_n_kv,
                ggml_row_size(k_raw->type, head_k_t),
                ggml_row_size(k_raw->type, n_embd_k_gqa),
                0);
        cb(k_view, "mtp_k_foreign", il);

        // V view: layout flips on v_trans (mirrors llama_kv_cache::get_v).
        ggml_tensor * v_view;
        if (!v_trans) {
            v_view = ggml_view_3d(ctx0, v_raw,
                    head_v_t, n_head_kv_t, target_n_kv,
                    ggml_row_size(v_raw->type, head_v_t),
                    ggml_row_size(v_raw->type, n_embd_v_gqa),
                    0);
        } else {
            v_view = ggml_view_3d(ctx0, v_raw,
                    target_n_kv, n_head_kv_t, head_v_t,
                    ggml_row_size(v_raw->type, (uint64_t) target_n_kv * head_v_t),
                    ggml_row_size(v_raw->type, target_n_kv),
                    0);
        }
        cb(v_view, "mtp_v_foreign", il);

        ggml_tensor * kq_mask_l = is_sliding ? inp_mask->mask_swa_cnv : inp_mask->mask_full_cnv;

        ggml_tensor * attn_out = build_attn_mha(Qcur, k_view, v_view,
                /*kq_b=*/nullptr, kq_mask_l, /*sinks=*/nullptr, /*v_mla=*/nullptr,
                hparams.f_attention_scale, il);
        cb(attn_out, "mtp_kqv_foreign", il);

        // wo — applied manually; build_attn_mha returns the post-MHA result.
        attn_out = build_lora_mm(model.layers[il].wo, attn_out);
        cb(attn_out, "mtp_attn_out", il);

        // attn_post_norm + residual.
        cur = build_norm(attn_out, model.layers[il].attn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);
        cur = ggml_add(ctx0, cur, inpL);
        cb(cur, "attn_residual", il);

        // FFN — assistant's own dense (non-MoE) GELU block.
        ggml_tensor * ffn_inp = cur;
        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   nullptr, nullptr,
                model.layers[il].ffn_gate, nullptr, nullptr,
                model.layers[il].ffn_down, nullptr, nullptr,
                nullptr,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(cur, "ffn_post_norm", il);

        // FFN residual. build_ffn does NOT add the residual — the caller must.
        // Dropping this collapses each layer to ffn_post_norm(FFN(norm(x))),
        // discarding the post-attn residual stream (the dominant accept-rate
        // bug in the frankenturbo2 reference: 1.2% → 78%).
        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "ffn_residual", il);

        // Per-layer output scale (Gemma 4 layer_scalar).
        if (model.layers[il].out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].out_scale);
            cb(cur, "out_scaled", il);
        }

        cur = build_cvec(cur, il);
        cur = build_sidecar(cur, il);
        cb(cur, "l_out", il);
    }

    // post_projection: [n_embd -> n_backbone]. This is the chained-hidden
    // output — the speculative driver reads it back through res->t_embd and
    // feeds it as the next draft step's t_mtp_states input.
    ggml_tensor * mtp_embd = build_lora_mm(model.assist_post_proj, cur);
    cb(mtp_embd, "result_mtp_embd", -1);
    res->t_embd = mtp_embd;
    ggml_build_forward_expand(gf, mtp_embd);

    // Final norm + output head. Non-centroid path (use_ordered_embeddings ==
    // false, e.g. 26B-A4B): the tied `output` head directly. Centroid path is
    // a follow-up commit (build_masked_embedding_logits).
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);

    if (use_ordered_embeddings) {
        cur = build_masked_embedding_logits(cur, model.output);
    } else {
        cur = build_lora_mm(model.output, cur);
    }
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}

ggml_tensor * llama_model_gemma4_assistant::graph::build_masked_embedding_logits(ggml_tensor * hidden, ggml_tensor * lm_head_w) {
    // Reference: transformers Gemma4AssistantMaskedEmbedder
    // 1. centroid_logits = hidden @ centroids^T                  [B, L, num_centroids]
    // 2. top_k_idx       = topk(centroid_logits, k=top_k)        [B, L, top_k]
    // 3. canonical[c, j] = token_ordering[c * (V/num_centroids) + j]
    //    (token_ordering is a vocab-sized i32 permutation buffer)
    // 4. selected_emb    = lm_head[canonical[top_k_idx]]         [B, L, top_k * V_per, D]
    // 5. selected_logits = hidden @ selected_emb^T               [B, L, top_k * V_per]
    // 6. scatter selected_logits into [B, L, V] positions; everything else
    //    is filled with `min(selected_logits) - 1`.
    //
    // Implementing the masked scatter cleanly in ggml requires `scatter_rows`
    // which is not yet a graph-level op. The piece is parked here as a method
    // on the graph so the runtime patch can swap in the implementation
    // without touching the loader or the architecture metadata.
    GGML_UNUSED(hidden);
    GGML_UNUSED(lm_head_w);
    GGML_ABORT("gemma4-assistant: build_masked_embedding_logits not yet implemented");
}
