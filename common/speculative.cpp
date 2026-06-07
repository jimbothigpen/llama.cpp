#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "phantom.h"
#include "sampling.h"

#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <map>
#include <cinttypes>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// Emit a one-time note when MTP is enabled on a detected iGPU/APU. After C1 (defer+batch the
// catch-up decode) MTP is iGPU-tuned to n_max=1, where it measures ~1.16x of pure decode on
// gfx1150 (32.4 vs 28.0 t/s). n_max>=2 remains a net slowdown even with C1, so the iGPU default
// is clamped to n_max=1 (see the constructor).
// Uses the generic backend device registry so CUDA, HIP, and Vulkan iGPUs are all detected.
static void mtp_warn_igpu_once() {
    static bool warned = false;
    if (warned) return;
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t d = 0; d < n_dev; d++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(d)) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            LOG_WRN("%s: MTP on integrated GPUs is tuned to n_max=1 (C1 catch-up batching): "
                    "measured ~1.16x of pure decode at n_max=1; n_max>=2 remains a net slowdown. "
                    "The iGPU default is n_max=1; override with --spec-draft-n-max.\n", __func__);
            warned = true;
            return;
        }
    }
    warned = true;
}

// True if any device in the generic backend registry is an integrated GPU (APU). Used to pick
// the iGPU-tuned MTP draft depth (n_max=1), where defer+batch catch-up (C1) makes MTP a net
// win (~1.09x of pure decode); n_max>=2 stays a slowdown even with C1.
// Covers CUDA, HIP, and Vulkan iGPUs uniformly via GGML_BACKEND_DEVICE_TYPE_IGPU.
static bool mtp_any_igpu() {
    const size_t n_dev = ggml_backend_dev_count();
    for (size_t d = 0; d < n_dev; d++) {
        if (ggml_backend_dev_type(ggml_backend_dev_get(d)) == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            return true;
        }
    }
    return false;
}

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"mtp",           COMMON_SPECULATIVE_TYPE_DRAFT_MTP},  // Alias retained for fork backward-compatibility 2026-05-25; "draft-mtp" is canonical per mainline.
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"phantom",       COMMON_SPECULATIVE_TYPE_PHANTOM},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH},
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const auto vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const auto vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    virtual llama_context * get_mtp_ctx() const { return nullptr; }

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const { return false; }

    // true if this implementation requires the target context to extract pre-norm embeddings
    virtual bool need_embd_nextn() const { return false; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        // Defense-in-depth: draft-simple dereferences ctx_dft immediately (llama_n_batch below,
        // and again every draft step). A null draft context here means the draft model's context
        // failed to initialize (e.g. the gemma4-assistant guard threw because ctx_other was not
        // wired). Fail loudly with an actionable message instead of segfaulting in n_batch().
        if (ctx_dft == nullptr) {
            throw std::runtime_error(
                "draft-simple speculator requires a draft context, but ctx_dft is null "
                "(the draft model's context failed to initialize — check that the draft "
                "context was created successfully and, for external MTP, that ctx_other was wired)");
        }

        LOG_INF("%s: adding speculative implementation 'draft-simple'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens; always keep the first
                if (cur_p->data[0].p < params.p_min) {
                    if (i == 0) {
                        common_sampler_accept(smpl, id, true);
                        dparams.at(seq_id).result->push_back(id);
                    }
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

// EAGLE3 speculative decoding: extract target hidden states → CPU FC encode → decode+sample loop
struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // FC weight dequantized to host F32 for CPU matmul
    std::vector<float> fc_weight_f32;
    int64_t n_embd;        // EAGLE3 hidden dim
    int64_t fc_input_size; // n_aux_layers × n_embd_tgt

    // draft-to-target vocab remap; empty when vocabs are identical
    std::vector<llama_token> d2t_map;

    common_speculative_impl_draft_eagle3(const common_params_speculative & sparams, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
        , params(sparams.draft)
    {
        auto * ctx_tgt = params.ctx_tgt;
        auto * ctx_dft = params.ctx_dft;
        const auto * model_eagle3 = llama_get_model(ctx_dft);
        const auto * model_tgt    = llama_get_model(ctx_tgt);

        n_embd = llama_model_n_embd(model_eagle3);
        const int32_t n_aux   = llama_model_eagle3_n_aux_layers(model_eagle3);
        const int64_t n_embd_tgt = llama_model_n_embd(model_tgt);
        fc_input_size = n_aux * n_embd_tgt;

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling sp;
            sp.no_perf = false;
            sp.top_k   = 10;
            sp.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            smpl.reset(common_sampler_init(model_eagle3, sp));
        }

        // EAGLE3 shares vocab with target — skip compat check
        // Enable target hidden state extraction
        llama_set_eagle3(ctx_tgt, model_eagle3);

        // Enable embeddings output on decoder for autoregressive recurrence
        llama_set_embeddings(ctx_dft, true);

        // Dequantize fc.weight to host F32
        const int64_t n_elements = n_embd * fc_input_size;
        fc_weight_f32.resize(n_elements);
        int64_t fc_in = llama_model_eagle3_get_fc_weight(model_eagle3, fc_weight_f32.data(), n_elements);
        GGML_ASSERT(fc_in == fc_input_size && "eagle3 fc.weight dimension mismatch");

        // Load optional draft-to-target vocab remap
        const int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_eagle3));
        d2t_map.resize(n_vocab);
        if (llama_model_eagle3_get_d2t(model_eagle3, d2t_map.data(), n_vocab) != n_vocab) {
            d2t_map.clear(); // tensor absent or unsupported type — no remap
        }

        LOG_INF("%s: EAGLE3 initialized (n_embd=%lld, fc_in=%lld, n_aux=%d, d2t=%s)\n",
                __func__, (long long)n_embd, (long long)fc_input_size, n_aux,
                d2t_map.empty() ? "none" : "present");
    }

    ~common_speculative_impl_draft_eagle3() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop — target extraction configured in constructor
    }

    bool process(const llama_batch & /*tgt_batch*/) override {
        // EAGLE3 draft doesn't mirror the target batch; extraction happens passively
        // in process_ubatch after the target decode. Just return true.
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto * ctx_tgt = params.ctx_tgt;
        auto * ctx_dft = params.ctx_dft;

        // Retrieve target hidden states extracted during last target decode
        int32_t n_features = 0;
        const float * all_features = llama_get_eagle3_target_features(ctx_tgt, &n_features);

        if (!all_features || n_features == 0) {
            LOG_DBG("%s: no target features available\n", __func__);
            return;
        }

        const int n_aux = (int)(fc_input_size / n_embd);
        const int n_tokens_batch = n_features / (n_aux * n_embd);
        if (n_tokens_batch <= 0 || n_features != n_aux * n_embd * n_tokens_batch) {
            LOG_WRN("%s: feature layout mismatch (n_features=%d)\n", __func__, n_features);
            return;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) continue;

            // Gather features for last token, all aux layers → feat_concat
            const int last_tok_idx = n_tokens_batch - 1;
            std::vector<float> feat_concat(fc_input_size);
            for (int layer = 0; layer < n_aux; layer++) {
                const float * layer_data = all_features + layer * n_embd * n_tokens_batch;
                memcpy(feat_concat.data() + layer * n_embd,
                       layer_data + last_tok_idx * n_embd,
                       n_embd * sizeof(float));
            }

            // CPU FC projection: g_embd = fc_weight × feat_concat
            std::vector<float> g_embd(n_embd, 0.0f);
            for (int64_t i = 0; i < n_embd; i++) {
                float sum = 0.0f;
                const float * row = fc_weight_f32.data() + i * fc_input_size;
                for (int64_t j = 0; j < fc_input_size; j++) {
                    sum += row[j] * feat_concat[j];
                }
                g_embd[i] = sum;
            }

            // Seed decoder with g_embd from FC and last target token
            llama_set_eagle3_g_embeddings(ctx_dft, g_embd.data(), 1);

            // The EAGLE3 draft KV is stateless per iteration — the driver rolls it back
            // to the prompt checkpoint each cycle and accepted tokens are never decoded
            // into it, so seq_pos_max stays pinned at prompt-end while dp.n_past grows.
            // Anchor the batch to the drafter's own KV (mirrors DFlash commit 003ecc2d1);
            // using dp.n_past trips llama_batch_allocr::init()'s Y=X+1 check from cycle 2.
            const llama_pos dft_pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id) + 1;

            common_batch_clear(batch);
            common_batch_add(batch, dp.id_last, dft_pos0, { seq_id }, true);

            if (llama_decode(ctx_dft, batch) != 0) {
                LOG_WRN("%s: eagle3 decoder failed (seq=%d)\n", __func__, (int) seq_id);
                return;
            }

            auto * smpl = smpls[seq_id].get();
            common_sampler_reset(smpl);

            for (int i = 0; i < params.n_max; ++i) {
                common_sampler_sample(smpl, ctx_dft, 0, true);
                const auto * cur_p = common_sampler_get_candidates(smpl, true);
                const llama_token id = cur_p->data[0].id;

                common_sampler_accept(smpl, id, true);
                dp.result->push_back(d2t_map.empty() ? id : d2t_map[id]);

                if (params.n_max <= (int) dp.result->size()) break;
                if (cur_p->data[0].p < params.p_min)          break;

                // Autoregressive: prenorm output becomes next g_embd
                const float * embd = llama_get_embeddings_ith(ctx_dft, -1);
                if (!embd) {
                    LOG_WRN("%s: no embeddings at step %d\n", __func__, i);
                    break;
                }
                llama_set_eagle3_g_embeddings(ctx_dft, embd, 1);

                common_batch_clear(batch);
                common_batch_add(batch, id, dft_pos0 + 1 + i, { seq_id }, true);

                if (llama_decode(ctx_dft, batch) != 0) {
                    LOG_WRN("%s: eagle3 decoder failed at step %d\n", __func__, i);
                    break;
                }
            }

            if ((int) dp.result->size() < params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    bool kv_shared_with_target = false;  // qwen-MTP: drafter has 0 own KV layers → shares target KV
    bool is_mem_shared = false;          // gemma4 external-assistant: ctx_dft->ctx_other == ctx_tgt

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    // === C1 (Path C): defer + batch the catch-up decode ===
    // Instead of running a standalone "catch-up" llama_decode in process() to write the draft
    // KV for the just-verified committed span, we STASH that span here and prepend it (KV-only,
    // logits off) to the NEXT cycle's lead draft decode. One decode then does both the catch-up
    // KV write and the lead — saving exactly one llama_decode per speculation cycle. The deferred
    // span occupies committed positions (< the driver's seq_rm rollback boundary ckpt.pos_max+1),
    // so it survives the per-cycle ctx_dft rollback automatically — no driver change needed.
    // This relies on the MTP draft context using partial seq_rm (load_dft a no-op, i.e.
    // use_ckpt_dft == false), which holds for all attention-KV MTP heads (Qwen NextN/MTP).
    std::vector<std::vector<llama_token>> defer_tokens;    // [n_seq][k+1] verified committed tokens
    std::vector<std::vector<float>>       defer_h;         // [n_seq][(k+1)*n_embd] verified h-feeds
    std::vector<int32_t>                  defer_n_full;    // [n_seq] rows captured (drafted span + 1)
    std::vector<uint16_t>                 last_n_accepted; // [n_seq] set by accept(), read by draft()

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd_out(llama_get_model(ctx_dft));
        GGML_ASSERT(n_embd == llama_model_n_embd(llama_get_model(ctx_tgt)) &&
                "MTP input row width must match the target h_nextn width");

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        // iGPU default: MTP only beats pure decode on an integrated GPU at n_max=1 (after
        // C1's defer+batch catch-up). Unless the user set --spec-draft-n-max explicitly,
        // clamp the draft depth to 1 on iGPU. Explicit override is always honored.
        if (!this->params.n_max_set && this->params.n_max > 1 && mtp_any_igpu()) {
            LOG_INF("%s: iGPU detected - defaulting MTP draft n_max %d -> 1 "
                    "(iGPU-tuned; ~1.09x of pure decode after C1 catch-up batching). "
                    "Override with --spec-draft-n-max.\n", __func__, this->params.n_max);
            this->params.n_max = 1;
        }

        mtp_warn_igpu_once();

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);
        llama_set_mtp_source(ctx_dft, ctx_tgt);

        kv_shared_with_target = llama_model_n_layer_kv(llama_get_model(ctx_dft)) == 0;

        is_mem_shared = llama_get_ctx_other(ctx_dft) == ctx_tgt;

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);

        defer_tokens.assign(n_seq, {});
        defer_h.assign(n_seq, {});
        defer_n_full.assign(n_seq, 0);
        last_n_accepted.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }

        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 1 && !kv_shared_with_target && !is_mem_shared) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        // single bulk GPU→CPU sync: one call forces host sync for all tgt embeddings;
        // subsequent rows read from this pointer with offset arithmetic (no extra sync per row).
        const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);

        // If kv is shared with target (e.g Gemma4) there is no catch-up at all.
        // Otherwise (Qwen NextN/MTP, own KV) C1 Path C DEFERS the catch-up decode: instead of
        // re-decoding the just-verified committed span into ctx_dft here, stash it (tokens +
        // positions + verified h-feeds) and let the next cycle's draft() prepend it (KV-only) to
        // the lead decode. One decode then writes both the catch-up KV and the lead — saving one
        // llama_decode per cycle. Must run BEFORE the pending_h update loop below: row 0's h-feed
        // is the *current* pending_h (h of the previously committed token), not the new one.
        if (!kv_shared_with_target) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                const int32_t beg = i_batch_beg[seq_id];
                const int32_t end = i_batch_end[seq_id];
                if (beg < 0) {
                    defer_n_full[seq_id] = 0;
                    continue;
                }

                // Assumes each sequence occupies a contiguous run in the batch (same assumption
                // the previous catch-up's right-shift made). Holds for the spec drivers.
                const int32_t n_rows = end - beg + 1;
                defer_n_full[seq_id] = n_rows;
                defer_tokens[seq_id].resize(n_rows);
                defer_h[seq_id].resize((size_t) n_rows * n_embd);

                for (int32_t j = 0; j < n_rows; ++j) {
                    defer_tokens[seq_id][j] = batch_in.token[beg + j];

                    // To write KV at this position we feed the verified pre-norm hidden of the
                    // PREVIOUS position: row 0 (the verified id_last) pairs with the current
                    // pending_h; row j>=1 pairs with h_tgt[beg+j-1] (target's verified hidden,
                    // shifted right by one — same shift the old catch-up batch applied).
                    const float * src = (j == 0)
                        ? pending_h[seq_id].data()
                        : (h_tgt + (size_t) (beg + j - 1) * n_embd);
                    std::memcpy(defer_h[seq_id].data() + (size_t) j * n_embd, src, row_bytes);
                }
            }
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = h_tgt + (size_t)(i_batch_beg[seq_id] + i) * n_embd;
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        // C1: batch index of each seq's lead token in the FIRST decode. Because we prepend the
        // deferred catch-up span (KV-only, logits off) before each lead, the lead is no longer at
        // batch index 0/1/2…; the chain-sampling logits/embd reads index by BATCH position
        // (get_logits_ith / get_embeddings_nextn_ith resolve via output_ids), so the first pass
        // must read at the lead's true batch index, not the sequential per-seq counter.
        std::vector<int32_t> lead_ibatch(n_seq, -1);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            // C1 Path C: prepend the deferred catch-up span - the verified committed tokens
            // (id_last + last cycle's accepted drafts) whose draft-KV was not written because
            // the standalone catch-up decode was removed. KV-only (logits off); their verified
            // h-feeds were stashed in process(). One decode below writes this span's KV and the
            // new lead together. No-op for shared-KV (defer_n_full=0) and the first cycle.
            const int32_t n_def = std::min<int32_t>((int32_t) last_n_accepted[seq_id] + 1, defer_n_full[seq_id]);
            // The deferred span is exactly the n_def committed tokens immediately before the new
            // lead, so derive their positions from dp.n_past (current/authoritative) rather than
            // storing absolute positions in process() — robust to a context-shift in between.
            for (int32_t j = 0; j < n_def; ++j) {
                const llama_pos p = dp.n_past - n_def + j;
                common_batch_add(batch, defer_tokens[seq_id][j], p, { seq_id }, false);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1),
                            defer_h[seq_id].data() + (size_t) j * n_embd, row_bytes);
            }

            // lead token (logits on). Record its batch index for the first chain-sampling pass
            // (the KV-only prefix above shifts it off the sequential position).
            lead_ibatch[seq_id] = batch.n_tokens;
            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;
        bool first_pass = true;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                // First pass reads the lead's logits/embd at its TRUE batch index (the KV-only
                // deferred prefix shifts the lead off the sequential position; get_logits_ith /
                // get_embeddings_nextn_ith index by batch position via output_ids). Later passes
                // rebuild a prefix-free batch, so the sequential counter equals the batch index.
                const int32_t s_idx = first_pass ? lead_ibatch[seq_id] : i_batch;
                common_sampler_sample(smpl, ctx_dft, s_idx, true);
                h_row = llama_get_embeddings_nextn_ith(ctx_dft, s_idx);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens; always keep the first
                if (cur_p->data[0].p < params.p_min) {
                    if (i == 0) {
                        common_sampler_accept(smpl, id, true);
                        dparams.at(seq_id).result->push_back(id);
                    }
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            first_pass = false;
            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        // C1: the next draft() prepends the verified committed span [id_last, accepted...]
        // - that is (n_accepted + 1) rows of the deferred stash captured in process().
        last_n_accepted[seq_id] = n_accepted;

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-simple'\n", __func__);
        LOG_INF("%s: - size_n=%d, size_m=%d, min_hits=%d\n", __func__,
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(this->type).c_str());
        LOG_INF("%s: - size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", __func__,
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: adding speculative implementation 'ngram-mod'\n", __func__);
        LOG_INF("%s: - n_match=%d, n_max=%d, n_min=%d\n", __func__,
                this->params.n_match, this->params.n_max, this->params.n_min);
        LOG_INF("%s: - mod size=%zu (%.3f MB)\n", __func__,
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) - resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-cache'\n", __func__);
        LOG_INF("%s: - n_draft=%d, cache_static=%s, cache_dynamic=%s\n", __func__,
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_phantom : public common_speculative_impl {
    common_params_speculative params;
    common_ngram_mod mod;

    // one phantom instance per seq (phantom is inherently single-seq per instance)
    std::vector<std::unique_ptr<common_speculative_state_phantom>> phantoms;

    common_speculative_impl_phantom(const common_params_speculative & p, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_PHANTOM, n_seq)
        , params(p)
        , mod(p.ngram_mod.n_match, 4*1024*1024)
    {
        phantoms.reserve(n_seq);
        for (uint32_t i = 0; i < n_seq; ++i) {
            phantoms.push_back(std::make_unique<common_speculative_state_phantom>(
                mod,
                p.phantom_bloom_bits,
                p.phantom_buffers,
                /*ghost_cap=*/ 64,
                p.ngram_mod.n_min,
                p.ngram_mod.n_max));
        }
        LOG_INF("%s: phantom n_seq=%u bloom_bits=%d ghost_buffers=%d ngram_n_match=%d\n",
                __func__, n_seq, p.phantom_bloom_bits, p.phantom_buffers,
                p.ngram_mod.n_match);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        if (seq_id >= 0 && seq_id < (llama_seq_id)phantoms.size()) {
            phantoms[seq_id]->begin(prompt);
        }
    }

    bool process(const llama_batch &) override { return true; }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id)n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }
            phantoms[seq_id]->draft(*dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id >= 0 && seq_id < (llama_seq_id)phantoms.size()) {
            phantoms[seq_id]->accept(n_accepted);
        }
    }
};

// ---- DFlash block-diffusion speculative decoding ----
// Port from buun/master (bc9340b97f4c) adapted to ygg's llama_cross cross-attn API.
// S2 scope: CPU ring buffer + eval-callback hidden state capture. GPU tape deferred to S3.

struct common_speculative_impl_dflash : public common_speculative_impl {
    llama_context       * ctx_tgt;
    llama_context       * ctx_dft;
    const llama_model   * model_dft;
    llama_seq_id    seq_id = 0;

    float p_min;

    int block_size;
    llama_token mask_token_id;
    int n_target_layers;
    int n_embd;
    int n_target_features;

    // Ring buffer for target hidden states
    static constexpr int RING_SIZE = 4096;
    std::vector<std::vector<float>> ring_buf; // [n_target_layers][RING_SIZE * n_embd]
    int ring_write_pos = 0;
    int ring_filled    = 0;
    int committed_len  = 0;

    // Interleaved cross-attention buffer
    std::vector<float> cross_buf;

    // Drafter ctx window (matches LLAMA_DFLASH_PER_SLOT_CTX)
    static constexpr int ctx_window = LLAMA_DFLASH_PER_SLOT_CTX;

    // GPU ring handle (nullptr = CPU path)
    void * gpu_ring_handle = nullptr;

    // Adaptive draft length
    int n_low_accept   = 0;
    int n_draft_last   = 0;
    int adaptive_n_draft = -1;

    llama_batch batch_dft;

    // Build interleaved cross-attention data and inject into drafter context.
    // Returns cross_len (number of context tokens), or 0 if ring is empty.
    int build_cross_data(llama_context * ctx) {
        if (gpu_ring_handle) {
            // GPU ring path (not available in S2 — handle is always nullptr)
            int gpu_write_pos = ring_write_pos % ctx_window;
            int gpu_filled    = std::min(ring_filled, ctx_window);
            llama_dflash_cross_ring_gpu_set_cross(ctx, gpu_ring_handle, seq_id,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, ctx_window);
            return gpu_filled;
        }

        int cross_len  = std::min(ring_filled, ctx_window > 0 ? ctx_window : ring_filled);
        cross_buf.resize((size_t)n_target_features * cross_len);
        int read_start = (ring_write_pos - cross_len + RING_SIZE) % RING_SIZE;

        for (int t = 0; t < cross_len; ++t) {
            int slot = (read_start + t) % RING_SIZE;
            for (int layer = 0; layer < n_target_layers; ++layer) {
                memcpy(&cross_buf[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                       ring_buf[layer].data() + (size_t)slot * n_embd,
                       n_embd * sizeof(float));
            }
        }

        // Inject into drafter context via ygg's llama_cross struct
        llama_set_cross_data(ctx, cross_buf.data(), n_target_features, cross_len);
        return cross_len;
    }

    common_speculative_impl_dflash(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DFLASH, n_seq)
        , ctx_tgt(params.draft.ctx_tgt)
        , ctx_dft(params.draft.ctx_dft)
        , model_dft(llama_get_model(params.draft.ctx_dft))
        , p_min(params.draft.p_min)
    {
        block_size        = llama_model_dflash_block_size(model_dft);
        mask_token_id     = (llama_token) llama_model_dflash_mask_token_id(model_dft);
        n_target_layers   = llama_model_dflash_n_target_layers(model_dft);
        n_embd            = llama_model_n_embd(model_dft);
        n_target_features = llama_model_dflash_n_target_features(model_dft);

        ring_buf.resize(n_target_layers);
        for (int i = 0; i < n_target_layers; ++i) {
            ring_buf[i].resize((size_t)RING_SIZE * n_embd, 0.0f);
        }

        std::vector<int32_t> capture_layers(n_target_layers);
        llama_model_dflash_target_layer_ids(model_dft, capture_layers.data(), n_target_layers);
        llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);

        batch_dft = llama_batch_init(block_size, 0, 1);

        // Try GPU ring (S2: always returns nullptr → CPU fallback)
        gpu_ring_handle = llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, ctx_window);

        {
            std::string ids_str;
            for (int i = 0; i < n_target_layers; ++i) {
                if (i) ids_str += ", ";
                ids_str += std::to_string(capture_layers[i]);
            }
            LOG_INF("dflash: block_size=%d, mask_token=%d, n_target_layers=%d, n_embd=%d, target_ids=[%s]\n",
                    block_size, mask_token_id, n_target_layers, n_embd, ids_str.c_str());
        }
    }

    ~common_speculative_impl_dflash() override {
        llama_dflash_cross_ring_gpu_free(gpu_ring_handle);
        llama_batch_free(batch_dft);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        capture_target_hiddens();
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    bool need_embd() const override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id sid = 0; sid < (llama_seq_id) n_seq; ++sid) {
            auto & dp = dparams[sid];
            if (!dp.drafting) {
                continue;
            }

            llama_tokens & result  = *dp.result;
            const llama_token id_last = dp.id_last;
            const int32_t n_max_eff   = dp.n_max > 0 ? dp.n_max : (block_size - 1);

            const int n_draft_base = adaptive_n_draft > 0 ? adaptive_n_draft : (block_size - 1);
            const int n_draft      = std::min(n_draft_base, n_max_eff);

            if (committed_len == 0) {
                continue;
            }

            const int64_t t0 = ggml_time_us();

            // build_cross_data injects the target-hidden ring into the drafter context via
            // llama_set_cross_data (side effect); its return value (the cross-attention ring
            // length) must NOT be used to position the drafter batch — see below.
            build_cross_data(ctx_dft);

            const int64_t t1 = ggml_time_us();

            // Drafter batch position must follow the drafter context's *own* KV, not the cross
            // length. The cross-attention ring grows by (n_accepted+1) every iteration as target
            // hiddens are committed (append_target_hiddens), but the drafter context is stateless
            // per iteration: speculative-simple trims it back to the prompt checkpoint each round,
            // so its last KV position stays put. Positioning the batch at the ring length only
            // coincides with the drafter KV on the first iteration; thereafter it exceeds kv_pos+1
            // and llama_batch_allocr::init() rejects the batch (Y = X+1 violated), so every draft
            // decode after the first fails. Anchor the batch to the drafter KV instead.
            const llama_pos dft_pos0 = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), sid) + 1;

            // Build drafter batch: [id_last, mask, mask, ..., mask]
            const int batch_len = n_draft + 1;
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, id_last,       dft_pos0,     { sid }, true);
            for (int i = 1; i < batch_len; ++i) {
                common_batch_add(batch_dft, mask_token_id, dft_pos0 + i, { sid }, true);
            }

            const int64_t t2 = ggml_time_us();

            int ret = llama_decode(ctx_dft, batch_dft);
            if (ret != 0) {
                LOG_ERR("dflash: drafter decode failed with %d\n", ret);
                continue;
            }

            const int64_t t3 = ggml_time_us();

            // CPU argmax (GPU argmax stubs return nullptr in S2)
            {
                int32_t * argmax       = llama_get_logits_argmax(ctx_dft);
                float   * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
                const int K_flat       = llama_get_logits_argmax_k(ctx_dft);

                if (argmax) {
                    // GPU argmax path (available if llama_get_logits_argmax returns non-null)
                    for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                        if (argmax_probs && p_min > 0.0f && i > 1) {
                            float log_prob  = argmax_probs[i * K_flat];
                            float log_p_min = logf(p_min);
                            if (log_prob < log_p_min) {
                                break;
                            }
                        }
                        result.push_back((llama_token) argmax[i * K_flat]);
                    }
                } else {
                    // CPU fallback: argmax over full vocab
                    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                    for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                        float * logits = llama_get_logits_ith(ctx_dft, i);
                        if (!logits) {
                            break;
                        }
                        llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
                        result.push_back(best);
                    }
                }
            }

            const int64_t t4 = ggml_time_us();

            n_draft_last = (int) result.size();

            LOG_DBG("dflash draft (ctx=%d): concat=%.1fms batch=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
                    committed_len,
                    (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3, (t4 - t0) / 1e3);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted, bool /*is_other*/) override {
        // Adaptive draft length
        if (n_draft_last > 0) {
            float f_acc = (float) n_accepted / (float) n_draft_last;
            if (f_acc < 0.3f) {
                n_low_accept++;
                if (n_low_accept >= 3) {
                    int base = adaptive_n_draft > 0 ? adaptive_n_draft : (block_size - 1);
                    adaptive_n_draft = std::max(1, base / 2);
                    LOG_DBG("dflash: low acceptance streak (%d) — reducing draft to %d\n",
                            n_low_accept, adaptive_n_draft);
                    n_low_accept = 0;
                }
            } else {
                n_low_accept = 0;
                if (f_acc > 0.6f && adaptive_n_draft > 0) {
                    adaptive_n_draft = std::min(block_size - 1, adaptive_n_draft + 1);
                }
            }
        }

        // Append hidden states: id_last (1) + accepted drafts (n_accepted)
        // The target context's eval callback captured all verification-batch hidden states;
        // we commit the first (n_accepted + 1) of them to the ring.
        append_target_hiddens((int)n_accepted + 1);
    }

private:
    void ring_write(int n_tokens, int src_offset = 0) {
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data    = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd    = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            int64_t ntok    = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0 || embd <= 0) continue;

            int to_write = std::min(n_tokens, (int)ntok - src_offset);
            if (to_write <= 0) continue;

            for (int t = 0; t < to_write; ++t) {
                int slot = (ring_write_pos + t) % RING_SIZE;
                memcpy(ring_buf[layer].data() + (size_t)slot * embd,
                       data + (size_t)(src_offset + t) * embd,
                       embd * sizeof(float));
            }

            if (gpu_ring_handle && to_write > 0) {
                int gpu_pos = ring_write_pos % ctx_window;
                llama_dflash_cross_ring_gpu_write(gpu_ring_handle, layer, gpu_pos,
                    data + (size_t)src_offset * embd, to_write, (int)embd);
            }
        }
        ring_write_pos = (ring_write_pos + n_tokens) % RING_SIZE;
        ring_filled    = std::min(ring_filled + n_tokens, RING_SIZE);
    }

    void capture_target_hiddens() {
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) return;

        int start_offset = std::max(0, (int)n_tokens - RING_SIZE);
        int to_store     = (int)n_tokens - start_offset;

        ring_write_pos = 0;
        ring_filled    = 0;
        ring_write(to_store, start_offset);
        committed_len  = (int)n_tokens;
    }

    void append_target_hiddens(int n_tokens) {
        if (n_tokens <= 0) return;
        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        ring_write(n_tokens);
        committed_len += n_tokens;
    }
};

struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:           return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_PHANTOM:       return "phantom";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_PHANTOM:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_DFLASH:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_model_path = !params.draft.mparams.path.empty();
        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_mtp          = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;
        bool has_draft_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3)) &&
                                params.draft.ctx_dft != nullptr &&
                                llama_model_eagle3_n_aux_layers(llama_get_model(params.draft.ctx_dft)) > 0;

        // Whether a draft-context speculator was *explicitly requested* (regardless of whether its
        // context actually built). has_mtp/has_draft_eagle3/has_dflash above gate on ctx_dft != null,
        // so a draft type whose context failed to initialize collapses them to false. We must not let
        // that silently downgrade to the draft-simple fallback below (which also needs ctx_dft and
        // would then segfault on a null draft ctx) — keep the user's explicit intent visible here.
        bool requested_mtp    = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP));
        bool requested_eagle3 = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3));
        bool requested_dflash = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DFLASH));

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));
        bool has_phantom       = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_PHANTOM));
        bool has_dflash        = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DFLASH)) &&
                                params.draft.ctx_dft != nullptr && params.draft.ctx_tgt != nullptr &&
                                llama_model_dflash_block_size(llama_get_model(params.draft.ctx_dft)) > 0;

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 11);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_phantom) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_PHANTOM, params));
        }
        if (has_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        }
        if (has_draft_simple) {
            if (!has_draft_model_path) {
                LOG_WRN("%s: draft model is not specified - cannot use 'draft' type\n", __func__);
                has_draft_simple = false;
            }
        } else if (has_draft_model_path && !(has_dflash || has_mtp || has_draft_eagle3) &&
                   !(requested_mtp || requested_eagle3 || requested_dflash)) {
            // Only auto-enable draft-simple when the user did NOT explicitly request a
            // draft-context speculator. If they requested mtp/eagle3/dflash but its context
            // failed to build (has_* collapsed to false), do NOT silently fall back to
            // draft-simple — fall through and fail loudly rather than running the wrong speculator.
            LOG_WRN("%s: draft model is specified but 'draft' speculative type is not explicitly enabled - enabling it\n", __func__);
            has_draft_simple = true;
        }

        if (has_draft_simple) {
            if (params.draft.ctx_dft == nullptr) {
                // draft-simple cannot run without a draft context; selecting it here would
                // segfault in its ctor (llama_n_batch on a null ctx). Refuse loudly.
                LOG_ERR("%s: 'draft-simple' was selected but the draft context is null - "
                        "the draft model's context failed to initialize; not enabling draft-simple\n", __func__);
            } else {
                configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
            }
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        // (removed duplicate has_mtp push 2026-05-25 — M-RoPE checkpoint fix)
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_PHANTOM: {
                impls.push_back(std::make_unique<common_speculative_impl_phantom>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_impl_dflash>(config.params, n_seq));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_setup_draft_model(struct llama_model * model_dft, const struct llama_model * model_tgt) {
    if (model_dft == nullptr || model_tgt == nullptr) {
        return;
    }
    // Compact-vocab EAGLE3 drafts have no token embeddings of their own; inherit the target's.
    if (llama_model_eagle3_get_tok_embd(model_dft) == nullptr) {
        struct ggml_tensor * tgt_tok_embd = llama_model_eagle3_get_tok_embd(model_tgt);
        if (tgt_tok_embd != nullptr) {
            llama_model_eagle3_set_tok_embd(model_dft, tgt_tok_embd);
            LOG_INF("%s: draft inheriting target's tok_embd (compact-vocab EAGLE3)\n", __func__);
        }
    }
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

llama_context * common_speculative_get_mtp_ctx(common_speculative * spec, llama_seq_id seq_id) {
    GGML_UNUSED(seq_id);

    if (spec == nullptr) {
        return nullptr;
    }

    for (auto & impl : spec->impls) {
        llama_context * ctx = impl->get_mtp_ctx();
        if (ctx != nullptr) {
            return ctx;
        }
    }

    return nullptr;
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}

static void mtp_update_kv_cache(struct llama_context * ctx, const llama_batch & batch, bool is_prompt_warmup) {
    if (batch.n_tokens == 0) {
        return;
    }

    // F5 (PR #1601): clear any stale MTP cells at or after the start of this
    // batch before writing. The MTP head has its own KV cache that may still
    // hold leftover positions from a previous WARMUP / UPDATE_ACCEPTED pass;
    // without this seq_rm the new write lands on top of stale cells and the
    // MTP head reads corrupted KV state (degenerate, non-deterministic drafts).
    {
        const llama_seq_id seq_id    = batch.seq_id[0][0];
        const llama_pos    start_pos = batch.pos[0];
        if (llama_memory_seq_pos_max(llama_get_memory(ctx), seq_id) >= start_pos) {
            llama_memory_seq_rm(llama_get_memory(ctx), seq_id, start_pos, -1);
        }
    }

    LOG_DBG("[MTP-UPDATE|%s] Updating %d tokens...\n", is_prompt_warmup ? "PROMPT_WARMUP" : "GEN_ACCEPTED", batch.n_tokens);

    llama_batch mtp_batch = batch;
    if (is_prompt_warmup) {
        llama_set_mtp_op_type(ctx, MTP_OP_WARMUP);
    } else {
        llama_set_mtp_op_type(ctx, MTP_OP_UPDATE_ACCEPTED);
    }

    for (int i = 0; i < mtp_batch.n_tokens; ++i) {
        mtp_batch.logits[i] = true;
    }
    llama_decode(ctx, mtp_batch);
    llama_set_mtp_op_type(ctx, MTP_OP_NONE);
}

static void mtp_accept_tokens(
    struct llama_context * ctx,
    const std::vector<llama_token> & ids,
    int32_t n_past_base,
    llama_seq_id seq_id
) {
    if (ids.empty()) {
        return;
    }

    llama_batch accepted_batch = llama_batch_init(ids.size(), 0, 1);
    for (size_t i = 0; i < ids.size(); ++i) {
        common_batch_add(accepted_batch, ids[i], n_past_base + i, { seq_id }, true);
    }

    mtp_update_kv_cache(ctx, accepted_batch, false);

    llama_batch_free(accepted_batch);
}
