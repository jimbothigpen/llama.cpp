#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;
    llama_context_ptr ctx_dft;

    // --spec-type draft-mtp with no separate draft model path => create a second context on the
    // same loaded model with ctx_type=MTP.  The graph_mtp inner-class on qwen35/qwen35moe
    // handles the MTP tail block; no override_arch needed.
    const bool want_mtp = std::find(params.speculative.types.begin(), params.speculative.types.end(),
                                    COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != params.speculative.types.end();

    // self-spec types (ngram-*, phantom, dflash without external draft) do not need a draft context
    const bool needs_draft_ctx = std::any_of(params.speculative.types.begin(), params.speculative.types.end(),
        [](common_speculative_type t) {
            return t == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE ||
                   t == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3 ||
                   t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP    ||
                   t == COMMON_SPECULATIVE_TYPE_DFLASH;
        });

    if (want_mtp && params.speculative.draft.mparams.path.empty()) {
        char trunk_arch[64] = {0};
        llama_model_meta_val_str(model_tgt, "general.architecture", trunk_arch, sizeof(trunk_arch));

        const bool is_mtp_capable =
            std::string(trunk_arch) == "qwen35" ||
            std::string(trunk_arch) == "qwen35moe";
        if (!is_mtp_capable) {
            LOG_ERR("MTP not supported for trunk architecture '%s'\n", trunk_arch);
            return 1;
        }

        // The arch may be MTP-capable while this particular GGUF carries no NextN heads
        // (e.g. a plain Qwen3.5-9B export). ctx_type=MTP would then silently downgrade to
        // DEFAULT (llama-context.cpp) and the combined token+embd draft batch would abort
        // the XOR assert in llama_context::decode. Fail fast with an actionable message.
        if (llama_model_n_nextn_layer(model_tgt) <= 0) {
            LOG_ERR("%s", "--spec-type draft-mtp requires an MTP-bundled model: the trunk GGUF "
                          "has no NextN heads (nextn_predict_layers=0)\n");
            return 1;
        }

        auto cparams_mtp = common_context_params_to_llama(params);
        cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;

        LOG_INF("creating MTP draft context (ctx_type=MTP, same model as target)\n");

        ctx_dft.reset(llama_init_from_model(model_tgt, cparams_mtp));
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create MTP draft context\n");
            return 1;
        }

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();

        // Wire DRAFT_MTP into the speculative types so common_speculative_init
        // creates the correct MTP implementation.
        if (std::find(params.speculative.types.begin(), params.speculative.types.end(),
                      COMMON_SPECULATIVE_TYPE_DRAFT_MTP) == params.speculative.types.end()) {
            params.speculative.types.push_back(COMMON_SPECULATIVE_TYPE_DRAFT_MTP);
        }
    } else if (needs_draft_ctx) {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        // Compact-vocab EAGLE3 drafts inherit the target's tok_embd. Must run before the draft
        // context is created, since graph reserve dereferences model.tok_embd.
        common_speculative_setup_draft_model(model_dft.get(), model_tgt);

        auto cparams = common_context_params_to_llama(params_dft);
        if (want_mtp) {
            // External MTP draft (e.g. gemma4-assistant): type the draft context MTP so its
            // graph reserve is deferred until llama_set_mtp_source wires the target context;
            // otherwise the assistant graph asserts on a missing MTP source during the
            // construction-time sched_reserve (gemma4-assistant.cpp:84).
            cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        }
        ctx_dft.reset(llama_init_from_model(model_dft.get(), cparams));

        params.speculative.draft.ctx_tgt = ctx_tgt;
        params.speculative.draft.ctx_dft = ctx_dft.get();
    }

    // Init the speculator BEFORE any draft-context decode (§-FLAG-B ordering fix).
    // common_speculative_init wires the MTP source (llama_set_mtp_source) and
    // embeddings_nextn onto ctx_dft.  The first draft decode happens just below at the
    // common_context_can_seq_rm probe (and again at the prompt-eval decode further down); for an
    // MTP-typed draft context that lazy graph_reserve asserts on a missing MTP source unless the
    // source is wired first.  Its only prerequisites are params.speculative.draft.ctx_tgt/ctx_dft,
    // both set just above.  common_speculative_begin stays below — it needs prompt_tgt.
    struct common_speculative * spec = common_speculative_init(params.speculative, 1);

    // check if the context supports partial sequence removal
    const bool use_ckpt_tgt = (common_context_can_seq_rm(ctx_tgt)       == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);
    const bool use_ckpt_dft = ctx_dft && (common_context_can_seq_rm(ctx_dft.get()) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt_tgt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    llama_seq_id seq_id = 0;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));
    if (ctx_dft) {
        llama_decode(ctx_dft.get(), llama_batch_get_one(inp.data(), inp.size() - 1));
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    const auto & params_spec = params.speculative;

    // spec was initialized above (before the first draft decode); begin needs prompt_tgt (built above)
    common_speculative_begin(spec, seq_id, prompt_tgt);

    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);

    size_t n_draft = 0;

    llama_tokens draft;
    common_prompt_checkpoint ckpt;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    while (true) {
        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            ckpt.update_pos(
                    prompt_tgt.size(),
                    llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), seq_id),
                    llama_memory_seq_pos_max(llama_get_memory(ctx_tgt), seq_id));

            if (use_ckpt_dft) {
                ckpt.update_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }

            // generate a new draft
            common_speculative_get_draft_params(spec, seq_id) = {
                /* .drafting   = */ true,
                /* .n_max      = */ -1,
                /* .n_past     = */ n_past,
                /* .id_last    = */ id_last,
                /* .prompt     = */ &prompt_tgt,
                /* .result     = */ &draft, // output
            };
            common_speculative_draft(spec);

            // save the original draft size
            n_draft = draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty()) {
                if (use_ckpt_tgt) {
                    ckpt.update_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                }
            }

            if (ctx_dft) {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt_tgt) {
                GGML_ASSERT(!ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { seq_id }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { seq_id }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            llama_decode(ctx_tgt, batch_tgt);
        }

        // evaluate the same batch with the draft model
        {
            // Trim ctx_dft back to pre-draft position so re-evaluation of batch_tgt doesn't
            // collide with draft-generated KV entries. Required for M-RoPE models (Qwen3.5 family)
            // where llama_batch_allocr::init() enforces last_kv_pos < min_batch_pos (strictly <).
            // Safe/no-op for standard-RoPE models.
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }

            // update the draft model with the target's hidden states (required for MTP).
            // Mirrors the server's main loop (tools/server/server-context.cpp). Without this
            // call, common_speculative_state_draft_mtp::process() never runs → pending_h stays
            // zero across iterations, the MTP head reads garbage h_input, and draft acceptance
            // collapses by ~33pp on Qwen3.5 (38% vs 71% mainline anchor).
            if (!common_speculative_process(spec, batch_tgt)) {
                LOG_ERR("%s", "failed to process speculative batch\n");
            }
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt_tgt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt_tgt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            // MTP progress guarantee (mirror of server fix 13d638b2f). `ids` holds the
            // validated draft prefix plus one resampled tail token. For MTP-driven
            // speculation the MTP head's per-step state is NOT captured by ckpt.load_dft,
            // so reusing `ids` verbatim — including the resampled tail — desyncs the next
            // accept_n trace from the original sampler trace and locks this partial-accept
            // restore loop indefinitely (observed as endless "partial acceptance: N < M,
            // restoring checkpoint" with n_predict frozen, GPU busy on redundant re-verifies).
            // Dropping the resampled tail forces the next iteration to re-verify only the
            // validated drafts, which strictly shrinks the draft each restore and converges.
            // Gate on ids.size() > 1: when only the resample survives (zero drafts accepted),
            // keep ids[0] so the loop re-verifies the target's own prediction (always
            // re-accepted), which is what breaks the full-rejection loop — popping there
            // would empty the draft and is actively harmful. The hang only manifests at
            // n_max >= 2, where a partial accept can leave ids.size() > 1.
            if (want_mtp && ids.size() > 1) {
                ids.pop_back();
            }

            draft = std::move(ids);

            {
                ckpt.load_tgt(ctx_tgt, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, ckpt.pos_max + 1, -1);
            }

            if (ctx_dft) {
                ckpt.load_dft(ctx_dft.get(), seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, ckpt.pos_max + 1, -1);
            }

            prompt_tgt.resize(ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        if (n_draft > 0) {
            common_speculative_accept(spec, seq_id, ids.size() - 1);
        }

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;
        n_predict += ids.size();

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        for (size_t i = 0; i < ids.size(); ++i) {
            prompt_tgt.push_back(id_last);

            id_last = ids[i];

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), seq_id, n_past, -1);
            if (ctx_dft) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft.get()), seq_id, n_past, -1);
            }
        }

        if ((params.n_predict >= 0 && n_predict > params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", 100.0f * n_accept / n_drafted);

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
