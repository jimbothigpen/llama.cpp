// Block-diffusion generation for diffusion-gemma (mainline DRAFT PR #24427, ported).
//
// Reference block-diffusion loop (EntropyBoundSampler + StableAndConfident stopping + linear
// temperature schedule) with KV-cache reuse:
//
//   * ENCODER phase (causal, no self-conditioning): the prompt is prefilled once into the unified
//     sliding-window KV cache; its per-layer K/V become the read-only prefix.
//   * DECODER phase (bidirectional, self-conditioned): each denoising step decodes only the canvas
//     tokens at positions [n_past, n_past+canvas). They read the cached prefix and attend the
//     canvas bidirectionally. After reading the logits the canvas K/V is rolled back so the cache
//     keeps only the committed prefix.
//
// This is the generic host-sampling port: full-softmax over the vocabulary on the CPU, with DENSE
// self-conditioning (the previous step's full softmax fed back through llama_set_diffusion_self_cond).
// The CUDA fast-sampling / sparse top-k / device-loop paths from the PR are intentionally omitted.

#include "arg.h"
#include "chat.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <random>
#include <string>
#include <vector>

// reference defaults from generation_config.json / DiffusionGemmaGenerationConfig
static constexpr int   DEF_CANVAS_LENGTH     = 256;
static constexpr int   DEF_MAX_DENOISE_STEPS = 48;
static constexpr float ENTROPY_BOUND         = 0.1f;   // EntropyBoundSamplerConfig.entropy_bound
static constexpr float TEMP_MIN              = 0.4f;   // LinearTemperatureScheduleConfig.t_min
static constexpr float TEMP_MAX              = 0.8f;   // LinearTemperatureScheduleConfig.t_max
static constexpr float CONFIDENCE_THRESHOLD  = 0.005f; // StableAndConfident.confidence_threshold
static constexpr int   STABILITY_THRESHOLD   = 1;      // StableAndConfident.stability_threshold

// apply the model's chat template to the user prompt (this is a chat-trained model)
static std::string format_chat(llama_model * model, const std::string & prompt) {
    auto tmpls = common_chat_templates_init(model, "");
    common_chat_templates_inputs inputs;
    common_chat_msg user;
    user.role = "user";
    user.content = prompt;
    inputs.messages.push_back(user);
    inputs.add_generation_prompt = true;
    return common_chat_templates_apply(tmpls.get(), inputs).prompt;
}

int main(int argc, char ** argv) {
    common_params params;
    params.diffusion.steps = DEF_MAX_DENOISE_STEPS;
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DIFFUSION)) {
        return 1;
    }
    common_init();

    const int   canvas_length = DEF_CANVAS_LENGTH;
    const int   n_steps       = std::max(params.diffusion.steps, 1);
    const int   blocks_from_n = params.n_predict > 0 ? (params.n_predict + canvas_length - 1) / canvas_length : 1;
    const int   max_canvases  = std::max(blocks_from_n, 1);
    const float entropy_bound = ENTROPY_BOUND;

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = params.n_gpu_layers >= 0 ? params.n_gpu_layers : 999;
    model_params.devices      = params.devices.data();
    model_params.use_mmap     = params.use_mmap;
    model_params.check_tensors = params.check_tensors;

    llama_model * model = llama_model_load_from_file(params.model.path.c_str(), model_params);
    if (!model) {
        LOG_ERR("error: failed to load model '%s'\n", params.model.path.c_str());
        return 1;
    }
    if (!llama_model_is_diffusion(model)) {
        LOG_ERR("error: not a diffusion model\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab * vocab   = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);

    // build + chat-format + tokenize the prompt prefix
    std::vector<llama_token> prompt_tokens;
    if (!params.prompt.empty()) {
        const std::string formatted = format_chat(model, params.prompt);
        LOG_INF("formatted prompt: %s\n", formatted.c_str());
        prompt_tokens = common_tokenize(vocab, formatted, /*add_special*/ false, /*parse_special*/ true);
    }
    const int prefix_len = (int) prompt_tokens.size();

    // ctx holds the committed prefix (prompt + finalized canvases), the canvas being denoised, plus
    // one canvas of headroom (the in-flight canvas K/V is written then rolled back each step).
    const int n_ctx_min = prefix_len + (max_canvases + 1) * canvas_length;
    const int n_ctx     = std::max<int>(n_ctx_min, (int) params.n_ctx);
    const int n_ub      = std::max(std::max(prefix_len, canvas_length), 1);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx    = n_ctx;
    ctx_params.n_batch  = n_ub;
    ctx_params.n_ubatch = n_ub;
    ctx_params.no_perf  = params.no_perf;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("error: failed to create context\n");
        llama_model_free(model);
        return 1;
    }
    llama_set_n_threads(ctx, params.cpuparams.n_threads, params.cpuparams_batch.n_threads);
    llama_memory_t mem = llama_get_memory(ctx);

    LOG_INF("diffusion-gemma: prefix=%d canvas=%d max_canvases=%d steps=%d entropy_bound=%.3f temp=[%.2f,%.2f] n_ctx=%d\n",
            prefix_len, canvas_length, max_canvases, n_steps, entropy_bound, TEMP_MIN, TEMP_MAX, n_ctx);

    std::mt19937 rng(params.sampling.seed == LLAMA_DEFAULT_SEED ? 1234u : params.sampling.seed);
    std::uniform_int_distribution<int>    rand_tok(0, n_vocab - 1);
    std::uniform_real_distribution<float> rand_unif(0.0f, 1.0f);

    llama_batch batch = llama_batch_init(n_ub, 0, 1);

    // ---- ENCODER phase: prefill the prompt prefix into the KV cache (no self-conditioning) ----
    int n_past = 0;
    llama_set_diffusion_self_cond(ctx, nullptr, 0, 0);
    if (prefix_len > 0) {
        llama_set_causal_attn(ctx, true);
        batch.n_tokens = prefix_len;
        for (int i = 0; i < prefix_len; ++i) {
            batch.token[i]     = prompt_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == prefix_len - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("error: prompt prefill (encoder) decode failed\n");
            return 1;
        }
        n_past = prefix_len;
        LOG_INF("prefill (encoder, no self-cond): %d tokens\n", prefix_len);
    }

    std::vector<llama_token> canvas(canvas_length);
    std::vector<llama_token> argmax_canvas(canvas_length, -1);
    std::vector<llama_token> prev_argmax(canvas_length, -1);
    std::vector<llama_token> accepted(canvas_length);
    // dense self-conditioning probs for the NEXT step: {n_vocab, n_tokens} column-major
    std::vector<float> sc_probs((size_t) n_vocab * canvas_length, 0.0f);

    std::vector<llama_token> generated;
    bool done = false;

    const auto t_gen_start = std::chrono::steady_clock::now();
    int n_blocks_run = 0, n_steps_total = 0;

    for (int block = 0; block < max_canvases && !done; ++block) {
        ++n_blocks_run;
        for (auto & t : canvas) t = rand_tok(rng);
        std::fill(prev_argmax.begin(), prev_argmax.end(), -1);
        llama_set_diffusion_self_cond(ctx, nullptr, 0, 0); // first step: zero self-conditioning

        for (int cur_step = n_steps; cur_step >= 1; --cur_step) {
            ++n_steps_total;

            // decode the canvas only (bidirectional, self-conditioned), reading the cached prefix
            llama_set_causal_attn(ctx, false);
            batch.n_tokens = canvas_length;
            for (int j = 0; j < canvas_length; ++j) {
                batch.token[j]     = canvas[j];
                batch.pos[j]       = n_past + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = 1;
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("error: llama_decode failed at step %d\n", cur_step);
                return 1;
            }

            const float temp = TEMP_MIN + (TEMP_MAX - TEMP_MIN) * ((float) cur_step / (float) n_steps);

            std::vector<float>       entropy(canvas_length);
            std::vector<llama_token> sampled(canvas_length);

            const float * logits = llama_get_logits(ctx); // canvas rows [0, canvas_length)
            std::vector<float> probs(n_vocab);
            std::fill(sc_probs.begin(), sc_probs.end(), 0.0f);

            for (int j = 0; j < canvas_length; ++j) {
                const float * lg = logits + (size_t) j * n_vocab;
                float maxl = -INFINITY;
                int   amax = 0;
                for (int v = 0; v < n_vocab; ++v) {
                    const float x = lg[v] / temp;
                    if (x > maxl) { maxl = x; amax = v; }
                }
                float sum = 0.0f;
                for (int v = 0; v < n_vocab; ++v) {
                    const float p = expf(lg[v] / temp - maxl);
                    probs[v] = p;
                    sum += p;
                }
                // entropy + multinomial sample over the full (renormalized) softmax
                float ent = 0.0f;
                const float r = rand_unif(rng) * sum;
                float cum = 0.0f;
                int   tok = amax;
                bool  picked = false;
                float * scj = sc_probs.data() + (size_t) j * n_vocab;
                for (int v = 0; v < n_vocab; ++v) {
                    const float p = probs[v] / sum;
                    if (p > 0.0f) ent -= p * logf(p);
                    scj[v] = p; // dense self-cond for the next step
                    cum += probs[v];
                    if (!picked && cum >= r) { tok = v; picked = true; }
                }
                entropy[j]       = ent;
                sampled[j]       = tok;
                argmax_canvas[j] = amax;
            }

            // roll back the canvas K/V written by this decode; keep only the committed prefix
            llama_memory_seq_rm(mem, 0, n_past, -1);

            // entropy-bound accept: sort positions by entropy ascending, accept the prefix where
            // sum(entropy of all-but-last) <= entropy_bound
            std::vector<int> order(canvas_length);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b) { return entropy[a] < entropy[b]; });

            std::vector<char> accept_mask(canvas_length, 0);
            float prefix = 0.0f;
            int   n_accept = 0;
            for (int k = 0; k < canvas_length; ++k) {
                if (prefix <= entropy_bound) {
                    accept_mask[order[k]] = 1;
                    prefix += entropy[order[k]];
                } else {
                    break;
                }
            }
            for (int i = 0; i < canvas_length; ++i) {
                if (accept_mask[i]) { accepted[i] = sampled[i]; ++n_accept; }
            }

            const float mean_ent = std::accumulate(entropy.begin(), entropy.end(), 0.0f) / canvas_length;
            const bool  stable    = (STABILITY_THRESHOLD == 0) || (argmax_canvas == prev_argmax);
            const bool  confident = mean_ent < CONFIDENCE_THRESHOLD;
            LOG_INF("step %3d  temp=%.3f  accepted=%4d/%d  mean_entropy=%.4f%s\n",
                    cur_step, temp, n_accept, canvas_length, mean_ent,
                    (stable && confident) ? "  [STOP]" : "");
            if (stable && confident) {
                break;
            }
            prev_argmax = argmax_canvas;

            // self-conditioning for the NEXT step (dense)
            llama_set_diffusion_self_cond(ctx, sc_probs.data(), n_vocab, canvas_length);

            // renoise non-accepted positions with fresh random tokens
            for (int i = 0; i < canvas_length; ++i) {
                canvas[i] = accept_mask[i] ? accepted[i] : rand_tok(rng);
            }
        }

        // block output = the last stable denoising step's per-position argmax
        const std::vector<llama_token> & block_out = argmax_canvas;
        generated.insert(generated.end(), block_out.begin(), block_out.end());
        for (int j = 0; j < canvas_length; ++j) {
            if (llama_vocab_is_eog(vocab, block_out[j])) { done = true; break; }
        }

        // COMMIT (encoder phase): if another block follows, write the finalized canvas K/V (plain,
        // causal) and advance the prefix pointer.
        if (!done && block + 1 < max_canvases) {
            llama_set_causal_attn(ctx, true);
            llama_set_diffusion_self_cond(ctx, nullptr, 0, 0);
            batch.n_tokens = canvas_length;
            for (int j = 0; j < canvas_length; ++j) {
                batch.token[j]     = block_out[j];
                batch.pos[j]       = n_past + j;
                batch.n_seq_id[j]  = 1;
                batch.seq_id[j][0] = 0;
                batch.logits[j]    = (j == canvas_length - 1) ? 1 : 0;
            }
            if (llama_decode(ctx, batch) != 0) {
                LOG_ERR("error: canvas commit (encoder) decode failed at block %d\n", block);
                break;
            }
            n_past += canvas_length;
            LOG_INF("committed block %d -> n_past=%d\n", block, n_past);
        }
    }

    const double gen_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_gen_start).count();

    LOG_INF("\n=== generated canvas ===\n%s\n", common_detokenize(vocab, generated, false).c_str());

    // extract the final response: drop a leading "<|channel>thought ... <channel|>" block if present
    llama_token chan_close = LLAMA_TOKEN_NULL;
    {
        auto t = common_tokenize(vocab, "<channel|>", false, true);
        if (t.size() == 1) chan_close = t[0];
    }
    const int n_gen = (int) generated.size();
    int start = 0;
    if (chan_close != LLAMA_TOKEN_NULL) {
        for (int j = 0; j < n_gen; ++j) if (generated[j] == chan_close) start = j + 1;
    }
    std::vector<llama_token> answer;
    for (int j = start; j < n_gen; ++j) {
        if (llama_vocab_is_eog(vocab, generated[j])) break;
        answer.push_back(generated[j]);
    }
    LOG_INF("=== answer ===\n%s\n", common_detokenize(vocab, answer, false).c_str());

    LOG_INF("=== perf ===\ngeneration: %d block(s), %d denoising steps, %d canvas tokens in %.2f s\n",
            n_blocks_run, n_steps_total, n_blocks_run * canvas_length, gen_s);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
