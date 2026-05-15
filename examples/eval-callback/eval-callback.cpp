#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    // Multi-seq diagnostic split (zaya1 P7). When n_parallel > 1 we partition the prompt
    // tokens into n_parallel sequences, each starting at pos 0. Seq 0 is the same as
    // the head of a single-seq run — comparable for activation diffing.
    const int n_seqs = params.n_parallel > 1 ? params.n_parallel : 1;
    if (n_seqs > 1) {
        const int n_per_seq = (int) tokens.size() / n_seqs;
        if (n_per_seq <= 0) {
            LOG_ERR("%s : prompt too short for n_parallel=%d (need >= %d tokens)\n", __func__, n_seqs, n_seqs);
            return false;
        }
        LOG_INF("multi-seq diag: splitting first %d tokens into %d seqs of %d tokens each\n",
                n_per_seq * n_seqs, n_seqs, n_per_seq);

        llama_batch batch = llama_batch_init(n_per_seq * n_seqs, 0, n_seqs);
        for (int s = 0; s < n_seqs; ++s) {
            for (int i = 0; i < n_per_seq; ++i) {
                const llama_token tok = tokens[s * n_per_seq + i];
                common_batch_add(batch, tok, i, { (llama_seq_id) s }, /*logits=*/false);
            }
        }
        const int r = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (r != 0) {
            LOG_ERR("%s : failed to eval (multi-seq, r=%d)\n", __func__, r);
            return false;
        }
        return true;
    }

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // Construct cb_data with --tensor-filter regex patterns from CLI; also wires cb_eval into params.
    common_debug_cb_user_data cb_data(params, params.tensor_filter, /*abort_on_nan=*/false);
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
