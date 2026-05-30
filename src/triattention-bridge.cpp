/*
 * triattention-bridge.cpp — C++ bridge for accessing llama internals from C
 *
 * Ported from ft2/feature/triattention for ygg.
 * Adaptations vs ft2:
 *   - kv->get_used_n_kv() → kv->get_n_used()
 *   - Removed Phase 3B indirection (has_indirection, get_active_kv_real_len)
 *   - tria_get_kv_positions: simplified sequential for single-seq AR decode
 *   - tria_compact_kv: Phase B — legacy physical compaction via triattention_compact
 */

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <vector>

/* Helper: extract llama_kv_cache from either pure KV or hybrid memory */
static llama_kv_cache * get_kv(void * ctx_void) {
    auto * ctx = (llama_context *)ctx_void;
    auto * mem = llama_get_memory(ctx);
    if (!mem) return nullptr;

    auto * kv = dynamic_cast<llama_kv_cache *>(mem);
    if (kv) return kv;

    auto * hybrid = dynamic_cast<llama_memory_hybrid *>(mem);
    if (hybrid) return hybrid->get_mem_attn();

    // Hybrid sliding-window models (e.g. Gemma-4) use llama_kv_cache_iswa, which
    // does NOT inherit llama_kv_cache. Return the base (full-attention) sub-cache:
    // it holds the full token-position space and is the cache TriAttention scoring,
    // position queries, and physical compaction operate over. SWA layers self-manage
    // their sliding window; their K/V is captured only to enrich the per-token score
    // (see llama_tria_capture_alloc).
    auto * iswa = dynamic_cast<llama_kv_cache_iswa *>(mem);
    if (iswa) return iswa->get_base();

    return nullptr;
}

extern "C" {
#include "triattention-runtime.h"

struct ggml_tensor * tria_get_k_tensor(void * ctx_void, int layer_idx) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return nullptr;
    return kv->get_layer_k_raw(layer_idx);
}

struct ggml_tensor * tria_get_v_tensor(void * ctx_void, int layer_idx) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return nullptr;
    return kv->get_layer_v_raw(layer_idx);
}

int tria_get_n_kv(void * ctx_void) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    auto * ctx = (llama_context *)ctx_void;
    const uint32_t n_seq = llama_n_seq_max(ctx);
    llama_pos pmax = -1;
    for (llama_seq_id s = 0; s < (llama_seq_id) n_seq; ++s) {
        pmax = std::max(pmax, kv->seq_pos_max(s));
    }
    return (pmax >= 0) ? (int) (pmax + 1) : 0;
}

int tria_get_used_n_kv(void * ctx_void) {
    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;
    return (int) kv->get_n_used();
}

int tria_get_n_ctx(void * ctx_void) {
    auto * ctx = (llama_context *)ctx_void;
    if (!ctx) return 0;
    return (int) llama_n_ctx(ctx);
}

int tria_get_kv_positions(void * ctx_void, int * positions, int max_positions) {
    auto * ctx = (llama_context *)ctx_void;
    if (!ctx || !positions || max_positions <= 0) {
        return 0;
    }

    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    /* Phase A simplification: assume sequential positions for single-seq AR decode.
     * Phase B: implement proper per-cell position tracking for multi-sequence. */
    int n = std::min(max_positions, (int) kv->get_n_used());
    for (int i = 0; i < n; i++) {
        positions[i] = i;
    }
    return n;
}

int tria_compact_kv(struct tria_runtime * rt, void * ctx_void) {
    auto * ctx = (llama_context *)ctx_void;
    if (!rt || !ctx || !rt->global_scores || rt->global_budget <= 0) {
        return 0;
    }

    auto * kv = get_kv(ctx_void);
    if (!kv) return 0;

    const int n_kv = (int) kv->get_n_used();
    const int n_old = n_kv - rt->window;
    if (n_old <= 0) {
        return 0;
    }

    int budget = rt->global_budget;
    budget = std::max(1, std::min(budget, n_old));

    /* Skip eviction if cache is already within budget */
    if (n_old <= budget) {
        return 0;
    }

    /* Protect sink/prefix tokens — always keep at least the first `prefix` tokens */
    int prefix = rt->sink > 0 ? rt->sink : 128;
    if (prefix > n_old) prefix = n_old;

    /* Budget must cover at least the protected prefix */
    if (budget < prefix) budget = prefix;

    /* Build keep set: prefix tokens + top-scoring non-prefix + window */
    std::vector<uint32_t> keep_positions;
    keep_positions.reserve(budget + rt->window);

    for (int i = 0; i < prefix; i++) {
        keep_positions.push_back((uint32_t)i);
    }

    int remaining_budget = budget - (int)keep_positions.size();
    if (remaining_budget > 0 && prefix < n_old) {
        std::vector<uint32_t> ranked;
        ranked.reserve(n_old - prefix);
        for (int i = prefix; i < n_old; i++) {
            ranked.push_back((uint32_t)i);
        }

        std::stable_sort(ranked.begin(), ranked.end(), [&](uint32_t a, uint32_t b) {
            if (rt->global_scores[a] == rt->global_scores[b]) return a < b;
            return rt->global_scores[a] > rt->global_scores[b];
        });

        int take = std::min(remaining_budget, (int)ranked.size());
        ranked.resize(take);
        std::sort(ranked.begin(), ranked.end());
        keep_positions.insert(keep_positions.end(), ranked.begin(), ranked.end());
    }

    /* Window tokens are always kept */
    for (int pos = n_old; pos < n_kv; ++pos) {
        keep_positions.push_back((uint32_t)pos);
    }

    if ((int)keep_positions.size() >= n_kv) {
        return 0;
    }

    llama_synchronize(ctx);

    /* Log on first eviction call so smoke tests can confirm the evictor fired */
    static bool tria_first_evict = true;
    if (tria_first_evict) {
        fprintf(stderr, "tria: Phase B evictor first call — n_kv=%d keep=%d evict=%d budget=%d window=%d\n",
                n_kv, (int)keep_positions.size(),
                n_kv - (int)keep_positions.size(),
                budget, rt->window);
        tria_first_evict = false;
    }

    if (!kv->triattention_compact(keep_positions)) {
        return 0;
    }

    return n_kv - (int)keep_positions.size();
}

} /* extern "C" */
