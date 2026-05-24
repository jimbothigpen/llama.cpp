/*
 * triattention-bridge.cpp — C++ bridge for accessing llama internals from C
 *
 * Ported from ft2/feature/triattention for ygg.
 * Adaptations vs ft2:
 *   - kv->get_used_n_kv() → kv->get_n_used()
 *   - Removed Phase 3B indirection (has_indirection, get_active_kv_real_len)
 *   - tria_get_kv_positions: simplified sequential for single-seq AR decode
 *   - tria_compact_kv: stubbed (Phase B)
 */

#include "llama.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"

#include <algorithm>
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
    /* Phase A stub: physical KV compaction deferred to Phase B.
     * Returns 0 so scoring continues without compaction. */
    (void)rt;
    (void)ctx_void;
    return 0;
}

} /* extern "C" */
