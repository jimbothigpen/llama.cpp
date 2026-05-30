#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstddef>
#include <vector>

/* Per-layer K/V capture buffers for TriAttention in-graph harness.
 * Populated each decode step by ggml_set_rows nodes in llama-graph.cpp.
 * Allocated on CPU backend — always host-accessible (bypasses ROCm sub-alloc bug). */
struct tria_kv_capture {
    ggml_tensor * k_buffer = nullptr;  /* [ne0_k, n_ctx] — same dtype as KV K tensor */
    ggml_tensor * v_buffer = nullptr;  /* [ne0_v, n_ctx] — same dtype as KV V tensor */
};

/* Global capture buffer array. Set when TriAttention is enabled.
 * Read by build_attn() (llama-graph.cpp) and tria_get_k_capture() (bridge). */
extern tria_kv_capture * g_tria_capture;
extern size_t            g_tria_capture_n;  /* == n_layer when set */

/* Allocate per-layer capture buffers mirroring KV cache K/V tensors.
 * Called in llama_context constructor after KV cache is initialized.
 * kv_ctx:     pointer to llama_kv_cache (via llama_get_memory cast). For a hybrid
 *             sliding-window model this is the BASE (full-attention) sub-cache.
 * kv_swa_ctx: pointer to the SWA sub-cache (llama_kv_cache*) for hybrid models,
 *             or nullptr for non-SWA models. Per layer, the template tensor is
 *             taken from whichever sub-cache owns that layer (base holds non-SWA
 *             layers, swa holds SWA layers — the cache filters partition them).
 * backend_cpu: CPU backend to allocate on
 * n_layer: number of attention layers */
void llama_tria_capture_alloc(
    void              * kv_ctx,
    void              * kv_swa_ctx,
    ggml_backend_t      backend_cpu,
    int                 n_layer,
    std::vector<tria_kv_capture> & out_capture,
    ggml_context    ** out_ggml_ctx,
    ggml_backend_buffer_t * out_buf);

/* Free capture state. Called in llama_context destructor. */
void llama_tria_capture_free(ggml_context * ggml_ctx, ggml_backend_buffer_t buf);

/* C-linkage accessors used by triattention-runtime.c */
extern "C" {
    struct ggml_tensor * tria_get_k_capture(int layer_idx);
    struct ggml_tensor * tria_get_v_capture(int layer_idx);
}
