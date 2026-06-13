/*
 * llama-triattention.cpp — in-graph K/V capture harness for TriAttention (Phase A)
 *
 * Allocates CPU-backend capture tensors mirroring the KV cache K/V tensors per layer.
 * These buffers are populated each decode step by ggml_set_rows nodes inserted in
 * llama-graph.cpp, bypassing the broken ggml_backend_tensor_get() sub-alloc path on ROCm.
 */

#include "llama-triattention.h"
#include "llama-kv-cache.h"
#include "llama-memory-hybrid.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstring>
#include <cassert>
#include <cstdio>

/* Global capture buffer pointers, set by llama_tria_capture_alloc() */
tria_kv_capture * g_tria_capture   = nullptr;
size_t            g_tria_capture_n = 0;

/* hparams of the model that owns g_tria_capture (the TriAttention target). Set
 * by the first context to allocate capture buffers; consulted to keep a draft
 * context from scattering into the target's buffers. See llama-triattention.h. */
const struct llama_hparams * g_tria_capture_hparams = nullptr;

/* ------------------------------------------------------------------ */
/* Public C++ API                                                      */
/* ------------------------------------------------------------------ */

void llama_tria_capture_alloc(
    void              * kv_ctx,
    void              * kv_swa_ctx,
    ggml_backend_t      backend_cpu,
    int                 n_layer,
    std::vector<tria_kv_capture> & out_capture,
    ggml_context    ** out_ggml_ctx,
    ggml_backend_buffer_t * out_buf)
{
    *out_ggml_ctx = nullptr;
    *out_buf      = nullptr;
    out_capture.clear();

    if (!kv_ctx || !backend_cpu || n_layer <= 0) return;

    auto * kv     = static_cast<llama_kv_cache *>(kv_ctx);
    auto * kv_swa = static_cast<llama_kv_cache *>(kv_swa_ctx);  /* nullptr for non-SWA models */

    /* Create a ggml context just big enough to hold all capture tensor metadata */
    const size_t overhead = ggml_tensor_overhead();
    ggml_init_params ctx_params = {
        /*.mem_size   =*/ (size_t)n_layer * 2 * overhead + 512,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ true,
    };
    ggml_context * tria_ctx = ggml_init(ctx_params);
    if (!tria_ctx) return;

    out_capture.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        /* Select the sub-cache that owns this layer. base holds non-SWA layers,
         * swa holds SWA layers (the iswa filters partition them), so exactly one
         * of base/swa returns a non-null template tensor per layer. For non-SWA
         * models kv_swa is null and the base cache owns every layer. */
        llama_kv_cache * src_kv = kv;
        ggml_tensor    * k_tpl  = kv->get_layer_k_raw(il);
        if (!k_tpl && kv_swa) {
            ggml_tensor * k_swa = kv_swa->get_layer_k_raw(il);
            if (k_swa) { src_kv = kv_swa; k_tpl = k_swa; }
        }
        if (k_tpl) {
            out_capture[il].k_buffer = ggml_dup_tensor(tria_ctx, k_tpl);
            if (out_capture[il].k_buffer) {
                char name[GGML_MAX_NAME];
                snprintf(name, sizeof(name), "tria_k_cap_%d", il);
                ggml_set_name(out_capture[il].k_buffer, name);
            }
        }
        /* V capture: skip transposed V (non-flash-attn path uses transposed layout).
         * Use the SAME sub-cache picked for K above so dims stay consistent. */
        ggml_tensor * v_tpl = src_kv->get_layer_v_raw(il);
        if (v_tpl && !src_kv->get_v_trans()) {
            out_capture[il].v_buffer = ggml_dup_tensor(tria_ctx, v_tpl);
            if (out_capture[il].v_buffer) {
                char name[GGML_MAX_NAME];
                snprintf(name, sizeof(name), "tria_v_cap_%d", il);
                ggml_set_name(out_capture[il].v_buffer, name);
            }
        }
    }

    /* Allocate all tensors on the CPU backend (always host-accessible) */
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(tria_ctx, backend_cpu);
    if (!buf) {
        ggml_free(tria_ctx);
        out_capture.clear();
        return;
    }

    /* Zero-initialize all capture buffers */
    ggml_backend_buffer_clear(buf, 0);

    size_t total_bytes = ggml_backend_buffer_get_size(buf);
    fprintf(stderr, "tria: allocated %d-layer K/V capture buffers (%.2f MiB) on CPU backend\n",
            n_layer, (double)total_bytes / (1024.0 * 1024.0));

    *out_ggml_ctx = tria_ctx;
    *out_buf      = buf;
}

void llama_tria_capture_free(ggml_context * ggml_ctx, ggml_backend_buffer_t buf) {
    if (buf) {
        ggml_backend_buffer_free(buf);
    }
    if (ggml_ctx) {
        ggml_free(ggml_ctx);
    }
}

/* ------------------------------------------------------------------ */
/* C-linkage accessors for triattention-runtime.c                     */
/* ------------------------------------------------------------------ */

extern "C" {

struct ggml_tensor * tria_get_k_capture(int layer_idx) {
    if (!g_tria_capture || layer_idx < 0 || (size_t)layer_idx >= g_tria_capture_n) {
        return nullptr;
    }
    return g_tria_capture[layer_idx].k_buffer;
}

struct ggml_tensor * tria_get_v_capture(int layer_idx) {
    if (!g_tria_capture || layer_idx < 0 || (size_t)layer_idx >= g_tria_capture_n) {
        return nullptr;
    }
    return g_tria_capture[layer_idx].v_buffer;
}

} /* extern "C" */
