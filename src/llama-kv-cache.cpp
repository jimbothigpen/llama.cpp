#include "llama-kv-cache.h"

#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

#ifdef LLAMA_KV_COMPACTION
#include "llama-kv-compact-pipeline.h"

#include <atomic>
#include <cstring>

namespace {
// KV cache compaction (Attention Matching) helpers — ported from
// jandhyala-dev/modelai-llama.cpp.
bool is_block_aligned_for_head(ggml_type type, uint32_t head_dim) {
    const int64_t blk = ggml_blck_size(type);
    return blk > 0 && head_dim > 0 && (head_dim % blk) == 0;
}

void type_to_float(const void * src, ggml_type type, float * dst, int64_t n) {
    if (type == GGML_TYPE_F32) {
        std::memcpy(dst, src, size_t(n) * sizeof(float));
        return;
    }

    auto to_float = ggml_get_type_traits(type)->to_float;
    GGML_ASSERT(to_float != nullptr);
    to_float(src, dst, n);
}
} // namespace
#endif // LLAMA_KV_COMPACTION

// Turbo TCQ prompt cache safety: compute a fingerprint from the codebook env
// vars so that loading a cache created with a different codebook is detected.
// The fingerprint is a CRC32 of the codebook FILE CONTENTS (not the path),
// so the check is relocatable — only the actual data matters.
static uint32_t turboq_tcq_codebook_crc32(const char * path, size_t n_floats) {
    if (!path || !path[0]) {
        return 0; // no custom codebook → use compiled-in default → hash 0
    }
    uint32_t crc = 0xFFFFFFFF;
    FILE * f = fopen(path, "rb");
    if (!f) { return 0; }
    float buf[512];
    size_t n = fread(buf, sizeof(float), n_floats, f);
    fclose(f);
    if (n != n_floats) { return 0; }
    const uint8_t * data = (const uint8_t *)buf;
    for (size_t i = 0; i < n_floats * sizeof(float); i++) {
        crc ^= data[i];
        for (int j = 0; j < 8; j++) {
            crc = (crc >> 1) ^ (0xEDB88320 & -(crc & 1));
        }
    }
    return crc ^ 0xFFFFFFFF;
}

static uint32_t turboq_tcq_fingerprint(void) {
    const char * cb3 = getenv("TURBO_TCQ_CB");
    const char * cb2 = getenv("TURBO_TCQ_CB2");
    uint32_t h3 = turboq_tcq_codebook_crc32(cb3, 512);
    uint32_t h2 = turboq_tcq_codebook_crc32(cb2, 256);
    return h3 ^ (h2 * 0x9E3779B9); // mix both hashes
}

static bool ggml_type_is_turboq_tcq(enum ggml_type t) {
    return t == GGML_TYPE_TURBOQ3_TCQ || t == GGML_TYPE_TURBOQ2_TCQ;
}

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

static ggml_tensor * ggml_mul_mat_aux(
        ggml_context * ctx,
        ggml_tensor * cur,
        ggml_tensor * rot) {
    const auto n = rot->ne[0];

    ggml_tensor * res;

    res = ggml_reshape_2d(ctx, cur, n, ggml_nelements(cur)/n);
    res = ggml_mul_mat   (ctx, rot, res);
    ggml_mul_mat_set_hint(res, GGML_HINT_SRC0_IS_HADAMARD);
    res = ggml_reshape_4d(ctx, res, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    return res;
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                     bool   offload,
                     bool   unified,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                 uint32_t   oscar_res_window,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
                int32_t   n_layers_high_precision,
                ggml_type type_k_low,
                ggml_type type_v_low
#ifdef LLAMA_KV_COMPACTION
    ,                bool   enable_compacted_prefix
#endif
    ) :
    model(model), hparams(hparams), v_trans(v_trans),
    n_seq_max(n_seq_max), n_stream(unified ? 1 : n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    oscar_residual_window(oscar_res_window),
    n_layers_high_precision(n_layers_high_precision),
    type_k_low(type_k_low),
    type_v_low(type_v_low),
    other(static_cast<llama_kv_cache *>(mem_other)),
    v_cells_impl(other ? other->v_cells_impl : std::make_shared<llama_kv_cells_vec>()),
    v_cells(*v_cells_impl) {

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            // +1 per layer for optional k_res tensor (OScaR residual window)
            ggml_init_params params = {
                /*.mem_size   =*/ size_t((2u*(1 + n_stream) + 1u)*n_layer*ggml_tensor_overhead() + ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    GGML_ASSERT(n_stream == 1 || n_stream == n_seq_max);

    v_heads.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_heads[s] = 0;
    }

    v_cells.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].resize(kv_size);
    }

    // by default, all sequence ids are mapped to the 0th stream
    seq_to_stream.resize(LLAMA_MAX_SEQ, 0);

    if (n_stream > 1) {
        seq_to_stream.resize(n_stream, 0);
        for (uint32_t s = 0; s < n_stream; ++s) {
            seq_to_stream[s] = s;
        }
    }

    // [TAG_V_CACHE_VARIABLE]
    if (v_trans && hparams.is_n_embd_v_gqa_variable()) {
        LLAMA_LOG_WARN("%s: the V embeddings have different sizes across layers and FA is not enabled - padding V cache to %d\n",
                __func__, hparams.n_embd_v_gqa_max());
    }

    const bool is_mla = hparams.is_mla();

    bool la_log_emitted = false;  // log TURBO_LAYER_ADAPTIVE banner once per kv_cache construction (fit probe + real run each get their own banner)

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (n_embd_head_v_all == 0) {
            n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
        } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
            n_embd_head_v_all = -1;
        }

        // [TAG_V_CACHE_VARIABLE]
        const uint32_t n_embd_k_gqa =            hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = !v_trans ? hparams.n_embd_v_gqa(il) : hparams.n_embd_v_gqa_max();

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
        ggml_backend_dev_t offload_dev = nullptr;

        if (offload) {
            offload_dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(offload_dev);

            dev_name = ggml_backend_dev_name(offload_dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        // TurboQuant requires head_dim (n_embd_head_k) divisible by 128.
        // For models with non-128-aligned heads (e.g. DeepSeek2 MLA with head_dim=192/576),
        // fall back to q8_0 with a clear message instead of asserting later.
        const bool is_turbo_type = (type_k == GGML_TYPE_TURBOQ2_0 || type_k == GGML_TYPE_TURBOQ3_0 || type_k == GGML_TYPE_TURBOQ4_0 || type_k == GGML_TYPE_TURBOQ8_0 || type_k == GGML_TYPE_TURBOQ5_0 || type_k == GGML_TYPE_TURBOQ6_0 || type_k == GGML_TYPE_TURBOQ2_TCQ || type_k == GGML_TYPE_TURBOQ3_TCQ || type_k == GGML_TYPE_KV_OSCAR_INT2 ||
                                    type_v == GGML_TYPE_TURBOQ2_0 || type_v == GGML_TYPE_TURBOQ3_0 || type_v == GGML_TYPE_TURBOQ4_0 || type_v == GGML_TYPE_TURBOQ8_0 || type_v == GGML_TYPE_TURBOQ5_0 || type_v == GGML_TYPE_TURBOQ6_0 || type_v == GGML_TYPE_TURBOQ2_TCQ || type_v == GGML_TYPE_TURBOQ3_TCQ || type_v == GGML_TYPE_KV_OSCAR_INT2);
        const uint32_t n_embd_head_k_layer = hparams.n_embd_head_k(il);
        if (is_turbo_type && n_embd_head_k_layer % 128 != 0) {
            if (il == 0) {
                LLAMA_LOG_WARN("%s: turbo KV cache requires head_dim divisible by 128, "
                               "but this model has n_embd_head_k=%u — falling back to q8_0\n",
                               __func__, n_embd_head_k_layer);
            }
            type_k = GGML_TYPE_Q8_0;
            type_v = GGML_TYPE_Q8_0;
        }

        // Layer-adaptive KV precision: TURBO_LAYER_ADAPTIVE env var selects strategy.
        // Default OFF (mode 0); each non-zero mode is an explicit opt-in.
        //
        //   0 = uniform (default — no behavioral change)
        //   1 = q8_0 K+V for first 4 + last 4 layers (turbo K+V outer-protection)
        //   2 = q8_0 K+V for last 8 layers
        //   5 = first2+last2 V=TURBOQ4_0, rest V=TURBOQ2_0   (K unchanged)
        //   6 = last8 V=TURBOQ4_0,         rest V=TURBOQ2_0   (K unchanged)
        //   7 = Boundary V (recommended): first2+last2 V=q8_0, rest V=TURBOQ2_0 (K unchanged)
        //
        // Source: TQ-KV 5aeb2fdbe llama-kv-cache.cpp lines 267-326; da4a02ec7 added
        // mode 7 + reorganized comment. Auto-enable on V=TURBO2_0 from TQ-KV is
        // intentionally dropped so default behavior stays "uniform" per recon/08
        // §Step 5 validation gate ("Boundary V default-off — no behavioral change
        // without explicit flag").
        ggml_type layer_type_k = type_k;
        ggml_type layer_type_v = type_v;
        {
            const char * env = getenv("TURBO_LAYER_ADAPTIVE");
            const int adaptive_mode = env ? atoi(env) : 0;
            const bool is_turbo = (type_k == GGML_TYPE_TURBOQ2_0 || type_k == GGML_TYPE_TURBOQ3_0 || type_k == GGML_TYPE_TURBOQ4_0 || type_k == GGML_TYPE_TURBOQ8_0 || type_k == GGML_TYPE_TURBOQ5_0 || type_k == GGML_TYPE_TURBOQ6_0 || type_k == GGML_TYPE_TURBOQ2_TCQ || type_k == GGML_TYPE_TURBOQ3_TCQ || type_k == GGML_TYPE_KV_OSCAR_INT2);
            const bool v_is_turbo = (type_v == GGML_TYPE_TURBOQ2_0 || type_v == GGML_TYPE_TURBOQ3_0 || type_v == GGML_TYPE_TURBOQ4_0 || type_v == GGML_TYPE_TURBOQ8_0 || type_v == GGML_TYPE_TURBOQ5_0 || type_v == GGML_TYPE_TURBOQ6_0 || type_v == GGML_TYPE_TURBOQ2_TCQ || type_v == GGML_TYPE_TURBOQ3_TCQ || type_v == GGML_TYPE_KV_OSCAR_INT2);
            const uint32_t n_layer = hparams.n_layer();
            if (!la_log_emitted && adaptive_mode > 0) {
                LLAMA_LOG_INFO("%s: layer-adaptive mode %d enabled (TURBO_LAYER_ADAPTIVE)\n", __func__, adaptive_mode);
                la_log_emitted = true;
            }
            if (adaptive_mode == 1 && is_turbo && n_layer >= 8) {
                if (il < 4 || il >= n_layer - 4) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 2 && is_turbo && n_layer >= 8) {
                if (il >= n_layer - 8) {
                    layer_type_k = GGML_TYPE_Q8_0;
                    layer_type_v = GGML_TYPE_Q8_0;
                }
            } else if (adaptive_mode == 5 && v_is_turbo && n_layer >= 8) {
                const bool is_boundary = (il < 2 || il >= n_layer - 2);
                layer_type_v = is_boundary ? GGML_TYPE_TURBOQ4_0 : GGML_TYPE_TURBOQ2_0;
            } else if (adaptive_mode == 6 && v_is_turbo && n_layer >= 8) {
                layer_type_v = (il >= n_layer - 8) ? GGML_TYPE_TURBOQ4_0 : GGML_TYPE_TURBOQ2_0;
            } else if (adaptive_mode == 7 && v_is_turbo && n_layer >= 8) {
                const bool is_boundary = (il < 2 || il >= n_layer - 2);
                layer_type_v = is_boundary ? GGML_TYPE_Q8_0 : GGML_TYPE_TURBOQ2_0;
            }
        }

        // Layer-wise adaptive KV cache precision (Phase A5 / CLI-driven)
        // Bottom (n_layer - n_layers_high_precision) layers use low precision types.
        // Composable with TURBO_LAYER_ADAPTIVE — this override applies on top of it.
        if (n_layers_high_precision > 0) {
            const uint32_t n_layer_hp = hparams.n_layer();
            const bool is_low_layer = (il < (n_layer_hp - (uint32_t)n_layers_high_precision));
            if (is_low_layer && type_k_low != GGML_TYPE_COUNT) {
                layer_type_k = type_k_low;
            }
            if (is_low_layer && type_v_low != GGML_TYPE_COUNT) {
                layer_type_v = type_v_low;
            }
        }

        // If the target GPU backend does not support SET_ROWS for the chosen KV type,
        // fall back to CPU buffer so the pre-allocated tensor is CPU-resident and the
        // CPU backend can run SET_ROWS. Applies generically to any type not registered
        // in the backend's SET_ROWS dispatch (e.g. RQ types).
        if (offload_dev) {
            struct ggml_tensor dummy_src0 = {};
            dummy_src0.type  = GGML_TYPE_F32;
            dummy_src0.ne[0] = n_embd_k_gqa;
            struct ggml_tensor dummy_src1 = {};
            dummy_src1.type  = GGML_TYPE_I32;
            struct ggml_tensor dummy_op = {};
            dummy_op.op      = GGML_OP_SET_ROWS;
            dummy_op.src[0]  = &dummy_src0;
            dummy_op.src[1]  = &dummy_src1;

            dummy_op.type = layer_type_k;
            const bool k_ok = ggml_backend_dev_supports_op(offload_dev, &dummy_op);
            dummy_op.type = layer_type_v;
            const bool v_ok = ggml_backend_dev_supports_op(offload_dev, &dummy_op);

            if (!k_ok || !v_ok) {
                LLAMA_LOG_DEBUG("%s: layer %3d: KV type (%s/%s) has no GPU SET_ROWS — CPU buffer\n",
                        __func__, il, ggml_type_name(layer_type_k), ggml_type_name(layer_type_v));
                buft = ggml_backend_cpu_buffer_type();
            }
        }

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, layer_type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_3d(ctx, layer_type_v, n_embd_v_gqa, kv_size, n_stream) : nullptr;

        has_k && ggml_format_name(k, "cache_k_l%d", il);
        has_v && ggml_format_name(v, "cache_v_l%d", il);

        // Allocate F16 residual buffer for OScaR INT2 K cache when residual window is enabled
        ggml_tensor * k_res = nullptr;
        if (has_k && layer_type_k == GGML_TYPE_KV_OSCAR_INT2 && oscar_residual_window > 0) {
            k_res = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, n_embd_k_gqa, kv_size, n_stream);
            ggml_format_name(k_res, "cache_k_res_l%d", il);
        }

        std::vector<ggml_tensor *> k_stream;
        std::vector<ggml_tensor *> v_stream;

        for (uint32_t s = 0; s < n_stream; ++s) {
            k_stream.push_back(has_k ? ggml_view_2d(ctx, k, n_embd_k_gqa, kv_size, k->nb[1], s*k->nb[2]) : nullptr);
            v_stream.push_back(has_v ? ggml_view_2d(ctx, v, n_embd_v_gqa, kv_size, v->nb[1], s*v->nb[2]) : nullptr);
        }

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v, k_res, k_stream, v_stream, });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

#ifdef LLAMA_KV_COMPACTION
    // KV cache compaction (Attention Matching): build per-layer layouts for the
    // compacted-prefix store. Ported from jandhyala-dev/modelai-llama.cpp.
    if (enable_compacted_prefix) {
        std::vector<llama_compacted_prefix_layer_layout> compacted_layouts;
        compacted_layouts.reserve(layers.size());

        for (const auto & layer : layers) {
            compacted_layouts.push_back({
                /* layer_id       = */ layer.il,
                /* n_head_kv      = */ hparams.n_head_kv(layer.il),
                /* n_embd_head_k  = */ hparams.n_embd_head_k(layer.il),
                /* n_embd_head_v  = */ layer.v ? hparams.n_embd_head_v(layer.il) : 0u,
                /* type_k         = */ layer.k ? layer.k->type : type_k,
                /* type_v         = */ layer.v ? layer.v->type : type_v,
            });
        }

        compacted_prefix = llama_compacted_prefix_store(std::move(compacted_layouts));
    }
#endif // LLAMA_KV_COMPACTION

    // allocate tensors and initialize the buffers to avoid NaNs in the padding
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max, n_stream,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LLAMA_ATTN_ROT_DISABLE = getenv("LLAMA_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LLAMA_ATTN_ROT_DISABLE ? atoi(LLAMA_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LLAMA_ATTN_ROT_DISABLE)\n", __func__);
        }

        // OScaR INT2 has its own inline full-dim WHT in the encode/decode kernels (set-rows.cu /
        // fattn-vec.cuh). Applying the graph-level attn_rot Hadamard on top of OScaR's own WHT
        // would compose H_D × H_D = I (normalized Hadamard is its own inverse), negating the
        // rotation and leaving K stored unrotated → poor INT2 quantization quality.
        // So exclude GGML_TYPE_KV_OSCAR_INT2 from attn_rot to let OScaR's own WHT be the sole rotation.
        // §-FLAG: K-shift (RoPE updates) for OScaR INT2 assumes attn_rot=false; live-inference
        // with context shifts should verify K-shift correctness with this change.
        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            type_k != GGML_TYPE_KV_OSCAR_INT2 &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek V3.2 DSA lightning indexer
        if (model.arch == LLM_ARCH_DEEPSEEK32 && hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    // OScaR INT2 K-shift fix (§-FLAG-ATTN_ROT_KSHIFT):
    // attn_rot_k=false for OScaR INT2 (avoids double H^2=I rotation on fresh tokens).
    // But the K-shift / RoPE-update path needs to un-rotate before RoPE and re-rotate
    // after, because K is stored in the WHT-rotated domain.  Pre-compute the Hadamard
    // matrix for the OScaR WHT dimension so build_graph_shift can build a valid rot tensor.
    {
        int64_t oscar_wht_max = 0;
        for (const auto & lyr : layers) {
            if (lyr.k && lyr.k->type == GGML_TYPE_KV_OSCAR_INT2) {
                oscar_wht_max = std::max(oscar_wht_max, (int64_t) hparams.n_embd_head_k(lyr.il));
            }
        }
        for (int64_t n = 64; n <= oscar_wht_max; n *= 2) {
            if (attn_rot_hadamard.count(n)) {
                continue; // already computed by attn_rot block above
            }
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr tmp_ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(tmp_ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LLAMA_KV_CACHE_DEBUG = getenv("LLAMA_KV_CACHE_DEBUG");
    debug = LLAMA_KV_CACHE_DEBUG ? atoi(LLAMA_KV_CACHE_DEBUG) : 0;
}

void llama_kv_cache::clear(bool data) {
    for (uint32_t s = 0; s < n_stream; ++s) {
        v_cells[s].reset();
        v_heads[s] = 0;
    }

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if (seq_id >= 0) {
        auto & cells = v_cells[seq_to_stream[seq_id]];
        auto & head  = v_heads[seq_to_stream[seq_id]];

        uint32_t new_head = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id) && cells.seq_rm(i, seq_id)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }

        // If we freed up a slot, set head to it so searching can start there.
        if (new_head != cells.size() && new_head < head) {
            head = new_head;
        }
    } else {
        // match any sequence
        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];
            auto & head  = v_heads[s];

            uint32_t new_head = cells.size();

            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.pos_in(i, p0, p1)) {
                    continue;
                }

                cells.rm(i);

                if (new_head == cells.size()) {
                    new_head = i;
                }
            }

            // If we freed up a slot, set head to it so searching can start there.
            if (new_head != cells.size() && new_head < head) {
                head = new_head;
            }
        }
    }

    return true;
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (size_t) seq_id_src < seq_to_stream.size());
    GGML_ASSERT(seq_id_dst >= 0 && (size_t) seq_id_dst < seq_to_stream.size());

    const auto s0 = seq_to_stream[seq_id_src];
    const auto s1 = seq_to_stream[seq_id_dst];

    if (s0 == s1) {
        // since both sequences are in the same stream, no data copy is necessary
        // we just have to update the cells meta data

        auto & cells = v_cells[s0];

        if (seq_id_src == seq_id_dst) {
            return;
        }

        if (p0 < 0) {
            p0 = 0;
        }

        if (p1 < 0) {
            p1 = std::numeric_limits<llama_pos>::max();
        }

        for (uint32_t i = 0; i < cells.size(); ++i) {
            if (!cells.pos_in(i, p0, p1)) {
                continue;
            }

            if (cells.seq_has(i, seq_id_src)) {
                cells.seq_add(i, seq_id_dst);
            }
        }

        return;
    }

    // cross-stream sequence copies require to copy the actual buffer data

    bool is_full = true;

    if (p0 > 0 && p0 + 1 < (int) get_size()) {
        is_full = false;
    }

    if (p1 > 0 && p1 + 1 < (int) get_size()) {
        is_full = false;
    }

    GGML_ASSERT(is_full && "seq_cp() is only supported for full KV buffers");

    // enqueue the copy operation - the buffer copy will be performed during the next update
    sc_info.ssrc.push_back(s0);
    sc_info.sdst.push_back(s1);

    v_cells[s1].reset();
    for (uint32_t i = 0; i < v_cells[s0].size(); ++i) {
        if (v_cells[s0].seq_has(i, seq_id_src)) {
            llama_pos pos   = v_cells[s0].pos_get(i);
            llama_pos shift = v_cells[s0].get_shift(i);

            llama_kv_cell_ext ext = v_cells[s0].ext_get(i);

            if (shift != 0) {
                pos -= shift;
                assert(pos >= 0);
            }

            v_cells[s1].pos_set(i, pos);
            v_cells[s1].seq_add(i, seq_id_dst);

            if (shift != 0) {
                v_cells[s1].pos_add(i, shift);
            }

            v_cells[s1].ext_set(i, ext);
        }
    }

    v_heads[s1] = v_heads[s0];

    //for (uint32_t s = 0; s < n_stream; ++s) {
    //    LLAMA_LOG_WARN("%s: seq %d: min = %d, max = %d\n", __func__, s, v_cells[s].seq_pos_min(s), v_cells[s].seq_pos_max(s));
    //}
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];
    auto & head  = v_heads[seq_to_stream[seq_id]];

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

    auto & cells = v_cells[seq_to_stream[seq_id]];

    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < seq_to_stream.size());

    const auto & cells = v_cells[seq_to_stream[seq_id]];

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all,
            llama_mtp_op_type mtp_op_type) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = n_stream == 1 ? balloc.split_simple(n_ubatch) : balloc.split_equal(n_ubatch, true);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches, mtp_op_type);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches), mtp_op_type);
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift, std::move(sc_info));
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches, llama_mtp_op_type mtp_op_type) {
    llama_kv_cache::slot_info_vec_t res;

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        std::vector<uint32_t> v_heads_old; // old positions of the heads, before placing the ubatch

        std::vector<llama_kv_cells> v_cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false, mtp_op_type);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        {
            state_t state = { sinfo_new, v_heads, {} };

            for (uint32_t s = 0; s < sinfo_new.n_stream(); ++s) {
                auto & cells = v_cells[sinfo_new.strm[s]];

                state.v_cells.push_back(cells.cp(sinfo_new.idxs[s]));
            }

            states.push_back(std::move(state));
        }

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch, mtp_op_type);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        const auto & sinfo = it->sinfo;

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            auto & cells = v_cells[sinfo.strm[s]];
            auto & head  = v_heads[sinfo.strm[s]];

            cells.set(sinfo.idxs[s], it->v_cells[s]);
            head = it->v_heads_old[s];
        }
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift, const stream_copy_info & sc_info) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (!sc_info.empty()) {
        assert(n_stream > 1 && "stream copy should never happen with a single stream");

        llama_synchronize(lctx);

        const size_t n_copy = sc_info.ssrc.size();

        for (size_t i = 0; i < n_copy; ++i) {
            const auto ssrc = sc_info.ssrc[i];
            const auto sdst = sc_info.sdst[i];

            assert(ssrc < n_stream);
            assert(sdst < n_stream);

            LLAMA_LOG_DEBUG("%s: copying KV buffer: stream %d to stream %d\n", __func__, ssrc, sdst);

            assert(ssrc != sdst);

            for (uint32_t il = 0; il < layers.size(); ++il) {
                const auto & layer = layers[il];

                ggml_backend_tensor_copy(layer.k_stream[ssrc], layer.k_stream[sdst]);

                if (layer.v_stream[ssrc]) {
                    ggml_backend_tensor_copy(layer.v_stream[ssrc], layer.v_stream[sdst]);
                }
            }
        }
    }

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        for (uint32_t s = 0; s < n_stream; ++s) {
            auto & cells = v_cells[s];

            cells.reset_shift();
        }
    }

    return updated;
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont, llama_mtp_op_type mtp_op_type) const {

    // MTP WARMUP / UPDATE_ACCEPTED fast-path:
    //
    // When MTP shares the main llama_context, the K/V cells at (target_pos, target_seq)
    // ALREADY exist (placed by the main-model decode) and the WARMUP/UPDATE_ACCEPTED pass
    // needs to OVERWRITE their tensor data without disturbing cell metadata. The bypass
    // locates those existing cells and apply_ubatch skips its metadata mutation.
    //
    // With F5's separate ctx_mtp, the secondary context's KV is independent. The server's
    // seq_rm-before-decode pattern in mtp_update_kv_cache() clears any cells at >= start_pos
    // before the decode, so the cells the bypass would look for never exist in ctx_mtp at
    // WARMUP time. Falling through to the standard path then allocates fresh cells at the
    // head and apply_ubatch (also relaxed) writes the right (pos, seq_id) metadata.
    //
    // We keep the fast-path as an opportunistic match: if cells at target_pos already exist
    // with matching seq_id, reuse them. Otherwise FALL THROUGH to standard allocation.
    if (mtp_op_type == MTP_OP_WARMUP || mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
        if (ubatch.n_tokens == 0 || ubatch.n_seqs_unq == 0) {
            LLAMA_LOG_ERROR("%s: MTP op with empty ubatch\n", __func__);
            return { };
        }

        const llama_pos    target_pos = ubatch.pos[0];
        const llama_seq_id target_seq = ubatch.seq_id[0][0];
        const uint32_t     stream_id  = seq_to_stream[target_seq];
        const auto &       cells      = v_cells[stream_id];

        bool found = false;
        uint32_t found_idx = 0;

        const uint32_t head_cur = v_heads[stream_id];
        if (head_cur < cells.size() &&
            !cells.is_empty(head_cur) &&
            cells.pos_get(head_cur) == target_pos &&
            cells.seq_has(head_cur, target_seq)) {
            found = true;
            found_idx = head_cur;
        } else {
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (!cells.is_empty(i) &&
                    cells.pos_get(i) == target_pos &&
                    cells.seq_has(i, target_seq)) {
                    found = true;
                    found_idx = i;
                    break;
                }
            }
        }

        if (found) {
            // The main-model decode placed the prompt's K/V into contiguous cells
            // starting at found_idx (since cells are allocated head-forward). The
            // WARMUP/UPDATE_ACCEPTED batch contains those same N tokens at positions
            // [target_pos, target_pos + N - 1]; locate ALL N cells, not just the first.
            // (set_input_k_idxs asserts ubatch.n_tokens == sinfo.size()*sinfo.n_stream().)
            const uint32_t n_tokens = ubatch.n_tokens;
            if (found_idx + n_tokens <= cells.size()) {
                slot_info res = {
                    /*.s0   =*/ stream_id,
                    /*.s1   =*/ stream_id,
                    /*.strm =*/ { },
                    /*.idxs =*/ { },
                };
                res.resize(1);
                res.strm[0] = stream_id;
                res.idxs[0].reserve(n_tokens);
                for (uint32_t t = 0; t < n_tokens; ++t) {
                    res.idxs[0].push_back(found_idx + t);
                }
                return res;
            }
            // Fall through if the contiguous range exceeds cache size.
        }
        // Not found / out-of-range: fall through to standard allocation. apply_ubatch
        // detects "cells need fresh metadata" via its own match check and writes them.
    }

    if (debug > 0) {
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const auto seq_id = ubatch.seq_id_unq[s];
            const auto stream_id = seq_to_stream[seq_id];
            const auto & cells = v_cells[stream_id];
            const uint32_t head_cur = v_heads[stream_id];

            LLAMA_LOG_DEBUG("%s: stream[%d], n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                    __func__, stream_id, cells.used_max_p1(), cells.get_used(), head_cur, get_size(), n_swa);

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    if (cells.is_empty(i)) {
                        ss += '.';
                    } else {
                        assert(cells.seq_count(i) >= 1);

                        if (cells.seq_count(i) == 1) {
                            ss += std::to_string(cells.seq_get(i));
                        } else {
                            ss += 'M';
                        }
                    }
                    if (i%256 == 255) {
                        ss += " *";
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            if ((debug == 2 && n_swa > 0) || debug > 2) {
                std::string ss;
                for (uint32_t i = 0; i < cells.size(); ++i) {
                    std::string cur;
                    if (cells.is_empty(i)) {
                        cur = '.';
                    } else {
                        cur = std::to_string(cells.pos_get(i));
                    }
                    const int n = cur.size();
                    for (int j = 0; j < 5 - n; ++j) {
                        cur += ' ';
                    }
                    ss += cur;
                    if (i%256 == 255) {
                        ss += " *";
                    }
                    if (i%64 == 63) {
                        ss += '\n';
                    }
                }
                LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (cells.seq_pos_min(s) < 0) {
                    continue;
                }

                LLAMA_LOG_DEBUG("%s: stream[%d] min[%d] = %5d, max[%d] = %5d\n", __func__, stream_id, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
            }
        }
    }

    uint32_t n_tokens = ubatch.n_tokens;
    uint32_t n_seqs   = 1;

    if (n_stream > 1) {
        GGML_ASSERT(n_tokens % ubatch.n_seqs_unq == 0);

        n_seqs   = ubatch.n_seqs_unq;
        n_tokens = n_tokens / n_seqs;
    }

    slot_info res = {
        /*.s0   =*/ LLAMA_MAX_SEQ,
        /*.s1   =*/ 0,
        /*.strm =*/ { },
        /*.idxs =*/ { },
    };

    res.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const auto seq_id = ubatch.seq_id_unq[s];

        if (n_stream > 1) {
            GGML_ASSERT(ubatch.n_seq_id[s*n_tokens]    == 1);
            GGML_ASSERT(ubatch.seq_id  [s*n_tokens][0] == seq_id);
        }

        res.s0 = std::min<uint32_t>(res.s0, seq_to_stream[seq_id]);
        res.s1 = std::max<uint32_t>(res.s1, seq_to_stream[seq_id]);

        res.strm[s] = seq_to_stream[seq_id];
        res.idxs[s].reserve(n_tokens);

        const auto & cells = v_cells[seq_to_stream[seq_id]];

        uint32_t head_cur = v_heads[seq_to_stream[seq_id]];

        // if we have enough unused cells before the current head ->
        //   better to start searching from the beginning of the cache, hoping to fill it
        if (head_cur > cells.get_used() + 2*n_tokens) {
            head_cur = 0;
        }

        if (n_tokens > cells.size()) {
            LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
            return { };
        }

        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs[s].push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs[s].size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs[s].clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs[s].size() < n_tokens) {
            return { };
        }
    }

    assert(res.s1 >= res.s0);

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch, llama_mtp_op_type mtp_op_type) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // MTP WARMUP / UPDATE_ACCEPTED fast-path (paired with the find_slot bypass above):
    // only valid when the cells located by find_slot ALREADY hold the correct
    // (pos, seq_id). In that case the K/V tensor contents will be rewritten by the
    // graph itself, so the destructive metadata mutation + SWA purge is unnecessary.
    // We still move the head to match upstream-MTP's `cache.head = i`.
    //
    // With F5's separate ctx_mtp, the secondary context's cells are typically empty
    // (or just cleared by mtp_update_kv_cache's seq_rm-before-decode), so the
    // matching-metadata precondition fails and we fall through to the standard
    // mutation path which sets pos/seq_id on the freshly-allocated cells.
    if (mtp_op_type == MTP_OP_WARMUP || mtp_op_type == MTP_OP_UPDATE_ACCEPTED) {
        bool cells_already_match = true;
        for (uint32_t s = 0; s < sinfo.n_stream() && cells_already_match; ++s) {
            const auto & cells = v_cells[sinfo.strm[s]];
            for (uint32_t ii = 0; ii < sinfo.size() && cells_already_match; ++ii) {
                const uint32_t i   = s*sinfo.size() + ii;
                const auto     idx = sinfo.idxs[s][ii];
                if (cells.is_empty(idx) ||
                    cells.pos_get(idx) != ubatch.pos[i] ||
                    !cells.seq_has(idx, ubatch.seq_id[i][0])) {
                    cells_already_match = false;
                }
            }
        }

        if (cells_already_match) {
            for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
                if (!sinfo.idxs[s].empty()) {
                    v_heads[sinfo.strm[s]] = sinfo.idxs[s].front();
                }
            }
            return;
        }
        // Fall through to standard mutation path so pos/seq_id get written.
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    llama_seq_id seq_pos_max_rm[LLAMA_MAX_SEQ];
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.n_stream()*sinfo.size());

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        for (uint32_t ii = 0; ii < sinfo.size(); ++ii) {
            const uint32_t i = s*sinfo.size() + ii;

            auto & cells = v_cells[sinfo.strm[s]];

            const auto idx = sinfo.idxs[s][ii];

            if (!cells.is_empty(idx)) {
                assert(cells.seq_count(idx) == 1);

                const llama_seq_id seq_id = cells.seq_get(idx);
                const llama_pos    pos    = cells.pos_get(idx);

                seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

                cells.rm(idx);
            }

            cells.pos_set(idx, ubatch.pos[i]);

            if (ubatch.is_pos_2d()) {
                llama_kv_cell_ext ext {
                    /*.x =*/ ubatch.pos[i + ubatch.n_tokens*2],
                    /*.y =*/ ubatch.pos[i + ubatch.n_tokens],
                };
                cells.ext_set(idx, ext);
            }

            for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
                cells.seq_add(idx, ubatch.seq_id[i][s]);
            }
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < LLAMA_MAX_SEQ; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        GGML_ASSERT(s < seq_to_stream.size());

        auto & cells = v_cells[seq_to_stream[s]];

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        auto & head = v_heads[sinfo.strm[s]];

        head = sinfo.idxs[s].back() + 1;
    }
}

bool llama_kv_cache::get_can_shift() const {
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    const auto & cells = v_cells[seq_to_stream[0]];

    return cells.size();
}

uint32_t llama_kv_cache::get_n_stream() const {
    return n_stream;
}

bool llama_kv_cache::get_has_shift() const {
    // TurboQuant uses kernel-level WHT rotation -- position shift is a no-op
    if (!layers.empty() && (layers[0].k->type == GGML_TYPE_TURBOQ2_0 || layers[0].k->type == GGML_TYPE_TURBOQ3_0 || layers[0].k->type == GGML_TYPE_TURBOQ4_0 || layers[0].k->type == GGML_TYPE_TURBOQ8_0 || layers[0].k->type == GGML_TYPE_TURBOQ5_0 || layers[0].k->type == GGML_TYPE_TURBOQ6_0)) { return false; }
    bool result = false;

    for (uint32_t s = 0; s < n_stream; ++s) {
        result |= v_cells[s].get_has_shift();
    }

    return result;
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

ggml_tensor * llama_kv_cache::get_layer_k_raw(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].k;
}

ggml_tensor * llama_kv_cache::get_layer_v_raw(int32_t il) const {
    auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) return nullptr;
    return layers[it->second].v;
}

uint32_t llama_kv_cache::get_n_used() const {
    GGML_ASSERT(n_stream == 1 && "get_n_used: foreign-KV consumers require a unified single-stream cache");
    return v_cells[seq_to_stream[0]].used_max_p1();
}

bool llama_kv_cache::get_v_trans() const {
    return v_trans;
}

bool llama_kv_cache::triattention_compact(const std::vector<uint32_t> & keep_positions) {
    if (n_stream != 1) {
        LLAMA_LOG_WARN("%s: TriAttention compaction only supports a unified single-stream KV cache\n", __func__);
        return false;
    }

    if (keep_positions.empty()) {
        return false;
    }

    auto & cells = v_cells[0];
    const uint32_t n_kv_old = cells.used_max_p1();
    const uint32_t n_kv_new = (uint32_t) keep_positions.size();

    if (n_kv_new > n_kv_old || n_kv_old > cells.size()) {
        return false;
    }

    for (uint32_t src : keep_positions) {
        if (src >= n_kv_old || cells.is_empty(src)) {
            LLAMA_LOG_WARN("%s: invalid TriAttention source row %u (n_kv_old=%u)\n", __func__, src, n_kv_old);
            return false;
        }
    }

    /* Skip rows that are already in their final destination */
    uint32_t first_move = 0;
    while (first_move < n_kv_new && keep_positions[first_move] == first_move) {
        first_move++;
    }

    /* Compact tensor rows: gather rows at keep_positions[first_move..] into [first_move..n_kv_new-1] */
    const auto compact_rows = [&](ggml_tensor * tensor) {
        if (tensor == nullptr || first_move >= n_kv_new) {
            return;
        }
        const size_t row_bytes = tensor->nb[1];
        const uint32_t n_move  = n_kv_new - first_move;

        std::vector<uint8_t> buf(static_cast<size_t>(n_move) * row_bytes);
        for (uint32_t i = 0; i < n_move; ++i) {
            ggml_backend_tensor_get(tensor, buf.data() + i * row_bytes,
                                    keep_positions[first_move + i] * row_bytes, row_bytes);
        }
        ggml_backend_tensor_set(tensor, buf.data(), first_move * row_bytes, n_move * row_bytes);
    };

    for (const auto & layer : layers) {
        compact_rows(layer.k);
        compact_rows(layer.v);
    }

    /* Rebuild cell metadata at the new physical positions */
    llama_kv_cells compacted;
    compacted.resize(cells.size());

    for (uint32_t dst = 0; dst < n_kv_new; ++dst) {
        const uint32_t src = keep_positions[dst];
        compacted.pos_set(dst, cells.pos_get(src));
        compacted.ext_set(dst, cells.ext_get(src));

        for (int32_t seq_id = 0; seq_id < LLAMA_MAX_SEQ; ++seq_id) {
            if (cells.seq_has(src, seq_id)) {
                compacted.seq_add(dst, seq_id);
            }
        }
    }

    cells      = std::move(compacted);
    v_heads[0] = n_kv_new < cells.size() ? n_kv_new : 0;

    LLAMA_LOG_INFO("%s: TriAttention compacted KV cache from %u to %u rows\n", __func__, n_kv_old, n_kv_new);
    return true;
}


uint32_t llama_kv_cache::get_n_kv(const slot_info & sinfo) const {
    uint32_t result = 0;

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const auto & cells = v_cells[sinfo.strm[s]];

        result = std::max(std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur))), result);
    }

    return result;
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    if (!v_trans) {
        // note: v->nb[1] <= v->nb[2]
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, ns,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),          // v->nb[1]
                ggml_row_size(v->type, n_embd_v_gqa),                   // v->nb[2]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size),           // v->nb[3]
                ggml_row_size(v->type, n_embd_v_gqa*kv_size)*sinfo.s0);
    }

    // note: v->nb[1] > v->nb[2]
    return ggml_view_4d(ctx, v,
            n_kv, hparams.n_head_kv(il), hparams.n_embd_head_v(il), ns,
            ggml_row_size(v->type, kv_size*hparams.n_embd_head_v(il)),  // v->nb[1]
            ggml_row_size(v->type, kv_size),                        // v->nb[2]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa),           // v->nb[3]
            ggml_row_size(v->type, kv_size*n_embd_v_gqa)*sinfo.s0);
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    const int64_t n_stream = k->ne[2];

    if (n_stream > 1) {
        const int64_t kv_size = get_size();

        assert(n_embd_gqa == k->ne[0]);
        assert(kv_size    == k->ne[1]);

        // merge the buffer across all streams because the idxs are global
        k = ggml_reshape_2d(ctx, k, n_embd_gqa, kv_size*n_stream);
    }

    // store the current K values into the cache
    // For KV_OSCAR_INT2: flag op_params[0]=1 so set_rows applies WHT during K encode.
    // V writes (cpy_v) leave op_params[0]=0 → plain ref encoding, no WHT.
    ggml_tensor * k_set = ggml_set_rows(ctx, k, k_cur, k_idxs);
    if (k->type == GGML_TYPE_KV_OSCAR_INT2) {
        k_set->op_params[0] = 1;
    }
    return k_set;
}

ggml_tensor * llama_kv_cache::cpy_k_res(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);
    ggml_tensor * k_res = layers[ikv].k_res;
    if (!k_res) {
        return nullptr;
    }

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_embd_gqa  = n_embd_head * n_head;

    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    // merge head dims so set_rows sees a 2D scatter [n_embd_gqa, n_tokens]
    ggml_tensor * k_cur_2d = ggml_view_2d(ctx, k_cur, n_embd_gqa, k_cur->ne[2], k_cur->nb[2], 0);

    const int64_t n_stream = k_res->ne[2];
    ggml_tensor * k_res_2d = k_res;
    if (n_stream > 1) {
        const int64_t kv_size = get_size();
        k_res_2d = ggml_reshape_2d(ctx, k_res, n_embd_gqa, kv_size * n_stream);
    }

    // k_cur_2d is F32, k_res_2d is F16 — ggml_set_rows handles the cast
    return ggml_set_rows(ctx, k_res_2d, k_cur_2d, k_idxs);
}

ggml_tensor * llama_kv_cache::get_k_res(ggml_context * ctx, int32_t il, uint32_t n_kv, const slot_info & sinfo) const {
    const int32_t ikv = map_layer_ids.at(il);
    auto * k_res = layers[ikv].k_res;
    if (!k_res) {
        return nullptr;
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k_res->ne[0];
    const uint32_t ns = sinfo.s1 - sinfo.s0 + 1;

    return ggml_view_4d(ctx, k_res,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, ns,
            ggml_row_size(k_res->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k_res->type, n_embd_k_gqa),
            ggml_row_size(k_res->type, n_embd_k_gqa * kv_size),
            ggml_row_size(k_res->type, n_embd_k_gqa * kv_size) * sinfo.s0);
}

uint32_t llama_kv_cache::get_oscar_res_window() const {
    return oscar_residual_window;
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il, const slot_info & sinfo) const {
    GGML_UNUSED(sinfo);

    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    const int64_t n_stream = v->ne[2];

    // take this branch when FA is enabled (the V cache is not transposed)
    if (!v_trans) {
        v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

        if (n_stream > 1) {
            const int64_t kv_size = get_size();

            assert(n_embd_gqa == v->ne[0]);
            assert(kv_size    == v->ne[1]);

            // merge the buffer across all streams because the idxs are global
            v = ggml_reshape_2d(ctx, v, n_embd_gqa, kv_size*n_stream);
        }

        return ggml_set_rows(ctx, v, v_cur, v_idxs);
    }

    if (ggml_row_size(v_cur->type, n_embd_gqa) == v_cur->nb[2]) {
        // we can merge dims 0, 1 and 2
        v_cur = ggml_reshape_2d(ctx, v_cur, n_embd_gqa, n_tokens);
    } else {
        // otherwise -> make a copy to get contiguous data
        v_cur = ggml_cont_2d   (ctx, v_cur, n_embd_gqa, n_tokens);
    }

    // [TAG_V_CACHE_VARIABLE]
    if (n_embd_gqa < v->ne[0]) {
        v_cur = ggml_pad(ctx, v_cur, v->ne[0] - n_embd_gqa, 0, 0, 0);
    }

    // in this branch the v_idxs are constructed in such a way that each row is a single head element
    ggml_tensor * v_view = ggml_reshape_2d(ctx, v, 1, ggml_nelements(v));

    v_cur = ggml_reshape_2d(ctx, v_cur, 1, ggml_nelements(v_cur));

    return ggml_set_rows(ctx, v_view, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs;

    if (!v_trans) {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);
    } else {
        v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens*hparams.n_embd_v_gqa_max());
    }

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
        const int64_t offs = sinfo.strm[s]*get_size();

        for (uint32_t i = 0; i < sinfo.size(); ++i) {
            data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
        }
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    const uint32_t n_tokens = ubatch->n_tokens;
    GGML_ASSERT(n_tokens == (int64_t) sinfo.size()*sinfo.n_stream());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    if (!v_trans) {
        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*get_size();

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                data[s*sinfo.size() + i] = offs + sinfo.idxs[s][i];
            }
        }
    } else {
        // note: the V cache is transposed when not using flash attention
        const int64_t kv_size = get_size();

        const int64_t n_embd_v_gqa = hparams.n_embd_v_gqa_max();

        for (uint32_t s = 0; s < sinfo.n_stream(); ++s) {
            const int64_t offs = sinfo.strm[s]*kv_size*n_embd_v_gqa;

            for (uint32_t i = 0; i < sinfo.size(); ++i) {
                for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                    data[s*sinfo.size()*n_embd_v_gqa + i*n_embd_v_gqa + j] = offs + j*kv_size + sinfo.idxs[s][i];
                }
            }
        }
    }
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t s = 0; s < n_stream; ++s) {
        const auto & cells = v_cells[s];

        for (uint32_t i = 0; i < cells.size(); ++i) {
            data[s*cells.size() + i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
        }
    }
}

struct args_set_input_kq_mask {
    const llama_hparams & hparams;
    const llama_ubatch  * ubatch;

    const std::vector<llama_kv_cells> & v_cells;
    const std::vector<uint32_t>       & seq_to_stream;

    uint32_t       n_swa;
    llama_swa_type swa_type;

    int64_t n_kv;
    int64_t n_stream;
    int64_t n_tps;
};

template<typename T, bool causal, bool swa, bool is_2d, bool alibi>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
  //const auto & hparams = args.hparams;
    const auto & ubatch  = args.ubatch;

    const auto & v_cells       = args.v_cells;
    const auto & seq_to_stream = args.seq_to_stream;

    const uint32_t       n_swa    = args.n_swa;
    const llama_swa_type swa_type = args.swa_type;

    const int64_t n_kv     = args.n_kv;
    const int64_t n_stream = args.n_stream;
    const int64_t n_tps    = args.n_tps;

    const T mask_keep = llama_cast<T>(0.0f);
    const T mask_drop = llama_cast<T>(-INFINITY);

    // the min position in the batch for each sequence
    llama_pos seq_pos_min[LLAMA_MAX_SEQ];
    std::fill(seq_pos_min, seq_pos_min + LLAMA_MAX_SEQ, INT32_MAX);

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        // bookkeeping of the KQ mask cells that could change for other tokens of the same sequence
        std::unordered_map<llama_seq_id, uint32_t>              seq_srct;
        std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idxs;

        for (uint32_t ii = 0; ii < n_tps; ++ii) {
            const uint32_t i = s*n_tps + ii;

            const llama_seq_id seq_id = ubatch->seq_id[i][0];

            const auto & cells = v_cells.at(seq_to_stream[seq_id]);

                  llama_pos p0 = -1;
            const llama_pos p1 = ubatch->pos[i];

            // for M-RoPE
            const llama_pos p1_x = is_2d ? ubatch->pos[i + ubatch->n_tokens*2] : 0;
            const llama_pos p1_y = is_2d ? ubatch->pos[i + ubatch->n_tokens]   : 0;

            const uint64_t idst = n_kv*i;

            // for tokens of the same sequence, the mask is mostly the same, so we can reuse it
            // the only cells that could change are the ones that are with similar positions as the
            //   ones in the batch (i.e. due to causal masking, SWA, etc.)
            // keep track of those cells and shortcut the loop to save time
            // note: this optimization is not compatible with Alibi position encoding
            // ref:  https://github.com/ggml-org/llama.cpp/pull/18842
            bool prev = false;

            auto & idxs = seq_idxs[seq_id];

            if (!alibi) {
                if (seq_srct.find(seq_id) != seq_srct.end()) {
                    const uint32_t srct = seq_srct[seq_id];

                    const uint64_t idst_prev = n_kv*srct;

                    std::copy(data + idst_prev, data + idst_prev + n_kv, data + idst);

                    prev = true;
                } else {
                    idxs.clear();
                    idxs.reserve(ubatch->n_tokens + n_swa + 32);

                    seq_srct[seq_id] = i;
                }
            }

            for (uint32_t jj = 0; jj < n_kv; ++jj) {
                uint32_t j = jj;

                // we have an exiting mask for this sequence -> update just seq_idxs
                if (!alibi) {
                    if (prev) {
                        if (jj >= idxs.size()) {
                            break;
                        }

                        j = idxs[jj];
                    }
                }

                if (cells.is_empty(j)) {
                    goto skip;
                }

                // mask the token if not the same sequence
                if (!cells.seq_has(j, seq_id)) {
                    goto skip;
                }

                p0 = cells.pos_get(j);

                if (!alibi) {
                    if (!prev) {
                        // record all cells for which: p0 >= seq_pos_min[seq_id] - n_swa - 32
                        if (p0 + (int32_t) (n_swa + 32) >= seq_pos_min[seq_id]) {
                            idxs.push_back(j);
                        }
                    }
                }

                if (causal) {
                    // mask future tokens
                    if (p0 > p1) {
                        goto skip;
                    }

                    // M-RoPE causal mask
                    if (is_2d) {
                        if (p0 == p1) {
                            const auto & p0_ext = cells.ext_get(j);

                            if (p0_ext.is_2d_gt(p1_x, p1_y)) {
                                goto skip;
                            }
                        }
                    }
                }

                // apply SWA if any
                if (swa) {
                    if (llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1)) {
                        goto skip;
                    }
                }

                if (alibi) {
                    data[idst + j] = llama_cast<T>(static_cast<float>(-std::abs(p0 - p1)));
                } else {
                    data[idst + j] = mask_keep;
                }

                continue;
skip:
                data[idst + j] = mask_drop;
            }
        }
    }
}

template<typename T, bool causal, bool swa, bool is_2d>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool alibi = args.hparams.use_alibi;
    if (alibi) {
        set_input_kq_mask_impl<T, causal, swa, is_2d, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, is_2d, false>(args, data);
    }
}

template<typename T, bool causal, bool swa>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool is_2d = args.ubatch->is_pos_2d();
    if (is_2d) {
        set_input_kq_mask_impl<T, causal, swa, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, swa, false>(args, data);
    }
}

template<typename T, bool causal>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data) {
    const bool swa = args.swa_type != LLAMA_SWA_TYPE_NONE;
    if (swa) {
        set_input_kq_mask_impl<T, causal, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, causal, false>(args, data);
    }
}

template<typename T>
static void set_input_kq_mask_impl(const args_set_input_kq_mask & args, T * data, bool causal_attn) {
    if (causal_attn) {
        set_input_kq_mask_impl<T, true> (args, data);
    } else {
        set_input_kq_mask_impl<T, false>(args, data);
    }
}

void llama_kv_cache::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const int64_t n_kv     = dst->ne[0];
    const int64_t n_stream = dst->ne[3]; // num streams in the current ubatch

    GGML_ASSERT(n_tokens%n_stream == 0);

    // n_tps == n_tokens_per_stream
    const int64_t n_tps = n_tokens/n_stream;

    //const int64_t t_start = ggml_time_us();

    const args_set_input_kq_mask args = {
        /*.hparams          =*/ hparams,
        /*.ubatch           =*/ ubatch,
        /*.v_cells          =*/ v_cells,
        /*.seq_to_stream    =*/ seq_to_stream,
        /*.n_swa            =*/ n_swa,
        /*.swa_type         =*/ swa_type,
        /*.n_kv             =*/ n_kv,
        /*.n_stream         =*/ n_stream,
        /*.n_tps            =*/ n_tps,
    };

    if (dst->type == GGML_TYPE_F16) {
        set_input_kq_mask_impl<ggml_fp16_t>(args, (ggml_fp16_t *) dst->data, causal_attn);
    } else {
        set_input_kq_mask_impl<float>(args, (float *) dst->data, causal_attn);
    }

    //const int64_t t_end = ggml_time_us();

    //LLAMA_LOG_ERROR("%s: kq mask time: %0.3f ms\n", __func__, (t_end - t_start)/1000.0);
}

void llama_kv_cache::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    const int64_t n_tokens = ubatch->n_tokens;

    GGML_ASSERT(n_stream == 1 && "TODO: support multiple streams");
    const auto & cells = v_cells[0];

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    GGML_ASSERT(!ubatch->equal_seqs()); // TODO: use ubatch->n_seqs instead of failing

    int32_t * data = (int32_t *) dst->data;

    const int32_t n_kv = dst->ne[0];

    for (int h = 0; h < 1; ++h) {
        for (int i = 0; i < n_tokens; ++i) {
            for (int j = 0; j < n_kv; ++j) {
                // the position when the cells is empty is irrelevant - it will be masked out later in the attention
                const llama_pos p0 = cells.is_empty(j) ? -1 : cells.pos_get(j);

                data[h*(n_kv*n_tokens) + i*n_kv + j] = llama_relative_position_bucket(p0, ubatch->pos[i], hparams.n_rel_attn_bkts, false);
            }
        }
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        size_k_bytes += ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        size_v_bytes += layer.v ? ggml_nbytes(layer.v) : 0;
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = ggml_mul_mat_aux(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size*n_stream]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    // OScaR INT2 K-shift WHT rotation (separate from k_rot since attn_rot_k=false for OScaR)
    ggml_tensor * k_oscar_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot) {
        kv_self->set_input_k_rot(k_rot);
    }

    if (k_oscar_rot) {
        kv_self->set_input_k_rot(k_oscar_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size()*n_stream);
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    // OScaR INT2 K-shift fix (§-FLAG-ATTN_ROT_KSHIFT):
    // K is stored in WHT-rotated domain.  Build a Hadamard rot tensor so that
    // build_rope_shift can: dequant → inv-WHT → RoPE → fwd-WHT → requant.
    // Use the first OScaR INT2 layer's head_dim (all OScaR layers must have the same WHT dim).
    for (const auto & layer : layers) {
        if (layer.k && layer.k->type == GGML_TYPE_KV_OSCAR_INT2 && !inp->k_oscar_rot) {
            const int64_t oscar_n = (int64_t) hparams.n_embd_head_k(layer.il);
            inp->k_oscar_rot = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, oscar_n, oscar_n);
            ggml_set_input(inp->k_oscar_rot);
            ggml_set_name(inp->k_oscar_rot, "oscar_inp_k_rot");
            break;
        }
    }

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;
        const bool is_turbo_k = (layer.k->type == GGML_TYPE_TURBOQ2_0 || layer.k->type == GGML_TYPE_TURBOQ3_0 || layer.k->type == GGML_TYPE_TURBOQ4_0 || layer.k->type == GGML_TYPE_TURBOQ8_0 || layer.k->type == GGML_TYPE_TURBOQ5_0 || layer.k->type == GGML_TYPE_TURBOQ6_0);
        if (is_turbo_k) { continue; }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size()*n_stream,
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        // For OScaR INT2 use the oscar WHT rot; for all others use the standard attn rot.
        ggml_tensor * layer_rot = (layer.k->type == GGML_TYPE_KV_OSCAR_INT2)
                                  ? inp->k_oscar_rot
                                  : inp->k_rot;

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, layer_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    io.write(&n_stream, sizeof(n_stream));

    for (uint32_t s = 0; s < n_stream; ++s) {
        cell_ranges_t cr { s, {} };

        uint32_t cell_count = 0;

        const auto & cells = v_cells[s];

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        // skip empty streams
        if (cell_count == 0) {
            continue;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (size_t) seq_id < seq_to_stream.size()));

    uint32_t n_stream_cur;
    io.read(&n_stream_cur, sizeof(n_stream_cur));
    if (n_stream_cur != n_stream) {
        throw std::runtime_error("n_stream mismatch");
    }

    for (uint32_t s = 0; s < n_stream; ++s) {
        uint32_t cell_count;
        io.read(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            continue;
        }

        const uint32_t strm = seq_id == -1 ? s : seq_to_stream[seq_id];

        slot_info sinfo;

        bool res = true;
        res = res && state_read_meta(io, strm, cell_count, sinfo, seq_id);
        res = res && state_read_data(io, strm, cell_count, sinfo);

        if (!res) {
            if (seq_id == -1) {
                clear(true);
            } else {
                seq_rm(seq_id, -1, -1);
            }
            throw std::runtime_error("failed to restore kv cache");
        }
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    const auto & cells = v_cells[cr.strm];

    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (hparams.n_pos_per_embd() > 1) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    const auto & cells = v_cells[cr.strm];

    const uint32_t v_trans = this->v_trans ? 1 : 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[cr.strm];

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        for (const auto & range : cr.data) {
            const size_t range_size = range.second - range.first;
            const size_t buf_size = range_size * k_size_row;
            io.write_tensor(k, range.first * k_size_row, buf_size);
        }
    }

    if (!v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write row size of value
            const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            io.write(&v_size_row, sizeof(v_size_row));

            // Read each range of cells of v_size length and write out
            for (const auto & range : cr.data) {
                const size_t range_size = range.second - range.first;
                const size_t buf_size = range_size * v_size_row;
                io.write_tensor(v, range.first * v_size_row, buf_size);
            }
        }
    } else {
        // When v is transposed, we also need the element size and get the element ranges from each row
        const uint32_t kv_size = cells.size();

        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[cr.strm];
            if (!v) {
                continue;
            }

            // Write value type
            const int32_t v_type_i = (int32_t) v->type;
            io.write(&v_type_i, sizeof(v_type_i));

            // Write element size
            const uint32_t v_size_el = ggml_type_size(v->type);
            io.write(&v_size_el, sizeof(v_size_el));

            // Write GQA embedding size
            io.write(&n_embd_v_gqa, sizeof(n_embd_v_gqa));

            // For each row, we get the element values of each cell
            for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                // Read each range of cells of v_size_el length and write out
                for (const auto & range : cr.data) {
                    const size_t range_size = range.second - range.first;
                    const size_t src_offset = (range.first + j * kv_size) * v_size_el;
                    const size_t buf_size = range_size * v_size_el;
                    io.write_tensor(v, src_offset, buf_size);
                }
            }
        }
    }

    // Turbo TCQ safety footer: embed codebook fingerprint so loading a cache
    // saved with a different TURBO_TCQ_CB/CB2 is detected at load time.
    bool has_tcq = false;
    for (const auto & layer : layers) {
        if (ggml_type_is_turboq_tcq(layer.k_stream[cr.strm]->type)) { has_tcq = true; break; }
        auto * v = layer.v_stream[cr.strm];
        if (v && ggml_type_is_turboq_tcq(v->type)) { has_tcq = true; break; }
    }
    if (has_tcq) {
        const uint32_t magic = 0x54514346; // "TQCF" — TurboQuant Cache Fingerprint
        const uint32_t fp    = turboq_tcq_fingerprint();
        io.write(&magic, sizeof(magic));
        io.write(&fp,    sizeof(fp));
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id) {
    auto & cells = v_cells[strm];
    auto & head  = v_heads[strm];

    if (dest_seq_id != -1) {
        // single sequence
        seq_rm(dest_seq_id, -1, -1);

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        sinfo = find_slot(ubatch, false);
        if (sinfo.empty()) {
            LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
            return false;
        }

        // TODO: we cannot yet restore llama_kv_cell_ext as the apply_ubatch() does not support it yet
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.n_stream() == 1);
        GGML_ASSERT(sinfo.idxs[0].size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[0][i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        clear(true);

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (hparams.n_pos_per_embd() > 1) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.s0 = strm;
        sinfo.s1 = strm;
        sinfo.resize(1);
        sinfo.strm[0] = strm;
        sinfo.idxs[0].resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[0][i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t strm, uint32_t cell_count, const slot_info & sinfo) {
    auto & cells = v_cells[strm];

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (this->v_trans != (bool) v_trans) {
        LLAMA_LOG_ERROR("%s: incompatible V transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k_stream[strm];

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            if (sinfo.is_contiguous()) {
                // Fast path: contiguous cells, single memcpy
                io.read_tensor(k, sinfo.head() * k_size_row, cell_count * k_size_row);
            } else {
                // Slow path: scatter to non-contiguous positions
                for (uint32_t i = 0; i < cell_count; ++i) {
                    const size_t dst_offset = sinfo.idxs[0][i] * k_size_row;
                    io.read_tensor(k, dst_offset, k_size_row);
                }
            }
        }
    }

    if (!this->v_trans) {
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read row size of value
            uint64_t v_size_row_ref;
            io.read(&v_size_row_ref, sizeof(v_size_row_ref));
            const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
            if (v_size_row != v_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells, single memcpy
                    io.read_tensor(v, sinfo.head() * v_size_row, cell_count * v_size_row);
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t i = 0; i < cell_count; ++i) {
                        const size_t dst_offset = sinfo.idxs[0][i] * v_size_row;
                        io.read_tensor(v, dst_offset, v_size_row);
                    }
                }
            }
        }
    } else {
        // For each layer, read the values for each cell (transposed)
        for (const auto & layer : layers) {
            const uint32_t il = layer.il;

            const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

            auto * v = layer.v_stream[strm];
            if (!v) {
                continue;
            }

            // Read type of value
            int32_t v_type_i_ref;
            io.read(&v_type_i_ref, sizeof(v_type_i_ref));
            const int32_t v_type_i = (int32_t) v->type;
            if (v_type_i != v_type_i_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
                return false;
            }

            // Read element size of value
            uint32_t v_size_el_ref;
            io.read(&v_size_el_ref, sizeof(v_size_el_ref));
            const size_t v_size_el = ggml_type_size(v->type);
            if (v_size_el != v_size_el_ref) {
                LLAMA_LOG_ERROR("%s: mismatched value element size (%zu != %zu, layer %d)\n", __func__, v_size_el, (size_t) v_size_el_ref, il);
                return false;
            }

            // Read GQA embedding size
            uint32_t n_embd_v_gqa_ref;
            io.read(&n_embd_v_gqa_ref, sizeof(n_embd_v_gqa_ref));
            if (n_embd_v_gqa != n_embd_v_gqa_ref) {
                LLAMA_LOG_ERROR("%s: mismatched GQA embedding size (%u != %u, layer %d)\n", __func__, n_embd_v_gqa, n_embd_v_gqa_ref, il);
                return false;
            }

            if (cell_count) {
                if (sinfo.is_contiguous()) {
                    // Fast path: contiguous cells
                    const uint32_t h = sinfo.head();
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        const size_t dst_offset = (h + j * cells.size()) * v_size_el;
                        io.read_tensor(v, dst_offset, cell_count * v_size_el);
                    }
                } else {
                    // Slow path: scatter to non-contiguous positions
                    for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
                        for (uint32_t i = 0; i < cell_count; ++i) {
                            const size_t dst_offset = (sinfo.idxs[0][i] + j * cells.size()) * v_size_el;
                            io.read_tensor(v, dst_offset, v_size_el);
                        }
                    }
                }
            }
        }
    }

    // Turbo TCQ safety: verify codebook fingerprint matches current process.
    bool has_tcq = false;
    for (const auto & layer : layers) {
        if (ggml_type_is_turboq_tcq(layer.k_stream[strm]->type)) { has_tcq = true; break; }
        auto * v = layer.v_stream[strm];
        if (v && ggml_type_is_turboq_tcq(v->type)) { has_tcq = true; break; }
    }
    if (has_tcq) {
        uint32_t magic_ref = 0;
        io.read(&magic_ref, sizeof(magic_ref));
        if (magic_ref != 0x54514346) { // "TQCF"
            LLAMA_LOG_ERROR("%s: turbo TCQ cache file missing codebook fingerprint — "
                            "file may have been saved by an older build without TCQ safety checks\n", __func__);
            return false;
        }
        uint32_t fp_ref = 0;
        io.read(&fp_ref, sizeof(fp_ref));
        const uint32_t fp_now = turboq_tcq_fingerprint();
        if (fp_ref != fp_now) {
            LLAMA_LOG_ERROR("%s: turbo TCQ codebook mismatch — cache was saved with fingerprint "
                            "0x%08X but current TURBO_TCQ_CB/CB2 gives 0x%08X. "
                            "Set the same codebook env vars as when the cache was created.\n",
                            __func__, fp_ref, fp_now);
            return false;
        }
        LLAMA_LOG_INFO("%s: turbo TCQ codebook fingerprint verified (0x%08X)\n", __func__, fp_ref);
    }

    return true;
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv) {
    n_kv = kv->get_size();

    const uint32_t n_stream = kv->get_n_stream();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].s0 = 0;
    sinfos[0].s1 = n_stream - 1;
    sinfos[0].idxs.resize(n_stream);
    for (uint32_t s = 0; s < n_stream; ++s) {
        sinfos[0].strm.push_back(s);
        sinfos[0].idxs[s].resize(1, 0);
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift,
        stream_copy_info sc_info) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift), sc_info(std::move(sc_info)) {
    if (!do_shift && this->sc_info.empty()) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches,
        llama_mtp_op_type mtp_op_type) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)), mtp_op_type(mtp_op_type) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift, sc_info);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur], mtp_op_type);
    n_kv = kv->get_n_kv(sinfos[i_cur]);

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

llama_pos llama_kv_cache_context::seq_pos_max(llama_seq_id seq_id) const {
    return kv->seq_pos_max(seq_id);
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}


ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_k_res(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k_res(ctx, k_cur, k_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il, sinfos[i_cur]);
}

ggml_tensor * llama_kv_cache_context::get_k_res(ggml_context * ctx, int32_t il) const {
    return kv->get_k_res(ctx, il, n_kv, sinfos[i_cur]);
}

uint32_t llama_kv_cache_context::get_oscar_res_window() const {
    return kv->get_oscar_res_window();
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_kq_mask(ggml_tensor * dst, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kq_mask(dst, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_pos_bucket(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_pos_bucket(dst, ubatch);
}

void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

#ifdef LLAMA_KV_COMPACTION
//
// KV cache compaction (Attention Matching) — first-landing slice (SELECT only).
// Ported from jandhyala-dev/modelai-llama.cpp. The graph-execution path and the
// SOLVER/OMP/NONUNIFORM/CHUNKED methods are deferred follow-ups; the solver
// wrappers below are intentionally stubbed (return false) in this slice.
//

bool llama_kv_cache::compacted_prefix_runtime_supported() const {
    // Check instance-level SWA config, not model-level hparams. When used as
    // kv_base inside llama_kv_cache_iswa, this instance has n_swa=0 and
    // swa_type=NONE even though the model has SWA layers.
    if (n_swa > 0 || swa_type != LLAMA_SWA_TYPE_NONE) {
        static std::atomic<bool> warned_swa{false};
        if (!warned_swa.exchange(true)) {
            LLAMA_LOG_WARN("%s: compacted prefix not supported for SWA sub-cache "
                           "(n_swa=%u, swa_type=%d) — models like Gemma3 use iSWA; "
                           "compaction only applies to the base (non-SWA) cache\n",
                           __func__, n_swa, (int)swa_type);
        }
        return false;
    }

    // M-RoPE models (Qwen2-VL, Qwen3-VL) use multi-dimensional positions that
    // the compacted-prefix pipeline cannot represent. IMROPE (Qwen3.5, etc.) is
    // safe for text-only compaction.
    if (hparams.n_pos_per_embd() > 1 && hparams.rope_type != LLAMA_ROPE_TYPE_IMROPE) {
        static std::atomic<bool> warned{false};
        if (!warned.exchange(true)) {
            LLAMA_LOG_WARN("%s: compacted prefix not supported for M-RoPE models (n_pos_per_embd=%u)\n",
                           __func__, hparams.n_pos_per_embd());
        }
        return false;
    }

    return true;
}

bool llama_kv_cache::supports_compaction() const {
    return compacted_prefix_runtime_supported();
}

// NOTE: must mirror the checks in compacted_prefix_runtime_supported().
std::string llama_kv_cache::compaction_unsupported_reason() const {
    if (n_swa > 0 || swa_type != LLAMA_SWA_TYPE_NONE) {
        return "swa_cache";
    }
    if (hparams.n_pos_per_embd() > 1 && hparams.rope_type != LLAMA_ROPE_TYPE_IMROPE) {
        return "mrope_positions";
    }
    return "";
}

bool llama_kv_cache::has_compacted_prefix() const {
    if (!compacted_prefix_runtime_supported()) {
        return false;
    }
    for (llama_seq_id sid = 0; sid < (llama_seq_id) seq_to_stream.size(); ++sid) {
        if (compacted_prefix.execution_enabled(sid)) {
            return true;
        }
    }
    return false;
}

const std::string & llama_kv_cache::compacted_prefix_method() const {
    return compacted_prefix_last_method;
}

bool llama_kv_cache::compacted_prefix_configure(
        llama_seq_id seq_id,
        uint32_t logical_token_count,
        const std::vector<llama_pos> & logical_positions,
        llama_pos live_suffix_pos0) {
    const bool is_imrope = (hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE);
    const bool ok = compacted_prefix.configure_seq(seq_id, logical_token_count, logical_positions, live_suffix_pos0, is_imrope);
    if (ok) {
        ++compacted_prefix_version_counter;
    }
    return ok;
}

void llama_kv_cache::compacted_prefix_clear(llama_seq_id seq_id, bool data) {
    if (seq_id < 0) {
        compacted_prefix.clear(data);
    } else {
        compacted_prefix.clear_seq(seq_id, data);
    }
    ++compacted_prefix_version_counter;
}

bool llama_kv_cache::compacted_prefix_enabled(llama_seq_id seq_id) const {
    return compacted_prefix.is_enabled(seq_id);
}

bool llama_kv_cache::compacted_prefix_set_execution(llama_seq_id seq_id, bool enabled) {
    if (!compacted_prefix_runtime_supported()) {
        return false;
    }
    return compacted_prefix.set_execution(seq_id, enabled);
}

bool llama_kv_cache::compacted_prefix_execution_enabled(llama_seq_id seq_id) const {
    return compacted_prefix_runtime_supported() && compacted_prefix.execution_enabled(seq_id);
}

bool llama_kv_cache::compacted_prefix_stream_owned_by_seq(
        uint32_t strm, llama_seq_id seq_id, std::vector<uint32_t> & live_cell_idxs) const {
    live_cell_idxs.clear();

    if (strm >= v_cells.size()) {
        return false;
    }

    const auto & cells = v_cells[strm];
    const uint32_t used_max_p1 = cells.used_max_p1();

    for (uint32_t idx = 0; idx < used_max_p1; ++idx) {
        if (cells.is_empty(idx)) {
            continue;
        }

        if (cells.seq_count(idx) != 1 || cells.seq_get(idx) != seq_id) {
            return false;
        }

        live_cell_idxs.push_back(idx);
    }

    return true;
}

void llama_kv_cache::compacted_prefix_pack_stream_tensors(
        uint32_t strm, const std::vector<uint32_t> & live_cell_idxs) {
    const uint32_t n_live = live_cell_idxs.size();
    const uint32_t kv_size = get_size();

    for (const auto & layer : layers) {
        auto * k = layer.k_stream[strm];
        if (k) {
            const size_t row_size = ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il));
            std::vector<uint8_t> packed(size_t(n_live) * row_size);

            for (uint32_t i = 0; i < n_live; ++i) {
                ggml_backend_tensor_get(k, packed.data() + size_t(i) * row_size, size_t(live_cell_idxs[i]) * row_size, row_size);
            }

            if (!packed.empty()) {
                ggml_backend_tensor_set(k, packed.data(), 0, packed.size());
            }
        }

        auto * v = layer.v_stream[strm];
        if (!v) {
            continue;
        }

        if (!v_trans) {
            const size_t row_size = ggml_row_size(v->type, hparams.n_embd_v_gqa(layer.il));
            std::vector<uint8_t> packed(size_t(n_live) * row_size);

            for (uint32_t i = 0; i < n_live; ++i) {
                ggml_backend_tensor_get(v, packed.data() + size_t(i) * row_size, size_t(live_cell_idxs[i]) * row_size, row_size);
            }

            if (!packed.empty()) {
                ggml_backend_tensor_set(v, packed.data(), 0, packed.size());
            }
            continue;
        }

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(layer.il);
        // V type must not be block-quantized (element-wise access assumes blck_size==1).
        GGML_ASSERT(ggml_blck_size(v->type) == 1 && "compacted V copy assumes non-block-quantized type");
        const size_t v_size_el = ggml_type_size(v->type);
        std::vector<uint8_t> packed(size_t(n_live) * v_size_el);

        for (uint32_t j = 0; j < n_embd_v_gqa; ++j) {
            for (uint32_t i = 0; i < n_live; ++i) {
                const size_t src_offset = (size_t(live_cell_idxs[i]) + size_t(j) * kv_size) * v_size_el;
                ggml_backend_tensor_get(v, packed.data() + size_t(i) * v_size_el, src_offset, v_size_el);
            }

            if (!packed.empty()) {
                const size_t dst_offset = size_t(j) * kv_size * v_size_el;
                ggml_backend_tensor_set(v, packed.data(), dst_offset, packed.size());
            }
        }
    }
}

// NOTE: not thread-safe; all llama_kv_cache operations assume single-threaded access.
bool llama_kv_cache::compacted_prefix_reclaim_live_kv(llama_seq_id seq_id) {
    if (!compacted_prefix_runtime_supported() || seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }

    const auto * state = compacted_prefix.get_seq(seq_id);
    if (state == nullptr || !state->enabled || state->live_suffix_pos0 < 0) {
        return false;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    auto & cells = v_cells[strm];
    if (cells.get_has_shift()) {
        return false;
    }

    std::vector<uint32_t> used_idxs;
    if (!compacted_prefix_stream_owned_by_seq(strm, seq_id, used_idxs)) {
        return false;
    }

    std::vector<uint32_t> keep_idxs;
    keep_idxs.reserve(used_idxs.size());
    for (const uint32_t idx : used_idxs) {
        if (cells.pos_get(idx) >= state->live_suffix_pos0) {
            keep_idxs.push_back(idx);
        }
    }

    const bool already_dense = [&]() {
        if (keep_idxs.empty()) {
            return cells.get_used() == 0;
        }

        if (keep_idxs.size() != cells.get_used()) {
            return false;
        }

        for (uint32_t i = 0; i < keep_idxs.size(); ++i) {
            if (keep_idxs[i] != i) {
                return false;
            }
        }

        return true;
    }();

    if (!already_dense) {
        compacted_prefix_pack_stream_tensors(strm, keep_idxs);

        llama_kv_cells packed;
        packed.resize(cells.size());
        if (!keep_idxs.empty()) {
            packed.set(0, cells.cp(keep_idxs));
        }

        cells = std::move(packed);
    }

    v_heads[strm] = keep_idxs.size();
    return true;
}

bool llama_kv_cache::compacted_prefix_select_from_live_kv(
        llama_seq_id seq_id,
        uint32_t target_tokens,
        llama_pos live_suffix_pos0,
        llama_kv_compact_pipeline_stats * stats,
        llama_pos p0) {
    const bool ok = llama_kv_compact_select_from_live_kv(*this, seq_id, target_tokens, live_suffix_pos0, stats, p0);
    if (ok) { compacted_prefix_last_method = "select"; }
    return ok;
}

// Deferred solver-based methods (first-landing slice ships SELECT only).
static bool llama_kv_compact_method_not_ported(const char * method) {
    static std::atomic<bool> warned{false};
    if (!warned.exchange(true)) {
        LLAMA_LOG_WARN("%s: KV compaction method '%s' is not yet ported "
                       "(first-landing slice implements SELECT only)\n", __func__, method);
    }
    return false;
}

bool llama_kv_cache::compacted_prefix_fit_from_live_kv(
        llama_seq_id, uint32_t, llama_pos, llama_kv_compact_pipeline_stats *, llama_pos,
        uint32_t, int, float) {
    return llama_kv_compact_method_not_ported("solver");
}

bool llama_kv_cache::compacted_prefix_omp_from_live_kv(
        llama_seq_id, uint32_t, llama_pos, llama_kv_compact_pipeline_stats *, llama_pos,
        uint32_t, int, float) {
    return llama_kv_compact_method_not_ported("omp");
}

bool llama_kv_cache::compacted_prefix_nonuniform_from_live_kv(
        llama_seq_id, uint32_t, llama_pos, llama_kv_compact_pipeline_stats *, uint32_t,
        int, float, llama_pos) {
    return llama_kv_compact_method_not_ported("nonuniform");
}

bool llama_kv_cache::compacted_prefix_chunked_from_live_kv(
        llama_seq_id, uint32_t, llama_pos, llama_kv_compact_pipeline_stats *, llama_pos,
        uint32_t, int, float) {
    return llama_kv_compact_method_not_ported("chunked");
}

bool llama_kv_cache::compacted_prefix_layer_layout_for_solver(int32_t il, llama_compacted_prefix_layer_layout & out) const {
    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end()) {
        return false;
    }

    const auto & layouts = compacted_prefix.get_layouts();
    const int32_t ikv = it->second;
    if (ikv < 0 || size_t(ikv) >= layouts.size()) {
        return false;
    }

    out = layouts[size_t(ikv)];
    return true;
}

bool llama_kv_cache::compacted_prefix_seq_positions(
        llama_seq_id seq_id, llama_pos p0, llama_pos p1, std::vector<llama_pos> & out) const {
    out.clear();

    if (seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }

    if (p0 < 0) {
        p0 = 0;
    }
    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    const uint32_t strm = seq_to_stream[seq_id];
    const auto & cells = v_cells[strm];

    out.reserve(cells.get_used());
    for (uint32_t idx = 0; idx < cells.used_max_p1(); ++idx) {
        if (cells.is_empty(idx) || !cells.seq_has(idx, seq_id)) {
            continue;
        }
        const llama_pos pos = cells.pos_get(idx);
        if (pos >= p0 && pos < p1) {
            out.push_back(pos);
        }
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return !out.empty();
}

bool llama_kv_cache::compacted_prefix_copy_k_head_f32(
        int32_t il,
        llama_seq_id seq_id,
        uint32_t head_kv,
        const std::vector<llama_pos> & positions,
        std::vector<float> & out) const {
    out.clear();

    llama_compacted_prefix_layer_layout layout;
    if (!compacted_prefix_layer_layout_for_solver(il, layout) || !is_block_aligned_for_head(layout.type_k, layout.n_embd_head_k)) {
        return false;
    }

    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end() || head_kv >= layout.n_head_kv || seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    const auto & cells = v_cells[strm];
    const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
    const uint32_t head_dim = layout.n_embd_head_k;
    const size_t row_size = ggml_row_size(layout.type_k, n_embd_k_gqa);
    const size_t head_offset = ggml_row_size(layout.type_k, size_t(head_kv) * head_dim);

    std::unordered_map<llama_pos, uint32_t> pos_to_idx;
    for (uint32_t idx = 0; idx < cells.used_max_p1(); ++idx) {
        if (!cells.is_empty(idx) && cells.seq_has(idx, seq_id)) {
            pos_to_idx.emplace(cells.pos_get(idx), idx);
        }
    }

    std::vector<uint8_t> row_bytes(row_size);
    std::vector<float> row_f32(head_dim);
    out.resize(size_t(positions.size()) * head_dim);

    const auto & layer = layers[size_t(it->second)];
    auto * k = layer.k_stream[strm];
    if (k == nullptr) {
        return false;
    }

    for (size_t i = 0; i < positions.size(); ++i) {
        const auto pos_it = pos_to_idx.find(positions[i]);
        if (pos_it == pos_to_idx.end()) {
            return false;
        }
        ggml_backend_tensor_get(k, row_bytes.data(), size_t(pos_it->second) * row_size, row_size);
        type_to_float(row_bytes.data() + head_offset, layout.type_k, row_f32.data(), head_dim);
        std::copy(row_f32.begin(), row_f32.end(), out.begin() + ptrdiff_t(i * head_dim));
    }

    return true;
}

bool llama_kv_cache::compacted_prefix_copy_v_head_f32(
        int32_t il,
        llama_seq_id seq_id,
        uint32_t head_kv,
        const std::vector<llama_pos> & positions,
        std::vector<float> & out) const {
    out.clear();

    llama_compacted_prefix_layer_layout layout;
    if (!compacted_prefix_layer_layout_for_solver(il, layout) || !is_block_aligned_for_head(layout.type_v, layout.n_embd_head_v)) {
        return false;
    }
    if (layout.n_embd_head_v == 0) {
        return true;
    }

    const auto it = map_layer_ids.find(il);
    if (it == map_layer_ids.end() || head_kv >= layout.n_head_kv || seq_id < 0 || (size_t) seq_id >= seq_to_stream.size()) {
        return false;
    }

    const uint32_t strm = seq_to_stream[seq_id];
    const auto & cells = v_cells[strm];
    const auto & layer = layers[size_t(it->second)];
    auto * v = layer.v_stream[strm];
    if (v == nullptr) {
        return false;
    }

    std::unordered_map<llama_pos, uint32_t> pos_to_idx;
    for (uint32_t idx = 0; idx < cells.used_max_p1(); ++idx) {
        if (!cells.is_empty(idx) && cells.seq_has(idx, seq_id)) {
            pos_to_idx.emplace(cells.pos_get(idx), idx);
        }
    }

    const uint32_t head_dim = layout.n_embd_head_v;
    const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);
    out.resize(size_t(positions.size()) * head_dim);

    if (!v_trans) {
        const size_t row_size = ggml_row_size(layout.type_v, n_embd_v_gqa);
        const size_t head_offset = ggml_row_size(layout.type_v, size_t(head_kv) * head_dim);
        std::vector<uint8_t> row_bytes(row_size);
        std::vector<float> row_f32(head_dim);

        for (size_t i = 0; i < positions.size(); ++i) {
            const auto pos_it = pos_to_idx.find(positions[i]);
            if (pos_it == pos_to_idx.end()) {
                return false;
            }
            ggml_backend_tensor_get(v, row_bytes.data(), size_t(pos_it->second) * row_size, row_size);
            type_to_float(row_bytes.data() + head_offset, layout.type_v, row_f32.data(), head_dim);
            std::copy(row_f32.begin(), row_f32.end(), out.begin() + ptrdiff_t(i * head_dim));
        }
        return true;
    }

    // Batch row extraction for transposed V. Transposed V is stored as a 1D
    // tensor of n_embd_v_gqa * kv_size elements; logical row d starts at linear
    // index d * kv_size. kv_size is block-aligned (KV cache padding), so each
    // row is independently quantized.
    const uint32_t kv_size = get_size();
    GGML_ASSERT(kv_size % ggml_blck_size(layout.type_v) == 0 &&
                "KV cache size must be block-aligned for transposed V extraction");
    const uint32_t head_offset = head_kv * head_dim;
    const size_t row_bytes = ggml_row_size(layout.type_v, kv_size);
    std::vector<uint8_t> row_buf(row_bytes);
    std::vector<float> row_f32(kv_size);

    std::vector<uint32_t> cell_indices(positions.size());
    for (size_t i = 0; i < positions.size(); ++i) {
        const auto pos_it = pos_to_idx.find(positions[i]);
        if (pos_it == pos_to_idx.end()) {
            return false;
        }
        cell_indices[i] = pos_it->second;
    }

    for (uint32_t j = 0; j < head_dim; ++j) {
        const size_t row_offset = size_t(head_offset + j) * row_bytes;
        ggml_backend_tensor_get(v, row_buf.data(), row_offset, row_bytes);
        type_to_float(row_buf.data(), layout.type_v, row_f32.data(), kv_size);

        for (size_t i = 0; i < positions.size(); ++i) {
            out[i * head_dim + j] = row_f32[cell_indices[i]];
        }
    }

    return true;
}
#endif // LLAMA_KV_COMPACTION
