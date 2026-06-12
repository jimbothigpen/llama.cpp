#include "pflash-graph.h"
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static constexpr int N_LOOKAHEAD = 8;

// ─── Placeholder scorer (used when scorer_path == "test") ─────────────────
void pflash_generate_placeholder_scores(float * out, int n_lookahead, int S,
        const int32_t * token_ids) {
    for (int n = 0; n < n_lookahead; n++) {
        for (int j = 0; j < S; j++) {
            float rel_pos    = (float)j / (float)(S - 1);
            float u_shape    = 1.0f - 4.0f * (rel_pos - 0.5f) * (rel_pos - 0.5f);
            float importance = 1.0f - u_shape;
            float noise      = (float)((token_ids[j] * 2654435761u) & 0xFFFF) / 65536.0f * 0.3f;
            out[n * S + j]   = importance + noise;
        }
    }
}

// ─── CPU helpers ──────────────────────────────────────────────────────────

// Download tensor to host as F32 (handles F32 and F16 source types)
static void tensor_get_f32(const ggml_tensor * t, std::vector<float> & dst) {
    size_t n = (size_t)ggml_nelements(t);
    dst.resize(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, dst.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        for (size_t i = 0; i < n; i++) dst[i] = ggml_fp16_to_fp32(tmp[i]);
    } else {
        fprintf(stderr, "pflash: unsupported scorer tensor type %d\n", (int)t->type);
        std::fill(dst.begin(), dst.end(), 0.0f);
    }
}

// ─── Arch-dispatched norm helper ──────────────────────────────────────────
// PFLASH_NORM_RMS:   rms_norm(x) * weight           (Llama, Qwen, Mistral)
// PFLASH_NORM_LAYER: rms_norm(x) * (1 + weight)     (Gemma3)
static ggml_tensor * pflash_build_norm(
        ggml_context    * ctx,
        ggml_tensor     * x,
        ggml_tensor     * weight,
        float             eps,
        pflash_norm_type  norm_type) {
    ggml_tensor * rn = ggml_rms_norm(ctx, x, eps);
    if (norm_type == PFLASH_NORM_LAYER) {
        // Gemma3 uses (1 + w) scaling: weight is initialized near 0, effective scale ≈ 1
        return ggml_mul(ctx, rn, ggml_add(ctx, weight, ggml_new_f32(ctx, 1.0f)));
    }
    return ggml_mul(ctx, rn, weight);
}

// ─── Real scorer forward pass ──────────────────────────────────────────────
//
// Builds a single ggml graph covering N_LOOKAHEAD layers.
// Each layer:
//   1. Computes Q_last (last-token query) and K_all (all-token keys) on GPU.
//   2. Computes the last-token attention scores entirely on-device (Q·Kᵀ →
//      softmax over tokens → mean over heads) and marks the [S] score vector
//      as the only output tensor.
//   3. Advances hidden state h through FFN (no attention residual) for the
//      next layer.
// After graph compute, only the per-layer [S] score vectors are downloaded
// (< 600 KB at S=24K) — the per-layer K tensor and the O(S²) attention matrix
// never leave VRAM. The same graph runs on the CPU backend when no GPU exists.
//
// Skipping attention residual keeps O(S²) attention matrix off GPU memory.
// RoPE is omitted (not required for relative importance ranking in S3).
//
pflash_scorer_result pflash_score(
        const std::vector<int32_t> & token_ids,
        const pflash_model         & model,
        const FlashPrefillConfig   & fp_cfg,
        int                          gpu_device) {

    const int S         = (int)token_ids.size();
    const int n_layers  = std::min(N_LOOKAHEAD, model.n_layers);
    const int n_embd    = model.n_embd;
    const int n_heads   = model.n_heads;
    const int n_kv      = model.n_kv_heads;
    const int d_head    = model.d_head;
    const int gqa_ratio = n_heads / n_kv;
    const float scale_attn = 1.0f / sqrtf((float)d_head);
    const float norm_eps   = model.norm_eps;

    pflash_scorer_result result;
    result.n_lookahead = n_layers; // actual scoring layers (may be < N_LOOKAHEAD for hybrid models)
    result.seq_len     = S;
    result.running_max.assign((size_t)n_layers * S, 0.0f);

    fprintf(stderr,
        "pflash: real scorer — %d tokens, %d layers, n_heads=%d n_kv=%d d=%d\n",
        S, n_layers, n_heads, n_kv, d_head);

    // Phase 3: use GPU backend for scorer compute; weights are in GPU VRAM.
    // Try discrete GPU first, then integrated GPU. Vulkan on iGPU (e.g. Strix Halo)
    // registers as GGML_BACKEND_DEVICE_TYPE_IGPU, not GPU — mirrors ggml_backend_init_best().
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    ggml_backend_t backend = dev ? ggml_backend_dev_init(dev, nullptr) : nullptr;
    if (!backend) {
        fprintf(stderr, "pflash: no GPU device; falling back to CPU scorer\n");
        backend = ggml_backend_cpu_init();
    }
    if (!backend) {
        fprintf(stderr, "pflash: scorer cannot init backend\n");
        return result;
    }
    (void)gpu_device;

    // ── Build compute context ──────────────────────────────────────────────
    // Estimate tensor count: ~30 per layer + embedding + input = ~270 total.
    struct ggml_init_params ctx_params = {
        /* .mem_size   = */ 16 * 1024 * 1024, // 16 MB: GPU tensor overhead headroom
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    ggml_context * ctx = ggml_init(ctx_params);
    if (!ctx) {
        fprintf(stderr, "pflash: scorer ggml_init failed\n");
        ggml_backend_free(backend);
        return result;
    }

    // ── Graph inputs ───────────────────────────────────────────────────────
    // inp_tokens: [S] (I32) — filled before graph compute
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, S);
    ggml_set_input(inp_tokens);

    // Initial embedding lookup: h = tok_embd[inp_tokens], cast to F32
    // tok_embd is always F32 (dequantized at load time if GGUF was quantized).
    ggml_tensor * h = ggml_cast(ctx,
        ggml_get_rows(ctx, model.tok_embd, inp_tokens),
        GGML_TYPE_F32);

    // Per-layer output tensors: per-token importance scores computed on-GPU.
    // out_score[l]: [S] — mean over heads of last-token softmax attention weights.
    // Replaces the old per-layer Q/K download + CPU scalar attention loop: the
    // O(S²) attention is never materialized and only a tiny [S] vector leaves VRAM.
    std::vector<ggml_tensor *> out_score(n_layers);

    // PARITY_HARNESS (strippable): when PFLASH_PARITY is set, also expose Q/K
    // so the post-graph block can recompute the original scalar reference.
    const bool pflash_parity = getenv("PFLASH_PARITY") != nullptr;
    std::vector<ggml_tensor *> out_Q, out_K;
    if (pflash_parity) { out_Q.assign(n_layers, nullptr); out_K.assign(n_layers, nullptr); }

    for (int l = 0; l < n_layers; l++) {
        const auto & lw = model.layers[l];

        // ── Attention norm ─────────────────────────────────────────────────
        ggml_tensor * h_norm = pflash_build_norm(ctx, h, lw.attn_norm, norm_eps, model.norm_type);

        // ── Q for last token only (position S-1) ──────────────────────────
        // h_norm: [n_embd, S] — extract last column (ne[0]=n_embd is fast dim)
        // byte offset of column (S-1) = (S-1) * n_embd * sizeof(F32 or F16)
        // We use F32 in the compute graph since rms_norm outputs F32.
        ggml_tensor * h_last = ggml_view_2d(ctx, h_norm,
            n_embd, 1,
            (size_t)n_embd * sizeof(float),          // row stride (bytes)
            (size_t)(S - 1) * n_embd * sizeof(float) // offset (bytes)
        );

        // Q_flat = wq^T @ h_last → [q_proj_dim, 1]
        // Qwen3:   q_proj_dim = n_heads * d_head (Q only)
        // Qwen3.5: q_proj_dim = n_heads * 2 * d_head (Q interleaved with gate per head)
        ggml_tensor * Q_flat = ggml_mul_mat(ctx, lw.wq, h_last);
        if (lw.attn_q_b) {
            Q_flat = ggml_add(ctx, Q_flat, lw.attn_q_b);
        }

        // Reshape to [d_head, n_heads], apply per-head Q-norm if present.
        // For Qwen3.5, extract Q (first d_head of each 2*d_head block) via stride view.
        ggml_tensor * Q_3d;
        if (Q_flat->ne[0] == (int64_t)n_heads * 2 * d_head) {
            // Qwen3.5 full-attention: reshape to [2*d_head, n_heads], view first d_head per head
            ggml_tensor * Q_both = ggml_reshape_3d(ctx, Q_flat, 2 * d_head, n_heads, 1);
            ggml_tensor * Q_nc   = ggml_view_3d(ctx, Q_both,
                d_head, n_heads, 1,
                Q_both->nb[1], Q_both->nb[2], 0);
            Q_3d = ggml_cont_2d(ctx, Q_nc, d_head, n_heads);
        } else {
            Q_3d = ggml_reshape_2d(ctx, Q_flat, d_head, n_heads);
        }

        // G4 fix: guard QK norm — lw.q_norm is NULL for arches without per-head norms
        ggml_tensor * Q_normed;
        if (lw.q_norm) {
            Q_normed = ggml_mul(ctx, ggml_rms_norm(ctx, Q_3d, norm_eps), lw.q_norm);
        } else {
            Q_normed = Q_3d;
        }

        // ── K for all tokens ───────────────────────────────────────────────
        // K_flat = wk^T @ h_norm → [n_kv*d_head, S]
        ggml_tensor * K_flat = ggml_mul_mat(ctx, lw.wk, h_norm);
        if (lw.attn_k_b) {
            K_flat = ggml_add(ctx, K_flat, lw.attn_k_b);
        }

        // Reshape to [d_head, n_kv, S]; apply per-head K-norm if present.
        ggml_tensor * K_3d = ggml_reshape_3d(ctx, K_flat, d_head, n_kv, S);

        // G5 fix: guard QK norm — lw.k_norm is NULL for arches without per-head norms
        ggml_tensor * K_normed;
        if (lw.k_norm) {
            K_normed = ggml_mul(ctx, ggml_rms_norm(ctx, K_3d, norm_eps), lw.k_norm);
        } else {
            K_normed = K_3d;
        }

        // ── On-GPU last-token attention scoring ────────────────────────────
        // CPU reference (per token t):
        //   score[t] = mean_h softmax_t( scale · Q[:,h] · K[:,kv(h),t] )
        // GQA: head h maps to kv group kv(h) = h / gqa_ratio. Group the heads
        // so each kv head broadcasts over its gqa_ratio query heads via mul_mat.
        //
        // Q3: [d_head, gqa_ratio, n_kv]  — Q_normed [d_head, n_heads] regrouped
        //     (head index = hh + kv·gqa_ratio, matching kv(h) = h / gqa_ratio)
        ggml_tensor * Q3 = ggml_reshape_3d(ctx, Q_normed, d_head, gqa_ratio, n_kv);
        // K3: [d_head, S, n_kv] — K_normed [d_head, n_kv, S] with S,n_kv swapped
        ggml_tensor * K3 = ggml_cont(ctx, ggml_permute(ctx, K_normed, 0, 2, 1, 3));
        // scores[t, hh, kv] = Q[:,head]·K[:,kv,t]  → [S, gqa_ratio, n_kv]
        ggml_tensor * scores = ggml_mul_mat(ctx, K3, Q3);
        scores = ggml_scale(ctx, scores, scale_attn);
        // softmax over t (ne[0] = S), per (query head, kv head)
        ggml_tensor * attn = ggml_soft_max(ctx, scores);            // [S, gqa_ratio, n_kv]
        // mean over all n_heads = gqa_ratio·n_kv: flatten head dims, sum, scale
        ggml_tensor * attn2d = ggml_reshape_2d(ctx, attn, S, n_heads);   // [S, n_heads]
        ggml_tensor * attn_t = ggml_cont(ctx, ggml_transpose(ctx, attn2d)); // [n_heads, S]
        ggml_tensor * score  = ggml_sum_rows(ctx, attn_t);          // [1, S]
        score = ggml_scale(ctx, score, 1.0f / (float)n_heads);
        score = ggml_reshape_1d(ctx, score, S);                     // [S]
        ggml_set_output(score);
        out_score[l] = score;

        if (pflash_parity) {
            ggml_set_output(Q_normed); out_Q[l] = Q_normed;
            ggml_set_output(K_normed); out_K[l] = K_normed;
        }

        // ── FFN residual (advances h for next layer without O(S²) attention) ─
        ggml_tensor * h_ffn_norm = pflash_build_norm(ctx, h, lw.ffn_norm, norm_eps, model.norm_type);

        ggml_tensor * gate    = ggml_mul_mat(ctx, lw.ffn_gate, h_ffn_norm); // [n_ff, S]
        ggml_tensor * up      = ggml_mul_mat(ctx, lw.ffn_up,   h_ffn_norm); // [n_ff, S]
        ggml_tensor * act;
        if (model.ffn_act == PFLASH_FFN_GELU) {
            act = ggml_mul(ctx, ggml_gelu(ctx, gate), up);                  // GeGLU (Gemma3)
        } else {
            act = ggml_mul(ctx, ggml_silu(ctx, gate), up);                  // SwiGLU (default)
        }
        ggml_tensor * ffn_out = ggml_mul_mat(ctx, lw.ffn_down, act);        // [n_embd, S]

        h = ggml_add(ctx, h, ffn_out);
    }

    // ── Allocate and run graph ─────────────────────────────────────────────
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, 512, false);
    for (int l = 0; l < n_layers; l++) {
        ggml_build_forward_expand(gf, out_score[l]);
    }
    ggml_build_forward_expand(gf, h); // ensure FFN chain executes

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    bool ok = ggml_gallocr_alloc_graph(alloc, gf);
    if (!ok) {
        fprintf(stderr, "pflash: scorer graph allocation failed\n");
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return result;
    }

    // Fill token ID input — clamp to scorer vocab bounds as a safety measure.
    // With a matched-vocab scorer (e.g. Qwen3.5-0.8B for Qwen3.5-9B target), all
    // token IDs should be in range; this guard prevents crashes with mismatched models.
    const int32_t n_vocab_max = (int32_t)(model.n_vocab - 1);
    std::vector<int32_t> clamped_ids(token_ids.size());
    for (int i = 0; i < S; i++) {
        clamped_ids[i] = std::max(0, std::min(token_ids[i], n_vocab_max));
    }
    ggml_backend_tensor_set(inp_tokens, clamped_ids.data(), 0, S * sizeof(int32_t));

    enum ggml_status status = ggml_backend_graph_compute(backend, gf);
    if (status != GGML_STATUS_SUCCESS) {
        fprintf(stderr, "pflash: scorer graph compute failed (status=%d)\n", (int)status);
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        ggml_backend_free(backend);
        return result;
    }

    // ── Download per-layer scores (computed on-device) ────────────────────
    // out_score[l]: [S] — already the mean-over-heads last-token attention
    // weights. Only n_layers · S floats leave VRAM (< 600 KB at S=24K); the
    // O(S²) attention matrix and the per-layer K tensor never touch the host.
    std::vector<float> score_host;
    for (int l = 0; l < n_layers; l++) {
        tensor_get_f32(out_score[l], score_host); // [S]
        float * score_l = result.running_max.data() + (size_t)l * S;
        std::copy(score_host.begin(), score_host.begin() + S, score_l);
    }

    // PARITY_HARNESS (strippable): recompute the original scalar reference from
    // downloaded Q/K and report max-abs-diff + Pearson corr vs the GPU scores.
    if (pflash_parity) {
        std::vector<float> Q_host, K_host, ref(S), attn_row(S);
        double worst = 0.0, sxy = 0.0, sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0;
        size_t npts = 0;
        for (int l = 0; l < n_layers; l++) {
            tensor_get_f32(out_Q[l], Q_host); // [d_head, n_heads]
            tensor_get_f32(out_K[l], K_host); // [d_head, n_kv, S]
            std::fill(ref.begin(), ref.end(), 0.0f);
            for (int hh = 0; hh < n_heads; hh++) {
                int kv_h = hh / gqa_ratio;
                const float * q    = Q_host.data() + (size_t)hh   * d_head;
                const float * K_kv = K_host.data() + (size_t)kv_h * d_head;
                for (int t = 0; t < S; t++) {
                    const float * k = K_kv + (size_t)t * (size_t)n_kv * d_head;
                    float dot = 0.0f;
                    for (int d = 0; d < d_head; d++) dot += q[d] * k[d];
                    attn_row[t] = dot * scale_attn;
                }
                float mx = *std::max_element(attn_row.data(), attn_row.data() + S);
                float sum = 0.0f;
                for (int t = 0; t < S; t++) { attn_row[t] = expf(attn_row[t] - mx); sum += attn_row[t]; }
                for (int t = 0; t < S; t++) ref[t] += attn_row[t] / sum;
            }
            const float * gpu = result.running_max.data() + (size_t)l * S;
            for (int t = 0; t < S; t++) {
                float r = ref[t] / (float)n_heads, g = gpu[t];
                worst = std::max(worst, (double)std::fabs(r - g));
                sx += r; sy += g; sxx += (double)r * r; syy += (double)g * g; sxy += (double)r * g; npts++;
            }
        }
        double cov = sxy - sx * sy / npts;
        double corr = cov / (sqrt(sxx - sx * sx / npts) * sqrt(syy - sy * sy / npts));
        fprintf(stderr, "pflash-parity: layers=%d S=%d  max_abs_diff=%.3e  pearson=%.8f\n",
            n_layers, S, worst, corr);
    }

    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    ggml_backend_free(backend);

    (void)fp_cfg;
    return result;
}
