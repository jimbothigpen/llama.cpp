/*
 * triattention-backend.h — function pointer ABI for optional HIP acceleration
 *
 * When GGML_BACKEND_DL is OFF (static linking), pointers resolve to
 * triattention-hip.hip symbols directly.
 * When GGML_BACKEND_DL is ON (dynamic loading), pointers are NULL
 * and scoring falls back to CPU.
 */
#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct tria_backend {
    int  (*stats_upload)(const float *, int, const float *, const float *, int,
                         float **, float **, float **);
    void (*stats_free)(float *, float *, float *);
    /* score_q8_0 args (Phase C, GQA-aware):
     *   k_data_host, n_tokens, score_start, cur_pos, n_embd_k_gqa,
     *   n_kv_heads, n_heads, head_dim, freq_count, rope_neox,
     *   key_pos, omega_dev, q_mean_real_dev, q_mean_imag_dev, q_abs_mean_dev,
     *   q_mean_offset, layer_weight, global_scores_dev, n_offsets, offsets */
    int  (*score_q8_0)(const void *, int, int, int, int, int, int, int, int, int,
                       const int *, const float *, const float *, const float *, const float *,
                       int, float, float *, int, const int *);
    int  (*scores_download)(float *, const float *, int);
    int  (*compact_rows)(void *, const uint32_t *, uint32_t, uint32_t, uint32_t);
};

/* Global backend — initialized by tria_backend_init() */
extern struct tria_backend g_tria_backend;

/* Returns 1 if GPU acceleration is available */
int tria_backend_init(void);

#ifdef __cplusplus
}
#endif
