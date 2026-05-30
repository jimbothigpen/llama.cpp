#pragma once

/*
 * triattention-vulkan.h — Vulkan compute-shader port of the TriAttention Phase-C
 * GPU GQA scoring kernel. The ABI mirrors triattention-hip.h exactly so that
 * triattention-backend.c can point g_tria_backend at either set of symbols.
 *
 * Memory model (same as the HIP path): per-layer K is captured into CPU-backend
 * buffers, so k_data_host is a HOST pointer that score_q8_0 uploads internally.
 * The persistent stats handles (omega/q_mean_real/q_mean_imag/q_abs_mean) and the
 * in/out global_scores handle are opaque DEVICE handles returned by
 * tria_vk_stats_upload (reinterpreted as float* for ABI compatibility — they are
 * never dereferenced as floats by the runtime, only passed back here). key_pos
 * and offsets are host arrays.
 *
 * See triattention-hip.h for the full per-argument documentation of
 * tria_vk_score_q8_0 — the contract is identical.
 */

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tria_vk_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes);

int tria_vk_score_q8_0(
        const void * k_data_host,
        int n_tokens,
        int score_start,
        int cur_pos,
        int n_embd_k_gqa,
        int n_kv_heads,
        int n_heads,
        int head_dim,
        int freq_count,
        int rope_neox,
        const int * key_pos,
        const float * omega_dev,
        const float * q_mean_real_dev,
        const float * q_mean_imag_dev,
        const float * q_abs_mean_dev,
        int q_mean_offset,
        float layer_weight,
        float * global_scores_dev,
        int n_offsets,
        const int * offsets);

int  tria_vk_stats_upload(const float * omega, int freq_count,
                          const float * q_mean_real, const float * q_mean_imag,
                          int n_kv_heads,
                          float ** omega_dev_out,
                          float ** q_mean_real_dev_out,
                          float ** q_mean_imag_dev_out);
void tria_vk_stats_free(float * omega_dev, float * q_mean_real_dev, float * q_mean_imag_dev);
int  tria_vk_scores_download(float * scores, const float * scores_dev, int n_scores);

#ifdef __cplusplus
}
#endif
