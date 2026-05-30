#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tria_hip_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes);

/*
 * GPU-side TriAttention scoring for q8_0 K cache (Phase C).
 *
 * IMPORTANT - host vs device memory:
 *   In this fork the per-layer K is captured into CPU-backend buffers
 *   (llama-triattention.cpp), so k_data_host is a HOST pointer. The
 *   scoring function uploads the scored slice to the GPU internally. The
 *   persistent stats pointers (omega, q_mean_real, q_mean_imag, q_abs_mean)
 *   and the in/out global_scores buffer are DEVICE pointers (allocated by
 *   tria_hip_stats_upload). key_pos and offsets are host arrays.
 *
 * GQA: the CPU reference (tria_score_kv_head) aggregates n_heads/n_kv_heads
 *   query heads per KV head: for each query head it z-normalizes the raw
 *   scores across tokens and takes the max across query heads, then the
 *   runtime z-normalizes that per-KV-head result and max-aggregates into
 *   global_scores with a layer weight. This kernel reproduces that two-stage
 *   reduction exactly, so it is correct for both GQA (n_heads != n_kv_heads)
 *   and MHA (n_heads == n_kv_heads).
 *
 * k_data_host  : HOST pointer to q8_0 K tensor (all tokens, this layer)
 * n_tokens     : number of tokens to score (n_new, starting at score_start)
 * score_start  : offset into K tensor (first token to score)
 * cur_pos      : current sequence length (n_kv) — delta = cur_pos - key_pos[s] + offset
 * n_embd_k_gqa : total K embedding dim (n_kv_heads * head_dim)
 * n_kv_heads   : number of KV heads
 * n_heads      : number of query heads (>= n_kv_heads, multiple of it)
 * head_dim     : per-head dimension (<= 128, multiple of 32)
 * freq_count   : head_dim / 2 (number of rotary frequency pairs)
 * rope_neox    : 1 = NEOX/IMROPE split-half layout, 0 = NORMAL interleaved
 * key_pos      : HOST array of token positions [n_tokens]
 * omega_dev    : device omega[freq_count]
 * q_mean_real_dev : device q_mean_real[num_layers * n_heads * freq_count]
 * q_mean_imag_dev : device q_mean_imag[num_layers * n_heads * freq_count]
 * q_abs_mean_dev  : device q_abs_mean[num_layers * n_heads * freq_count], or NULL
 *                   (NULL disables the per-frequency residual "extra" term)
 * q_mean_offset: element offset for this layer = layer_idx * n_heads * freq_count
 * layer_weight : normalized layer importance weight (applied to positive z)
 * global_scores_dev : device global_scores[n_tokens] (in/out, max-aggregated)
 * n_offsets    : number of future offsets (TRIA_N_OFFSETS = 17)
 * offsets      : HOST array of future offsets [n_offsets]
 */
int tria_hip_score_q8_0(
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

/* Allocate/free persistent GPU buffers for scoring stats.
 * Any of the *_out pointers may be NULL to skip that array; a single array
 * can be uploaded by passing only its data + out pointer (used to push the
 * per-pass global_scores slice as well). */
int  tria_hip_stats_upload(const float * omega, int freq_count,
                            const float * q_mean_real, const float * q_mean_imag,
                            int n_kv_heads,
                            float ** omega_dev_out,
                            float ** q_mean_real_dev_out,
                            float ** q_mean_imag_dev_out);
void tria_hip_stats_free(float * omega_dev, float * q_mean_real_dev, float * q_mean_imag_dev);
int  tria_hip_scores_download(float * scores, const float * scores_dev, int n_scores);

#ifdef __cplusplus
}
#endif
