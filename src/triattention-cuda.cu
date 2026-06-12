#include "triattention-cuda.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

/*
 * TriAttention GPU scorer — CUDA arm (T4 / sm_75).
 *
 * Functionally identical to triattention-hip.hip, with one deliberate
 * difference in the raw-score kernel: it is REGISTER-FRUGAL. The HIP kernel
 * materializes per-thread arrays k_f32[256] + rel_r[128] + rel_i[128] (~512
 * floats), which spill on sm_75. This kernel interchanges the freq/offset
 * loops and dequantizes K on the fly, so the only per-thread arrays are
 * delta[17] + trig_acc[17] (TRIA_N_OFFSETS = 17). The result is BIT-IDENTICAL
 * to the HIP/CPU path — see worker-scratch/triattn-cuda-scorer-2026-06-11/design.md.
 */

/* ── compact kernel (gather retained rows, shift down in place) ──────── */

static __global__ void tria_cuda_gather_rows_kernel(
        const uint8_t * tensor_data,
        const uint32_t * indices,
        uint8_t * scratch,
        size_t row_bytes,
        uint32_t n_move) {
    const size_t total_bytes = row_bytes * n_move;
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_bytes) {
        return;
    }
    const size_t row = idx / row_bytes;
    const size_t col = idx - row * row_bytes;
    const size_t src_row = indices[row];
    scratch[idx] = tensor_data[src_row * row_bytes + col];
}

static inline void tria_cuda_free_ptr(void * ptr) {
    if (ptr) {
        (void) cudaFree(ptr);
    }
}

extern "C" int tria_cuda_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes) {
    if (tensor_data == nullptr || h_indices == nullptr) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (first_move >= n_keep || row_bytes == 0) {
        return 0;
    }

    const uint32_t n_move = n_keep - first_move;
    const size_t indices_bytes = static_cast<size_t>(n_move) * sizeof(uint32_t);
    const size_t scratch_bytes = static_cast<size_t>(n_move) * row_bytes;

    uint32_t * d_indices = nullptr;
    uint8_t * d_scratch = nullptr;

    cudaError_t err = cudaMalloc(reinterpret_cast<void **>(&d_indices), indices_bytes);
    if (err != cudaSuccess) {
        return static_cast<int>(err);
    }
    err = cudaMalloc(reinterpret_cast<void **>(&d_scratch), scratch_bytes);
    if (err != cudaSuccess) {
        (void) cudaFree(d_indices);
        return static_cast<int>(err);
    }
    err = cudaMemcpy(d_indices, h_indices + first_move, indices_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        (void) cudaFree(d_scratch);
        (void) cudaFree(d_indices);
        return static_cast<int>(err);
    }

    constexpr uint32_t block_size = 256;
    const size_t total_bytes = scratch_bytes;
    const uint32_t n_blocks = static_cast<uint32_t>((total_bytes + block_size - 1) / block_size);

    tria_cuda_gather_rows_kernel<<<dim3(n_blocks), dim3(block_size), 0, 0>>>(
            static_cast<const uint8_t *>(tensor_data),
            d_indices, d_scratch,
            static_cast<size_t>(row_bytes), n_move);

    err = cudaGetLastError();
    if (err == cudaSuccess) {
        err = cudaMemcpy(
                static_cast<uint8_t *>(tensor_data) + static_cast<size_t>(first_move) * row_bytes,
                d_scratch, scratch_bytes, cudaMemcpyDeviceToDevice);
    }
    if (err == cudaSuccess) {
        err = cudaDeviceSynchronize();
    }

    (void) cudaFree(d_scratch);
    (void) cudaFree(d_indices);
    return static_cast<int>(err);
}

/* ── GPU scoring for q8_0 K cache (Phase C, GQA-aware) ───────────────── */

#define TRIA_QK8_0 32
#define TRIA_Q8_0_BLOCK_SIZE (2 + TRIA_QK8_0)  /* fp16 scale + 32 int8 */
#define TRIA_MAX_HD 256                         /* max supported head_dim (large-head: hd=256) */
#define TRIA_MAX_FC (TRIA_MAX_HD / 2)           /* head_dim <= 256 => fc <= 128 */
#define TRIA_MAX_OFFSETS 17                     /* == TRIA_N_OFFSETS (triattention.h) */
#define TRIA_REDUCE_BLOCK 256

static __device__ __forceinline__ float tria_fp16_to_fp32(uint16_t d_bits) {
    const uint32_t sign = (d_bits >> 15) & 1u;
    const uint32_t exp  = (d_bits >> 10) & 0x1fu;
    const uint32_t mant = d_bits & 0x3ffu;
    float d;
    if (exp == 0) {
        d = (sign ? -1.f : 1.f) * ldexpf((float)mant, -24);
    } else if (exp == 31) {
        uint32_t f32bits = (sign << 31) | 0x7f800000u;  /* ±inf (matches HIP) */
        memcpy(&d, &f32bits, 4);
    } else {
        uint32_t f32bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        memcpy(&d, &f32bits, 4);
    }
    return d;
}

/* Dequantize a single head-local element idx (0..head_dim-1) from a q8_0 row.
 * Bit-identical to indexing the HIP kernel's precomputed k_f32[idx]:
 * d = fp16(block scale); value = d * (float)int8. */
static __device__ __forceinline__ float tria_dequant_one(
        const uint8_t * __restrict__ row, int head_block_start, int idx) {
    const int blk_idx = idx / TRIA_QK8_0;
    const int j       = idx - blk_idx * TRIA_QK8_0;
    const uint8_t * blk = row + (size_t)(head_block_start + blk_idx) * TRIA_Q8_0_BLOCK_SIZE;
    uint16_t d_bits;
    memcpy(&d_bits, blk, 2);
    const float d = tria_fp16_to_fp32(d_bits);
    const int8_t q = (int8_t)blk[2 + j];
    return d * (float)q;
}

/*
 * Pass 1 — raw per-token score for one query head (eq 6-11, max_beta=0).
 * out[s] = mean_offsets( sum_f rel_r*cos(omega*delta) - rel_i*sin(omega*delta) ) + extra
 *   rel_r = qmr*kr + qmi*ki ; rel_i = qmi*kr - qmr*ki ; delta = cur_pos - pos + offset
 *   extra = sum_f max(0, q_abs_mean - |q_mean|) * |k_f|   (skipped if q_abs_mean == NULL)
 * One thread per token. K head slice is shared across the KV head's query heads (GQA).
 *
 * REGISTER-FRUGAL: f is the outer loop; per-offset accumulation lives in
 * trig_acc[TRIA_MAX_OFFSETS] (constant-indexed, fully unrolled -> register
 * resident). K is dequantized two elements at a time on the fly, so no
 * k_f32[256]/rel_r[128]/rel_i[128] arrays are materialized. The per-offset
 * accumulation order (ascending f) and the outer reduction order (ascending
 * offset) match the HIP/CPU reference exactly, so the result is bit-identical.
 */
static __global__ void tria_raw_score_kernel(
        const uint8_t * __restrict__ k_q8,        /* device slice, row 0 == score_start */
        int n_tokens,
        int cur_pos,
        int n_embd_k_gqa,
        int kvi,
        int head_dim,
        int freq_count,
        int rope_neox,
        const int * __restrict__ key_pos,         /* device [n_tokens] */
        const float * __restrict__ q_mean_real,   /* [freq_count] this head */
        const float * __restrict__ q_mean_imag,
        const float * __restrict__ q_abs_mean,    /* [freq_count] this head or NULL */
        const float * __restrict__ omega,         /* [freq_count] */
        const int * __restrict__ offsets,
        int n_offsets,
        float * __restrict__ scores_out) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= n_tokens) return;

    const int row_bytes = (n_embd_k_gqa / TRIA_QK8_0) * TRIA_Q8_0_BLOCK_SIZE;
    const uint8_t * row = k_q8 + (size_t)s * row_bytes;
    const int head_block_start = kvi * (head_dim / TRIA_QK8_0);

    const int pos = key_pos[s];

    /* per-offset delta (s-local, freq-independent) — kept register-resident */
    float delta[TRIA_MAX_OFFSETS];
    float trig_acc[TRIA_MAX_OFFSETS];
    #pragma unroll
    for (int oi = 0; oi < TRIA_MAX_OFFSETS; oi++) {
        delta[oi]    = (oi < n_offsets) ? (float)(cur_pos - pos + offsets[oi]) : 0.f;
        trig_acc[oi] = 0.f;
    }

    float extra = 0.f;
    for (int f = 0; f < freq_count; f++) {
        int ridx, iidx;
        if (rope_neox) { ridx = f;        iidx = freq_count + f; }
        else           { ridx = 2 * f;    iidx = 2 * f + 1;      }
        const float kr = tria_dequant_one(row, head_block_start, ridx);
        const float ki = tria_dequant_one(row, head_block_start, iidx);
        const float qr = q_mean_real[f];
        const float qi = q_mean_imag[f];
        const float rel_r = qr * kr + qi * ki;
        const float rel_i = qi * kr - qr * ki;
        if (q_abs_mean) {
            const float qma = sqrtf(qr * qr + qi * qi);
            const float residual = q_abs_mean[f] - qma;
            if (residual > 0.f) {
                extra += residual * sqrtf(kr * kr + ki * ki);
            }
        }
        const float w = omega[f];
        #pragma unroll
        for (int oi = 0; oi < TRIA_MAX_OFFSETS; oi++) {
            if (oi < n_offsets) {
                float c, sn;
                sincosf(w * delta[oi], &sn, &c);
                trig_acc[oi] += rel_r * c - rel_i * sn;
            }
        }
    }

    float trig_sum = 0.f;
    #pragma unroll
    for (int oi = 0; oi < TRIA_MAX_OFFSETS; oi++) {
        if (oi < n_offsets) trig_sum += trig_acc[oi];
    }
    scores_out[s] = trig_sum / (float)n_offsets + extra;
}

/* block-wide sum reduction over smem[0..blockDim) — blockDim must be pow2 */
static __device__ __forceinline__ float tria_block_sum(float * smem, float v) {
    smem[threadIdx.x] = v;
    __syncthreads();
    for (int st = blockDim.x / 2; st > 0; st >>= 1) {
        if (threadIdx.x < st) smem[threadIdx.x] += smem[threadIdx.x + st];
        __syncthreads();
    }
    return smem[0];
}

/*
 * Pass 2 — z-normalize raw scores across tokens, max-aggregate across the
 * KV head's query heads. is_first selects assign vs max (mirrors CPU `first`).
 * Single block. CPU level-1 std = sqrt(var/n), floored to 1e-6.
 */
static __global__ void tria_znorm_into_agg_kernel(
        const float * __restrict__ scores,
        int n_tokens,
        int is_first,
        float * __restrict__ agg) {
    extern __shared__ float smem[];

    float ls = 0.f;
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) ls += scores[i];
    const float mean = tria_block_sum(smem, ls) / (float)n_tokens;
    __syncthreads();

    float lv = 0.f;
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        const float d = scores[i] - mean;
        lv += d * d;
    }
    float std = sqrtf(tria_block_sum(smem, lv) / (float)n_tokens);
    if (std < 1e-6f) std = 1e-6f;

    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        const float z = (scores[i] - mean) / std;
        if (is_first)        agg[i] = z;
        else if (z > agg[i]) agg[i] = z;
    }
}

/*
 * Pass 3 — z-normalize per-KV-head aggregate across tokens, weight positive z
 * by layer importance, max-aggregate into global_scores. Single block.
 * CPU level-2 (runtime) std = sqrt(var/n + 1e-8).
 */
static __global__ void tria_znorm_lw_global_kernel(
        const float * __restrict__ agg,
        int n_tokens,
        float layer_weight,
        float * __restrict__ global_scores) {
    extern __shared__ float smem[];

    float ls = 0.f;
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) ls += agg[i];
    const float mean = tria_block_sum(smem, ls) / (float)n_tokens;
    __syncthreads();

    float lv = 0.f;
    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        const float d = agg[i] - mean;
        lv += d * d;
    }
    const float std = sqrtf(tria_block_sum(smem, lv) / (float)n_tokens + 1e-8f);

    for (int i = threadIdx.x; i < n_tokens; i += blockDim.x) {
        const float z = (agg[i] - mean) / std;
        const float wz = z > 0.f ? z * layer_weight : z;
        if (wz > global_scores[i]) global_scores[i] = wz;
    }
}

extern "C" int tria_cuda_stats_upload(
        const float * omega, int freq_count,
        const float * q_mean_real, const float * q_mean_imag,
        int n_kv_heads,
        float ** omega_dev_out,
        float ** q_mean_real_dev_out,
        float ** q_mean_imag_dev_out) {
    cudaError_t err;
    const size_t omega_bytes = (size_t)freq_count * sizeof(float);
    const size_t qm_bytes    = (size_t)n_kv_heads * freq_count * sizeof(float);

    if (omega_dev_out && omega && freq_count > 0) {
        *omega_dev_out = nullptr;
        err = cudaMalloc((void**)omega_dev_out, omega_bytes);
        if (err != cudaSuccess) return (int)err;
        err = cudaMemcpy(*omega_dev_out, omega, omega_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            tria_cuda_free_ptr(*omega_dev_out);
            *omega_dev_out = nullptr;
            return (int)err;
        }
    }

    if (q_mean_real_dev_out && q_mean_real && qm_bytes > 0) {
        *q_mean_real_dev_out = nullptr;
        err = cudaMalloc((void**)q_mean_real_dev_out, qm_bytes);
        if (err != cudaSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_cuda_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            return (int)err;
        }
        err = cudaMemcpy(*q_mean_real_dev_out, q_mean_real, qm_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_cuda_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            tria_cuda_free_ptr(*q_mean_real_dev_out);
            *q_mean_real_dev_out = nullptr;
            return (int)err;
        }
    }

    if (q_mean_imag_dev_out && q_mean_imag && qm_bytes > 0) {
        *q_mean_imag_dev_out = nullptr;
        err = cudaMalloc((void**)q_mean_imag_dev_out, qm_bytes);
        if (err != cudaSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_cuda_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            if (q_mean_real_dev_out && *q_mean_real_dev_out) { tria_cuda_free_ptr(*q_mean_real_dev_out); *q_mean_real_dev_out = nullptr; }
            return (int)err;
        }
        err = cudaMemcpy(*q_mean_imag_dev_out, q_mean_imag, qm_bytes, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_cuda_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            if (q_mean_real_dev_out && *q_mean_real_dev_out) { tria_cuda_free_ptr(*q_mean_real_dev_out); *q_mean_real_dev_out = nullptr; }
            tria_cuda_free_ptr(*q_mean_imag_dev_out);
            *q_mean_imag_dev_out = nullptr;
            return (int)err;
        }
    }

    return 0;
}

extern "C" void tria_cuda_stats_free(float * omega_dev, float * q_mean_real_dev, float * q_mean_imag_dev) {
    tria_cuda_free_ptr(omega_dev);
    tria_cuda_free_ptr(q_mean_real_dev);
    tria_cuda_free_ptr(q_mean_imag_dev);
}

extern "C" int tria_cuda_scores_download(float * scores, const float * scores_dev, int n_scores) {
    if (!scores || !scores_dev || n_scores <= 0) return 0;
    return (int)cudaMemcpy(scores, scores_dev, (size_t)n_scores * sizeof(float), cudaMemcpyDeviceToHost);
}

extern "C" int tria_cuda_score_q8_0(
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
        const int * offsets) {

    if (!k_data_host || n_tokens <= 0 || n_kv_heads <= 0) return 0;
    if (n_heads < n_kv_heads || (n_heads % n_kv_heads) != 0) return (int)cudaErrorInvalidValue;
    if (freq_count > TRIA_MAX_FC || head_dim > TRIA_MAX_HD || (head_dim % TRIA_QK8_0) != 0) {
        return (int)cudaErrorInvalidValue;
    }
    if (n_offsets <= 0 || n_offsets > TRIA_MAX_OFFSETS) {
        return (int)cudaErrorInvalidValue;
    }
    const int gqa = n_heads / n_kv_heads;
    const int row_bytes = (n_embd_k_gqa / TRIA_QK8_0) * TRIA_Q8_0_BLOCK_SIZE;
    const size_t k_slice_bytes = (size_t)n_tokens * row_bytes;

    uint8_t * d_k   = nullptr;
    int     * d_key = nullptr;
    int     * d_off = nullptr;
    float   * d_raw = nullptr;
    float   * d_agg = nullptr;
    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void**)&d_k,   k_slice_bytes);              if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc((void**)&d_key, (size_t)n_tokens * sizeof(int));   if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc((void**)&d_off, (size_t)n_offsets * sizeof(int));  if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc((void**)&d_raw, (size_t)n_tokens * sizeof(float)); if (err != cudaSuccess) goto cleanup;
    err = cudaMalloc((void**)&d_agg, (size_t)n_tokens * sizeof(float)); if (err != cudaSuccess) goto cleanup;

    err = cudaMemcpy(d_k, (const uint8_t *)k_data_host + (size_t)score_start * row_bytes,
                    k_slice_bytes, cudaMemcpyHostToDevice);       if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_key, key_pos, (size_t)n_tokens * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;
    err = cudaMemcpy(d_off, offsets, (size_t)n_offsets * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto cleanup;

    {
        const int block = 256;
        const int grid  = (n_tokens + block - 1) / block;
        const int rblk  = TRIA_REDUCE_BLOCK;
        const size_t smem = (size_t)rblk * sizeof(float);

        for (int kvi = 0; kvi < n_kv_heads; kvi++) {
            for (int g = 0; g < gqa; g++) {
                const int qh = kvi * gqa + g;
                const float * qmr = q_mean_real_dev + q_mean_offset + (size_t)qh * freq_count;
                const float * qmi = q_mean_imag_dev + q_mean_offset + (size_t)qh * freq_count;
                const float * qab = q_abs_mean_dev
                    ? q_abs_mean_dev + q_mean_offset + (size_t)qh * freq_count : nullptr;

                tria_raw_score_kernel<<<dim3(grid), dim3(block), 0, 0>>>(
                    d_k, n_tokens, cur_pos, n_embd_k_gqa, kvi, head_dim, freq_count, rope_neox,
                    d_key, qmr, qmi, qab, omega_dev, d_off, n_offsets, d_raw);
                err = cudaGetLastError();
                if (err != cudaSuccess) goto sync;

                tria_znorm_into_agg_kernel<<<dim3(1), dim3(rblk), smem, 0>>>(
                    d_raw, n_tokens, (g == 0) ? 1 : 0, d_agg);
                err = cudaGetLastError();
                if (err != cudaSuccess) goto sync;
            }

            tria_znorm_lw_global_kernel<<<dim3(1), dim3(rblk), smem, 0>>>(
                d_agg, n_tokens, layer_weight, global_scores_dev);
            err = cudaGetLastError();
            if (err != cudaSuccess) goto sync;
        }
    }

sync:
    {
        const cudaError_t se = cudaDeviceSynchronize();
        if (err == cudaSuccess) err = se;
    }
cleanup:
    tria_cuda_free_ptr(d_k);
    tria_cuda_free_ptr(d_key);
    tria_cuda_free_ptr(d_off);
    tria_cuda_free_ptr(d_raw);
    tria_cuda_free_ptr(d_agg);
    return (int)err;
}
