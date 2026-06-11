#include "triattention-hip.h"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

/* ── compact kernel (gather retained rows, shift down in place) ──────── */

static __global__ void tria_hip_gather_rows_kernel(
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

static inline void tria_hip_free_ptr(void * ptr) {
    if (ptr) {
        (void) hipFree(ptr);
    }
}

extern "C" int tria_hip_compact_rows(
        void * tensor_data,
        const uint32_t * h_indices,
        uint32_t n_keep,
        uint32_t first_move,
        uint32_t row_bytes) {
    if (tensor_data == nullptr || h_indices == nullptr) {
        return static_cast<int>(hipErrorInvalidValue);
    }
    if (first_move >= n_keep || row_bytes == 0) {
        return 0;
    }

    const uint32_t n_move = n_keep - first_move;
    const size_t indices_bytes = static_cast<size_t>(n_move) * sizeof(uint32_t);
    const size_t scratch_bytes = static_cast<size_t>(n_move) * row_bytes;

    uint32_t * d_indices = nullptr;
    uint8_t * d_scratch = nullptr;

    hipError_t err = hipMalloc(reinterpret_cast<void **>(&d_indices), indices_bytes);
    if (err != hipSuccess) {
        return static_cast<int>(err);
    }
    err = hipMalloc(reinterpret_cast<void **>(&d_scratch), scratch_bytes);
    if (err != hipSuccess) {
        (void) hipFree(d_indices);
        return static_cast<int>(err);
    }
    err = hipMemcpy(d_indices, h_indices + first_move, indices_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        (void) hipFree(d_scratch);
        (void) hipFree(d_indices);
        return static_cast<int>(err);
    }

    constexpr uint32_t block_size = 256;
    const size_t total_bytes = scratch_bytes;
    const uint32_t n_blocks = static_cast<uint32_t>((total_bytes + block_size - 1) / block_size);

    hipLaunchKernelGGL(
            tria_hip_gather_rows_kernel,
            dim3(n_blocks), dim3(block_size), 0, 0,
            static_cast<const uint8_t *>(tensor_data),
            d_indices, d_scratch,
            static_cast<size_t>(row_bytes), n_move);

    err = hipGetLastError();
    if (err == hipSuccess) {
        err = hipMemcpy(
                static_cast<uint8_t *>(tensor_data) + static_cast<size_t>(first_move) * row_bytes,
                d_scratch, scratch_bytes, hipMemcpyDeviceToDevice);
    }
    if (err == hipSuccess) {
        err = hipDeviceSynchronize();
    }

    (void) hipFree(d_scratch);
    (void) hipFree(d_indices);
    return static_cast<int>(err);
}

/* ── GPU scoring for q8_0 K cache (Phase C, GQA-aware) ───────────────── */

#define TRIA_QK8_0 32
#define TRIA_Q8_0_BLOCK_SIZE (2 + TRIA_QK8_0)  /* fp16 scale + 32 int8 */
#define TRIA_MAX_HD 256                         /* max supported head_dim (large-head: hd=256) */
#define TRIA_MAX_FC (TRIA_MAX_HD / 2)           /* head_dim <= 256 => fc <= 128 */
#define TRIA_REDUCE_BLOCK 256

static __device__ __forceinline__ float tria_fp16_to_fp32(uint16_t d_bits) {
    const uint32_t sign = (d_bits >> 15) & 1u;
    const uint32_t exp  = (d_bits >> 10) & 0x1fu;
    const uint32_t mant = d_bits & 0x3ffu;
    float d;
    if (exp == 0) {
        d = (sign ? -1.f : 1.f) * ldexpf((float)mant, -24);
    } else if (exp == 31) {
        d = sign ? -__builtin_huge_valf() : __builtin_huge_valf();
    } else {
        uint32_t f32bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
        memcpy(&d, &f32bits, 4);
    }
    return d;
}

/*
 * Pass 1 — raw per-token score for one query head (eq 6-11, max_beta=0).
 * out[s] = mean_offsets( sum_f rel_r*cos(omega*delta) - rel_i*sin(omega*delta) ) + extra
 *   rel_r = qmr*kr + qmi*ki ; rel_i = qmi*kr - qmr*ki ; delta = cur_pos - pos + offset
 *   extra = sum_f max(0, q_abs_mean - |q_mean|) * |k_f|   (skipped if q_abs_mean == NULL)
 * One thread per token. K head slice is shared across the KV head's query heads (GQA).
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
    const int n_head_blocks = head_dim / TRIA_QK8_0;

    float k_f32[TRIA_MAX_HD];
    for (int b = 0; b < n_head_blocks; b++) {
        const uint8_t * blk = row + (head_block_start + b) * TRIA_Q8_0_BLOCK_SIZE;
        uint16_t d_bits;
        memcpy(&d_bits, blk, 2);
        const float d = tria_fp16_to_fp32(d_bits);
        const int8_t * qs = (const int8_t *)(blk + 2);
        for (int j = 0; j < TRIA_QK8_0; j++) {
            k_f32[b * TRIA_QK8_0 + j] = d * (float)qs[j];
        }
    }

    float rel_r[TRIA_MAX_FC];
    float rel_i[TRIA_MAX_FC];
    float extra = 0.f;
    for (int f = 0; f < freq_count; f++) {
        float kr, ki;
        if (rope_neox) { kr = k_f32[f];        ki = k_f32[freq_count + f]; }
        else           { kr = k_f32[2 * f];    ki = k_f32[2 * f + 1];      }
        const float qr = q_mean_real[f];
        const float qi = q_mean_imag[f];
        rel_r[f] = qr * kr + qi * ki;
        rel_i[f] = qi * kr - qr * ki;
        if (q_abs_mean) {
            const float qma = sqrtf(qr * qr + qi * qi);
            const float residual = q_abs_mean[f] - qma;
            if (residual > 0.f) {
                extra += residual * sqrtf(kr * kr + ki * ki);
            }
        }
    }

    const int pos = key_pos[s];
    float trig_sum = 0.f;
    for (int oi = 0; oi < n_offsets; oi++) {
        const float delta = (float)(cur_pos - pos + offsets[oi]);
        float trig = 0.f;
        for (int f = 0; f < freq_count; f++) {
            float c, sn;
            sincosf(omega[f] * delta, &sn, &c);
            trig += rel_r[f] * c - rel_i[f] * sn;
        }
        trig_sum += trig;
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

extern "C" int tria_hip_stats_upload(
        const float * omega, int freq_count,
        const float * q_mean_real, const float * q_mean_imag,
        int n_kv_heads,
        float ** omega_dev_out,
        float ** q_mean_real_dev_out,
        float ** q_mean_imag_dev_out) {
    hipError_t err;
    const size_t omega_bytes = (size_t)freq_count * sizeof(float);
    const size_t qm_bytes    = (size_t)n_kv_heads * freq_count * sizeof(float);

    if (omega_dev_out && omega && freq_count > 0) {
        *omega_dev_out = nullptr;
        err = hipMalloc((void**)omega_dev_out, omega_bytes);
        if (err != hipSuccess) return (int)err;
        err = hipMemcpy(*omega_dev_out, omega, omega_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            tria_hip_free_ptr(*omega_dev_out);
            *omega_dev_out = nullptr;
            return (int)err;
        }
    }

    if (q_mean_real_dev_out && q_mean_real && qm_bytes > 0) {
        *q_mean_real_dev_out = nullptr;
        err = hipMalloc((void**)q_mean_real_dev_out, qm_bytes);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_hip_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            return (int)err;
        }
        err = hipMemcpy(*q_mean_real_dev_out, q_mean_real, qm_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_hip_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            tria_hip_free_ptr(*q_mean_real_dev_out);
            *q_mean_real_dev_out = nullptr;
            return (int)err;
        }
    }

    if (q_mean_imag_dev_out && q_mean_imag && qm_bytes > 0) {
        *q_mean_imag_dev_out = nullptr;
        err = hipMalloc((void**)q_mean_imag_dev_out, qm_bytes);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_hip_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            if (q_mean_real_dev_out && *q_mean_real_dev_out) { tria_hip_free_ptr(*q_mean_real_dev_out); *q_mean_real_dev_out = nullptr; }
            return (int)err;
        }
        err = hipMemcpy(*q_mean_imag_dev_out, q_mean_imag, qm_bytes, hipMemcpyHostToDevice);
        if (err != hipSuccess) {
            if (omega_dev_out && *omega_dev_out) { tria_hip_free_ptr(*omega_dev_out); *omega_dev_out = nullptr; }
            if (q_mean_real_dev_out && *q_mean_real_dev_out) { tria_hip_free_ptr(*q_mean_real_dev_out); *q_mean_real_dev_out = nullptr; }
            tria_hip_free_ptr(*q_mean_imag_dev_out);
            *q_mean_imag_dev_out = nullptr;
            return (int)err;
        }
    }

    return 0;
}

extern "C" void tria_hip_stats_free(float * omega_dev, float * q_mean_real_dev, float * q_mean_imag_dev) {
    tria_hip_free_ptr(omega_dev);
    tria_hip_free_ptr(q_mean_real_dev);
    tria_hip_free_ptr(q_mean_imag_dev);
}

extern "C" int tria_hip_scores_download(float * scores, const float * scores_dev, int n_scores) {
    if (!scores || !scores_dev || n_scores <= 0) return 0;
    return (int)hipMemcpy(scores, scores_dev, (size_t)n_scores * sizeof(float), hipMemcpyDeviceToHost);
}

extern "C" int tria_hip_score_q8_0(
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
    if (n_heads < n_kv_heads || (n_heads % n_kv_heads) != 0) return (int)hipErrorInvalidValue;
    if (freq_count > TRIA_MAX_FC || head_dim > TRIA_MAX_HD || (head_dim % TRIA_QK8_0) != 0) {
        return (int)hipErrorInvalidValue;
    }
    const int gqa = n_heads / n_kv_heads;
    const int row_bytes = (n_embd_k_gqa / TRIA_QK8_0) * TRIA_Q8_0_BLOCK_SIZE;
    const size_t k_slice_bytes = (size_t)n_tokens * row_bytes;

    uint8_t * d_k   = nullptr;
    int     * d_key = nullptr;
    int     * d_off = nullptr;
    float   * d_raw = nullptr;
    float   * d_agg = nullptr;
    hipError_t err = hipSuccess;

    err = hipMalloc((void**)&d_k,   k_slice_bytes);              if (err != hipSuccess) goto cleanup;
    err = hipMalloc((void**)&d_key, (size_t)n_tokens * sizeof(int));   if (err != hipSuccess) goto cleanup;
    err = hipMalloc((void**)&d_off, (size_t)n_offsets * sizeof(int));  if (err != hipSuccess) goto cleanup;
    err = hipMalloc((void**)&d_raw, (size_t)n_tokens * sizeof(float)); if (err != hipSuccess) goto cleanup;
    err = hipMalloc((void**)&d_agg, (size_t)n_tokens * sizeof(float)); if (err != hipSuccess) goto cleanup;

    err = hipMemcpy(d_k, (const uint8_t *)k_data_host + (size_t)score_start * row_bytes,
                    k_slice_bytes, hipMemcpyHostToDevice);       if (err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_key, key_pos, (size_t)n_tokens * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) goto cleanup;
    err = hipMemcpy(d_off, offsets, (size_t)n_offsets * sizeof(int), hipMemcpyHostToDevice);
    if (err != hipSuccess) goto cleanup;

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

                hipLaunchKernelGGL(tria_raw_score_kernel, dim3(grid), dim3(block), 0, 0,
                    d_k, n_tokens, cur_pos, n_embd_k_gqa, kvi, head_dim, freq_count, rope_neox,
                    d_key, qmr, qmi, qab, omega_dev, d_off, n_offsets, d_raw);
                err = hipGetLastError();
                if (err != hipSuccess) goto sync;

                hipLaunchKernelGGL(tria_znorm_into_agg_kernel, dim3(1), dim3(rblk), smem, 0,
                    d_raw, n_tokens, (g == 0) ? 1 : 0, d_agg);
                err = hipGetLastError();
                if (err != hipSuccess) goto sync;
            }

            hipLaunchKernelGGL(tria_znorm_lw_global_kernel, dim3(1), dim3(rblk), smem, 0,
                d_agg, n_tokens, layer_weight, global_scores_dev);
            err = hipGetLastError();
            if (err != hipSuccess) goto sync;
        }
    }

sync:
    {
        const hipError_t se = hipDeviceSynchronize();
        if (err == hipSuccess) err = se;
    }
cleanup:
    tria_hip_free_ptr(d_k);
    tria_hip_free_ptr(d_key);
    tria_hip_free_ptr(d_off);
    tria_hip_free_ptr(d_raw);
    tria_hip_free_ptr(d_agg);
    return (int)err;
}
