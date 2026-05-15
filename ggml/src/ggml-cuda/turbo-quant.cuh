/*
 * TurboQuant CUDA kernels for KV cache compression
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBOQ3_0 (3-bit PolarQuant, block size 32)
 * Constants, WHT rotation, quantize/dequantize device functions.
 */

#pragma once

#include "common.cuh"
#include "turbo-innerq.cuh"
#include <cstdlib>
#include <cmath>

// ---- Quantization ratios for dequantize_block template ----
#define QR_TURBOQ3 1  // Each dequantize call produces 2 consecutive elements (like q8_0)
#define QR_TURBOQ4 1  // Each dequantize call produces 2 consecutive elements (like q8_0)

// ---- 2-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_2BIT[4] = {
    -0.133462f, -0.039994f, 0.039994f, 0.133462f
};

static __constant__ float TURBO_MID_2BIT[3] = {
    -0.086728f, 0.0f, 0.086728f
};

// ---- 3-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

// ---- Midpoints for nearest centroid lookup ----

static __constant__ float TURBO_MID_3BIT[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f
};

// ---- WHT sign arrays (seed=42) ----

static __constant__ float TURBO_WHT_SIGNS1[128] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f,
    -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2[128] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f,
    -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f
};

// ---- 64-element WHT sign arrays (first 64 of the 128-element arrays) ----

static __constant__ float TURBO_WHT_SIGNS1_64[64] = {
    -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f,
    1.0f, 1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f
};

static __constant__ float TURBO_WHT_SIGNS2_64[64] = {
    1.0f, 1.0f, 1.0f, 1.0f, -1.0f, 1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, 1.0f, 1.0f, 1.0f,
    1.0f, 1.0f, -1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f,
    1.0f, -1.0f, 1.0f, 1.0f, 1.0f, -1.0f, -1.0f, 1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, -1.0f, 1.0f, 1.0f
};

// ---- Fast Walsh-Hadamard Transform (in-place, normalized) ----
// O(n log n) = 896 ops for n=128

static __device__ __forceinline__ void turbo_fwht_128(float * x) {
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; i++) {
        x[i] *= inv_sqrt_128;
    }
}

// ---- Fast Walsh-Hadamard Transform for 64-element groups ----
// O(n log n) = 384 ops for n=64

static __device__ __forceinline__ void turbo_fwht_64(float * x) {
    for (int h = 1; h < 64; h *= 2) {
        for (int i = 0; i < 64; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j];
                float b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }
    const float inv_sqrt_64 = 0.125f;
    for (int i = 0; i < 64; i++) {
        x[i] *= inv_sqrt_64;
    }
}

// ---- Forward rotation: signs1 → FWHT → signs2 ----

static __device__ __forceinline__ void turbo_rotate_forward(float * x) {
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS1[i];
    turbo_fwht_128(x);
    for (int i = 0; i < 128; i++) x[i] *= TURBO_WHT_SIGNS2[i];
}

// ---- Forward rotation for 64-element groups ----

static __device__ __forceinline__ void turbo_rotate_forward_64(float * x) {
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS1_64[i];
    turbo_fwht_64(x);
    for (int i = 0; i < 64; i++) x[i] *= TURBO_WHT_SIGNS2_64[i];
}

// ---- InnerQ per-channel equalization ----
// Equalizes K channel variances before WHT rotation to reduce quantization error.
// Enabled via TURBO_INNERQ=N env var (N = calibration token count).
// Math: <Q/s, s*K> = <Q, K> preserves dot products.
// INNERQ_MAX_CHANNELS is defined in turbo-innerq.cuh

static __device__ float d_innerq_scale[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_scale_inv[INNERQ_MAX_CHANNELS];
static __device__ float d_innerq_sq_accum[INNERQ_MAX_CHANNELS];
static __device__ int   d_innerq_count;
static __device__ int   d_innerq_active;       // 0 = scales are identity, 1 = scales applied
static __device__ int   d_innerq_calibrating;  // 1 = accumulating K² stats

static int  innerq_enabled       = 0;  // host: 0=off, 1=calibrating, 2=active
static int  innerq_target_tokens = 0;
static float innerq_strength     = 0.5f;
static bool  innerq_initialized  = false;

// Host: read TURBO_INNERQ env, start calibration if enabled
static void turbo_innerq_init(void) {
    if (innerq_initialized) return;
    innerq_initialized = true;

    const char * env = getenv("TURBO_INNERQ");
    if (!env || atoi(env) <= 0) {
        innerq_enabled = 0;
        return;
    }
    innerq_target_tokens = atoi(env);
    innerq_enabled = 1;  // calibrating

    const char * env_str = getenv("TURBO_INNERQ_STRENGTH");
    if (env_str) innerq_strength = atof(env_str);
    if (innerq_strength <= 0.0f || innerq_strength > 1.0f) innerq_strength = 0.5f;

    // Zero accumulators and set calibrating flag on device
    float zeros[INNERQ_MAX_CHANNELS] = {0};
    int zero = 0, one = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_sq_accum, zeros, sizeof(zeros)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_count, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_active, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &one, sizeof(int)));

    GGML_LOG_INFO("%s: InnerQ calibration started (target=%d tokens, strength=%.2f)\n",
                   __func__, innerq_target_tokens, innerq_strength);
}

// Host: finalize calibration — compute scales, upload, activate
static void turbo_innerq_finalize(int group_size) {
    // Read accumulators from device
    float sq_accum[INNERQ_MAX_CHANNELS];
    int count = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(sq_accum, d_innerq_sq_accum, group_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int)));

    if (count <= 0) {
        GGML_LOG_WARN("%s: InnerQ calibration got 0 tokens, disabling\n", __func__);
        innerq_enabled = 0;
        int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
        return;
    }

    // Compute per-channel RMS
    float rms[INNERQ_MAX_CHANNELS];
    float mean_rms = 0.0f;
    float max_ratio = 0.0f, min_ratio = 1e30f;
    for (int i = 0; i < group_size; i++) {
        rms[i] = sqrtf(sq_accum[i] / (float)count);
        mean_rms += rms[i];
    }
    mean_rms /= (float)group_size;

    // Compute scale[i] = (mean_rms / channel_rms[i])^strength, clamp to [0.5, 2.0]
    float scale[INNERQ_MAX_CHANNELS];
    float scale_inv[INNERQ_MAX_CHANNELS];
    for (int i = 0; i < group_size; i++) {
        float ratio = (rms[i] > 1e-10f) ? (mean_rms / rms[i]) : 1.0f;
        float s = powf(ratio, innerq_strength);
        if (s < 0.5f) s = 0.5f;
        if (s > 2.0f) s = 2.0f;
        scale[i] = s;
        scale_inv[i] = 1.0f / s;
        if (ratio > max_ratio) max_ratio = ratio;
        if (ratio < min_ratio) min_ratio = ratio;
    }

    // Auto-skip if max channel ratio < 1.2 (already balanced)
    if (max_ratio < 1.2f && min_ratio > (1.0f / 1.2f)) {
        GGML_LOG_INFO("%s: InnerQ auto-disabled (channels already balanced, max_ratio=%.3f)\n",
                       __func__, max_ratio);
        innerq_enabled = 0;
        int zero = 0;
        CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
        return;
    }

    // Stop calibrating, upload scales, activate
    int zero = 0, one = 1;
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_scale, scale, group_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_scale_inv, scale_inv, group_size * sizeof(float)));
    CUDA_CHECK(cudaDeviceSynchronize());  // ensure scales are visible before activating
    CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_active, &one, sizeof(int)));

    innerq_enabled = 2;  // active

    // Publish scale_inv to shared host state for cross-TU tensor update
    turbo_innerq_publish(scale_inv, group_size);

    GGML_LOG_INFO("%s: InnerQ finalized (%d tokens, max_ratio=%.3f, min_ratio=%.3f)\n",
                   __func__, count, max_ratio, min_ratio);
}

// Host: called before each set_rows kernel launch
static void turbo_innerq_check_finalize(int group_size, int64_t ne00) {
    if (!innerq_initialized) {
        turbo_innerq_init();
    }
    if (innerq_enabled == 0) return;

    // InnerQ only works when each WHT group = one head (group_size == head_dim).
    // For standard models: ne00 = n_heads * head_dim, group_size = head_dim → ne00 % group_size == 0, fine.
    // For non-standard models (head_dim > group_size, e.g. GLM 576 → 64-group):
    //   ne00 = head_dim (single head), group_size = 64, ne00/group_size = 9 groups per head → WRONG.
    // Detect: if ne00 / group_size doesn't divide evenly into standard head counts (1,2,4,8,16,32,64,128),
    // it's likely multi-group-per-head. Simpler check: group_size < 128 means head_dim > 128.
    const bool multi_group_per_head = (group_size < 128);  // 64-group → head_dim > 128, multi-group
    if (multi_group_per_head) {
        if (innerq_enabled == 1) {
            GGML_LOG_WARN("%s: InnerQ disabled (ne00=%lld != group_size=%d, multi-group heads)\n",
                           __func__, (long long)ne00, group_size);
            innerq_enabled = 0;
            int zero = 0;
            CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_calibrating, &zero, sizeof(int)));
        }
        return;
    }

    // Check if calibration is complete
    if (innerq_enabled == 1) {
        int count = 0;
        CUDA_CHECK(cudaMemcpyFromSymbol(&count, d_innerq_count, sizeof(int)));
        if (count >= innerq_target_tokens) {
            turbo_innerq_finalize(group_size);
        }
    }
}

// Host: check if InnerQ is currently active (finalized)
static bool turbo_innerq_is_active(void) {
    return innerq_enabled == 2;
}

// ---- 4-bit centroids (Lloyd-Max for N(0, 1/128)) ----

static __constant__ float TURBO_CENTROIDS_4BIT[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};

// ---- Midpoints for nearest 4-bit centroid lookup ----

static __constant__ float TURBO_MID_4BIT[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.000000f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f
};

// ---- Nearest 4-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_4bit(float val) {
    if      (val < TURBO_MID_4BIT[ 0]) return  0;
    else if (val < TURBO_MID_4BIT[ 1]) return  1;
    else if (val < TURBO_MID_4BIT[ 2]) return  2;
    else if (val < TURBO_MID_4BIT[ 3]) return  3;
    else if (val < TURBO_MID_4BIT[ 4]) return  4;
    else if (val < TURBO_MID_4BIT[ 5]) return  5;
    else if (val < TURBO_MID_4BIT[ 6]) return  6;
    else if (val < TURBO_MID_4BIT[ 7]) return  7;
    else if (val < TURBO_MID_4BIT[ 8]) return  8;
    else if (val < TURBO_MID_4BIT[ 9]) return  9;
    else if (val < TURBO_MID_4BIT[10]) return 10;
    else if (val < TURBO_MID_4BIT[11]) return 11;
    else if (val < TURBO_MID_4BIT[12]) return 12;
    else if (val < TURBO_MID_4BIT[13]) return 13;
    else if (val < TURBO_MID_4BIT[14]) return 14;
    else                               return 15;
}

// ---- Per-block quantize for turbo4 (128 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turboq4_0_block(const float * __restrict__ src,
                                                    block_turboq4_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBOQ4 / 2; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBOQ4; j++) {
        uint8_t idx = turbo_nearest_centroid_4bit(src[j]);
        dst->qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
    }
}

// ---- Inline dequant helper: extract one float from turbo4 block ----

static __device__ __forceinline__ float turboq4_dequant_element(
        const block_turboq4_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 2] >> ((j % 2) * 4)) & 0xF;
    return TURBO_CENTROIDS_4BIT[idx] * norm;
}

// ---- Nearest 3-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_3bit(float val) {
    if      (val < TURBO_MID_3BIT[0]) return 0;
    else if (val < TURBO_MID_3BIT[1]) return 1;
    else if (val < TURBO_MID_3BIT[2]) return 2;
    else if (val < TURBO_MID_3BIT[3]) return 3;
    else if (val < TURBO_MID_3BIT[4]) return 4;
    else if (val < TURBO_MID_3BIT[5]) return 5;
    else if (val < TURBO_MID_3BIT[6]) return 6;
    else                              return 7;
}

// ---- Per-block quantize (32 elements, expects already-rotated input) ----
// Used by set_rows after group-level WHT rotation

static __device__ void quantize_f32_turboq3_0_block(const float * __restrict__ src,
                                                    block_turboq3_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBOQ3 / 4; j++) dst->qs[j] = 0;
    for (int j = 0; j < QK_TURBOQ3 / 8; j++) dst->signs[j] = 0;

    for (int j = 0; j < QK_TURBOQ3; j++) {
        uint8_t idx = turbo_nearest_centroid_3bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst->signs[j / 8] |= (1 << (j % 8));
        }
    }
}

// ---- Inline dequant helper: extract one float from turbo3 block ----

static __device__ __forceinline__ float turboq3_dequant_element(
        const block_turboq3_0 * __restrict__ x, int j, float norm) {
    uint8_t low2 = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    uint8_t hi1  = (x->signs[j / 8] >> (j % 8)) & 0x1;
    uint8_t idx  = low2 | (hi1 << 2);
    return TURBO_CENTROIDS_3BIT[idx] * norm;
}

// ---- Nearest 2-bit centroid index ----

static __device__ __forceinline__ uint8_t turbo_nearest_centroid_2bit(float val) {
    if      (val < TURBO_MID_2BIT[0]) return 0;
    else if (val < TURBO_MID_2BIT[1]) return 1;
    else if (val < TURBO_MID_2BIT[2]) return 2;
    else                              return 3;
}

// ---- Per-block quantize for turboq2 (QK_TURBOQ2 elements, expects already-rotated input) ----

static __device__ void quantize_f32_turboq2_0_block(const float * __restrict__ src,
                                                     block_turboq2_0 * __restrict__ dst) {
    for (int j = 0; j < QK_TURBOQ2 / 4; j++) dst->qs[j] = 0;

    for (int j = 0; j < QK_TURBOQ2; j++) {
        uint8_t idx = turbo_nearest_centroid_2bit(src[j]);
        dst->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// ---- Inline dequant helper: extract one float from turboq2 block ----

static __device__ __forceinline__ float turboq2_dequant_element(
        const block_turboq2_0 * __restrict__ x, int j, float norm) {
    uint8_t idx = (x->qs[j / 4] >> ((j % 4) * 2)) & 0x3;
    return TURBO_CENTROIDS_2BIT[idx] * norm;
}

// ============================================================================
// Weight compression types (WHT3_0, WHT4_0)
// These use N(0,1) centroids (NOT N(0,1/128) like KV cache types)
// and require inverse WHT (RHT) after centroid lookup.
// ============================================================================

#define QR_WHT4_0 1  // dequantize produces 2 consecutive elements
#define QR_WHT3_0 1

// ---- Weight centroids: Lloyd-Max for N(0,1) ----

static __constant__ float TQ4_CENTROIDS_WEIGHT[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f
};

static __constant__ float TQ3_CENTROIDS_WEIGHT[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

// ---- Sign array for weight WHT (golden ratio hash, 32 elements) ----

static __constant__ float TQ_WEIGHT_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f
};

// =====================================================================================
// TCQ KV cache (Trellis-Coded Quantization) — Phase 3a port from buun master TURBO[23]_TCQ
// Codebooks + GET_ROWS dequant helpers. SET_ROWS Viterbi encode kernels live in set-rows.cu.
// =====================================================================================

// === TURBOQ3_TCQ: Trellis-Coded Quantization (right-shift bitshift trellis, k=3, L=9) ===
// GLA-trained free-init TCQ codebook (512 entries) for N(0, 1/sqrt(128)) post-FWHT data
// MSE reduction: 37.6% vs Lloyd-Max 3-bit, +2.05 dB. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turboq3_tcq_codebook[512] = {
    -0.19075318f, -0.12398477f, -0.08053825f, -0.04337945f, -0.02360115f, +0.01870265f, +0.07576828f, +0.15711791f,
    -0.17111190f, -0.12162214f, -0.08470646f, -0.04852028f, -0.01371993f, +0.02535509f, +0.08013468f, +0.14563999f,
    -0.23385642f, -0.13636887f, -0.07996625f, -0.04284568f, -0.01378520f, +0.02527046f, +0.08126875f, +0.19733478f,
    -0.17217710f, -0.12501276f, -0.08301722f, -0.04618388f, -0.01582557f, +0.01849815f, +0.05651660f, +0.11781682f,
    -0.26939890f, -0.11554235f, -0.07074665f, -0.03676226f, -0.01378042f, +0.02288926f, +0.07751006f, +0.21598307f,
    -0.16721224f, -0.12556323f, -0.08082666f, -0.04102167f, -0.01442464f, +0.02706698f, +0.06868703f, +0.12768870f,
    -0.17612142f, -0.12177497f, -0.07355501f, -0.04208433f, -0.01214733f, +0.02949718f, +0.07909346f, +0.15018134f,
    -0.23495452f, -0.12467323f, -0.07873887f, -0.04478245f, -0.01067369f, +0.02844658f, +0.07484870f, +0.14291016f,
    -0.20845117f, -0.12025491f, -0.07898818f, -0.03999034f, -0.00396196f, +0.03149235f, +0.07821322f, +0.14260191f,
    -0.18444445f, -0.11889985f, -0.07379119f, -0.03679606f, -0.00808100f, +0.02833046f, +0.07491008f, +0.13134058f,
    -0.19901366f, -0.12241073f, -0.07129523f, -0.03430970f, -0.00634336f, +0.03164584f, +0.06921050f, +0.12507342f,
    -0.22138300f, -0.11838018f, -0.07095155f, -0.03446699f, -0.00752457f, +0.02620806f, +0.07400409f, +0.15958642f,
    -0.16634685f, -0.10892222f, -0.06854335f, -0.02767931f, -0.00510447f, +0.03830038f, +0.09252869f, +0.13887878f,
    -0.21289924f, -0.11350111f, -0.06690028f, -0.03032817f, -0.00054839f, +0.03241062f, +0.07777942f, +0.14089005f,
    -0.16115880f, -0.11725200f, -0.07240758f, -0.03489496f, -0.00463092f, +0.03327753f, +0.07979671f, +0.13508332f,
    -0.18059183f, -0.11007259f, -0.06711663f, -0.02841142f, +0.00008600f, +0.03609043f, +0.08622773f, +0.18401953f,
    -0.15190504f, -0.10264046f, -0.06591309f, -0.03053302f, +0.00219368f, +0.03783871f, +0.08697283f, +0.17363742f,
    -0.16044058f, -0.10606719f, -0.06668835f, -0.02990519f, +0.00298238f, +0.04131254f, +0.09152508f, +0.16726999f,
    -0.16298678f, -0.10606801f, -0.06302952f, -0.02649282f, +0.00338007f, +0.03691096f, +0.08051851f, +0.19143041f,
    -0.15842708f, -0.10271062f, -0.06741970f, -0.02783111f, +0.00129675f, +0.04058053f, +0.08952771f, +0.12665890f,
    -0.14287122f, -0.10702290f, -0.06360254f, -0.02298262f, +0.00504083f, +0.03929205f, +0.07607899f, +0.17748189f,
    -0.15732529f, -0.10472551f, -0.06157213f, -0.02291222f, +0.00406915f, +0.04300021f, +0.09802638f, +0.19737541f,
    -0.16368793f, -0.10786568f, -0.06302504f, -0.02213908f, +0.00705703f, +0.04387142f, +0.09279074f, +0.17373691f,
    -0.15563499f, -0.09970366f, -0.05740117f, -0.02069011f, +0.00532867f, +0.04516702f, +0.09245405f, +0.15705084f,
    -0.22633528f, -0.11082206f, -0.06271142f, -0.02594333f, +0.00196982f, +0.03854224f, +0.07979941f, +0.13428254f,
    -0.20595677f, -0.10630489f, -0.06029190f, -0.02214403f, -0.00260620f, +0.03775614f, +0.07463138f, +0.13103214f,
    -0.25072671f, -0.10346837f, -0.06094402f, -0.02491104f, +0.00614344f, +0.04080280f, +0.08221361f, +0.13847503f,
    -0.20928229f, -0.10634761f, -0.05699658f, -0.02148475f, -0.00035151f, +0.03748212f, +0.07271124f, +0.12825825f,
    -0.18312579f, -0.09889935f, -0.06073723f, -0.02458788f, +0.00436764f, +0.04666018f, +0.09222218f, +0.14264482f,
    -0.25463980f, -0.10378968f, -0.05824099f, -0.02155519f, +0.00609332f, +0.04016074f, +0.08052604f, +0.13524376f,
    -0.20022215f, -0.09820325f, -0.05344592f, -0.02058924f, +0.00430976f, +0.04488201f, +0.08667631f, +0.14100030f,
    -0.23726417f, -0.10697613f, -0.05615639f, -0.01963419f, +0.00929481f, +0.04763221f, +0.08734125f, +0.14092055f,
    -0.13854847f, -0.08281066f, -0.04378172f, -0.00652702f, +0.02368154f, +0.05515453f, +0.10098024f, +0.21544034f,
    -0.13675106f, -0.08835772f, -0.04778416f, -0.01087520f, +0.01662638f, +0.05679985f, +0.09930499f, +0.25459621f,
    -0.13744516f, -0.07804402f, -0.04053756f, -0.00156069f, +0.01937795f, +0.05717912f, +0.10366104f, +0.19898203f,
    -0.12785788f, -0.08260384f, -0.04168846f, -0.00836940f, +0.02032687f, +0.05140464f, +0.09839836f, +0.17357632f,
    -0.14337727f, -0.07776439f, -0.04075604f, -0.00035689f, +0.02425877f, +0.06102493f, +0.10354523f, +0.26100360f,
    -0.13787537f, -0.08036437f, -0.03951768f, -0.00204148f, +0.02145062f, +0.05740400f, +0.10506784f, +0.19793756f,
    -0.12882150f, -0.07994786f, -0.04003095f, -0.00191794f, +0.02359812f, +0.06184931f, +0.10233122f, +0.23810753f,
    -0.14044366f, -0.07837795f, -0.04160599f, -0.00048596f, +0.02446058f, +0.05855361f, +0.10956655f, +0.22929512f,
    -0.17846599f, -0.09742940f, -0.04639398f, -0.01092025f, +0.02348794f, +0.05447743f, +0.09550074f, +0.15359668f,
    -0.17422996f, -0.08763111f, -0.04266620f, -0.00590155f, +0.02432001f, +0.06166173f, +0.10203922f, +0.15632069f,
    -0.16551951f, -0.09271351f, -0.04697642f, -0.00990860f, +0.02472535f, +0.06128802f, +0.10103604f, +0.14517386f,
    -0.17118861f, -0.08584806f, -0.03829585f, +0.00053346f, +0.02704928f, +0.06109060f, +0.09696287f, +0.15332595f,
    -0.12697297f, -0.08251215f, -0.04329925f, -0.00899454f, +0.02452956f, +0.06064569f, +0.11392346f, +0.18405104f,
    -0.19098167f, -0.09401987f, -0.03961263f, -0.00091159f, +0.02620175f, +0.06351430f, +0.10044691f, +0.14884785f,
    -0.15357839f, -0.08420967f, -0.03983079f, -0.00441110f, +0.02716057f, +0.06522659f, +0.11198404f, +0.16775683f,
    -0.19805412f, -0.09481380f, -0.04197457f, -0.00466698f, +0.02339645f, +0.06436768f, +0.11203527f, +0.16789078f,
    -0.13746277f, -0.08557623f, -0.03912223f, -0.00399355f, +0.03151713f, +0.06573500f, +0.11236197f, +0.18292049f,
    -0.14053986f, -0.08499924f, -0.03501216f, -0.00172963f, +0.02630023f, +0.06582417f, +0.11766521f, +0.19003936f,
    -0.13166662f, -0.07917286f, -0.03360028f, +0.00095822f, +0.02770623f, +0.07172356f, +0.11358009f, +0.18991790f,
    -0.23290175f, -0.08433987f, -0.03867760f, +0.00061902f, +0.03305846f, +0.06233019f, +0.10861871f, +0.15443935f,
    -0.12210833f, -0.06640679f, -0.02985525f, +0.00214670f, +0.02966577f, +0.07318296f, +0.11824244f, +0.21638604f,
    -0.15819124f, -0.08219178f, -0.03493502f, +0.00624893f, +0.03856357f, +0.07096187f, +0.11145671f, +0.15940793f,
    -0.12626326f, -0.07091254f, -0.02856854f, +0.00733897f, +0.03200106f, +0.07230481f, +0.12070683f, +0.21324470f,
    -0.13749853f, -0.07346727f, -0.03025852f, +0.00530487f, +0.03579740f, +0.07030963f, +0.11728036f, +0.17899297f,
    -0.18793107f, -0.07859394f, -0.03031515f, +0.01418602f, +0.04532805f, +0.07363716f, +0.12567619f, +0.19763788f,
    -0.12486269f, -0.07178514f, -0.02911957f, +0.00866743f, +0.03677420f, +0.07358893f, +0.11658713f, +0.16348342f,
    -0.18465906f, -0.08903159f, -0.03331701f, +0.00903627f, +0.04149811f, +0.07646608f, +0.12565799f, +0.22711519f,
    -0.16195340f, -0.07480428f, -0.01911557f, +0.01691384f, +0.03921197f, +0.07628624f, +0.11136164f, +0.16702954f,
    -0.12647923f, -0.07496141f, -0.03331255f, +0.01061243f, +0.04254632f, +0.07620428f, +0.12315008f, +0.25389046f,
    -0.12756266f, -0.07329518f, -0.02324664f, +0.01344221f, +0.04260113f, +0.08009208f, +0.12919118f, +0.18493628f,
    -0.19126476f, -0.07707876f, -0.02340527f, +0.01554000f, +0.04223934f, +0.08060503f, +0.11884624f, +0.16863864f,
    -0.13215958f, -0.06856741f, -0.01997532f, +0.01749025f, +0.04587398f, +0.08523111f, +0.14069217f, +0.23933266f
};

// TCQ GET_ROWS dequantize (for non-FA paths)
#define QR_TURBOQ3_TCQ 2
static __device__ __forceinline__
void dequantize_turboq3_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turboq3_tcq * blk = (const block_turboq3_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs
    {
        const int t = iqs;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.x = d_turboq3_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64 (stride = half block size)
    {
        const int t = iqs + 64;
        const int bit_pos = t * 3;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0x1FF;
        v.y = d_turboq3_tcq_codebook[state] * norm;
    }
}

// =====================================================================================
// TURBOQ2_TCQ: 2-bit Trellis-Coded Quantization (k=2, L=8, 256 states, free initial state)
// =====================================================================================

// GLA-trained free-init 2-bit TCQ codebook (256 entries) for N(0, 1/sqrt(128)) post-FWHT data
// MSE reduction: 4.2% vs Lloyd-Max 2-bit, +0.18 dB. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turboq2_tcq_codebook[256] = {
    -0.17377298f, -0.08762707f, -0.01300744f, +0.10467077f, -0.15621400f, -0.07807468f, -0.00244975f, +0.10971872f,
    -0.17391683f, -0.07507965f, -0.00158784f, +0.08619211f, -0.22927522f, -0.08869821f, -0.01062877f, +0.08292897f,
    -0.14736083f, -0.07461455f, -0.00156869f, +0.08953861f, -0.20510331f, -0.07670021f, +0.00562693f, +0.09755767f,
    -0.17821995f, -0.07306755f, +0.01162439f, +0.12019700f, -0.14980938f, -0.06716545f, +0.01804089f, +0.11784229f,
    -0.17945849f, -0.06972521f, +0.00976605f, +0.11559892f, -0.16441021f, -0.06922967f, +0.00837952f, +0.09737813f,
    -0.15496514f, -0.06655134f, +0.01073252f, +0.09873007f, -0.16154034f, -0.06512384f, +0.01120347f, +0.09844273f,
    -0.16629047f, -0.07160361f, +0.01689301f, +0.10389574f, -0.15270690f, -0.06608909f, +0.01531757f, +0.10876989f,
    -0.15495242f, -0.06025202f, +0.02097986f, +0.12120320f, -0.21677839f, -0.06544403f, +0.01845107f, +0.12382485f,
    -0.16529795f, -0.06390794f, +0.01756180f, +0.10582994f, -0.17867196f, -0.06164099f, +0.02126243f, +0.11631798f,
    -0.14439308f, -0.06022475f, +0.01772231f, +0.11524636f, -0.16398476f, -0.05841067f, +0.02710701f, +0.12722188f,
    -0.14742752f, -0.05213630f, +0.02244631f, +0.10951075f, -0.14269118f, -0.05402560f, +0.02561049f, +0.11615862f,
    -0.14039113f, -0.05273549f, +0.02707237f, +0.13126772f, -0.15737704f, -0.05754378f, +0.02594541f, +0.10646760f,
    -0.14971745f, -0.05049292f, +0.03509529f, +0.13929558f, -0.14467933f, -0.05133092f, +0.03106021f, +0.12962434f,
    -0.16401061f, -0.05091477f, +0.02959540f, +0.11717260f, -0.14241236f, -0.04143231f, +0.04110209f, +0.15503085f,
    -0.14888643f, -0.04547486f, +0.03337607f, +0.12928898f, -0.13315155f, -0.04334711f, +0.03357259f, +0.12295390f,
    -0.13933571f, -0.04168339f, +0.04251146f, +0.14801516f, -0.12695345f, -0.04017735f, +0.03470594f, +0.12149578f,
    -0.13630760f, -0.03725725f, +0.04573099f, +0.14982770f, -0.13279556f, -0.03731158f, +0.03788514f, +0.14134987f,
    -0.14634417f, -0.03906009f, +0.04341434f, +0.13156858f, -0.11998180f, -0.03818642f, +0.04197899f, +0.12642762f,
    -0.15277894f, -0.03935205f, +0.04568923f, +0.16831640f, -0.11562648f, -0.03303958f, +0.04737825f, +0.12890437f,
    -0.13040864f, -0.03364901f, +0.04606153f, +0.14526574f, -0.13061834f, -0.03017139f, +0.05168760f, +0.14875662f,
    -0.12403387f, -0.03103612f, +0.04867485f, +0.12266303f, -0.10907682f, -0.02440896f, +0.05311224f, +0.15778596f,
    -0.11341729f, -0.02520524f, +0.05340497f, +0.15747784f, -0.11050928f, -0.02731021f, +0.05552406f, +0.13477354f,
    -0.11251016f, -0.02502996f, +0.05742991f, +0.15073479f, -0.12924648f, -0.02710250f, +0.05662459f, +0.16618961f,
    -0.12142910f, -0.02062330f, +0.06006443f, +0.14212358f, -0.12225247f, -0.01665350f, +0.05721657f, +0.16113346f,
    -0.10689972f, -0.01877897f, +0.06295932f, +0.15178648f, -0.11211861f, -0.01892951f, +0.06142450f, +0.16882628f,
    -0.09920592f, -0.01426363f, +0.06212827f, +0.15953216f, -0.14424184f, -0.01482532f, +0.06397840f, +0.15215315f,
    -0.10688859f, -0.01768018f, +0.06197682f, +0.13406777f, -0.10552422f, -0.01222899f, +0.06173200f, +0.16649240f,
    -0.11628240f, -0.01624644f, +0.06856942f, +0.16076413f, -0.08317817f, -0.00401934f, +0.07239269f, +0.17973306f,
    -0.09375231f, -0.00648847f, +0.06751947f, +0.18814264f, -0.10010364f, -0.00831303f, +0.07526674f, +0.15066913f,
    -0.11472419f, -0.01041994f, +0.07350467f, +0.16431492f, -0.10648406f, -0.00818389f, +0.07277713f, +0.17116972f,
    -0.10591904f, -0.00222131f, +0.07526167f, +0.15777809f, -0.09636197f, +0.00382409f, +0.07966353f, +0.15233697f,
    -0.09117776f, +0.00184235f, +0.07894982f, +0.21859670f, -0.07993965f, +0.00638250f, +0.09275463f, +0.19285717f
};


// 2-bit TCQ GET_ROWS dequantize
#define QR_TURBOQ2_TCQ 2
static __device__ __forceinline__
void dequantize_turboq2_tcq(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_turboq2_tcq * blk = (const block_turboq2_tcq *)vx + ib;
    const float norm = __half2float(blk->norm);

    // Decode element iqs: read 8-bit state via sliding window
    {
        const int t = iqs;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.x = d_turboq2_tcq_codebook[state] * norm;
    }
    // Decode element iqs + 64
    {
        const int t = iqs + 64;
        const int bit_pos = t * 2;
        const int byte_idx = bit_pos / 8;
        const int bit_off = bit_pos % 8;
        const uint16_t raw = (uint16_t)blk->qs[byte_idx] | ((uint16_t)blk->qs[byte_idx + 1] << 8);
        const int state = (raw >> bit_off) & 0xFF;
        v.y = d_turboq2_tcq_codebook[state] * norm;
    }
}
