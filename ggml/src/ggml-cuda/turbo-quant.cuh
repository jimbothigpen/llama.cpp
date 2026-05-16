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

// MSE reduction: 50.1% vs Lloyd-Max 3-bit, +3.02 dB. numpy GLA: n_train=4000, 100 iters, seed=99. Decode: state_t = read_9_bits(qs, t*3)
static __constant__ float d_turboq3_tcq_codebook[512] = {
    -0.24244059f, -0.12586778f, -0.06693592f, -0.02260770f, +0.01492950f, +0.05467265f, +0.10069778f, +0.18883320f,
    -0.19693744f, -0.14152811f, -0.09539399f, -0.06046141f, -0.02731707f, +0.01163860f, +0.05423523f, +0.11278591f,
    -0.11856443f, -0.06727399f, -0.02913110f, +0.00417571f, +0.03549468f, +0.07371171f, +0.11926779f, +0.18401266f,
    -0.25362726f, -0.15759121f, -0.10456934f, -0.06284792f, -0.01789622f, +0.03435958f, +0.08292559f, +0.14658904f,
    -0.16766223f, -0.09932603f, -0.04795861f, -0.00316137f, +0.03350896f, +0.07203513f, +0.12375449f, +0.24558071f,
    -0.21340639f, -0.11273975f, -0.05969454f, -0.02112451f, +0.01584557f, +0.05606037f, +0.09979239f, +0.18008010f,
    -0.14495838f, -0.08746232f, -0.05134764f, -0.02051995f, +0.00527687f, +0.03116450f, +0.06451037f, +0.11952747f,
    -0.20422562f, -0.11092815f, -0.05362599f, -0.00892124f, +0.02997769f, +0.07779223f, +0.13904265f, +0.22305706f,
    -0.17846867f, -0.11931835f, -0.08258449f, -0.04965282f, -0.01742237f, +0.01840734f, +0.06431029f, +0.13633489f,
    -0.16413829f, -0.09193468f, -0.05326458f, -0.01892892f, +0.02155236f, +0.07144441f, +0.12140869f, +0.16414391f,
    -0.16097247f, -0.11795594f, -0.07439681f, -0.03746189f, -0.00202306f, +0.03708065f, +0.07964572f, +0.25785580f,
    -0.21079398f, -0.08501953f, -0.04986830f, -0.02919976f, -0.00307999f, +0.01172143f, +0.04960671f, +0.10403022f,
    -0.13488377f, -0.08804465f, -0.05803939f, -0.02886309f, -0.00121364f, +0.03075502f, +0.07380433f, +0.14234027f,
    -0.17733476f, -0.11930768f, -0.08073044f, -0.05102654f, -0.02174008f, +0.01207697f, +0.05188434f, +0.10528153f,
    -0.27424359f, -0.14814219f, -0.09042648f, -0.04750653f, -0.00688004f, +0.03821837f, +0.08375420f, +0.15444848f,
    -0.17072668f, -0.11573062f, -0.07891619f, -0.04997802f, -0.02360069f, +0.00884181f, +0.04775132f, +0.09702020f,
    -0.13396922f, -0.08187833f, -0.03989934f, -0.00285008f, +0.03193579f, +0.06714261f, +0.10630646f, +0.19974332f,
    -0.10977794f, -0.05588607f, -0.01988312f, +0.00588292f, +0.02463065f, +0.04931722f, +0.08140395f, +0.11857282f,
    -0.11285258f, -0.06842930f, -0.03478571f, +0.00135103f, +0.04282236f, +0.08846240f, +0.14403294f, +0.18865710f,
    -0.16570802f, -0.13114756f, -0.08916780f, -0.01495983f, +0.02156897f, +0.05788230f, +0.10420620f, +0.15807896f,
    -0.09603385f, -0.05330852f, -0.01872682f, +0.01128407f, +0.04181543f, +0.07901397f, +0.12809893f, +0.20628030f,
    -0.12671234f, -0.07713382f, -0.04176560f, -0.01075576f, +0.02093381f, +0.05861618f, +0.10125964f, +0.16253341f,
    -0.12186792f, -0.07046833f, -0.02827389f, +0.00582941f, +0.03785510f, +0.07531738f, +0.13185618f, +0.20784822f,
    -0.11580890f, -0.06750206f, -0.03211596f, -0.00041264f, +0.02880913f, +0.06547855f, +0.11221221f, +0.17096693f,
    -0.20808545f, -0.15288957f, -0.09920800f, -0.05654906f, -0.02077297f, +0.01662349f, +0.06161885f, +0.11496038f,
    -0.25925224f, -0.12740968f, -0.07758909f, -0.03847224f, -0.00659505f, +0.02506258f, +0.05676728f, +0.15852313f,
    -0.20711072f, -0.15256361f, -0.09078260f, -0.04651003f, -0.01428200f, +0.02046691f, +0.06122406f, +0.11168941f,
    -0.29227489f, -0.10113064f, -0.06318919f, -0.04224788f, -0.01237292f, +0.01916771f, +0.05288843f, +0.08860565f,
    -0.18939137f, -0.13610712f, -0.07454449f, -0.03508454f, -0.00070383f, +0.03791586f, +0.07589655f, +0.12249952f,
    -0.23639095f, -0.16088664f, -0.10112434f, -0.05671202f, -0.02441828f, +0.01108337f, +0.04704943f, +0.08991196f,
    -0.18832129f, -0.10863860f, -0.06105676f, -0.02453765f, +0.00571738f, +0.03372482f, +0.06261604f, +0.10699216f,
    -0.22723058f, -0.15697967f, -0.09283338f, -0.04977757f, -0.00658221f, +0.03587560f, +0.07948679f, +0.13286804f,
    -0.10523734f, -0.05894853f, -0.01794997f, +0.01681460f, +0.05235468f, +0.08761997f, +0.12857652f, +0.27355651f,
    -0.16195214f, -0.08037403f, -0.03931148f, -0.00205822f, +0.03885316f, +0.08372425f, +0.14362726f, +0.19498548f,
    -0.11559617f, -0.06571012f, -0.02964597f, -0.00045770f, +0.02920656f, +0.06525015f, +0.11007642f, +0.23265810f,
    -0.12326290f, -0.06516271f, -0.02775460f, +0.00911453f, +0.03682196f, +0.07574877f, +0.13758171f, +0.19163566f,
    -0.09889923f, -0.05620999f, -0.01514455f, +0.01793674f, +0.05562053f, +0.10430135f, +0.16772700f, +0.28700828f,
    -0.14117142f, -0.08234416f, -0.03966629f, -0.00272311f, +0.03102731f, +0.07227346f, +0.13315912f, +0.20565525f,
    -0.09793990f, -0.05264642f, -0.01436317f, +0.01968854f, +0.05324087f, +0.09480734f, +0.16667446f, +0.25740325f,
    -0.14365898f, -0.07946859f, -0.03025317f, +0.01447767f, +0.05407316f, +0.09543498f, +0.14146231f, +0.20799392f,
    -0.16657843f, -0.10643959f, -0.06051657f, -0.02209583f, +0.01260932f, +0.04745538f, +0.09038523f, +0.16133716f,
    -0.21383845f, -0.13881313f, -0.09221762f, -0.05544837f, -0.02178388f, +0.01677356f, +0.05674765f, +0.10728363f,
    -0.17472305f, -0.11292139f, -0.06834519f, -0.03219563f, +0.00094835f, +0.03451309f, +0.07811368f, +0.14950613f,
    -0.21735978f, -0.14172379f, -0.09016410f, -0.05325706f, -0.02099085f, +0.01431495f, +0.05746740f, +0.10986551f,
    -0.16108559f, -0.09852512f, -0.05524211f, -0.01762269f, +0.01394665f, +0.05029779f, +0.09104291f, +0.15619289f,
    -0.18963714f, -0.12396694f, -0.07575205f, -0.03500398f, -0.00238001f, +0.03088680f, +0.06744511f, +0.11232874f,
    -0.15580266f, -0.11168178f, -0.07526547f, -0.04145918f, -0.00974866f, +0.03212880f, +0.07638067f, +0.13532050f,
    -0.18869418f, -0.12704822f, -0.07090112f, -0.03539131f, -0.00940597f, +0.01779585f, +0.05332254f, +0.10070462f,
    -0.10802900f, -0.05559649f, -0.01134203f, +0.02766773f, +0.06135347f, +0.09766156f, +0.13701990f, +0.20283839f,
    -0.15064489f, -0.08763143f, -0.05088234f, -0.01813378f, +0.01489159f, +0.05492927f, +0.10086069f, +0.16056357f,
    -0.10806098f, -0.05308804f, -0.01607634f, +0.01716060f, +0.04692414f, +0.08323829f, +0.12591397f, +0.19397456f,
    -0.14325471f, -0.07795846f, -0.03858727f, -0.01405432f, +0.01490734f, +0.04949452f, +0.09137284f, +0.14710410f,
    -0.08884655f, -0.04006506f, +0.00188640f, +0.03342239f, +0.06599921f, +0.10063411f, +0.13860981f, +0.21032288f,
    -0.12697365f, -0.06556813f, -0.02598349f, +0.00936226f, +0.04159310f, +0.07441500f, +0.11134150f, +0.15887714f,
    -0.24867819f, -0.09192626f, -0.04786854f, -0.01128089f, +0.02160387f, +0.05966909f, +0.10629620f, +0.19008696f,
    -0.10229637f, -0.05171858f, -0.01251010f, +0.01547078f, +0.03557770f, +0.06102134f, +0.09990518f, +0.15643583f,
    -0.22377374f, -0.14769778f, -0.08583978f, -0.04202831f, -0.00770624f, +0.02992121f, +0.07128810f, +0.12562713f,
    -0.11691130f, -0.05988247f, -0.02030350f, +0.01373520f, +0.04911441f, +0.09225391f, +0.15575270f, +0.23753035f,
    -0.18158022f, -0.09763482f, -0.05659763f, -0.02414636f, +0.00537057f, +0.03960274f, +0.07579990f, +0.12346489f,
    -0.26066830f, -0.12511689f, -0.06579913f, -0.00571045f, +0.03891529f, +0.08188405f, +0.12671439f, +0.18494135f,
    -0.15464623f, -0.08975428f, -0.04408587f, -0.01101469f, +0.02199709f, +0.05924839f, +0.10465596f, +0.26287255f,
    -0.19226125f, -0.11309006f, -0.07365324f, -0.03543059f, -0.00178878f, +0.03501295f, +0.07791925f, +0.14937145f,
    -0.12649273f, -0.06018269f, -0.01573098f, +0.02200219f, +0.05903495f, +0.09840808f, +0.13520581f, +0.18245036f,
    -0.16474872f, -0.09278035f, -0.04699890f, -0.00779894f, +0.03187623f, +0.07828258f, +0.13561429f, +0.23917313f
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

// MSE reduction: 33.1% vs Lloyd-Max 2-bit, +1.75 dB. numpy GLA: n_train=4000, 100 iters, 5 restarts. Decode: state_t = read_8_bits(qs, t*2)
static __constant__ float d_turboq2_tcq_codebook[256] = {
    -0.08176727f, -0.00033508f, +0.06850938f, +0.16613583f, -0.14090237f, -0.05715980f, +0.01615283f, +0.11012612f,
    -0.10581727f, -0.04260033f, -0.00423828f, +0.06296677f, -0.17352516f, -0.07213694f, +0.02485547f, +0.10813029f,
    -0.12736021f, -0.06026637f, +0.00177779f, +0.06987048f, -0.08498892f, -0.01943354f, +0.06211906f, +0.01397950f,
    -0.17903381f, -0.01989968f, +0.03569642f, +0.09051796f, -0.09042171f, -0.02577177f, +0.02050355f, +0.10467158f,
    -0.21265116f, -0.11087410f, -0.04349163f, +0.01669601f, -0.12012258f, -0.01521601f, +0.07030928f, +0.13750617f,
    -0.06601736f, -0.04198077f, +0.02279012f, +0.10377382f, -0.07896508f, -0.00657534f, +0.06652649f, +0.17177304f,
    -0.07452555f, -0.00981928f, +0.04254026f, +0.11680857f, -0.12769225f, -0.04400226f, +0.01111500f, +0.08063783f,
    -0.05339707f, +0.01173677f, +0.07039803f, +0.14338760f, -0.12492259f, -0.05478338f, -0.01731757f, +0.04320757f,
    -0.00530445f, -0.15542837f, -0.06801344f, +0.04485723f, -0.07050634f, +0.01234248f, +0.11757696f, +0.22165567f,
    -0.01849510f, +0.04277446f, +0.08655161f, +0.15533215f, -0.10084474f, -0.00810490f, -0.03715962f, +0.04786975f,
    -0.02117090f, +0.04766359f, +0.08838871f, +0.16277327f, -0.24295192f, -0.12420259f, -0.05557786f, +0.12114887f,
    -0.12861997f, -0.06805481f, -0.05590313f, +0.01283404f, -0.01349204f, +0.05466014f, +0.10226475f, +0.19152307f,
    -0.09299547f, -0.02196216f, +0.03284279f, +0.09021873f, -0.07505369f, +0.08066312f, -0.03999974f, +0.04350512f,
    +0.00485651f, +0.05240202f, +0.12679257f, +0.19781399f, -0.18016882f, -0.11454904f, -0.06387294f, +0.01354196f,
    -0.17339253f, -0.10154387f, -0.03942726f, +0.03053090f, -0.01029367f, +0.05617156f, +0.10911176f, +0.18613949f,
    -0.21304886f, -0.11837386f, -0.06452254f, +0.01450099f, -0.03497068f, +0.03907030f, +0.06927501f, +0.13114283f,
    -0.15195946f, -0.06528903f, +0.00816301f, +0.09342197f, -0.00768985f, +0.08454979f, -0.06193831f, +0.04520382f,
    -0.18858465f, -0.12311971f, -0.08049614f, +0.00820490f, -0.03343302f, +0.04559230f, +0.09504822f, +0.16720207f,
    -0.08559455f, -0.00763808f, -0.07567421f, +0.03534968f, -0.03516657f, +0.07333340f, +0.00215530f, +0.06659426f,
    -0.02403073f, +0.04535064f, +0.10581165f, +0.14817812f, -0.16961506f, -0.10086726f, -0.04851092f, +0.02657260f,
    -0.03184498f, +0.03237205f, +0.09189106f, +0.14247570f, -0.18240723f, -0.09515552f, +0.01455373f, +0.24037592f,
    -0.13847726f, -0.10706620f, -0.04225504f, +0.02279146f, -0.02027496f, +0.06288219f, +0.14652734f, +0.24736365f,
    -0.01184501f, +0.06392768f, +0.12518647f, +0.20364036f, -0.06881002f, -0.14446024f, -0.04796625f, +0.02247028f,
    -0.11420977f, -0.03750149f, +0.03140424f, +0.10375965f, -0.15867621f, -0.07792078f, -0.00786463f, +0.07086110f,
    -0.05512634f, +0.01544903f, +0.08794563f, +0.18253894f, -0.12583706f, -0.04047658f, +0.03500937f, +0.12212106f,
    -0.07983117f, -0.02346017f, +0.02269844f, +0.09270003f, -0.14228862f, -0.05948335f, +0.01340374f, +0.08643699f,
    -0.17088441f, -0.08146483f, +0.01637994f, +0.11269872f, -0.12229883f, -0.02740963f, +0.06919862f, +0.17516392f,
    -0.23416011f, -0.08861073f, -0.00531799f, +0.04334467f, -0.07542395f, -0.00959691f, +0.03128058f, +0.11384328f,
    -0.12321154f, -0.05411436f, -0.00802293f, +0.04527715f, -0.02979034f, +0.01261100f, +0.08631871f, +0.14489119f,
    -0.06713610f, -0.01768748f, +0.04439952f, +0.08539781f, -0.10447017f, -0.03861764f, +0.01176727f, +0.08397588f,
    -0.09664737f, -0.03306058f, +0.01965956f, +0.08313737f, -0.15701702f, -0.03552708f, +0.03436711f, +0.12348684f,
    -0.07465987f, +0.03148096f, -0.01592258f, +0.07807118f, -0.08365041f, -0.00777653f, +0.06189138f, +0.16461129f
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
