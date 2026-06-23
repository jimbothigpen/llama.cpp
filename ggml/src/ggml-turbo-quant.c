/*
 * TurboQuant: KV cache compression via PolarQuant + QJL
 * Based on: arXiv 2504.19874 (ICLR 2026)
 *
 * Implements GGML_TYPE_TURBOQ3_0 (3-bit) and GGML_TYPE_TURBOQ4_0 (4-bit)
 * for use as --cache-type-k turboq3 --cache-type-v turboq3 in llama-server.
 */

#include "ggml-quants.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <assert.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Global: WHT group size for CPU quantize path (set by CPU SET_ROWS handler) */
GGML_API int turboq3_cpu_wht_group_size = 0;

/* ---------- constants ---------- */

#define TURBO_SEED_ROTATION 42
#define TURBO_SEED_QJL      1042
#define TURBO_D             128  /* rotation group size = head_dim (independent of block size) */
#define TURBO_QJL_CONST     1.2533141373155003f  /* sqrt(pi/2) */

/* Optimal centroids from paper (scaled by 1/sqrt(d)) */
/* 2-bit: {±0.453, ±1.51} / sqrt(d) */
static const float CENTROIDS_2BIT[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };

/* 3-bit: Lloyd-Max for N(0, 1/128), pre-computed */
static const float CENTROIDS_3BIT[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};

/* ---------- rotation matrix (lazy init) ---------- */

static float turbo_rotation[TURBO_D * TURBO_D];
static float turbo_rotation_t[TURBO_D * TURBO_D]; /* transpose */
static int   turbo_rotation_initialized = 0;

/* Simple LCG PRNG for deterministic rotation generation */
static uint64_t turbo_prng_state;

static void turbo_prng_seed(uint64_t seed) {
    turbo_prng_state = seed;
}

static double turbo_prng_normal(void) {
    /* Box-Muller transform from uniform LCG */
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u1 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-15) u1 = 1e-15;
    turbo_prng_state = turbo_prng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    double u2 = (double)(turbo_prng_state >> 11) / (double)(1ULL << 53);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void turbo_init_rotation(void) {
    if (turbo_rotation_initialized) return;

    const int d = TURBO_D;

    /* Generate random Gaussian matrix */
    turbo_prng_seed(TURBO_SEED_ROTATION);
    float G[TURBO_D * TURBO_D];
    for (int i = 0; i < d * d; i++) {
        G[i] = (float)turbo_prng_normal();
    }

    /* QR decomposition via modified Gram-Schmidt */
    /* Q stored column-major in turbo_rotation */
    memcpy(turbo_rotation, G, d * d * sizeof(float));

    for (int j = 0; j < d; j++) {
        /* Normalize column j */
        float norm = 0.0f;
        for (int i = 0; i < d; i++) {
            norm += turbo_rotation[i * d + j] * turbo_rotation[i * d + j];
        }
        norm = sqrtf(norm);
        if (norm > 1e-10f) {
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + j] /= norm;
            }
        }

        /* Orthogonalize remaining columns against j */
        for (int k = j + 1; k < d; k++) {
            float dot = 0.0f;
            for (int i = 0; i < d; i++) {
                dot += turbo_rotation[i * d + j] * turbo_rotation[i * d + k];
            }
            for (int i = 0; i < d; i++) {
                turbo_rotation[i * d + k] -= dot * turbo_rotation[i * d + j];
            }
        }
    }

    /* Compute transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_rotation_t[i * d + j] = turbo_rotation[j * d + i];
        }
    }

    turbo_rotation_initialized = 1;
}

/* ---------- QJL projection matrix (lazy init, seed-based) ---------- */

static float turbo_qjl_matrix[TURBO_D * TURBO_D];
static float turbo_qjl_matrix_t[TURBO_D * TURBO_D];
static int   turbo_qjl_initialized = 0;

static void turbo_init_qjl(void) {
    if (turbo_qjl_initialized) return;

    const int d = TURBO_D;
    turbo_prng_seed(TURBO_SEED_QJL);

    for (int i = 0; i < d * d; i++) {
        turbo_qjl_matrix[i] = (float)turbo_prng_normal();
    }

    /* Transpose */
    for (int i = 0; i < d; i++) {
        for (int j = 0; j < d; j++) {
            turbo_qjl_matrix_t[i * d + j] = turbo_qjl_matrix[j * d + i];
        }
    }

    turbo_qjl_initialized = 1;
}

/* ---------- helper: matrix-vector multiply ---------- */

static void matvec(const float * M, const float * x, float * y, int d) {
    /* y = M @ x, M is row-major d×d */
    for (int i = 0; i < d; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            sum += M[i * d + j] * x[j];
        }
        y[i] = sum;
    }
}

/* ---------- nearest centroid ---------- */

static int nearest_centroid_2bit(float val) {
    /* Binary search on midpoints: {-0.133, -0.040, 0.040, 0.133} */
    if (val < -0.086728f) return 0;       /* midpoint(-0.133, -0.040) */
    if (val <  0.000000f) return 1;       /* midpoint(-0.040, 0.040) */
    if (val <  0.086728f) return 2;       /* midpoint(0.040, 0.133) */
    return 3;
}

static int nearest_centroid_3bit(float val) {
    /* 8 centroids, find nearest via midpoints */
    if (val < -0.154259f) return 0;
    if (val < -0.091775f) return 1;
    if (val < -0.043589f) return 2;
    if (val <  0.000000f) return 3;
    if (val <  0.043589f) return 4;
    if (val <  0.091775f) return 5;
    if (val <  0.154259f) return 6;
    return 7;
}

static int nearest_centroid_4bit(float val) {
    /* 16 centroids, optimal for N(0, 1/sqrt(128)), find nearest via midpoints */
    if (val < -0.145560f) return 0;
    if (val < -0.103361f) return 1;
    if (val < -0.079142f) return 2;
    if (val < -0.060009f) return 3;
    if (val < -0.043430f) return 4;
    if (val < -0.028293f) return 5;
    if (val < -0.013963f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.013963f) return 8;
    if (val <  0.028293f) return 9;
    if (val <  0.043430f) return 10;
    if (val <  0.060009f) return 11;
    if (val <  0.079142f) return 12;
    if (val <  0.103361f) return 13;
    if (val <  0.145560f) return 14;
    return 15;
}

/* ---------- WHT sign arrays (must match CUDA/Metal, seed=42) ---------- */

static const float turbo_cpu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};

static const float turbo_cpu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};

/* ---------- CPU forward WHT (in-place, group_size elements) ---------- */

static void turbo_cpu_fwht(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs1
    for (int i = 0; i < group_size; i++) x[i] *= s1[i];

    // butterfly stages
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs2
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s2[i];
}

/* ---------- CPU inverse WHT (in-place, group_size elements) ----------
 *
 * Forward is  y = D(s2) * N * H * D(s1) * x   (N = 1/sqrt(group_size))
 * H is the unnormalized Hadamard butterfly with H*H = group_size * I, so
 * (N*H) is self-inverse.  s1 and s2 are ±1 diagonals, also self-inverse.
 * The inverse therefore has the same structure with s1 and s2 swapped:
 *     x = D(s1) * N * H * D(s2) * y
 */
GGML_API void turbo_cpu_fwht_inverse(float * x, int group_size) {
    const float * s1 = turbo_cpu_s1;
    const float * s2 = turbo_cpu_s2;
    const float inv_sqrt = (group_size == 128) ? 0.08838834764831845f : 0.125f;

    // signs2 (undoes the s2 that was applied last in the forward pass)
    for (int i = 0; i < group_size; i++) x[i] *= s2[i];

    // butterfly stages (same as forward — self-inverse up to the inv_sqrt scaling below)
    for (int h = 1; h < group_size; h *= 2) {
        for (int i = 0; i < group_size; i += h * 2) {
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
        }
    }

    // normalize + signs1
    for (int i = 0; i < group_size; i++) x[i] *= inv_sqrt * s1[i];
}

/* ---------- TURBOQ2_0: 2-bit PolarQuant with WHT rotation (no QJL) ---------- */

void quantize_row_turboq2_0_ref(const float * GGML_RESTRICT x, block_turboq2_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBOQ2 == 0);

    extern int turboq3_cpu_wht_group_size;
    int group_size = turboq3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBOQ2;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turboq2_0 * grp_dst = y + g * blocks_per_group;

        /* 1. L2 norm over the group */
        float norm_sq = 0.0f;
        float buf[128];
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        /* 2. Normalize */
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        /* 3. Forward WHT rotation */
        turbo_cpu_fwht(buf, group_size);

        /* 4. Quantize + pack into sub-blocks */
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turboq2_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBOQ2;

            memset(blk->qs, 0, QK_TURBOQ2 / 4);

            for (int j = 0; j < QK_TURBOQ2; j++) {
                int idx = nearest_centroid_2bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                recon_sq += CENTROIDS_2BIT[idx] * CENTROIDS_2BIT[idx];
            }
        }

        /* 5. Corrected norm */
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turboq2_0(const block_turboq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBOQ2 == 0);
    const int nb = k / QK_TURBOQ2;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBOQ2; j++) {
            uint8_t idx = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            y[block * QK_TURBOQ2 + j] = CENTROIDS_2BIT[idx] * norm;
        }
    }
}

size_t quantize_turboq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ2 == 0);

    size_t row_size = (n_per_row / QK_TURBOQ2) * sizeof(block_turboq2_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turboq2_0_ref(
            src + row * n_per_row,
            (block_turboq2_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBOQ3_0: 3-bit PolarQuant with WHT rotation ---------- */

void quantize_row_turboq3_0_ref(const float * GGML_RESTRICT x, block_turboq3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBOQ3 == 0);

    // Read WHT group size from global (set by CPU SET_ROWS handler before each call).
    // Fallback: 128 if row is 128-aligned, else 64.
    extern int turboq3_cpu_wht_group_size;
    int group_size = turboq3_cpu_wht_group_size;
    if (group_size != 64 && group_size != 128) {
        group_size = (k % 128 == 0) ? 128 : 64;
    }
    if (k % group_size != 0) group_size = (group_size == 128) ? 64 : 128;
    assert(k % group_size == 0);

    const int n_groups = k / group_size;
    const int blocks_per_group = group_size / QK_TURBOQ3;

    for (int g = 0; g < n_groups; g++) {
        const float * grp_src = x + g * group_size;
        block_turboq3_0 * grp_dst = y + g * blocks_per_group;

        // 1. L2 norm over the group
        float norm_sq = 0.0f;
        float buf[128];  // max group_size
        for (int j = 0; j < group_size; j++) {
            buf[j] = grp_src[j];
            norm_sq += buf[j] * buf[j];
        }
        float grp_norm = sqrtf(norm_sq);
        float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

        // 2. Normalize
        for (int j = 0; j < group_size; j++) buf[j] *= inv_norm;

        // 3. Forward WHT rotation
        turbo_cpu_fwht(buf, group_size);

        // 4. Quantize + pack into sub-blocks
        float recon_sq = 0.0f;
        for (int b = 0; b < blocks_per_group; b++) {
            block_turboq3_0 * blk = &grp_dst[b];
            const int off = b * QK_TURBOQ3;

            memset(blk->qs, 0, QK_TURBOQ3 / 4);
            memset(blk->signs, 0, QK_TURBOQ3 / 8);

            for (int j = 0; j < QK_TURBOQ3; j++) {
                int idx = nearest_centroid_3bit(buf[off + j]);
                blk->qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) {
                    blk->signs[j / 8] |= (1 << (j % 8));
                }
                recon_sq += CENTROIDS_3BIT[idx] * CENTROIDS_3BIT[idx];
            }
        }

        // 5. Corrected norm: grp_norm / recon_norm (matching CUDA kernel)
        float recon_norm = sqrtf(recon_sq);
        float corrected = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            grp_dst[b].norm = GGML_FP32_TO_FP16(corrected);
        }
    }
}

void dequantize_row_turboq3_0(const block_turboq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    // Stub — Metal shader handles dequant on GPU.
    assert(k % QK_TURBOQ3 == 0);
    const int nb = k / QK_TURBOQ3;
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        for (int j = 0; j < QK_TURBOQ3; j++) {
            uint8_t low2 = (x[block].qs[j/4] >> ((j%4)*2)) & 0x3;
            uint8_t hi1 = (x[block].signs[j/8] >> (j%8)) & 0x1;
            uint8_t idx = low2 | (hi1 << 2);
            y[block * QK_TURBOQ3 + j] = CENTROIDS_3BIT[idx] * norm;
        }
    }
}

size_t quantize_turboq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ3 == 0);

    size_t row_size = (n_per_row / QK_TURBOQ3) * sizeof(block_turboq3_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turboq3_0_ref(
            src + row * n_per_row,
            (block_turboq3_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBOQ3_TCQ: Trellis-Coded Quantization (Viterbi on GPU; CPU is stub) ---------- */

void quantize_row_turboq3_tcq_ref(const float * GGML_RESTRICT x, block_turboq3_tcq * GGML_RESTRICT y, int64_t k) {
    /* Stub — CUDA kernel handles TCQ quantize (Viterbi). CPU path records norm only. */
    assert(k % QK_TURBOQ3_TCQ == 0);
    const int nb = k / QK_TURBOQ3_TCQ;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBOQ3_TCQ; j++) norm += x[i*QK_TURBOQ3_TCQ + j] * x[i*QK_TURBOQ3_TCQ + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, 49);
        y[i].pad = 0;
    }
}

void dequantize_row_turboq3_tcq(const block_turboq3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    /* CPU dequant stub — placeholder (no codebook on CPU yet) */
    GGML_UNUSED(x);
    assert(k % QK_TURBOQ3_TCQ == 0);
    const int nb = k / QK_TURBOQ3_TCQ;
    for (int block = 0; block < nb; block++) {
        for (int j = 0; j < QK_TURBOQ3_TCQ; j++) {
            y[block * QK_TURBOQ3_TCQ + j] = 0.0f;
        }
    }
}

size_t quantize_turboq3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ3_TCQ == 0);

    size_t row_size = (n_per_row / QK_TURBOQ3_TCQ) * sizeof(block_turboq3_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turboq3_tcq_ref(
            src + row * n_per_row,
            (block_turboq3_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBOQ2_TCQ: 2-bit Trellis-Coded Quantization (Viterbi on GPU; CPU is stub) ---------- */

void quantize_row_turboq2_tcq_ref(const float * GGML_RESTRICT x, block_turboq2_tcq * GGML_RESTRICT y, int64_t k) {
    /* Stub — CUDA kernel handles TCQ quantize (Viterbi). CPU path records norm only. */
    assert(k % QK_TURBOQ2_TCQ == 0);
    const int nb = k / QK_TURBOQ2_TCQ;
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_TURBOQ2_TCQ; j++) norm += x[i*QK_TURBOQ2_TCQ + j] * x[i*QK_TURBOQ2_TCQ + j];
        y[i].norm = GGML_FP32_TO_FP16(sqrtf(norm));
        memset(y[i].qs, 0, 33);
        y[i].pad = 0;
    }
}

void dequantize_row_turboq2_tcq(const block_turboq2_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    /* CPU dequant stub — placeholder (no codebook on CPU yet) */
    GGML_UNUSED(x);
    assert(k % QK_TURBOQ2_TCQ == 0);
    const int nb = k / QK_TURBOQ2_TCQ;
    for (int block = 0; block < nb; block++) {
        for (int j = 0; j < QK_TURBOQ2_TCQ; j++) {
            y[block * QK_TURBOQ2_TCQ + j] = 0.0f;
        }
    }
}

size_t quantize_turboq2_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                            int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ2_TCQ == 0);

    size_t row_size = (n_per_row / QK_TURBOQ2_TCQ) * sizeof(block_turboq2_tcq);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turboq2_tcq_ref(
            src + row * n_per_row,
            (block_turboq2_tcq *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- KV_OSCAR_INT2: OScaR 2-bit KV (FHT + min-max uniform INT2) — CPU stub ---------- */

void quantize_row_kv_oscar_int2_ref(const float * GGML_RESTRICT x, block_kv_oscar_int2 * GGML_RESTRICT y, int64_t k) {
    // Scalar min-max INT2 quantize (WHT already applied externally; no WHT here).
    // Used as CPU fallback for ggml_cpy during K-shift re-encode.
    assert(k % QK_OSCAR_INT2 == 0);
    const int nb = k / QK_OSCAR_INT2;
    for (int i = 0; i < nb; i++) {
        const float * blk = x + i * QK_OSCAR_INT2;
        float mn = blk[0], mx = blk[0];
        for (int j = 1; j < QK_OSCAR_INT2; j++) {
            if (blk[j] < mn) mn = blk[j];
            if (blk[j] > mx) mx = blk[j];
        }
        const float range = mx - mn;
        const float d = (range > 1e-10f) ? range / 3.0f : 1.0f;
        const float id = 1.0f / d;
        y[i].d = GGML_FP32_TO_FP16(d);
        y[i].m = GGML_FP32_TO_FP16(mn);
        memset(y[i].qs, 0, QK_OSCAR_INT2 / 4);
        for (int j = 0; j < QK_OSCAR_INT2; j++) {
            int q = (int)((blk[j] - mn) * id + 0.5f);
            q = q < 0 ? 0 : (q > 3 ? 3 : q);
            y[i].qs[j / 4] |= (uint8_t)(q << (2 * (j % 4)));
        }
    }
}

// SET_ROWS encode for fresh-token K writes: full-dim Walsh-Hadamard rotation + per-128-subblock
// uniform min-max INT2. This is the CPU/Vulkan-portable mirror of the CUDA k_set_rows_oscar_int2
// kernel (set-rows.cu); the matching D-pt WHT is applied to Q at decode by the flash-attention
// vec_dot (ggml-cpu: ggml_vec_dot_kv_oscar_int2_f32). K is stored in the WHT-rotated domain.
//   wht_group D = 256 when the combined-GQA row width is a multiple of 256, else 128 — this
//   matches the CUDA dispatch heuristic exactly (set_rows_cuda_oscar_int2: ne00 % 256 == 0).
//   H_D is normalized by 1/sqrt(D) so it is orthonormal (H_D^T H_D = I), preserving the q·k dot.
// NOTE: distinct from quantize_row_kv_oscar_int2_ref above, which is the plain (no-WHT) re-encode
// used by the K-shift ggml_cpy path (whose input is already WHT-rotated by the shift graph).
void quantize_row_kv_oscar_int2_wht(const float * GGML_RESTRICT x, block_kv_oscar_int2 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_OSCAR_INT2 == 0);
    const int D        = (k % (2 * QK_OSCAR_INT2) == 0) ? (2 * QK_OSCAR_INT2) : QK_OSCAR_INT2;
    const int n_groups = (int)(k / D);
    const int n_sb     = D / QK_OSCAR_INT2; // sub-blocks per group: 1 (D=128) or 2 (D=256)
    const float inv_sqrt_D = 1.0f / sqrtf((float)D);

    for (int g = 0; g < n_groups; g++) {
        float xr[2 * QK_OSCAR_INT2]; // max D = 256
        for (int j = 0; j < D; j++) {
            xr[j] = x[(int64_t)g * D + j];
        }

        // In-place full-dim D-pt Walsh-Hadamard transform, then 1/sqrt(D) normalize.
        for (int h = 1; h < D; h <<= 1) {
            for (int i = 0; i < D; i += 2 * h) {
                for (int j = i; j < i + h; j++) {
                    const float a = xr[j], b = xr[j + h];
                    xr[j]     = a + b;
                    xr[j + h] = a - b;
                }
            }
        }
        for (int j = 0; j < D; j++) {
            xr[j] *= inv_sqrt_D;
        }

        // Per-128 sub-block uniform min-max INT2.
        for (int ib = 0; ib < n_sb; ib++) {
            const float * blk = xr + ib * QK_OSCAR_INT2;
            float mn = blk[0], mx = blk[0];
            for (int j = 1; j < QK_OSCAR_INT2; j++) {
                if (blk[j] < mn) mn = blk[j];
                if (blk[j] > mx) mx = blk[j];
            }
            const float range = mx - mn;
            const float d  = (range > 1e-10f) ? range / 3.0f : 1.0f;
            const float id = 1.0f / d;
            block_kv_oscar_int2 * out = &y[(int64_t)g * n_sb + ib];
            out->d = GGML_FP32_TO_FP16(d);
            out->m = GGML_FP32_TO_FP16(mn);
            memset(out->qs, 0, QK_OSCAR_INT2 / 4);
            for (int j = 0; j < QK_OSCAR_INT2; j++) {
                int q = (int)lrintf((blk[j] - mn) * id);
                q = q < 0 ? 0 : (q > 3 ? 3 : q);
                out->qs[j / 4] |= (uint8_t)(q << (2 * (j % 4)));
            }
        }
    }
}

void dequantize_row_kv_oscar_int2(const block_kv_oscar_int2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_OSCAR_INT2 == 0);
    const int nb = k / QK_OSCAR_INT2;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float m = GGML_FP16_TO_FP32(x[i].m);
        for (int j = 0; j < QK_OSCAR_INT2; j++) {
            const uint8_t byte = x[i].qs[j / 4];
            const int q = (byte >> (2 * (j % 4))) & 0x3;
            y[i * QK_OSCAR_INT2 + j] = d * q + m;
        }
    }
}

size_t quantize_kv_oscar_int2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                              int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_OSCAR_INT2 == 0);
    size_t row_size = (n_per_row / QK_OSCAR_INT2) * sizeof(block_kv_oscar_int2);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_kv_oscar_int2_ref(
            src + row * n_per_row,
            (block_kv_oscar_int2 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBOQ4_0: 4-bit PolarQuant (default) / 3-bit + QJL (legacy) ---------- */

void quantize_row_turboq4_0_ref(const float * GGML_RESTRICT x, block_turboq4_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();
    turbo_init_qjl();

    assert(k % QK_TURBOQ4 == 0);
    const int nb = k / QK_TURBOQ4;
    const int d  = QK_TURBOQ4;

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: Extract norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        float norm = sqrtf(norm_sq);

        /* Normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 2: Forward WHT rotation (matches CUDA set_rows) */
        float rotated[TURBO_D];
        memcpy(rotated, normalized, d * sizeof(float));
        turbo_cpu_fwht(rotated, d);

#if TURBOQ4_USE_4BIT
        /* Step 3: 4-bit quantization (16 centroids) */
        static const float CENTROIDS_4BIT[16] = {
            -0.173926f, -0.117195f, -0.089527f, -0.068756f,
            -0.051262f, -0.035597f, -0.020989f, -0.006938f,
             0.006938f,  0.020989f,  0.035597f,  0.051262f,
             0.068756f,  0.089527f,  0.117195f,  0.173926f
        };
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_4bit(rotated[i]);
        }

        /* Norm correction */
        float recon_norm_sq = 0.0f;
        for (int i = 0; i < d; i++) {
            recon_norm_sq += CENTROIDS_4BIT[indices[i]] * CENTROIDS_4BIT[indices[i]];
        }
        float recon_norm = sqrtf(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? norm / recon_norm : norm;
        y[block].norm = GGML_FP32_TO_FP16(corrected_norm);
#else
        /* Step 3: 3-bit quantization (8 centroids) */
        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            indices[i] = (uint8_t)nearest_centroid_3bit(rotated[i]);
        }

        /* Step 4: Residual */
        float reconstructed[TURBO_D];
        for (int i = 0; i < d; i++) {
            reconstructed[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, reconstructed, mse_recon, d);

        float residual[TURBO_D];
        for (int i = 0; i < d; i++) {
            residual[i] = normalized[i] - mse_recon[i];
        }

        /* Step 5: QJL */
        float projected[TURBO_D];
        matvec(turbo_qjl_matrix, residual, projected, d);
#endif

        /* Pack */
#if !TURBOQ4_USE_4BIT
        y[block].norm  = GGML_FP32_TO_FP16(norm);
#endif

#if TURBOQ4_USE_4BIT
        /* 4-bit PolarQuant: nibble pack into qs[64] */
        memset(y[block].qs, 0, d / 2);
        for (int i = 0; i < d; i++) {
            y[block].qs[i / 2] |= (uint8_t)((indices[i] & 0xF) << ((i % 2) * 4));
        }
        y[block].rnorm = GGML_FP32_TO_FP16(0.0f);
#else
        /* Legacy 3-bit + QJL: pack 3-bit indices + QJL signs */
        memset(y[block].qs, 0, d * 3 / 8);
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t val   = (uint16_t)(indices[i] & 0x7);
            y[block].qs[byte_idx] |= (uint8_t)(val << bit_pos);
            if (bit_pos > 5 && byte_idx + 1 < d * 3 / 8) {
                y[block].qs[byte_idx + 1] |= (uint8_t)(val >> (8 - bit_pos));
            }
        }
        memset(y[block].signs, 0, d / 8);
        for (int i = 0; i < d; i++) {
            if (projected[i] >= 0.0f) {
                y[block].signs[i / 8] |= (1 << (i % 8));
            }
        }
#endif
    }
}

void dequantize_row_turboq4_0(const block_turboq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBOQ4 == 0);
    const int nb = k / QK_TURBOQ4;
    const int d  = QK_TURBOQ4;

#if TURBOQ4_USE_4BIT
    /* 4-bit PolarQuant: nibble unpack → centroid → inverse rotate → scale */
    static const float CENTROIDS_4BIT[16] = {
        -0.173926f, -0.117195f, -0.089527f, -0.068756f,
        -0.051262f, -0.035597f, -0.020989f, -0.006938f,
         0.006938f,  0.020989f,  0.035597f,  0.051262f,
         0.068756f,  0.089527f,  0.117195f,  0.173926f
    };
    for (int block = 0; block < nb; block++) {
        float norm = GGML_FP16_TO_FP32(x[block].norm);
        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            uint8_t idx = (x[block].qs[i / 2] >> ((i % 2) * 4)) & 0xF;
            dst[i] = CENTROIDS_4BIT[idx] * norm;
        }
        /* No inverse WHT, dequant stays in the rotated domain.
        * Q is WHT-rotated by the graph, so <Q_rot, K_rot> gives correct attention scores.
        * The inverse WHT is applied to the attention output via GGML_OP_TURBO_WHT (direction=1) in the graph.
        */
    }
#else
    /* Legacy 3-bit + QJL dequant */
    turbo_init_qjl();
    for (int block = 0; block < nb; block++) {
        float norm  = GGML_FP16_TO_FP32(x[block].norm);

        uint8_t indices[TURBO_D];
        for (int i = 0; i < d; i++) {
            int bit_offset = i * 3;
            int byte_idx   = bit_offset / 8;
            int bit_pos    = bit_offset % 8;
            uint16_t raw   = (uint16_t)x[block].qs[byte_idx];
            if (byte_idx + 1 < d * 3 / 8) {
                raw |= (uint16_t)x[block].qs[byte_idx + 1] << 8;
            }
            indices[i] = (uint8_t)((raw >> bit_pos) & 0x7);
        }

        float signs[TURBO_D];
        for (int i = 0; i < d; i++) {
            signs[i] = (x[block].signs[i / 8] & (1 << (i % 8))) ? 1.0f : -1.0f;
        }

        float rnorm = GGML_FP16_TO_FP32(x[block].rnorm);
        const float qjl_scale = TURBO_QJL_CONST / (float)d * rnorm;

        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = CENTROIDS_3BIT[indices[i]];
        }
        float mse_recon[TURBO_D];
        matvec(turbo_rotation_t, rotated_recon, mse_recon, d);

        float qjl_recon[TURBO_D];
        matvec(turbo_qjl_matrix_t, signs, qjl_recon, d);
        for (int i = 0; i < d; i++) {
            qjl_recon[i] *= qjl_scale;
        }

        float * dst = y + block * d;
        for (int i = 0; i < d; i++) {
            dst[i] = (mse_recon[i] + qjl_recon[i]) * norm;
        }
    }
#endif
}

size_t quantize_turboq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                         int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ4 == 0);

    size_t row_size = (n_per_row / QK_TURBOQ4) * sizeof(block_turboq4_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turboq4_0_ref(
            src + row * n_per_row,
            (block_turboq4_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ---------- TURBOQ8_0: 8-bit, uniform 256-level grid + per-block absmax (no QJL) ----------
 * Unified codec: byte-identical math to the GPU encode (set-rows.cu k_set_rows_turboq8 /
 * turbo-quant.cuh quantize_f32_turboq8_0_block) so CPU encode ↔ GPU decode (and vice versa)
 * agree on every backend. Centroid grid is uniform: centroid[i] = (i-127.5)/127.5 ∈ [-1,1];
 * the per-block absmax scale is folded into the stored norm (= grp_norm * scale).
 * NOTE: this deliberately does NOT use Lloyd-Max centroids or reconstruction-norm correction
 * (the 4-bit/3-bit path's approach) — the Hadamard rotation suppresses outliers so a uniform
 * grid in the rotated domain is the design buun measured/shipped.                              */

void quantize_row_turboq8_0_ref(const float * GGML_RESTRICT x, block_turboq8_0 * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBOQ8 == 0);
    const int nb = k / QK_TURBOQ8;
    const int d  = QK_TURBOQ8;  /* == TURBO_D == 128 */

    for (int block = 0; block < nb; block++) {
        const float * src = x + block * d;

        /* Step 1: extract L2 norm */
        float norm_sq = 0.0f;
        for (int i = 0; i < d; i++) norm_sq += src[i] * src[i];
        const float norm = sqrtf(norm_sq);

        /* Step 2: normalize */
        float normalized[TURBO_D];
        if (norm > 1e-10f) {
            const float inv = 1.0f / norm;
            for (int i = 0; i < d; i++) normalized[i] = src[i] * inv;
        } else {
            memset(normalized, 0, d * sizeof(float));
        }

        /* Step 3: forward FWHT rotation */
        float rotated[TURBO_D];
        matvec(turbo_rotation, normalized, rotated, d);

        /* Step 4: per-block absmax → uniform 256-level grid */
        float absmax = 0.0f;
        for (int i = 0; i < d; i++) {
            const float a = fabsf(rotated[i]);
            if (a > absmax) absmax = a;
        }
        const float scale     = absmax > 1e-10f ? absmax : 1e-10f;
        const float inv_scale = 1.0f / scale;
        for (int i = 0; i < d; i++) {
            int idx = (int)lrintf(rotated[i] * inv_scale * 127.5f + 127.5f);
            idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
            y[block].qs[i] = (uint8_t)idx;
        }

        /* Step 5: stored norm folds in the absmax scale (decode = centroid*norm) */
        y[block].norm = GGML_FP32_TO_FP16(norm * scale);
    }
}

void dequantize_row_turboq8_0(const block_turboq8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    turbo_init_rotation();

    assert(k % QK_TURBOQ8 == 0);
    const int nb = k / QK_TURBOQ8;
    const int d  = QK_TURBOQ8;

    for (int block = 0; block < nb; block++) {
        const float norm = GGML_FP16_TO_FP32(x[block].norm);

        /* Reconstruct in rotated domain via the uniform grid */
        float rotated_recon[TURBO_D];
        for (int i = 0; i < d; i++) {
            rotated_recon[i] = ((float)x[block].qs[i] - 127.5f) * (1.0f / 127.5f);
        }

        /* Inverse rotate, then scale by stored norm */
        float * dst = y + block * d;
        matvec(turbo_rotation_t, rotated_recon, dst, d);
        for (int i = 0; i < d; i++) dst[i] *= norm;
    }
}

size_t quantize_turboq8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                          int64_t nrows, int64_t n_per_row, const float * imatrix) {
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ8 == 0);

    size_t row_size = (n_per_row / QK_TURBOQ8) * sizeof(block_turboq8_0);
    for (int64_t row = 0; row < nrows; row++) {
        quantize_row_turboq8_0_ref(
            src + row * n_per_row,
            (block_turboq8_0 *)((char *)dst + row * row_size),
            n_per_row
        );
    }
    return nrows * row_size;
}

/* ================================================================== */
/* WHT3_0 / WHT4_0: WHT-rotated weight quantization                  */
/* ================================================================== */

/* Lloyd-Max centroids for N(0,1) — shared with Metal shaders */
static const float TQ3_0_CENTROIDS[8] = {
    -1.996684f, -1.291398f, -0.740341f, -0.247508f,
     0.230106f,  0.725222f,  1.277503f,  1.988943f
};

static const float TQ4_0_CENTROIDS[16] = {
    -2.732590f, -2.069017f, -1.618046f, -1.256231f,
    -0.942340f, -0.656759f, -0.388048f, -0.128395f,
     0.128395f,  0.388048f,  0.656759f,  0.942340f,
     1.256231f,  1.618046f,  2.069017f,  2.732590f,
};

/* WHT sign pattern (golden ratio hash, 32-element blocks) — shared by TQ3 and TQ4 */
static const float TQ3_0_SIGNS[32] = {
    +1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, +1.0f, -1.0f, +1.0f,
    -1.0f, -1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
    -1.0f, +1.0f, +1.0f, -1.0f, +1.0f, -1.0f, -1.0f, +1.0f,
};

#define TQ_BLOCK_SIZE 32
#define TQ_INV_SQRT32 0.17677669529663688f  /* 1/sqrt(32) */

/* Forward RHT: sign flips -> WHT butterfly -> normalize */
static void tq3_0_rht_forward(float * buf) {
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ3_0_SIGNS[i];
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32;
}

/* Inverse RHT: WHT butterfly -> normalize + unsign */
static void tq3_0_rht_inverse(float * buf) {
    for (int step = 1; step < TQ_BLOCK_SIZE; step <<= 1) {
        for (int i = 0; i < TQ_BLOCK_SIZE; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j]     = a + b;
                buf[j + step] = a - b;
            }
        }
    }
    for (int i = 0; i < TQ_BLOCK_SIZE; i++) buf[i] *= TQ_INV_SQRT32 * TQ3_0_SIGNS[i];
}

/* Nearest centroid for TQ3 (8 centroids) */
static int tq3_0_choose_index(float val) {
    /* Binary search on midpoints of TQ3_0_CENTROIDS */
    if (val < -1.644041f) return 0;
    if (val < -1.015870f) return 1;
    if (val < -0.493925f) return 2;
    if (val < -0.008701f) return 3;
    if (val <  0.477664f) return 4;
    if (val <  1.001363f) return 5;
    if (val <  1.633223f) return 6;
    return 7;
}

/* Nearest centroid for TQ4 (16 centroids) */
static int tq4_0_choose_index(float val) {
    /* Binary search on midpoints of TQ4_0_CENTROIDS */
    if (val < -2.400804f) return 0;
    if (val < -1.843532f) return 1;
    if (val < -1.437139f) return 2;
    if (val < -1.099286f) return 3;
    if (val < -0.799550f) return 4;
    if (val < -0.522404f) return 5;
    if (val < -0.258222f) return 6;
    if (val <  0.000000f) return 7;
    if (val <  0.258222f) return 8;
    if (val <  0.522404f) return 9;
    if (val <  0.799550f) return 10;
    if (val <  1.099286f) return 11;
    if (val <  1.437139f) return 12;
    if (val <  1.843532f) return 13;
    if (val <  2.400804f) return 14;
    return 15;
}

/* ---------- WHT3_0 quantization ---------- */

/* Per-block quantize helper. iw is optional per-element importance weights
 * (NULL = unweighted). When set, the scale search uses weighted error and
 * the WLS refinement weights the inner products — ADR-016 imatrix integration.
 * Note: iw is applied to post-RHT positions directly (per recon/08 §720
 * TheTom pattern). This is approximate (RHT mixes positions) but
 * empirically effective. */
static void quantize_block_wht3_0(const float * GGML_RESTRICT src_blk,
                                  block_wht3_0 * GGML_RESTRICT blk,
                                  const float * iw) {
    /* NOTE (WHT imatrix audit, 2026-06-04): the importance vector `iw` is a
     * per-INPUT-COLUMN quantity in the ORIGINAL weight basis. The forward RHT
     * below rotates/mixes all 32 columns of the block, so post-RHT coefficient
     * buf[j] no longer corresponds to original column j. Weighting the post-RHT
     * residual by iw[j] (the ADR-016 port did this) misaligns importance with
     * the rotated coefficient and measurably HURT PPL (9B WHT3_0: 8.89 vs the
     * unweighted TheTom reference 7.6776, +16%). We therefore quantize the
     * rotated coefficients UNWEIGHTED, byte-for-byte matching TheTom's proven
     * quantize_row_tq3_1s_ref. iw is intentionally ignored. */
    (void) iw;

    /* 1. Forward RHT */
    float buf[TQ_BLOCK_SIZE];
    memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
    tq3_0_rht_forward(buf);

    /* 2. Split into two halves, compute RMS per half */
    float rms0 = 0.0f, rms1 = 0.0f;
    for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
    for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
    rms0 = sqrtf(rms0 / 16.0f);
    rms1 = sqrtf(rms1 / 16.0f);

    /* 3. Scale search (9 points), unweighted least-squares in rotated space */
    static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
    float best_d0 = rms0, best_d1 = rms1;
    float best_err = 1e30f;

    for (int si = 0; si < 9; si++) {
        float d0 = rms0 * scales[si];
        float d1 = rms1 * scales[si];
        float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
        float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

        float err = 0.0f;
        for (int j = 0; j < 16; j++) {
            int idx = tq3_0_choose_index(buf[j] * inv0);
            float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d0;
            err += diff * diff;
        }
        for (int j = 16; j < 32; j++) {
            int idx = tq3_0_choose_index(buf[j] * inv1);
            float diff = buf[j] - TQ3_0_CENTROIDS[idx] * d1;
            err += diff * diff;
        }
        if (err < best_err) {
            best_err = err;
            best_d0 = d0;
            best_d1 = d1;
        }
    }

    /* 4. Iterative LS refinement (6 iterations) in rotated space. */
    for (int iter = 0; iter < 6; iter++) {
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        float num0 = 0.0f, den0 = 0.0f;
        float num1 = 0.0f, den1 = 0.0f;
        for (int j = 0; j < 16; j++) {
            int idx = tq3_0_choose_index(buf[j] * inv0);
            float c = TQ3_0_CENTROIDS[idx];
            num0 += buf[j] * c;
            den0 += c * c;
        }
        for (int j = 16; j < 32; j++) {
            int idx = tq3_0_choose_index(buf[j] * inv1);
            float c = TQ3_0_CENTROIDS[idx];
            num1 += buf[j] * c;
            den1 += c * c;
        }
        if (den0 > 1e-10f) best_d0 = num0 / den0;
        if (den1 > 1e-10f) best_d1 = num1 / den1;
    }

    /* 5. Final quantize + pack */
    float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
    float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

    blk->d0 = GGML_FP32_TO_FP16(best_d0);
    blk->d1 = GGML_FP32_TO_FP16(best_d1);
    memset(blk->qs, 0, QK_TQ3_0 * 3 / 8);

    /* TQ3 packing: 4 groups of 8 indices packed into 3 bytes each */
    for (int g = 0; g < 4; g++) {
        uint8_t indices[8];
        for (int i = 0; i < 8; i++) {
            int j = g * 8 + i;
            float inv = (j < 16) ? inv0 : inv1;
            indices[i] = (uint8_t)tq3_0_choose_index(buf[j] * inv);
        }
        uint8_t * qp = blk->qs + g * 3;
        qp[0] = (indices[0] & 7) | ((indices[1] & 7) << 3) | ((indices[2] & 3) << 6);
        qp[1] = ((indices[2] >> 2) & 1) | ((indices[3] & 7) << 1) | ((indices[4] & 7) << 4) | ((indices[5] & 1) << 7);
        qp[2] = ((indices[5] >> 1) & 3) | ((indices[6] & 7) << 2) | ((indices[7] & 7) << 5);
    }
}

/* Public ref entry — unweighted (no imatrix). */
void quantize_row_wht3_0_ref(const float * GGML_RESTRICT x, block_wht3_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;
    for (int block = 0; block < nb; block++) {
        quantize_block_wht3_0(x + block * QK_TQ3_0, &y[block], NULL);
    }
}

void dequantize_row_wht3_0(const block_wht3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TQ3_0 == 0);
    const int nb = k / QK_TQ3_0;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        /* Unpack 3-bit indices */
        float buf[32];
        for (int g = 0; g < 4; g++) {
            const uint8_t * qp = x[blk_i].qs + g * 3;
            uint8_t idx[8];
            idx[0] =  qp[0]       & 7;
            idx[1] = (qp[0] >> 3) & 7;
            idx[2] = ((qp[0] >> 6) | (qp[1] << 2)) & 7;
            idx[3] = (qp[1] >> 1) & 7;
            idx[4] = (qp[1] >> 4) & 7;
            idx[5] = ((qp[1] >> 7) | (qp[2] << 1)) & 7;
            idx[6] = (qp[2] >> 2) & 7;
            idx[7] = (qp[2] >> 5) & 7;

            for (int i = 0; i < 8; i++) {
                int j = g * 8 + i;
                float d = (j < 16) ? d0 : d1;
                buf[j] = TQ3_0_CENTROIDS[idx[i]] * d;
            }
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_TQ3_0, buf, QK_TQ3_0 * sizeof(float));
    }
}

size_t quantize_wht3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_TQ3_0 == 0);

    const int64_t nb_per_row = n_per_row / QK_TQ3_0;
    size_t row_size = nb_per_row * sizeof(block_wht3_0);
    for (int64_t row = 0; row < nrows; row++) {
        block_wht3_0 * y_row = (block_wht3_0 *)((char *)dst + row * row_size);
        const float * x_row = src + row * n_per_row;
        for (int64_t b = 0; b < nb_per_row; b++) {
            const float * iw = imatrix ? (imatrix + b * QK_TQ3_0) : NULL;
            quantize_block_wht3_0(x_row + b * QK_TQ3_0, &y_row[b], iw);
        }
    }
    return nrows * row_size;
}

/* ---------- WHT4_0 quantization ---------- */

/* Per-block quantize helper. iw is optional per-element importance weights
 * (NULL = unweighted). See quantize_block_wht3_0 for the rationale. */
static void quantize_block_wht4_0(const float * GGML_RESTRICT src_blk,
                                  block_wht4_0 * GGML_RESTRICT blk,
                                  const float * iw) {
    /* NOTE (WHT imatrix audit, 2026-06-04): see quantize_block_wht3_0 — iw is a
     * per-INPUT-COLUMN importance in the ORIGINAL basis and does not align with
     * the post-RHT rotated coefficients, so weighting by it hurt PPL. Quantize
     * UNWEIGHTED to match TheTom's proven quantize_row_tq4_1s_ref. */
    (void) iw;

    /* 1. Forward RHT */
    float buf[TQ_BLOCK_SIZE];
    memcpy(buf, src_blk, TQ_BLOCK_SIZE * sizeof(float));
    tq3_0_rht_forward(buf);

    /* 2. Split into two halves, compute RMS per half */
    float rms0 = 0.0f, rms1 = 0.0f;
    for (int j = 0; j < 16; j++) rms0 += buf[j] * buf[j];
    for (int j = 16; j < 32; j++) rms1 += buf[j] * buf[j];
    rms0 = sqrtf(rms0 / 16.0f);
    rms1 = sqrtf(rms1 / 16.0f);

    /* 3. Scale search (9 points), unweighted least-squares in rotated space */
    static const float scales[] = { 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.35f, 1.5f };
    float best_d0 = rms0, best_d1 = rms1;
    float best_err = 1e30f;

    for (int si = 0; si < 9; si++) {
        float d0 = rms0 * scales[si];
        float d1 = rms1 * scales[si];
        float inv0 = (d0 > 1e-10f) ? 1.0f / d0 : 0.0f;
        float inv1 = (d1 > 1e-10f) ? 1.0f / d1 : 0.0f;

        float err = 0.0f;
        for (int j = 0; j < 16; j++) {
            int idx = tq4_0_choose_index(buf[j] * inv0);
            float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d0;
            err += diff * diff;
        }
        for (int j = 16; j < 32; j++) {
            int idx = tq4_0_choose_index(buf[j] * inv1);
            float diff = buf[j] - TQ4_0_CENTROIDS[idx] * d1;
            err += diff * diff;
        }
        if (err < best_err) {
            best_err = err;
            best_d0 = d0;
            best_d1 = d1;
        }
    }

    /* 4. Iterative LS refinement (6 iterations) in rotated space. */
    for (int iter = 0; iter < 6; iter++) {
        float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
        float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

        float num0 = 0.0f, den0 = 0.0f;
        float num1 = 0.0f, den1 = 0.0f;
        for (int j = 0; j < 16; j++) {
            int idx = tq4_0_choose_index(buf[j] * inv0);
            float c = TQ4_0_CENTROIDS[idx];
            num0 += buf[j] * c;
            den0 += c * c;
        }
        for (int j = 16; j < 32; j++) {
            int idx = tq4_0_choose_index(buf[j] * inv1);
            float c = TQ4_0_CENTROIDS[idx];
            num1 += buf[j] * c;
            den1 += c * c;
        }
        if (den0 > 1e-10f) best_d0 = num0 / den0;
        if (den1 > 1e-10f) best_d1 = num1 / den1;
    }

    /* 5. Final quantize + pack (nibble packing) */
    float inv0 = (best_d0 > 1e-10f) ? 1.0f / best_d0 : 0.0f;
    float inv1 = (best_d1 > 1e-10f) ? 1.0f / best_d1 : 0.0f;

    blk->d0 = GGML_FP32_TO_FP16(best_d0);
    blk->d1 = GGML_FP32_TO_FP16(best_d1);
    memset(blk->qs, 0, QK_WHT4_0 / 2);

    for (int j = 0; j < QK_WHT4_0; j++) {
        float inv = (j < 16) ? inv0 : inv1;
        int idx = tq4_0_choose_index(buf[j] * inv);
        blk->qs[j / 2] |= (uint8_t)((idx & 0xF) << ((j & 1) * 4));
    }
}

/* Public ref entry — unweighted (no imatrix). */
void quantize_row_wht4_0_ref(const float * GGML_RESTRICT x, block_wht4_0 * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_WHT4_0 == 0);
    const int nb = k / QK_WHT4_0;
    for (int block = 0; block < nb; block++) {
        quantize_block_wht4_0(x + block * QK_WHT4_0, &y[block], NULL);
    }
}

void dequantize_row_wht4_0(const block_wht4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_WHT4_0 == 0);
    const int nb = k / QK_WHT4_0;

    for (int blk_i = 0; blk_i < nb; blk_i++) {
        float d0 = GGML_FP16_TO_FP32(x[blk_i].d0);
        float d1 = GGML_FP16_TO_FP32(x[blk_i].d1);

        float buf[32];
        for (int j = 0; j < 32; j++) {
            uint8_t idx = (x[blk_i].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
            float d = (j < 16) ? d0 : d1;
            buf[j] = TQ4_0_CENTROIDS[idx] * d;
        }

        /* Inverse RHT */
        tq3_0_rht_inverse(buf);

        memcpy(y + blk_i * QK_WHT4_0, buf, QK_WHT4_0 * sizeof(float));
    }
}

size_t quantize_wht4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_WHT4_0 == 0);

    const int64_t nb_per_row = n_per_row / QK_WHT4_0;
    size_t row_size = nb_per_row * sizeof(block_wht4_0);
    for (int64_t row = 0; row < nrows; row++) {
        block_wht4_0 * y_row = (block_wht4_0 *)((char *)dst + row * row_size);
        const float * x_row = src + row * n_per_row;
        for (int64_t b = 0; b < nb_per_row; b++) {
            const float * iw = imatrix ? (imatrix + b * QK_WHT4_0) : NULL;
            quantize_block_wht4_0(x_row + b * QK_WHT4_0, &y_row[b], iw);
        }
    }
    return nrows * row_size;
}

/* ============================================================================
 * WQ3_TCQ — 3-bit Trellis-Coded WEIGHT quant (k=3, L=10, 1024 states) + FWHT
 *
 * CPU encoder (from_float_ref) + CPU dequant (to_float) for buun's WQ3 weight
 * runtime (ggml-cuda/wq3-tcq.cu). The CUDA decode is the authoritative contract:
 *
 *   y[t] = norm * (1/sqrt128) * s1[t] * FWHT( s2 ⊙ codebook[state] )[t]
 *
 * The encoder inverts it: normalize x -> forward FWHT (s1 -> Hadamard butterfly
 * -> 1/sqrt128*s2) -> Viterbi trellis encode (right-shift, init state 0) ->
 * states; corrected_norm = ||x|| / ||codebook[states]|| stored as fp16.
 *
 * codebook + signs are DETERMINISTIC and hardcoded so the C encoder, the C
 * dequant and the emitted GGUF (turbo.tcq.codebook.weight + turbo.tcq.sign_seed=42)
 * are byte-identical, giving exact CPU<->CUDA parity. Signs are copied verbatim
 * from wq3-tcq.cu (h_wq3_tcq_signs1_seed42 / h_wq3_tcq_signs2_seed1084) — these
 * are the WEIGHT-path signs and differ from turbo_cpu_s1/s2 (the KV-cache path).
 *
 * Block layout (block_turboq3_tcq, 52B): fp16 norm + qs[49] (390-bit stream:
 * 7-bit zero prefix + 128*3-bit outputs) + 1 pad byte. Decode reads a sliding
 * 10-bit window at bit t*3.
 * ==========================================================================*/

#define WQ3_QK          128
#define WQ3_K_BITS      3
#define WQ3_L_BITS      10
#define WQ3_N_STATES    1024
#define WQ3_N_OUT       8
#define WQ3_STATE_MASK  0x3FF
#define WQ3_PRED_MASK   0x7F                 /* (1<<(L-K))-1 */
static const float WQ3_INV_SQRT128 = 0.08838834764831845f;

/* seed-42 FWHT signs — verbatim from ggml-cuda/wq3-tcq.cu */
static const float wq3_s1[128] = {
    -1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,
     1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
     1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,
     1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1, 1,
    -1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1,
     1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,
};
static const float wq3_s2[128] = {
    -1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,
    -1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,
     1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,
    -1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,
    -1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
    -1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1,
};

#include "wq3-tcq-codebook.inc"   /* static const float WQ3_TCQ_CODEBOOK[1024] */

/* Accessors so llama-quant.cpp can emit the codebook + sign_seed into the GGUF. */
const float * ggml_wq3_tcq_codebook(int * n_entries) {
    if (n_entries) *n_entries = WQ3_N_STATES;
    return WQ3_TCQ_CODEBOOK;
}
uint32_t ggml_wq3_tcq_sign_seed(void) { return 42u; }

/* unnormalized 128-point Hadamard butterfly (matches wq3_fwht128 in CUDA) */
static void wq3_butterfly(float * x) {
    for (int h = 1; h < WQ3_QK; h *= 2)
        for (int i = 0; i < WQ3_QK; i += h * 2)
            for (int j = i; j < i + h; j++) {
                float a = x[j], b = x[j + h];
                x[j] = a + b; x[j + h] = a - b;
            }
}

/* forward rotate (encoder): x*=s1 -> butterfly -> x*=inv_sqrt128*s2 (orthonormal) */
static void wq3_fwht_forward(float * x) {
    for (int i = 0; i < WQ3_QK; i++) x[i] *= wq3_s1[i];
    wq3_butterfly(x);
    for (int i = 0; i < WQ3_QK; i++) x[i] *= WQ3_INV_SQRT128 * wq3_s2[i];
}

/* Viterbi encode a unit-norm rotated 128-block into trellis states (start state 0).
 * scratch must hold: cost[1024], ncost[1024] (float) and bt[128*1024] (uint16_t).
 * Returns recon_norm = ||codebook[states]||. */
static float wq3_viterbi(const float * data, int * states,
                         float * cost, float * ncost, uint16_t * bt) {
    for (int s = 0; s < WQ3_N_STATES; s++) cost[s] = 1e30f;
    cost[0] = 0.0f;
    for (int t = 0; t < WQ3_QK; t++) {
        uint16_t * bt_t = bt + (size_t)t * WQ3_N_STATES;
        for (int s = 0; s < WQ3_N_STATES; s++) ncost[s] = 1e30f;
        const float xt = data[t];
        for (int s = 0; s < WQ3_N_STATES; s++) {
            const float bc = cost[s];
            if (bc >= 1e30f) continue;
            const int sh = s >> WQ3_K_BITS;            /* shared high bits of next state */
            for (int out = 0; out < WQ3_N_OUT; out++) {
                const int ns = sh | (out << (WQ3_L_BITS - WQ3_K_BITS));
                const float d = xt - WQ3_TCQ_CODEBOOK[ns];
                const float tot = bc + d * d;
                if (tot < ncost[ns]) { ncost[ns] = tot; bt_t[ns] = (uint16_t) s; }
            }
        }
        float * tmp = cost; cost = ncost; ncost = tmp;
    }
    int st = 0; float best = 1e30f;
    for (int s = 0; s < WQ3_N_STATES; s++) if (cost[s] < best) { best = cost[s]; st = s; }
    float rn = 0.0f;
    for (int t = WQ3_QK - 1; t >= 0; t--) {
        states[t] = st;
        rn += WQ3_TCQ_CODEBOOK[st] * WQ3_TCQ_CODEBOOK[st];
        st = bt[(size_t)t * WQ3_N_STATES + st];
    }
    return sqrtf(rn);
}

/* pack states -> qs[49]: 7-bit zero prefix, then out[t]=states[t]>>7 at bit 7+t*3 */
static void wq3_pack(const int * states, uint8_t * qs) {
    memset(qs, 0, 49);
    for (int t = 0; t < WQ3_QK; t++) {
        const int out = states[t] >> (WQ3_L_BITS - WQ3_K_BITS);
        const int bitpos = (WQ3_L_BITS - WQ3_K_BITS) + t * WQ3_K_BITS;
        for (int b = 0; b < WQ3_K_BITS; b++)
            if (out & (1 << b)) qs[(bitpos + b) >> 3] |= (uint8_t)(1 << ((bitpos + b) & 7));
    }
}

/* encode one 128-block using caller-provided Viterbi scratch */
static void wq3_encode_block(const float * x, block_turboq3_tcq * blk,
                             float * cost, float * ncost, uint16_t * bt) {
    float buf[WQ3_QK];
    float n2 = 0.0f;
    for (int i = 0; i < WQ3_QK; i++) { buf[i] = x[i]; n2 += x[i] * x[i]; }
    const float saved = sqrtf(n2);
    const float inv = (saved > 1e-10f) ? 1.0f / saved : 0.0f;
    for (int i = 0; i < WQ3_QK; i++) buf[i] *= inv;
    wq3_fwht_forward(buf);                     /* into unit-norm codebook domain */
    int states[WQ3_QK];
    const float rn = wq3_viterbi(buf, states, cost, ncost, bt);
    const float corrected = (rn > 1e-10f) ? saved / rn : saved;
    blk->norm = GGML_FP32_TO_FP16(corrected);
    wq3_pack(states, blk->qs);
    blk->pad = 0;
}

void quantize_row_wq3_tcq_ref(const float * GGML_RESTRICT x, block_turboq3_tcq * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBOQ3_TCQ == 0);
    const int nb = (int)(k / QK_TURBOQ3_TCQ);
    float  * cost  = (float *)   malloc(sizeof(float)    * WQ3_N_STATES);
    float  * ncost = (float *)   malloc(sizeof(float)    * WQ3_N_STATES);
    uint16_t * bt  = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)WQ3_QK * WQ3_N_STATES);
    for (int i = 0; i < nb; i++)
        wq3_encode_block(x + (size_t)i * QK_TURBOQ3_TCQ, &y[i], cost, ncost, bt);
    free(cost); free(ncost); free(bt);
}

void dequantize_row_wq3_tcq(const block_turboq3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_TURBOQ3_TCQ == 0);
    const int nb = (int)(k / QK_TURBOQ3_TCQ);
    for (int blk = 0; blk < nb; blk++) {
        const block_turboq3_tcq * b = &x[blk];
        const float norm = GGML_FP16_TO_FP32(b->norm);
        float r[WQ3_QK];
        /* sliding 10-bit window decode (qs[49] + implicit 0 pad byte read as qs[49]=0) */
        uint8_t qs[51]; memcpy(qs, b->qs, 49); qs[49] = 0; qs[50] = 0;
        for (int t = 0; t < WQ3_QK; t++) {
            const int bitpos = t * WQ3_K_BITS, byte = bitpos >> 3, off = bitpos & 7;
            const uint32_t raw = (uint32_t)qs[byte] | ((uint32_t)qs[byte+1] << 8) | ((uint32_t)qs[byte+2] << 16);
            const int state = (int)((raw >> off) & WQ3_STATE_MASK);
            r[t] = WQ3_TCQ_CODEBOOK[state];
        }
        /* inverse rotation matching CUDA: s2 -> butterfly -> inv_sqrt128 * s1 * norm */
        for (int i = 0; i < WQ3_QK; i++) r[i] *= wq3_s2[i];
        wq3_butterfly(r);
        for (int i = 0; i < WQ3_QK; i++) y[(size_t)blk * WQ3_QK + i] = r[i] * WQ3_INV_SQRT128 * wq3_s1[i] * norm;
    }
}

size_t quantize_wq3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    /* imatrix passthrough — intentionally NOT applied. The FWHT rotates each
     * 128-group, so original-basis per-column importance is misaligned with the
     * rotated trellis coefficients; moreover the Hadamard has equal-magnitude
     * entries (F_tj^2 = 1/128 for all t,j), so the diagonal of the rotated weight
     * matrix is uniform (= mean importance) and per-element weighting degenerates
     * to a constant that cannot change the Viterbi path. This matches the audited
     * WHT3_0/WHT4_0 decision (imatrix weighting HURT PPL +16% there). */
    GGML_UNUSED(imatrix);
    assert(n_per_row % QK_TURBOQ3_TCQ == 0);
    const int64_t nb_per_row = n_per_row / QK_TURBOQ3_TCQ;
    const size_t row_size = nb_per_row * sizeof(block_turboq3_tcq);

    float  * cost  = (float *)   malloc(sizeof(float)    * WQ3_N_STATES);
    float  * ncost = (float *)   malloc(sizeof(float)    * WQ3_N_STATES);
    uint16_t * bt  = (uint16_t *)malloc(sizeof(uint16_t) * (size_t)WQ3_QK * WQ3_N_STATES);
    for (int64_t row = 0; row < nrows; row++) {
        block_turboq3_tcq * y_row = (block_turboq3_tcq *)((char *)dst + row * row_size);
        const float * x_row = src + row * n_per_row;
        for (int64_t b = 0; b < nb_per_row; b++)
            wq3_encode_block(x_row + b * QK_TURBOQ3_TCQ, &y_row[b], cost, ncost, bt);
    }
    free(cost); free(ncost); free(bt);
    return nrows * row_size;
}
