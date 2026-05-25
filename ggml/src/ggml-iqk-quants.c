// IQK weight quantization types ported from ik_llama.cpp via frankenturbo2
// Source: /usr/src/llama-forks/frankenturbo2 @ feature/turboquant-kv-cache
// Type IDs renumbered: ft2 60/59/58 → ygg canonical 137(IQ2_K)/138(IQ3_K)/139(IQ4_K)
// Reference scalar implementations only — no SIMD optimization.

#include "ggml-impl.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

// IQ4_K centroid tables (CPU side; GPU equivalents are in turbo-quant.cuh)
static const int8_t iq4k_values[32] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
    -123, -100, -79, -61, -45, -31, -18,  -6, 5, 17, 29, 42, 57, 73, 93, 117,
};

// IQ3_K centroid table — 8 standard + 8 shifted values
static const int8_t iq3nl_values[16] = {
    -63, -40, -23, -10, 1, 13, 28,  47,
    -59, -36, -19,  -6, 5, 17, 32,  51,
};

// Best-index for IQ3 — only 8 entries, simple linear scan
static inline int iqk_best_index_iq3nl(const int8_t * values, float x) {
    int best = 0;
    float best_diff = fabsf(x - (float)values[0]);
    for (int i = 1; i < 8; ++i) {
        float diff = fabsf(x - (float)values[i]);
        if (diff < best_diff) { best_diff = diff; best = i; }
    }
    return best;
}

static const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15,
};

static inline int iqk_nearest_int(float fval) {
    assert(fabsf(fval) <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// Best-index lookup for iq4_k centroid table.  values is a 16-entry table,
// passed as either iq4k_values or iq4k_values+16 (the shifted variant).
static inline int iqk_best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : (x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15);
}

// =============================================================================
// IQ4_K
// =============================================================================

void dequantize_row_iq4_k(const block_iq4_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * qs = x[i].qs;
        const float d = GGML_FP16_TO_FP32(x[i].d);
        uint16_t extra = x[i].extra;

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const uint8_t sh = x[i].scales_h[ib/2] >> (4*(ib%2));
            const float dl1 = d * (((x[i].scales_l[ib] & 0xf) | ((sh << 4) & 0x30)) - 32);
            const float dl2 = d * (((x[i].scales_l[ib] >>  4) | ((sh << 2) & 0x30)) - 32);
            const int8_t * values1 = (extra & 1) ? iq4k_values + 16 : iq4k_values;
            const int8_t * values2 = (extra & 2) ? iq4k_values + 16 : iq4k_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl1 * values1[qs[j] & 0xf];
                y[j + 16] = dl2 * values2[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}

// Core per-superblock quantizer.  Imatrix-aware when quant_weights != NULL.
// ntry controls the scale-search radius (7 in upstream).
static void quantize_row_iq4_k_impl(
        const float * x, block_iq4_k * y,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        int ntry) {

    const int super_block_size = QK_K;
    const int block_size = 16;

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f / super_block_size;

    memset(y, 0, sizeof(block_iq4_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    uint16_t * scales_h = (uint16_t *)y->scales_h;
    const int8_t * shifted_values = values + 16;

    float max_scale = 0, amax_scale = 0;
    uint16_t extra = 0;

    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        if (amax < 1e-16f) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx_p = 0, sumq2_p = 0;
        float sumqx_m = 0, sumq2_m = 0;
        for (int j = 0; j < block_size; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = iqk_best_index_iq4nl(values, al);
            float q = values[l];
            sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
            l = iqk_best_index_iq4nl(values, -al);
            q = values[l];
            sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
        }
        d = sumqx_p/sumq2_p;
        bool is_shifted = false;
        float best = d*sumqx_p;
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
            d = sumqx_m/sumq2_m; best = d*sumqx_m;
        }
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = iqk_best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = iqk_best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
            }
            id = (itry + shifted_values[0])/max;
            sumqx_p = sumq2_p = 0;
            sumqx_m = sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = iqk_best_index_iq4nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = iqk_best_index_iq4nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
            }
        }
        if (is_shifted) extra |= (1u << ib);
        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) { amax_scale = abs_d; max_scale = d; }
    }

    float d = -max_scale/32;
    y->d = GGML_FP32_TO_FP16(d);
    y->extra = extra;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;

    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const int8_t * block_values = (extra & (1u << ib)) ? shifted_values : values;
        int l = iqk_nearest_int(id*scales[ib]);
        l = MAX(-32, MIN(31, l));
        float dl = d * l;
        float idl = dl ? 1/dl : 0.f;
        uint8_t * Lb = L + ib*block_size;
        const float * xb = x + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        for (int j = 0; j < block_size; ++j) {
            Lb[j] = iqk_best_index_iq4nl(block_values, idl*xb[j]);
            float w = weight[j];
            float q = block_values[Lb[j]]*l;
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        l += 32;
        uint8_t l_l = l & 0xf;
        uint8_t l_h = l >>  4;
        if (ib%2 == 0) y->scales_l[ib/2] = l_l;
        else            y->scales_l[ib/2] |= (l_l << 4);
        scales_h[ib/8] |= (l_h << 2*(ib%8));
    }
    if (sumq2 > 0) y->d = GGML_FP32_TO_FP16(sumqx/sumq2);

    // Pack 4-bit indices: low nibble is L[0..15], high nibble is L[16..31] for each sub-block
    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            y->qs[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}

void quantize_row_iq4_k_ref(const float * GGML_RESTRICT x, block_iq4_k * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_k(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq4_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq4_k_ref(x, (block_iq4_k *)y, k);
}

size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                      int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_K, n_per_row);
    const int nblock = n_per_row / QK_K;

    uint8_t L[QK_K];
    float weight[16];
    float scales[QK_K/16];

    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        block_iq4_k * y = (block_iq4_k *)((char *)dst + row * row_size);
        const float * qw_row = imatrix;

        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw_blk = qw_row ? qw_row + QK_K*ibl : NULL;
            quantize_row_iq4_k_impl(x + QK_K*ibl, y + ibl,
                                    scales, weight, L, iq4k_values, qw_blk, 7);
        }
    }
    return nrows * row_size;
}

// CPU dot product against q8_K activations (mainline pattern).
void ggml_vec_dot_iq4_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const int nb = n / QK_K;
    const block_iq4_k * x = (const block_iq4_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        uint16_t extra = x[ibl].extra;
        uint32_t h = *((const uint32_t *)x[ibl].scales_h);
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        int32_t sum = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls1 = ((x[ibl].scales_l[ib] & 0xf) | ((h << 4) & 0x30)) - 32;
            const int ls2 = ((x[ibl].scales_l[ib] >>  4) | ((h << 2) & 0x30)) - 32;
            h >>= 4;
            const int8_t * values1 = iq4k_values + 16*(extra & 1);
            const int8_t * values2 = iq4k_values +  8*(extra & 2);
            extra >>= 2;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j +  0] * values1[qs[j] & 0xf];
                sumi2 += q8[j + 16] * values2[qs[j] >>  4];
            }
            sum += ls1*sumi1 + ls2*sumi2;
            qs += 16;
            q8 += 32;
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;
}

// =============================================================================
// IQ3_K
// =============================================================================

void dequantize_row_iq3_k(const block_iq3_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        uint16_t sh = x[i].scales_h;
        uint16_t extra = x[i].extra;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1));
            float dl2 = d * ((2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1));
            sh >>= 2;
            const int8_t * values1 = (extra & 1) ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = (extra & 2) ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2)];
                y[j+16] = dl2 * values2[((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2)];
            }
            y += 32;
            if (shift_l == 6) qs += 32;
        }
    }
}

static void quantize_row_iq3_k_impl(
        const float * x, block_iq3_k * y,
        const float * quant_weights) {

    const int ntry = 3;
    float scales[QK_K/16];
    float weight[16];
    const int8_t * shifted_values = iq3nl_values + 8;

    memset(y, 0, sizeof(block_iq3_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    float sumx2 = 0;
    for (int j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
    const float sigma2 = 1.5f*sumx2/QK_K;

    uint16_t extra = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K/16; ++ib) {
        const float * xb = x + 16*ib;
        if (quant_weights) {
            const float * qw = quant_weights + ib*16;
            for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < 16; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) { amax = ax; max = xb[j]; }
        }
        if (amax < 1e-16f) { scales[ib] = 0; continue; }

        float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
        float id = 1/d;
        float sumqx_p = 0, sumq2_p = 0, sumqx_m = 0, sumq2_m = 0;
        float best = 0;
        for (int j = 0; j < 16; ++j) {
            float w = weight[j];
            float al = id*xb[j];
            int l = iqk_best_index_iq3nl(iq3nl_values, al);
            float q = iq3nl_values[l];
            sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
            l = iqk_best_index_iq3nl(iq3nl_values, -al);
            q = iq3nl_values[l];
            sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
        }
        if (sumq2_p > 0) { d = sumqx_p/sumq2_p; best = d*sumqx_p; }
        if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; }

        bool is_shifted = false;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (2*itry + iq3nl_values[0])/max;
            sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = iqk_best_index_iq3nl(iq3nl_values, al);
                float q = iq3nl_values[l];
                sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = iqk_best_index_iq3nl(iq3nl_values, -al);
                q = iq3nl_values[l];
                sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = false; }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = false; }
            id = (2*itry + shifted_values[0])/max;
            sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = iqk_best_index_iq3nl(shifted_values, al);
                float q = shifted_values[l];
                sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = iqk_best_index_iq3nl(shifted_values, -al);
                q = shifted_values[l];
                sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = true; }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = true; }
        }
        if (!d) { scales[ib] = 0; continue; }

        if (is_shifted) extra |= (1u << ib);
        scales[ib] = d;

        float abs_scale = fabsf(scales[ib]);
        if (abs_scale > max_abs_scale) max_abs_scale = abs_scale;
    }

    if (max_abs_scale == 0) return;

    float d = max_abs_scale/31;
    y->extra = extra;
    float id = 1/d;
    float sumqx = 0, sumq2 = 0;

    for (int ib = 0; ib < QK_K/16; ++ib) {
        int ls = iqk_nearest_int(0.5f*(id*fabsf(scales[ib])-1));
        ls = MAX(0, MIN(15, ls));
        y->scales_l[ib/2] |= (ls << 4*(ib%2));
        if (scales[ib] < 0) y->scales_h |= (1u << ib);
        ls = (2*ls + 1) * (scales[ib] < 0 ? -1 : 1);
        float dl = d * ls;
        if (!dl) continue;

        const int8_t * block_values = (y->extra & (1u << ib)) ? shifted_values : iq3nl_values;
        const float * xb = x + 16*ib;
        if (quant_weights) {
            const float * qw = quant_weights + ib*16;
            for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
        }
        float idl = 1/dl;
        int ib32 = ib/2;
        int offset = 16*(ib%2);
        uint8_t * qs = y->qs + 32*(ib32/4) + offset;
        uint8_t * qh = y->qh + 32*(ib32/8) + offset;
        for (int j = 0; j < 16; ++j) {
            const float al = idl*xb[j];
            int ibest = iqk_best_index_iq3nl(block_values, al);
            qs[j] |= ((ibest &  3) << 2*(ib32%4));
            qh[j] |= (((ibest >> 2) & 1) << (ib32%8));
            float w = weight[j];
            float q = block_values[ibest]*ls;
            sumqx += w*q*xb[j]; sumq2 += w*q*q;
        }
    }
    y->d = GGML_FP32_TO_FP16(1.01f * (sumq2 > 0 ? sumqx/sumq2 : d));
}

void quantize_row_iq3_k_ref(const float * GGML_RESTRICT x, block_iq3_k * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; ++i) quantize_row_iq3_k_impl(x + i*QK_K, y + i, NULL);
}

void quantize_row_iq3_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq3_k_ref(x, (block_iq3_k *)y, k);
}

size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                      int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ3_K, n_per_row);
    const int nblock = n_per_row / QK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        block_iq3_k * y = (block_iq3_k *)((char *)dst + row * row_size);
        const float * qw_row = imatrix;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw_blk = qw_row ? qw_row + QK_K*ibl : NULL;
            quantize_row_iq3_k_impl(x + QK_K*ibl, y + ibl, qw_blk);
        }
    }
    return nrows * row_size;
}

void ggml_vec_dot_iq3_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const int nb = n / QK_K;
    const block_iq3_k * x = (const block_iq3_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t sh = x[ibl].scales_h;
        uint16_t extra = x[ibl].extra;
        int32_t sum = 0;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const int ls1 = (2*(x[ibl].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1);
            const int ls2 = (2*(x[ibl].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1);
            sh >>= 2;
            const int8_t * values1 = (extra & 1) ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = (extra & 2) ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                int idx1 = ((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2);
                int idx2 = ((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2);
                sumi1 += q8[j+ 0] * values1[idx1];
                sumi2 += q8[j+16] * values2[idx2];
            }
            sum += ls1*sumi1 + ls2*sumi2;
            q8 += 32;
            if (shift_l == 6) qs += 32;
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;
}

// =============================================================================
// IQ2_K
// =============================================================================

// IQ2_K centroid table — 4 standard + 4 shifted values
static const int8_t iq2nl_values[8] = {
    -31, -13,  1, 17,
    -26,  -8,  6, 22,
};

// Best-index for IQ2 — only 4 entries, simple linear scan
static inline int iqk_best_index_iq2nl(const int8_t * values, float x) {
    int best = 0;
    float best_diff = fabsf(x - (float)values[0]);
    for (int i = 1; i < 4; ++i) {
        float diff = fabsf(x - (float)values[i]);
        if (diff < best_diff) { best_diff = diff; best = i; }
    }
    return best;
}

// qsort comparator for (value, index) pairs sorted by value (then index for stability)
typedef struct { float v; int i; } iqk_pair_t;
static int iqk_pair_cmp(const void * a, const void * b) {
    const iqk_pair_t * pa = (const iqk_pair_t *)a;
    const iqk_pair_t * pb = (const iqk_pair_t *)b;
    if (pa->v < pb->v) return -1;
    if (pa->v > pb->v) return 1;
    return pa->i - pb->i;
}

// Port of make_qx_quants from ggml-quants.c.  Quantizes scales themselves: finds
// a row-level scale d such that round(L[i]/d) maps each scales[i] to [-nmax, nmax-1].
// rmse_type controls weight inside the inner search (1 = x*x, 2 = uniform, 3 = |x|, 4 = sqrt|x|).
// qw, when non-NULL, overrides the rmse weighting.
static float iqk_make_qx_quants(int n, int nmax, const float * x, int8_t * L, int rmse_type, const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = iqk_nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) { rmse_type = -rmse_type; return_early = true; }

    float sumlx = 0, suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = iqk_nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i]
                : rmse_type == 1 ? x[i] * x[i]
                : rmse_type == 2 ? 1.f
                : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) continue;
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = iqk_nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i]
                    : rmse_type == 1 ? x[i] * x[i]
                    : rmse_type == 2 ? 1.f
                    : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = iqk_nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
        }
    }
    return scale;
}

void dequantize_row_iq2_k(const block_iq2_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        uint16_t extra = x[i].extra;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((x[i].scales[ib32] & 0xf) - 8);
            float dl2 = d * ((x[i].scales[ib32] >>  4) - 8);
            const int8_t * values1 = (extra & 1) ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = (extra & 2) ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl1 * values1[(qs[j +  0] >> shift) & 3];
                y[j + 16] = dl2 * values2[(qs[j + 16] >> shift) & 3];
            }
            y += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }
    }
}

// Per-superblock imatrix-aware quantizer.  Mirrors ik_llama.cpp's
// quantize_row_iq2_k_impl: triple-nested cumulative-sum search per 16-elem
// sub-block over standard and shifted iq2nl_values, both forward and reversed.
static void quantize_row_iq2_k_impl(const float * x, block_iq2_k * y, const float * quant_weights) {
    const int kBlockSize = 16;

    float scales[QK_K/16];
    float weight[16];
    float sumx[17], sumw[17];
    float sw[QK_K/16];
    int8_t Ls[QK_K/16];
    iqk_pair_t pairs[16];

    const int8_t * shifted_values = iq2nl_values + 4;

    memset(y, 0, sizeof(block_iq2_k));
    y->d = GGML_FP32_TO_FP16(0.f);

    float sumx2 = 0;
    for (int j = 0; j < QK_K; ++j) sumx2 += x[j]*x[j];
    const float sigma2 = 1.5f * sumx2 / QK_K;

    uint16_t extra = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
        const float * xb = x + kBlockSize*ib;
        if (quant_weights) {
            const float * qw = quant_weights + ib*kBlockSize;
            for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
        }
        sw[ib] = 0;
        float amax = 0;
        for (int j = 0; j < kBlockSize; ++j) {
            sw[ib] += weight[j];
            pairs[j].v = xb[j];
            pairs[j].i = j;
            float ax = fabsf(xb[j]);
            if (ax > amax) amax = ax;
        }
        if (amax < 1e-16f) {
            scales[ib] = 0;
            continue;
        }
        qsort(pairs, kBlockSize, sizeof(iqk_pair_t), iqk_pair_cmp);
        sumx[0] = sumw[0] = 0;
        for (int j = 0; j < kBlockSize; ++j) {
            int jj = pairs[j].i;
            sumw[j+1] = sumw[j] + weight[jj];
            sumx[j+1] = sumx[j] + weight[jj]*xb[jj];
        }
        float best = 0, d = 0;
        bool is_shifted = false;
        for (int i1 = 0; i1 < kBlockSize; ++i1) {
            for (int i2 = i1; i2 < kBlockSize; ++i2) {
                for (int i3 = i2; i3 < kBlockSize; ++i3) {
                    // Forward, standard
                    float sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[0] + (sumx[i2] - sumx[i1])*iq2nl_values[1]
                                + (sumx[i3] - sumx[i2])*iq2nl_values[2] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[3];
                    float sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[0]*iq2nl_values[0] + (sumw[i2] - sumw[i1])*iq2nl_values[1]*iq2nl_values[1]
                                + (sumw[i3] - sumw[i2])*iq2nl_values[2]*iq2nl_values[2] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[3]*iq2nl_values[3];
                    if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx/sumq2; best = d*sumqx; is_shifted = false; }
                    // Forward, shifted
                    sumqx = (sumx[i1] - sumx[ 0])*shifted_values[0] + (sumx[i2] - sumx[i1])*shifted_values[1]
                          + (sumx[i3] - sumx[i2])*shifted_values[2] + (sumx[kBlockSize] - sumx[i3])*shifted_values[3];
                    sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[0]*shifted_values[0] + (sumw[i2] - sumw[i1])*shifted_values[1]*shifted_values[1]
                          + (sumw[i3] - sumw[i2])*shifted_values[2]*shifted_values[2] + (sumw[kBlockSize] - sumw[i3])*shifted_values[3]*shifted_values[3];
                    if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx/sumq2; best = d*sumqx; is_shifted = true; }
                    // Reversed, standard
                    sumqx = (sumx[i1] - sumx[ 0])*iq2nl_values[3] + (sumx[i2] - sumx[i1])*iq2nl_values[2]
                          + (sumx[i3] - sumx[i2])*iq2nl_values[1] + (sumx[kBlockSize] - sumx[i3])*iq2nl_values[0];
                    sumq2 = (sumw[i1] - sumw[ 0])*iq2nl_values[3]*iq2nl_values[3] + (sumw[i2] - sumw[i1])*iq2nl_values[2]*iq2nl_values[2]
                          + (sumw[i3] - sumw[i2])*iq2nl_values[1]*iq2nl_values[1] + (sumw[kBlockSize] - sumw[i3])*iq2nl_values[0]*iq2nl_values[0];
                    if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx/sumq2; best = d*sumqx; is_shifted = false; }
                    // Reversed, shifted
                    sumqx = (sumx[i1] - sumx[ 0])*shifted_values[3] + (sumx[i2] - sumx[i1])*shifted_values[2]
                          + (sumx[i3] - sumx[i2])*shifted_values[1] + (sumx[kBlockSize] - sumx[i3])*shifted_values[0];
                    sumq2 = (sumw[i1] - sumw[ 0])*shifted_values[3]*shifted_values[3] + (sumw[i2] - sumw[i1])*shifted_values[2]*shifted_values[2]
                          + (sumw[i3] - sumw[i2])*shifted_values[1]*shifted_values[1] + (sumw[kBlockSize] - sumw[i3])*shifted_values[0]*shifted_values[0];
                    if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx/sumq2; best = d*sumqx; is_shifted = true; }
                }
            }
        }
        scales[ib] = d;
        if (is_shifted) extra |= (1u << ib);
        float abs_scale = fabsf(d);
        if (abs_scale > max_abs_scale) max_abs_scale = abs_scale;
    }

    if (!max_abs_scale) return;

    float d_super = iqk_make_qx_quants(QK_K/kBlockSize, 8, scales, Ls, 0, sw);
    if (!d_super) return;

    y->extra = extra;
    float id = 1.f / d_super;

    float sumqx = 0, sumq2 = 0;
    for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
        int ls = iqk_nearest_int(id*scales[ib]);
        if (ls < -8) ls = -8;
        if (ls >  7) ls =  7;
        y->scales[ib/2] |= ((ls + 8) << (4*(ib%2)));
        float dl = d_super * ls;
        if (dl) {
            const int8_t * block_values = (extra & (1u << ib)) ? shifted_values : iq2nl_values;
            const float * xb = x + kBlockSize*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ib*kBlockSize;
                for (int j = 0; j < kBlockSize; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < kBlockSize; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float idl = 1.f / dl;
            int ib32 = ib/2;
            int offset = 16*(ib%2);
            uint8_t * qs = y->qs + 32*(ib32/4) + offset;
            for (int j = 0; j < 16; ++j) {
                float al = idl*xb[j];
                int ibest = iqk_best_index_iq2nl(block_values, al);
                qs[j] |= (ibest << (2*(ib32%4)));
                float w = weight[j];
                float q = block_values[ibest]*ls;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    y->d = GGML_FP32_TO_FP16(1.030f * (sumq2 > 0 ? sumqx/sumq2 : d_super));
}

size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                      int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ2_K, n_per_row);
    const int nblock = n_per_row / QK_K;

    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        block_iq2_k * y = (block_iq2_k *)((char *)dst + row * row_size);
        const float * qw_row = imatrix;

        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * qw_blk = qw_row ? qw_row + QK_K*ibl : NULL;
            quantize_row_iq2_k_impl(x + QK_K*ibl, y + ibl, qw_blk);
        }
    }
    return nrows * row_size;
}

void quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_k(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq2_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq2_k_ref(x, (block_iq2_k *)y, k);
}

void ggml_vec_dot_iq2_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const int nb = n / QK_K;
    const block_iq2_k * x = (const block_iq2_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;

    float sumf = 0;
    for (int ibl = 0; ibl < nb; ++ibl) {
        const float d4d8 = GGML_FP16_TO_FP32(x[ibl].d) * y[ibl].d;
        const uint8_t * qs = x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        uint16_t extra = x[ibl].extra;
        int32_t sum = 0;

        int shift = 0;
        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            const int ls1 = (x[ibl].scales[ib32] & 0xf) - 8;
            const int ls2 = (x[ibl].scales[ib32] >>  4) - 8;
            const int8_t * values1 = (extra & 1) ? iq2nl_values + 4 : iq2nl_values;
            const int8_t * values2 = (extra & 2) ? iq2nl_values + 4 : iq2nl_values;
            extra >>= 2;
            int sumi1 = 0, sumi2 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j +  0] * values1[(qs[j +  0] >> shift) & 3];
                sumi2 += q8[j + 16] * values2[(qs[j + 16] >> shift) & 3];
            }
            sum += ls1*sumi1 + ls2*sumi2;
            q8 += 32;
            shift += 2;
            if (shift == 8) { qs += 32; shift = 0; }
        }
        sumf += d4d8 * sum;
    }
    *s = sumf;
}
// =============================================================================
// IQ4_KS — 4.25 bpw (row_meta_size = 4: per-row float scale)
// Row layout: [float d][block_iq4_ks blocks[n_per_row/QK_K]]
// Each block_iq4_ks: 8 scale bytes (1 codebook-shift bit + 7-bit signed scale offset)
//                    + 128 qs bytes (4-bit indices, 2 per byte)
// =============================================================================

void dequantize_row_iq4_ks(const block_iq4_ks * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    const int kBlockSize = 32;
    assert(k % QK_K == 0);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    const block_iq4_ks * x = (const block_iq4_ks *)(dptr + 1);
    const int nblock = k / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint8_t * qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float dl = d * (float)((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j               ] = dl * (float)values[qs[j] & 0xf];
                y[j + kBlockSize/2] = dl * (float)values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

// Per-row imatrix-aware quantizer.  Mirrors ik_llama.cpp's quantize_row_iq4_k_impl_bs128
// instantiated with super_block_size=QK_K, block_size=32, ntry=7.
// cy points to row_meta + blocks; we write the float row scale at cy and blocks after.
static void quantize_row_iq4_ks_impl(int n_per_row, const float * x, char * cy,
                                     float * all_scales, float * weight,
                                     const float * quant_weights) {
    const int super_block_size = QK_K;
    const int block_size = 32;
    const int ntry = 7;
    const int8_t * values = iq4k_values;
    const int8_t * shifted_values = values + 16;

    float * dptr = (float *)cy;
    block_iq4_ks * y = (block_iq4_ks *)(dptr + 1);

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        float * scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f / super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) { amax = ax; max = xb[j]; }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = -max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = iqk_best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = iqk_best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = iqk_best_index_iq4nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = iqk_best_index_iq4nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = false; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = false; }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = iqk_best_index_iq4nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = iqk_best_index_iq4nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = true; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = true; }
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            float ad = fabsf(d);
            if (ad > amax_scale) amax_scale = ad;
        }
    }

    float d_row = amax_scale / 127.f;
    *dptr = d_row;
    if (!d_row) return;

    float id = 1.f / d_row;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f / super_block_size;
        float * scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = (y[ibl].scales[ib] & 0x01) ? shifted_values : values;
            int l = iqk_nearest_int(0.5f*(id*scales[ib] + 127.f));
            if (l < 0) l = 0;
            if (l > 127) l = 127;
            l <<= 1;
            y[ibl].scales[ib] |= (uint8_t)l;
            l -= 127;
            float dl = d_row * l;
            float idl = dl ? 1.f/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            uint8_t * qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = (uint8_t)iqk_best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = (uint8_t)iqk_best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2;
}

size_t quantize_iq4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);

    const int kBlockSize = 32;
    float * all_scales = (float *)malloc(sizeof(float) * (n_per_row/kBlockSize));
    float weight[32];

    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        char * y_row = (char *)dst + row * row_size;
        const float * qw_row = imatrix;
        quantize_row_iq4_ks_impl(n_per_row, x, y_row, all_scales, weight, qw_row);
    }
    free(all_scales);
    return nrows * row_size;
}

void quantize_row_iq4_ks_ref(const float * GGML_RESTRICT x, block_iq4_ks * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_ks(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq4_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq4_ks_ref(x, (block_iq4_ks *)y, k);
}

void ggml_vec_dot_iq4_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    const int kBlockSize = 32;
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const float * dptr = (const float *)vx;
    const float d = *dptr;
    const block_iq4_ks * x = (const block_iq4_ks *)(dptr + 1);
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const int nblock = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const int8_t * qy = y[ibl].qs;
        const uint8_t * qx = x[ibl].qs;
        const float db = d * y[ibl].d;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            const float dl = db * (float)((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            int suml = 0;
            for (int j = 0; j < kBlockSize/2; ++j) {
                suml += qy[j               ] * values[qx[j] & 0xf]
                      + qy[j + kBlockSize/2] * values[qx[j] >>  4];
            }
            sumf += dl * (float)suml;
            qy += kBlockSize;
            qx += kBlockSize/2;
        }
    }
    *s = sumf;
}

// =============================================================================
// IQ3_KS — 3.1875 bpw (row_meta_size = 2: per-row ggml_half scale)
// Row layout: [ggml_half d][block_iq3_ks blocks[n_per_row/QK_K]]
// Each block_iq3_ks: 16-bit extra (8 high-bits-of-scale | 8 codebook-shift-bits)
//                  + 4 bytes scales (4-bit low parts of 8 sub-block scales)
//                  + 64 bytes qs (2-bit low indices)
//                  + 32 bytes qh (1-bit high indices)
// 8 sub-blocks of 32 elements; 5-bit signed scale per sub-block (offset by 16).
// =============================================================================

void dequantize_row_iq3_ks(const block_iq3_ks * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    const int kBlockSize = 32;
    assert(k % QK_K == 0);
    const ggml_half * dptr = (const ggml_half *)vx;
    const float d = GGML_FP16_TO_FP32(*dptr);
    const block_iq3_ks * x = (const block_iq3_ks *)(dptr + 1);
    const int nblock = k / QK_K;

    for (int ibl = 0; ibl < nblock; ++ibl) {
        float dl[8];
        for (int j = 0; j < 4; ++j) {
            int ls1 = (x[ibl].scales[j] & 0xf) | (((x[ibl].extra >> (j + 0)) & 1) << 4);
            int ls2 = (x[ibl].scales[j] >>  4) | (((x[ibl].extra >> (j + 4)) & 1) << 4);
            dl[j + 0] = d * (float)(ls1 - 16);
            dl[j + 4] = d * (float)(ls2 - 16);
        }
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        for (int i128 = 0; i128 < QK_K/128; ++i128) {
            for (int ib = 0; ib < 4; ++ib) {
                const int8_t * values = iq3nl_values + (((x[ibl].extra >> (8 + 4*i128 + ib)) & 1) << 3);
                for (int j = 0; j < kBlockSize; ++j) {
                    y[j] = dl[4*i128 + ib] *
                           (float)values[((qs[j] >> 2*ib) & 3) | (((qh[j] >> (4*i128 + ib)) & 1) << 2)];
                }
                y += kBlockSize;
            }
            qs += kBlockSize;
        }
    }
}

// Per-row imatrix-aware quantizer.  Mirrors ik_llama.cpp's quantize_row_iq3_ks_impl
// with super_block_size=QK_K, block_size=32, ntry=5.
static void quantize_row_iq3_ks_impl(int n_per_row, const float * x, char * cy,
                                     float * all_scales, float * weight,
                                     const float * quant_weights) {
    const int super_block_size = QK_K;
    const int block_size = 32;
    const int ntry = 5;
    const int8_t * values = iq3nl_values;
    const int8_t * shifted_values = values + 8;

    ggml_half * dptr = (ggml_half *)cy;
    block_iq3_ks * y = (block_iq3_ks *)(dptr + 1);

    float amax_scale = 0;
    float max_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq3_ks));
        const float * xbl = x + ibl*super_block_size;
        float * scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f / super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) { amax = ax; max = xb[j]; }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = -max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = iqk_best_index_iq3nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = iqk_best_index_iq3nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) { d = sumqx_p/sumq2_p; best = d*sumqx_p; }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = iqk_best_index_iq3nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = iqk_best_index_iq3nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = false; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = false; }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = iqk_best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = iqk_best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = true; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = true; }
            }
            if (is_shifted) y[ibl].extra |= (uint16_t)(1u << (8 + ib));
            scales[ib] = d;
            float ascale = fabsf(d);
            if (ascale > amax_scale) { amax_scale = ascale; max_scale = d; }
        }
    }

    float d_row = -max_scale / 16.f;
    *dptr = GGML_FP32_TO_FP16(d_row);
    if (!d_row) return;

    float id = 1.f / d_row;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f / super_block_size;
        float * scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = ((y[ibl].extra >> (8 + ib)) & 1) ? shifted_values : values;
            int l = iqk_nearest_int(id*scales[ib]);
            if (l < -16) l = -16;
            if (l > 15) l = 15;
            uint8_t ul = (uint8_t)(l + 16);
            y[ibl].scales[ib%4] |= (uint8_t)((ul & 0xf) << (4*(ib/4)));
            y[ibl].extra |= (uint16_t)((ul >> 4) << ib);
            float dl = d_row * (float)l;
            float idl = dl ? 1.f/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            uint8_t * qs = y[ibl].qs + (ib/4)*block_size;
            uint8_t * qh = y[ibl].qh;  // (ib/8) is always 0 for ib<8
            for (int j = 0; j < block_size; ++j) {
                int idx = iqk_best_index_iq3nl(block_values, idl*xb[j]);
                qs[j] |= (uint8_t)((idx & 3) << (2*(ib%4)));
                qh[j] |= (uint8_t)(((idx >> 2) & 1) << (ib%8));
                float w = weight[j];
                float q = block_values[idx]*l;
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
        }
    }
    if (sumq2 > 0) *dptr = GGML_FP32_TO_FP16(sumqx/sumq2);
}

size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);

    const int kBlockSize = 32;
    float * all_scales = (float *)malloc(sizeof(float) * (n_per_row/kBlockSize));
    float weight[32];

    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        char * y_row = (char *)dst + row * row_size;
        const float * qw_row = imatrix;
        quantize_row_iq3_ks_impl(n_per_row, x, y_row, all_scales, weight, qw_row);
    }
    free(all_scales);
    return nrows * row_size;
}

void quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_ks(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq3_ks(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq3_ks_ref(x, (block_iq3_ks *)y, k);
}

void ggml_vec_dot_iq3_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const ggml_half * dptr = (const ggml_half *)vx;
    const float d = GGML_FP16_TO_FP32(*dptr);
    const block_iq3_ks * x = (const block_iq3_ks *)(dptr + 1);
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const int nblock = n / QK_K;

    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float db = d * y[ibl].d;
        const uint8_t * qs = x[ibl].qs;
        const uint8_t * qh = x[ibl].qh;
        const int8_t  * q8 = y[ibl].qs;
        const uint16_t extra = x[ibl].extra;
        int32_t sum = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int nibble  = (x[ibl].scales[ib%4] >> (4*(ib/4))) & 0xf;
            const int hi      = (extra >> ib) & 1;
            const int ls      = (nibble | (hi << 4)) - 16;
            const int shift   = (extra >> (8 + ib)) & 1;
            const int8_t * values = iq3nl_values + (shift << 3);
            const int qs_off  = (ib/4) * 32;
            const int shift_l = 2*(ib%4);
            const int shift_h = ib%8;
            int32_t sumi = 0;
            for (int j = 0; j < 32; ++j) {
                int idx = ((qs[qs_off + j] >> shift_l) & 3) | (((qh[j] >> shift_h) & 1) << 2);
                sumi += q8[j] * values[idx];
            }
            sum += ls * sumi;
            q8 += 32;
        }
        sumf += db * (float)sum;
    }
    *s = sumf;
}

// =============================================================================
// IQ4_KSS — 4.0 bpw (row_meta_size = 4: per-row float scale)
// Row layout: [float d][block_iq4_kss blocks[n_per_row/QK_K]]
// Each block_iq4_kss: 32 uint32_t = 128 bytes (8 uint32_t per 32-element sub-block).
//
// Per sub-block packing: the 32 4-bit indices are grouped into 8 uint16_t pairs.
// Each uint16_t holds:
//   - bit 0: one of 8 bits of the per-sub-block scale byte (the byte spans 8 uint16_ts)
//   - bits 1..15: the 4 nibble indices, scrambled via Gray-code (j ^ (j<<1)) and
//                 forced to even parity in 16 bits via prune_iq4ks
// Decode: aux = qs & 0xfffe; aux ^= (aux >> 1) inverts the scramble.
// The closed-form inverse-Gray-code (used at encode) replaces ik_llama's 32K-entry table.
// =============================================================================

// Inverse of f(j) = j ^ (j<<1).  For 16-bit v, j[i] = XOR of v[0..i] (LSB to bit i).
// Computed via doubling: v ^= v<<1; v ^= v<<2; v ^= v<<4; v ^= v<<8.
static inline uint16_t iqk_inverse_gray(uint16_t v) {
    v ^= (uint16_t)(v << 1);
    v ^= (uint16_t)(v << 2);
    v ^= (uint16_t)(v << 4);
    v ^= (uint16_t)(v << 8);
    return v;
}

// Force even parity of the 16-bit value v by perturbing one of 4 nibbles.  Picks
// the nibble whose perturbation has the smallest weighted MSE increase at scale dl.
static uint16_t iqk_prune_iq4kss(uint16_t v, const int8_t * values,
                                  const float * x, const float * w, float dl) {
    if ((__builtin_popcount(v) & 1) == 0) return v;
    float best_score = 1e30f;
    uint8_t q4[4];
    int jbest = -1;
    uint8_t bestq = 0;
    for (int j = 0; j < 4; ++j) {
        uint8_t q = (v >> 4*j) & 0xf;
        q4[j] = q;
        int pc = __builtin_popcount(q);
        float diff0 = dl*(float)values[q] - x[j];
        int qmin = q < 2 ? 0 : q - 2;
        int qmax = q > 13 ? 15 : q + 2;
        for (int iq = qmin; iq <= qmax; ++iq) {
            uint8_t qq = (uint8_t)iq;
            if (qq == q) continue;
            int pci = __builtin_popcount(qq);
            if (((pci - pc) & 1) != 0) {
                float diff1 = dl*(float)values[qq] - x[j];
                float score = w[j]*(diff1*diff1 - diff0*diff0);
                if (score < best_score) { best_score = score; jbest = j; bestq = qq; }
            }
        }
    }
    if (jbest < 0) return v;  // no parity-flipping neighbor found; should not happen
    q4[jbest] = bestq;
    return (uint16_t)(q4[0] | (q4[1] << 4) | (q4[2] << 8) | (q4[3] << 12));
}

void dequantize_row_iq4_kss(const block_iq4_kss * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const float * dptr = (const float *)vx;
    const float d = *dptr;
    const block_iq4_kss * x = (const block_iq4_kss *)(dptr + 1);
    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;
    const int nblock = k / QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const uint16_t * qs = (const uint16_t *)x[ibl].qs;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t ls = 0;
            for (int kk = 0; kk < 8; ++kk) {
                aux16[kk] = qs[kk] & 0xfffe;
                aux16[kk] ^= (uint16_t)(aux16[kk] >> 1);
                ls |= (int16_t)((qs[kk] & 1) << kk);
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            const float dl = d * (float)((int)(ls & 254) - 127);
            for (int j = 0; j < 16; ++j) {
                y[j +  0] = dl * (float)values[aux8[j] & 0xf];
                y[j + 16] = dl * (float)values[aux8[j] >>  4];
            }
            y  += 32;
            qs += 8;
        }
    }
}

static void quantize_row_iq4_kss_impl(int n_per_row, const float * x, char * cy,
                                      float * all_scales, float * weight,
                                      const float * quant_weights) {
    const int super_block_size = QK_K;
    const int block_size = 32;
    const int ntry = 7;
    const int8_t * values = iq4k_values;
    const int8_t * shifted_values = values + 16;

    float * dptr = (float *)cy;
    *dptr = 0;
    block_iq4_kss * y = (block_iq4_kss *)(dptr + 1);

    uint16_t vps[block_size/2], vms[block_size/2], vs[block_size/2];
    float xv[4], wv[4];

    float amax_scale = 0;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_kss));
        const float * xbl = x + ibl*super_block_size;
        float * scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f / super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) { amax = ax; max = xb[j]; }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float d = -max/values[0];
            memset(vs, 0, sizeof(vs));
            for (int itry = -ntry; itry <= ntry; ++itry) {
                // Standard codebook
                float id = (itry + values[0])/max;
                float this_d = 1.f / id;
                float sumqx_p = 0, sumq2_p = 0;
                float sumqx_m = 0, sumq2_m = 0;
                for (int kk = 0; kk < block_size/4; ++kk) {
                    xv[0] =     xb[2*kk+0]; xv[1] =     xb[2*kk+0+block_size/2];
                    xv[2] =     xb[2*kk+1]; xv[3] =     xb[2*kk+1+block_size/2];
                    wv[0] = weight[2*kk+0]; wv[1] = weight[2*kk+0+block_size/2];
                    wv[2] = weight[2*kk+1]; wv[3] = weight[2*kk+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (uint16_t)(iqk_best_index_iq4nl(values,  al) << 4*j);
                        vm |= (uint16_t)(iqk_best_index_iq4nl(values, -al) << 4*j);
                    }
                    vp = iqk_prune_iq4kss(vp, values, xv, wv, this_d);
                    vm = iqk_prune_iq4kss(vm, values, xv, wv, this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j]; sumq2_p += w*q*q;
                        q = values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j]; sumq2_m += w*q*q;
                    }
                    vps[kk] = vp; vms[kk] = vm;
                }
                bool copy_p = false, copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; copy_p = true; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; copy_m = true; }
                if (copy_m) memcpy(vs, vms, sizeof(vs));
                else if (copy_p) memcpy(vs, vps, sizeof(vs));

                // Shifted codebook
                id = (itry + shifted_values[0])/max;
                this_d = 1.f / id;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int kk = 0; kk < block_size/4; ++kk) {
                    xv[0] =     xb[2*kk+0]; xv[1] =     xb[2*kk+0+block_size/2];
                    xv[2] =     xb[2*kk+1]; xv[3] =     xb[2*kk+1+block_size/2];
                    wv[0] = weight[2*kk+0]; wv[1] = weight[2*kk+0+block_size/2];
                    wv[2] = weight[2*kk+1]; wv[3] = weight[2*kk+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = id*xv[j];
                        vp |= (uint16_t)(iqk_best_index_iq4nl(shifted_values,  al) << 4*j);
                        vm |= (uint16_t)(iqk_best_index_iq4nl(shifted_values, -al) << 4*j);
                    }
                    vp = iqk_prune_iq4kss(vp, shifted_values, xv, wv, this_d);
                    vm = iqk_prune_iq4kss(vm, shifted_values, xv, wv, this_d);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = shifted_values[(vp >> 4*j) & 0xf];
                        sumqx_p += w*q*xv[j]; sumq2_p += w*q*q;
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        sumqx_m += w*q*xv[j]; sumq2_m += w*q*q;
                    }
                    vps[kk] = vp; vms[kk] = vm;
                }
                copy_p = copy_m = false;
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; copy_p = true; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; copy_m = true; }
                if (copy_m) memcpy(vs, vms, sizeof(vs));
                else if (copy_p) memcpy(vs, vps, sizeof(vs));
            }
            scales[ib] = d;
            float ad = fabsf(d);
            if (ad > amax_scale) amax_scale = ad;
        }
    }

    float d_row = amax_scale / 127.f;
    *dptr = d_row;
    if (!d_row) return;

    float id = 1.f / d_row;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        float * scales = all_scales + (super_block_size/block_size)*ibl;
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f / super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            int l = iqk_nearest_int(0.5f*(id*scales[ib] + 127.f));
            if (l < 0) l = 0;
            if (l > 127) l = 127;
            l = (l << 1) - 127;
            if (l) {
                float dl = d_row * (float)l;
                float idl = 1.f / dl;
                float mse_p = 0, mse_m = 0;
                for (int kk = 0; kk < block_size/4; ++kk) {
                    xv[0] =     xb[2*kk+0]; xv[1] =     xb[2*kk+0+block_size/2];
                    xv[2] =     xb[2*kk+1]; xv[3] =     xb[2*kk+1+block_size/2];
                    wv[0] = weight[2*kk+0]; wv[1] = weight[2*kk+0+block_size/2];
                    wv[2] = weight[2*kk+1]; wv[3] = weight[2*kk+1+block_size/2];
                    uint16_t vp = 0, vm = 0;
                    for (int j = 0; j < 4; ++j) {
                        float al = idl*xv[j];
                        vp |= (uint16_t)(iqk_best_index_iq4nl(values,         al) << 4*j);
                        vm |= (uint16_t)(iqk_best_index_iq4nl(shifted_values, al) << 4*j);
                    }
                    vp = iqk_prune_iq4kss(vp,         values, xv, wv, dl);
                    vm = iqk_prune_iq4kss(vm, shifted_values, xv, wv, dl);
                    for (int j = 0; j < 4; ++j) {
                        float w = wv[j];
                        float q = values[(vp >> 4*j) & 0xf];
                        mse_p += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                        q = shifted_values[(vm >> 4*j) & 0xf];
                        mse_m += w*(xv[j] - dl*q)*(xv[j] - dl*q);
                    }
                    vps[kk] = vp; vms[kk] = vm;
                }
                const uint16_t * v = vps;
                const int8_t * block_values = values;
                if (mse_m < mse_p) { v = vms; block_values = shifted_values; }
                for (int kk = 0; kk < block_size/4; ++kk) {
                    xv[0] =     xb[2*kk+0]; xv[1] =     xb[2*kk+0+block_size/2];
                    xv[2] =     xb[2*kk+1]; xv[3] =     xb[2*kk+1+block_size/2];
                    wv[0] = weight[2*kk+0]; wv[1] = weight[2*kk+0+block_size/2];
                    wv[2] = weight[2*kk+1]; wv[3] = weight[2*kk+1+block_size/2];
                    for (int j = 0; j < 4; ++j) {
                        float q = block_values[(v[kk] >> 4*j) & 0xf] * l;
                        sumqx += wv[j]*q*xv[j];
                        sumq2 += wv[j]*q*q;
                    }
                }
                int ll = l + 127;
                if (mse_m < mse_p) ll |= 1;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int kk = 0; kk < block_size/4; ++kk) {
                    uint16_t scrambled = iqk_inverse_gray(v[kk] & 0x7fff);
                    q16[kk] = (uint16_t)((scrambled << 1) | ((ll >> kk) & 1));
                }
            } else {
                int ll = l + 127;
                uint16_t * q16 = (uint16_t *)y[ibl].qs + (block_size/4)*ib;
                for (int kk = 0; kk < block_size/4; ++kk) {
                    q16[kk] = (uint16_t)((ll >> kk) & 1);
                }
            }
        }
    }
    if (sumq2 > 0) *dptr = sumqx/sumq2 * 1.01f;
}

size_t quantize_iq4_kss(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);

    const int kBlockSize = 32;
    float * all_scales = (float *)malloc(sizeof(float) * (n_per_row/kBlockSize));
    float weight[32];

    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        char * y_row = (char *)dst + row * row_size;
        const float * qw_row = imatrix;
        quantize_row_iq4_kss_impl(n_per_row, x, y_row, all_scales, weight, qw_row);
    }
    free(all_scales);
    return nrows * row_size;
}

void quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_kss(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq4_kss(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq4_kss_ref(x, (block_iq4_kss *)y, k);
}

void ggml_vec_dot_iq4_kss_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                                const void * GGML_RESTRICT vx, size_t bx,
                                const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const float * dptr = (const float *)vx;
    const float d = *dptr;
    const block_iq4_kss * x = (const block_iq4_kss *)(dptr + 1);
    const block_q8_K  * y = (const block_q8_K  *)vy;
    const int nblock = n / QK_K;

    uint16_t aux16[8];
    const uint8_t * aux8 = (const uint8_t *)aux16;

    float sumf = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float db = d * y[ibl].d;
        const uint16_t * qs = (const uint16_t *)x[ibl].qs;
        const int8_t  * q8 = y[ibl].qs;
        int32_t sum = 0;
        for (int ib = 0; ib < QK_K/32; ++ib) {
            int16_t ls = 0;
            for (int kk = 0; kk < 8; ++kk) {
                aux16[kk] = qs[kk] & 0xfffe;
                aux16[kk] ^= (uint16_t)(aux16[kk] >> 1);
                ls |= (int16_t)((qs[kk] & 1) << kk);
            }
            const int8_t * values = iq4k_values + ((ls & 1) << 4);
            const int dl = (int)(ls & 254) - 127;
            int32_t sumi = 0;
            for (int j = 0; j < 16; ++j) {
                sumi += q8[j     ] * (int32_t)values[aux8[j] & 0xf];
                sumi += q8[j + 16] * (int32_t)values[aux8[j] >>  4];
            }
            sum += dl * sumi;
            qs += 8;
            q8 += 32;
        }
        sumf += db * (float)sum;
    }
    *s = sumf;
}

// ============================================================
// IQ5_K: ik_llama 5-bit imatrix-aware weight quant (5.50 bpw)
// Lifted from ikllama/main ggml/src/iqk/iqk_quantize.cpp
// C++ stripped: namespace removed, bool→int, QHelper→row loop
// ============================================================

static const int8_t iq5nl_values[64] = {
    -126, -114, -103,  -92,  -83,  -74,  -65,  -57,  -50,  -43,  -36,  -30,  -24,  -18,  -12,   -6,
      -1,    5,   11,   17,   23,   29,   36,   43,   51,   59,   68,   77,   87,   97,  109,  121,
    -124, -112, -101,  -90,  -81,  -72,  -63,  -55,  -48,  -41,  -34,  -28,  -22,  -16,  -10,   -4,
       1,    7,   13,   19,   25,   31,   38,   45,   53,   61,   70,   79,   89,   99,  111,  123,
};

void dequantize_row_iq5_k(const block_iq5_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;
        uint16_t extra = x[i].extra;
        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * (float)(((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            float dl2 = d * (float)(((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            float dl3 = d * (float)(((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            float dl4 = d * (float)(((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                y[j+16] = dl2 * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                y[j+32] = dl3 * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                y[j+48] = dl4 * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            y  += 64;
            qs += 32;
            extra >>= 4;
            shift += 2;
            if (shift == 8) { qh += 32; shift = 0; }
        }
    }
}

static const int8_t iq5nl_index[248] = {
     0,  0,  0,  0,  0,  0, 32,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 33, 33,  2,  2,  2,  2,  2,  2,  2,  2,  2, 34, 34,  3,  3,
     3,  3,  3,  3,  3,  3, 35, 35,  4,  4,  4,  4,  4,  4,  4, 36, 36,  5,  5,  5,  5,  5,  5,  5, 37, 37,  6,  6,  6,  6,  6,  6,
     6, 38,  7,  7,  7,  7,  7,  7, 39, 39,  8,  8,  8,  8,  8, 40, 40,  9,  9,  9,  9,  9, 41, 41, 10, 10, 10, 10, 10, 42, 11, 11,
    11, 11, 11, 43, 12, 12, 12, 12, 12, 44, 13, 13, 13, 13, 13, 45, 14, 14, 14, 14, 14, 46, 15, 15, 15, 15, 47, 47, 16, 16, 16, 16,
    48, 17, 17, 17, 17, 17, 49, 18, 18, 18, 18, 18, 50, 19, 19, 19, 19, 19, 51, 20, 20, 20, 20, 20, 52, 21, 21, 21, 21, 21, 53, 53,
    22, 22, 22, 22, 22, 54, 54, 23, 23, 23, 23, 23, 23, 55, 24, 24, 24, 24, 24, 24, 24, 56, 25, 25, 25, 25, 25, 25, 25, 57, 57, 26,
    26, 26, 26, 26, 26, 26, 58, 58, 27, 27, 27, 27, 27, 27, 27, 27, 59, 28, 28, 28, 28, 28, 28, 28, 28, 28, 60, 29, 29, 29, 29, 29,
    29, 29, 29, 29, 29, 61, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 62, 31, 31, 31, 31, 31, 31,
};

static inline int best_index_iq5nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 247) return ix < 0 ? 0 : 31;
    ix = iq5nl_index[ix];
    return ix < 32 ? ix : x - values[ix-32] < values[ix-31] - x ? ix-32 : ix-31;
}

static void quantize_row_iq5_k_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy,
                                    int n_per_row, const float * quant_weights) {
    const int ntry = 5;
    const float step = 1.f;
    block_iq5_k * y = (block_iq5_k *)vy;
    float scales[QK_K/16];
    float weight[16];
    const int8_t * shifted_values = iq5nl_values + 32;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq5_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;
        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) { amax = ax; max = xb[j]; }
            }
            if (amax < 1e-16f) { scales[ib] = 0; continue; }
            float d = ntry > 0 ? -max/iq5nl_values[0] : max/iq5nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0, sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j]; float al = id*xb[j];
                int l = best_index_iq5nl(iq5nl_values, al);
                float q = iq5nl_values[l]; sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = best_index_iq5nl(iq5nl_values, -al);
                q = iq5nl_values[l]; sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; }
            int is_shifted = 0;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + iq5nl_values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j]; float al = id*xb[j];
                    int l = best_index_iq5nl(iq5nl_values, al);
                    float q = iq5nl_values[l]; sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = best_index_iq5nl(iq5nl_values, -al);
                    q = iq5nl_values[l]; sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = 0; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = 0; }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j]; float al = id*xb[j];
                    int l = best_index_iq5nl(shifted_values, al);
                    float q = shifted_values[l]; sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = best_index_iq5nl(shifted_values, -al);
                    q = shifted_values[l]; sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = 1; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = 1; }
            }
            if (d) {
                const int8_t * bv = is_shifted ? shifted_values : iq5nl_values;
                float sumqx = 0, sumq2 = 0; id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j]; float al = id*xb[j];
                    int l = best_index_iq5nl(bv, al); float q = bv[l];
                    sumqx += w*q*xb[j]; sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (uint16_t)(1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) { max_abs_scale = abs_scale; max_scale = scales[ib]; }
        }

        if (!max_abs_scale) continue;
        float d = -max_scale/32;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        y[ibl].extra = extra;
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = iqk_nearest_int(id*scales[ib]);
            ls = MAX(-32, MIN(31, ls));
            int uls = ls + 32;
            y[ibl].scales_l[ib/2] |= (uint8_t)((uls & 0xf) << 4*(ib%2));
            y[ibl].scales_h[ib/4] |= (uint8_t)((uls >>  4) << 2*(ib%4));
            float dl = d * ls;
            if (dl) {
                const int8_t * bv = y[ibl].extra & (1 << ib) ? shifted_values : iq5nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    int ibest = best_index_iq5nl(bv, idl*xb[j]);
                    qs[j] |= (uint8_t)((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= (uint8_t)((ibest >>  4) << (ib32%8));
                    float q = bv[ibest]*(float)ls;
                    sumqx += weight[j]*q*xb[j]; sumq2 += weight[j]*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(sumqx/sumq2);
    }
}

void quantize_row_iq5_k_ref(const float * GGML_RESTRICT x, block_iq5_k * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq5_k(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq5_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq5_k_ref(x, (block_iq5_k *)y, k);
}

size_t quantize_iq5_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                      int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ5_K, n_per_row);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq5_k_impl(src + row*n_per_row, (char *)dst + row*row_size, n_per_row, imatrix);
    }
    return nrows * row_size;
}

void ggml_vec_dot_iq5_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(bs);
    const int nb = n / QK_K;
    const block_iq5_k * x = (const block_iq5_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * sl = x[i].scales_l;
        const uint8_t * sh = x[i].scales_h;
        const int8_t  * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        int shift = 0, sumb = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            int dl1 = (int)(((sl[2*ib64+0] & 0xf) | ((sh[ib64] << 4) & 0x30)) - 32);
            int dl2 = (int)(((sl[2*ib64+0] >>  4) | ((sh[ib64] << 2) & 0x30)) - 32);
            int dl3 = (int)(((sl[2*ib64+1] & 0xf) | ((sh[ib64] >> 0) & 0x30)) - 32);
            int dl4 = (int)(((sl[2*ib64+1] >>  4) | ((sh[ib64] >> 2) & 0x30)) - 32);
            const int8_t * values1 = iq5nl_values + ((extra & 1) << 5);
            const int8_t * values2 = iq5nl_values + ((extra & 2) << 4);
            const int8_t * values3 = iq5nl_values + ((extra & 4) << 3);
            const int8_t * values4 = iq5nl_values + ((extra & 8) << 2);
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * (int32_t)values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 1) << 4)];
                sumi2 += q8[j+16] * (int32_t)values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 1) << 4)];
                sumi3 += q8[j+32] * (int32_t)values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 2) << 3)];
                sumi4 += q8[j+48] * (int32_t)values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 2) << 3)];
            }
            sumb += dl1*sumi1 + dl2*sumi2 + dl3*sumi3 + dl4*sumi4;
            q8 += 64; qs += 32; extra >>= 4; shift += 2;
        }
        sumf += d * sumb;
    }
    *s = sumf;
}

// ============================================================
// IQ6_K: ik_llama 6-bit imatrix-aware weight quant (6.625 bpw)
// Lifted from ikllama/ik/iq6_k ggml/src/iqk/iqk_quantize.cpp
// vec_dot written from scratch (ikllama has GGML_ABORT there)
// Dequantize uses int8 table lookup instead of polynomial approx
// ============================================================

static const int8_t iq6nl_values[128] = {
    -127, -121, -115, -109, -104,  -98,  -93,  -88,  -84,  -79,  -74,  -70,  -66,  -62,  -58,  -54,
     -51,  -47,  -44,  -40,  -37,  -34,  -31,  -28,  -25,  -22,  -19,  -16,  -13,  -11,   -8,   -5,
      -2,    0,    3,    6,    9,   12,   14,   17,   20,   23,   27,   30,   33,   36,   40,   44,
      47,   51,   55,   59,   63,   68,   72,   77,   82,   87,   92,   98,  103,  109,  115,  121,
    -126, -120, -114, -108, -103,  -97,  -92,  -87,  -83,  -78,  -73,  -69,  -65,  -61,  -57,  -53,
     -50,  -46,  -43,  -39,  -36,  -33,  -30,  -27,  -24,  -21,  -18,  -15,  -12,  -10,   -7,   -4,
      -1,    1,    4,    7,   10,   13,   15,   18,   21,   24,   28,   31,   34,   37,   41,   45,
      48,   52,   56,   60,   64,   69,   73,   78,   83,   88,   93,   99,  104,  110,  116,  122,
};

void dequantize_row_iq6_k(const block_iq6_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;
        uint16_t extra = x[i].extra;
        int shift = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            float dl1 = d * sl[4*ib64 + 0];
            float dl2 = d * sl[4*ib64 + 1];
            float dl3 = d * sl[4*ib64 + 2];
            float dl4 = d * sl[4*ib64 + 3];
            const int8_t * values1 = iq6nl_values + ((extra & 1) ? 64 : 0);
            const int8_t * values2 = iq6nl_values + ((extra & 2) ? 64 : 0);
            const int8_t * values3 = iq6nl_values + ((extra & 4) ? 64 : 0);
            const int8_t * values4 = iq6nl_values + ((extra & 8) ? 64 : 0);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4)];
                y[j+16] = dl2 * values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4)];
                y[j+32] = dl3 * values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2)];
                y[j+48] = dl4 * values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2)];
            }
            y  += 64; qs += 32; extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }
    }
}

static const uint8_t iq6nl_index[249] = {
     0,   0,   0,  64,   1,   1,   1,   1,   1,  65,   2,   2,   2,   2,   2,  66,
     3,   3,   3,   3,  67,  67,   4,   4,   4,   4,  68,   5,   5,   5,   5,  69,
    69,   6,   6,   6,  70,  70,   7,   7,   7,  71,   8,   8,   8,  72,  72,   9,
     9,   9,  73,  73,  10,  10,  10,  74,  11,  11,  11,  75,  12,  12,  12,  76,
    13,  13,  13,  77,  14,  14,  14,  78,  15,  15,  79,  79,  16,  16,  80,  17,
    17,  81,  81,  18,  18,  82,  19,  19,  83,  83,  20,  84,  84,  21,  85,  85,
    22,  86,  86,  23,  87,  87,  24,  88,  88,  25,  89,  89,  26,  90,  90,  27,
    91,  91,  28,  92,  29,  93,  93,  30,  94,  94,  31,  95,  95,  32,  96,  33,
    97,  97,  34,  98,  98,  35,  99,  99,  36, 100, 100,  37, 101,  38, 102, 102,
    39, 103, 103,  40, 104, 104,  41,  41, 105,  42,  42, 106, 106,  43, 107, 107,
    44, 108, 108,  45,  45, 109,  46,  46,  46, 110,  47,  47, 111, 111,  48,  48,
   112,  49,  49,  49, 113,  50,  50,  50, 114,  51,  51,  51, 115,  52,  52,  52,
   116, 116,  53,  53,  53, 117,  54,  54,  54, 118, 118,  55,  55,  55, 119, 119,
    56,  56,  56, 120, 120,  57,  57,  57, 121, 121,  58,  58,  58,  58, 122,  59,
    59,  59,  59, 123, 123,  60,  60,  60,  60, 124,  61,  61,  61,  61,  61, 125,
    62,  62,  62,  62,  62, 126,  63,  63,  63,
};

static inline int best_index_iq6nl(const float * values, float x) {
    int ix = (int)(x - values[0]);
    if (ix < 0 || ix >= 249) return ix < 0 ? 0 : 63;
    ix = iq6nl_index[ix];
    return ix < 64 ? ix : x - values[ix-64] < values[ix-63] - x ? ix-64 : ix-63;
}

static void quantize_row_iq6_k_impl(const float * GGML_RESTRICT x, void * GGML_RESTRICT vy,
                                    int n_per_row, const float * quant_weights,
                                    const float * values, const float * shifted_values) {
    const int ntry = 5;
    const float step = 1.f;
    block_iq6_k * y = (block_iq6_k *)vy;
    float scales[QK_K/16];
    float weight[16];

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq6_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);
        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 2*sumx2/QK_K;
        float max_scale = 0, max_abs_scale = 0;
        uint16_t extra = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) { amax = ax; max = xb[j]; }
            }
            if (amax < 1e-16f) { scales[ib] = 0; continue; }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0, sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j]; float al = id*xb[j];
                int l = best_index_iq6nl(values, al);
                float q = values[l]; sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                l = best_index_iq6nl(values, -al);
                q = values[l]; sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; }
            int is_shifted = 0;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry*step + values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j]; float al = id*xb[j];
                    int l = best_index_iq6nl(values, al);
                    float q = values[l]; sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = best_index_iq6nl(values, -al);
                    q = values[l]; sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = 0; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = 0; }
                id = (itry*step + shifted_values[0])/max;
                sumqx_p = sumq2_p = sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j]; float al = id*xb[j];
                    int l = best_index_iq6nl(shifted_values, al);
                    float q = shifted_values[l]; sumqx_p += w*q*xb[j]; sumq2_p += w*q*q;
                    l = best_index_iq6nl(shifted_values, -al);
                    q = shifted_values[l]; sumqx_m += w*q*xb[j]; sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) { d = sumqx_p/sumq2_p; best = d*sumqx_p; is_shifted = 1; }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) { d = sumqx_m/sumq2_m; best = d*sumqx_m; is_shifted = 1; }
            }
            if (d) {
                const float * bv = is_shifted ? shifted_values : values;
                float sumqx = 0, sumq2 = 0; id = 1/d;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j]; float al = id*xb[j];
                    int l = best_index_iq6nl(bv, al); float q = bv[l];
                    sumqx += w*q*xb[j]; sumq2 += w*q*q;
                }
                if (sumq2 > 0) d = sumqx/sumq2;
            }
            scales[ib] = d;
            if (is_shifted) extra |= (uint16_t)(1 << ib);
            float abs_scale = fabsf(scales[ib]);
            if (abs_scale > max_abs_scale) { max_abs_scale = abs_scale; max_scale = scales[ib]; }
        }

        if (!max_abs_scale) continue;
        float d = -max_scale/127;
        y[ibl].d = GGML_FP32_TO_FP16(d);
        y[ibl].extra = extra;
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = iqk_nearest_int(id*scales[ib]);
            ls = MAX(-127, MIN(127, ls));
            y[ibl].scales[ib] = (int8_t)ls;
            float dl = d * ls;
            if (dl) {
                const float * bv = y[ibl].extra & (1 << ib) ? shifted_values : values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/2) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/4) + offset;
                for (int j = 0; j < 16; ++j) {
                    int ibest = best_index_iq6nl(bv, idl*xb[j]);
                    qs[j] |= (uint8_t)((ibest & 0xf) << 4*(ib32%2));
                    qh[j] |= (uint8_t)((ibest >>  4) << 2*(ib32%4));
                    float q = bv[ibest]*(float)ls;
                    sumqx += weight[j]*q*xb[j]; sumq2 += weight[j]*q*q;
                }
            }
        }
        if (sumq2 > 0) y[ibl].d = GGML_FP32_TO_FP16(sumqx/sumq2);
    }
}

void quantize_row_iq6_k_ref(const float * GGML_RESTRICT x, block_iq6_k * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq6_k(x, (void *)y, 1, k, NULL);
}

void quantize_row_iq6_k(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq6_k_ref(x, (block_iq6_k *)y, k);
}

size_t quantize_iq6_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                      int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    float fvalues[128];
    for (int i = 0; i < 64; ++i) {
        fvalues[i]    = (float)iq6nl_values[i];
        fvalues[i+64] = (float)iq6nl_values[i] + 1.0f;
    }
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ6_K, n_per_row);
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq6_k_impl(src + row*n_per_row, (char *)dst + row*row_size,
                                n_per_row, imatrix, fvalues, fvalues + 64);
    }
    return nrows * row_size;
}

void ggml_vec_dot_iq6_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                              const void * GGML_RESTRICT vx, size_t bx,
                              const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    GGML_UNUSED(nrc); GGML_UNUSED(bx); GGML_UNUSED(by); GGML_UNUSED(bs);
    const int nb = n / QK_K;
    const block_iq6_k * x = (const block_iq6_k *)vx;
    const block_q8_K  * y = (const block_q8_K  *)vy;
    float sumf = 0;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d) * y[i].d;
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sl = x[i].scales;
        const int8_t  * q8 = y[i].qs;
        uint16_t extra = x[i].extra;
        int shift = 0, sumb = 0;
        for (int ib64 = 0; ib64 < QK_K/64; ++ib64) {
            int dl1 = (int)sl[4*ib64 + 0];
            int dl2 = (int)sl[4*ib64 + 1];
            int dl3 = (int)sl[4*ib64 + 2];
            int dl4 = (int)sl[4*ib64 + 3];
            const int8_t * values1 = iq6nl_values + ((extra & 1) ? 64 : 0);
            const int8_t * values2 = iq6nl_values + ((extra & 2) ? 64 : 0);
            const int8_t * values3 = iq6nl_values + ((extra & 4) ? 64 : 0);
            const int8_t * values4 = iq6nl_values + ((extra & 8) ? 64 : 0);
            int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
            for (int j = 0; j < 16; ++j) {
                sumi1 += q8[j+ 0] * (int32_t)values1[(qs[j+ 0] & 0xf) | (((qh[j+ 0] >> shift) & 0x03) << 4)];
                sumi2 += q8[j+16] * (int32_t)values2[(qs[j+16] & 0xf) | (((qh[j+16] >> shift) & 0x03) << 4)];
                sumi3 += q8[j+32] * (int32_t)values3[(qs[j+ 0] >>  4) | (((qh[j+ 0] >> shift) & 0x0c) << 2)];
                sumi4 += q8[j+48] * (int32_t)values4[(qs[j+16] >>  4) | (((qh[j+16] >> shift) & 0x0c) << 2)];
            }
            sumb += dl1*sumi1 + dl2*sumi2 + dl3*sumi3 + dl4*sumi4;
            q8 += 64; qs += 32; extra >>= 4;
            shift += 4;
            if (shift == 8) { qh += 32; shift = 0; }
        }
        sumf += d * sumb;
    }
    *s = sumf;
}
