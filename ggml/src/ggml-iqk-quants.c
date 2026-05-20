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
