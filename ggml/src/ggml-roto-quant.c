// ggml-roto-quant.c — RotorQuant KV cache CPU encode/decode
//
// Implements PlanarQuant (Givens-rotation) and IsoQuant (Hadamard rotation) for
// KV cache compression. CPU code handles encode + decode of the block format;
// GPU decode (FA kernel) is in ggml-cuda/fattn-common.cuh.
//
// Block layouts (see ggml-common.h for canonical struct definitions):
//   planar3_0: norm(fp16) + 2-bit mag[128] + 1-bit sign[128] = 50 bytes (3.125 bpv)
//   planar4_0: norm(fp16) + rnorm(fp16) + 4-bit idx[128] nibble-packed = 68 bytes (4.25 bpv)
//   iso3_0:    identical layout to planar3_0 (rotation differs at encode context)
//   iso4_0:    identical layout to planar4_0 (rotation differs at encode context)
//
// Source: carlosfundora/llama.cpp-1-bit-turbo @85ba5b945 (April 16 2025)
//         ggml-planar-quant.c + ggml-iso-quant.c, lifted to yggdrasil Phase 4a-1.
//         Type IDs renumbered: PLANAR3_0(44)→RQ_PLANAR3_0(72), PLANAR4_0(45)→RQ_PLANAR4_0(73),
//                              ISO3_0(46)→RQ_ISO3_0(74), ISO4_0(47)→RQ_ISO4_0(75).

#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-impl.h"

#include <math.h>
#include <string.h>
#include <assert.h>
#include <float.h>

// ---------------------------------------------------------------------------
// PlanarQuant 3-bit
// ---------------------------------------------------------------------------

// Lloyd-Max-optimal centroids for 4-level uniform distribution on [0,1]:
// boundaries at 0.25, 0.50, 0.75 → centroids 0.125, 0.375, 0.625, 0.875
static const float PLANAR3_MAG_CENTROIDS[4] = { 0.125f, 0.375f, 0.625f, 0.875f };

static inline int planar3_mag_index(float abs_norm) {
    int idx = (int)(abs_norm * 4.0f);
    if (idx < 0) idx = 0;
    if (idx > 3) idx = 3;
    return idx;
}

void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x,
                                 block_planar3_0 * GGML_RESTRICT y,
                                 int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    const int nb = (int)(k / QK_PLANAR3);
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PLANAR3; j++) {
            float av = fabsf(x[i * QK_PLANAR3 + j]);
            if (av > norm) norm = av;
        }
        y[i].norm = GGML_FP32_TO_FP16(norm);
        memset(y[i].qs,    0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));
        if (norm == 0.0f) continue;
        const float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_PLANAR3; j++) {
            const float val  = x[i * QK_PLANAR3 + j];
            const int   sign = (val < 0.0f) ? 1 : 0;
            const float anv  = fabsf(val) * inv_norm;
            const int   midx = planar3_mag_index(anv);
            y[i].qs[j / 4]    |= (uint8_t)((midx & 0x3) << ((j % 4) * 2));
            if (sign) y[i].signs[j / 8] |= (uint8_t)(1 << (j % 8));
        }
    }
}

void quantize_row_planar3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_planar3_0_ref(x, (block_planar3_0 *)y, k);
}

void dequantize_row_planar3_0(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR3 == 0);
    const int nb = (int)(k / QK_PLANAR3);
    const block_planar3_0 * x = (const block_planar3_0 *)vx;
    for (int i = 0; i < nb; i++) {
        const float norm = GGML_FP16_TO_FP32(x[i].norm);
        for (int j = 0; j < QK_PLANAR3; j++) {
            const int midx = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            const int sign = (x[i].signs[j / 8] >> (j % 8)) & 0x1;
            const float mag = PLANAR3_MAG_CENTROIDS[midx] * norm;
            y[i * QK_PLANAR3 + j] = sign ? -mag : mag;
        }
    }
}

size_t quantize_planar3_0(const float * src, void * dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR3 == 0);
    const size_t row_size = (n_per_row / QK_PLANAR3) * sizeof(block_planar3_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_planar3_0_ref(src + r * n_per_row,
                                   (block_planar3_0 *)((char *)dst + r * row_size),
                                   n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}

// ---------------------------------------------------------------------------
// PlanarQuant 4-bit
// ---------------------------------------------------------------------------

void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x,
                                 block_planar4_0 * GGML_RESTRICT y,
                                 int64_t k) {
    assert(k % QK_PLANAR4 == 0);
    const int nb = (int)(k / QK_PLANAR4);
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_PLANAR4; j++) {
            float av = fabsf(x[i * QK_PLANAR4 + j]);
            if (av > norm) norm = av;
        }
        y[i].norm  = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(norm > 0.0f ? 1.0f / norm : 0.0f);
        memset(y[i].qs, 0, sizeof(y[i].qs));
        if (norm == 0.0f) continue;
        const float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_PLANAR4; j++) {
            float fq = x[i * QK_PLANAR4 + j] * inv_norm * 7.5f + 7.5f;
            int   q  = (int)(fq + 0.5f);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;
            if (j % 2 == 0) y[i].qs[j / 2]  = (uint8_t)(q & 0xF);
            else             y[i].qs[j / 2] |= (uint8_t)((q & 0xF) << 4);
        }
    }
}

void quantize_row_planar4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_planar4_0_ref(x, (block_planar4_0 *)y, k);
}

void dequantize_row_planar4_0(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_PLANAR4 == 0);
    const int nb = (int)(k / QK_PLANAR4);
    const block_planar4_0 * x = (const block_planar4_0 *)vx;
    for (int i = 0; i < nb; i++) {
        const float scale = GGML_FP16_TO_FP32(x[i].norm) / 7.5f;
        for (int j = 0; j < QK_PLANAR4; j++) {
            const int q = (j % 2 == 0) ? (x[i].qs[j / 2] & 0xF) : ((x[i].qs[j / 2] >> 4) & 0xF);
            y[i * QK_PLANAR4 + j] = ((float)q - 7.5f) * scale;
        }
    }
}

size_t quantize_planar4_0(const float * src, void * dst,
                           int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_PLANAR4 == 0);
    const size_t row_size = (n_per_row / QK_PLANAR4) * sizeof(block_planar4_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_planar4_0_ref(src + r * n_per_row,
                                   (block_planar4_0 *)((char *)dst + r * row_size),
                                   n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}

// ---------------------------------------------------------------------------
// IsoQuant 3-bit (identical CPU encode/decode to planar3; rotation context differs)
// ---------------------------------------------------------------------------

static const float ISO3_MAG_CENTROIDS[4] = { 0.125f, 0.375f, 0.625f, 0.875f };

static inline int iso3_mag_index(float abs_norm) {
    int idx = (int)(abs_norm * 4.0f);
    if (idx < 0) idx = 0;
    if (idx > 3) idx = 3;
    return idx;
}

void quantize_row_iso3_0_ref(const float * GGML_RESTRICT x,
                               block_iso3_0 * GGML_RESTRICT y,
                               int64_t k) {
    assert(k % QK_ISO3 == 0);
    const int nb = (int)(k / QK_ISO3);
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_ISO3; j++) {
            float av = fabsf(x[i * QK_ISO3 + j]);
            if (av > norm) norm = av;
        }
        y[i].norm = GGML_FP32_TO_FP16(norm);
        memset(y[i].qs,    0, sizeof(y[i].qs));
        memset(y[i].signs, 0, sizeof(y[i].signs));
        if (norm == 0.0f) continue;
        const float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_ISO3; j++) {
            const float val  = x[i * QK_ISO3 + j];
            const int   sign = (val < 0.0f) ? 1 : 0;
            const float anv  = fabsf(val) * inv_norm;
            const int   midx = iso3_mag_index(anv);
            y[i].qs[j / 4]    |= (uint8_t)((midx & 0x3) << ((j % 4) * 2));
            if (sign) y[i].signs[j / 8] |= (uint8_t)(1 << (j % 8));
        }
    }
}

void quantize_row_iso3_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iso3_0_ref(x, (block_iso3_0 *)y, k);
}

void dequantize_row_iso3_0(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ISO3 == 0);
    const int nb = (int)(k / QK_ISO3);
    const block_iso3_0 * x = (const block_iso3_0 *)vx;
    for (int i = 0; i < nb; i++) {
        const float norm = GGML_FP16_TO_FP32(x[i].norm);
        for (int j = 0; j < QK_ISO3; j++) {
            const int midx = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            const int sign = (x[i].signs[j / 8] >> (j % 8)) & 0x1;
            const float mag = ISO3_MAG_CENTROIDS[midx] * norm;
            y[i * QK_ISO3 + j] = sign ? -mag : mag;
        }
    }
}

size_t quantize_iso3_0(const float * src, void * dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO3 == 0);
    const size_t row_size = (n_per_row / QK_ISO3) * sizeof(block_iso3_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_iso3_0_ref(src + r * n_per_row,
                                (block_iso3_0 *)((char *)dst + r * row_size),
                                n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}

// ---------------------------------------------------------------------------
// IsoQuant 4-bit (identical CPU encode/decode to planar4; rotation context differs)
// ---------------------------------------------------------------------------

void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x,
                               block_iso4_0 * GGML_RESTRICT y,
                               int64_t k) {
    assert(k % QK_ISO4 == 0);
    const int nb = (int)(k / QK_ISO4);
    for (int i = 0; i < nb; i++) {
        float norm = 0.0f;
        for (int j = 0; j < QK_ISO4; j++) {
            float av = fabsf(x[i * QK_ISO4 + j]);
            if (av > norm) norm = av;
        }
        y[i].norm  = GGML_FP32_TO_FP16(norm);
        y[i].rnorm = GGML_FP32_TO_FP16(norm > 0.0f ? 1.0f / norm : 0.0f);
        memset(y[i].qs, 0, sizeof(y[i].qs));
        if (norm == 0.0f) continue;
        const float inv_norm = 1.0f / norm;
        for (int j = 0; j < QK_ISO4; j++) {
            float fq = x[i * QK_ISO4 + j] * inv_norm * 7.5f + 7.5f;
            int   q  = (int)(fq + 0.5f);
            if (q < 0)  q = 0;
            if (q > 15) q = 15;
            if (j % 2 == 0) y[i].qs[j / 2]  = (uint8_t)(q & 0xF);
            else             y[i].qs[j / 2] |= (uint8_t)((q & 0xF) << 4);
        }
    }
}

void quantize_row_iso4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iso4_0_ref(x, (block_iso4_0 *)y, k);
}

void dequantize_row_iso4_0(const void * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_ISO4 == 0);
    const int nb = (int)(k / QK_ISO4);
    const block_iso4_0 * x = (const block_iso4_0 *)vx;
    for (int i = 0; i < nb; i++) {
        const float scale = GGML_FP16_TO_FP32(x[i].norm) / 7.5f;
        for (int j = 0; j < QK_ISO4; j++) {
            const int q = (j % 2 == 0) ? (x[i].qs[j / 2] & 0xF) : ((x[i].qs[j / 2] >> 4) & 0xF);
            y[i * QK_ISO4 + j] = ((float)q - 7.5f) * scale;
        }
    }
}

size_t quantize_iso4_0(const float * src, void * dst,
                        int64_t nrows, int64_t n_per_row, const float * imatrix) {
    (void)imatrix;
    assert(n_per_row % QK_ISO4 == 0);
    const size_t row_size = (n_per_row / QK_ISO4) * sizeof(block_iso4_0);
    for (int64_t r = 0; r < nrows; r++) {
        quantize_row_iso4_0_ref(src + r * n_per_row,
                                (block_iso4_0 *)((char *)dst + r * row_size),
                                n_per_row);
    }
    return (size_t)(nrows * (int64_t)row_size);
}
