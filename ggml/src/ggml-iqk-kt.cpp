// IQ4_KT (and future IQ_KT family) ported from ik_llama.cpp.
//
// Architecture summary:
//   The codebook is NOT stored — it's regenerated on demand via a deterministic
//   bit-mixing function (iqkt_gen_group_int from ggml-iqk-kt-family.hpp).
//   A 15-bit index + offset selects a group_size-element float vector via a
//   hash-like state machine.
//
//   Quantize: nearest-neighbor search over the implicit 32K codebook per group
//   (~140ms for 4096 elements with brute force; soft-bin clustering ~100× faster).
//   Dequant: trivial — just call the formula.
//
// IQ4_KT specifically: IQ4KTParams = IQKTParams<GROUP_SIZE=4, NUM_BITS=15, IS_ABS=false>.
//   kNumVal = 32768.  Two implicit codebooks (offset 4096 vs 36864) selected
//   per-sub-block by shb[ib] & 1.
//
// Block layout (128 bytes per QK_K=256 elements):
//   shb[0..7]   = 8 uint32_t (32 B): bit 0 = "use offset2" flag, bits 1..7 = signed
//                 6-bit-offset-by-64 scale, bits 8..31 = 24 high bits across 4 groups
//   ql[0..63]   = 64 B: 8-bit low part of 15-bit index, one byte per group
//                 (kNumGroups = 32 groups per superblock; 8 sub-blocks × 4 groups)
//   qh[0..15]   = 16 B: 4-bit mid part of index, two groups packed per byte
//   bytes 80..127 unused (block size is allocated padded to 128 B).
//
// Row layout: [float row_scale][block_iq4_kt blocks[n_per_row/QK_K]]; row_meta=4.

#include "ggml-iqk-kt-family.hpp"

#include "ggml-impl.h"
#include "ggml-common.h"
#include "ggml-quants.h"

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <mutex>
#include <vector>

extern "C" {

// =============================================================================
// IQ4_KT — 4.0 bpw trellis-coded (row_meta_size = 4: per-row float scale)
// =============================================================================

namespace {

// IQ4_KT block-layout constants.
constexpr int kIQ4KT_BlockSize  = 32;
constexpr int kIQ4KT_GroupSize  = IQ4KTParams::kGroupSize;   // 4
constexpr int kIQ4KT_NumBits    = IQ4KTParams::kNumBits;     // 15
constexpr int kIQ4KT_NumVal     = IQ4KTParams::kNumVal;      // 32768
constexpr int kIQ4KT_Ng         = kIQ4KT_BlockSize / kIQ4KT_GroupSize; // 8
constexpr int kIQ4KT_Nblock     = QK_K / kIQ4KT_BlockSize;             // 8
constexpr int kIQ4KT_NumGroups  = QK_K / kIQ4KT_GroupSize;             // 64
constexpr int kIQ4KT_OffsetA    = 4096;
constexpr int kIQ4KT_OffsetB    = 4096 + 32768;
constexpr int kIQ4KT_NeighboursPB = 6;    // soft-bin replication factor

// Dual codebook (offset A + offset B) for IQ4_KT.
// Other IQ_KT family members use a single codebook each; see family header.
struct IQ4KT_Codebook {
    IQKTCookedBook<kIQ4KT_GroupSize, kIQ4KT_NumBits> a;   // offset = kIQ4KT_OffsetA
    IQKTCookedBook<kIQ4KT_GroupSize, kIQ4KT_NumBits> b;   // offset = kIQ4KT_OffsetB
    bool initialized = false;
};

static IQ4KT_Codebook g_iq4kt_codebook;

// Thread-safe lazy initialization.  test-backend-ops parallelizes quantize across
// threads via std::async; without synchronization, multiple threads racing on the
// vector resizes causes heap corruption ("double free", "unaligned chunk").
static std::once_flag g_iq4kt_init_once;

static void iq4kt_codebook_do_init() {
    iqkt_cooked_book_init<kIQ4KT_GroupSize, kIQ4KT_NumBits, false>(
        g_iq4kt_codebook.a, kIQ4KT_OffsetA, kIQ4KT_NeighboursPB);
    iqkt_cooked_book_init<kIQ4KT_GroupSize, kIQ4KT_NumBits, false>(
        g_iq4kt_codebook.b, kIQ4KT_OffsetB, kIQ4KT_NeighboursPB);
    g_iq4kt_codebook.initialized = true;
}

static inline void iq4kt_codebook_init() {
    std::call_once(g_iq4kt_init_once, iq4kt_codebook_do_init);
}

// Find best per-sub-block scale d such that d * codebook[best_idx[g]] approximates xb.
// IQ4_KT-specific: takes both codebook_a and codebook_b (dual-codebook).
// IQ2/3/1_KT use a single codebook; their scale-finding is simpler and defined in P3a/b/c.
static float iq4kt_find_best_scale(const float * xb, const float * weight,
                                   const int * best_idx, const float * codebook_a,
                                   const float * codebook_b, const uint8_t * use_b) {
    float sumqx = 0, sumq2 = 0;
    for (int g = 0; g < kIQ4KT_Ng; ++g) {
        const float * cb = use_b[g] ? codebook_b : codebook_a;
        const float * v  = cb + (size_t)best_idx[g] * kIQ4KT_GroupSize;
        const float * xl = xb + g * kIQ4KT_GroupSize;
        const float * wl = weight + g * kIQ4KT_GroupSize;
        for (int k = 0; k < kIQ4KT_GroupSize; ++k) {
            sumqx += wl[k] * v[k] * xl[k];
            sumq2 += wl[k] * v[k] * v[k];
        }
    }
    return sumq2 > 0.f ? sumqx / sumq2 : 0.f;
}

// Per-row IQ4_KT quantizer.  Uses iqkt_find_best_index from the family header
// for cluster-accelerated NN search over both codebooks.
static void quantize_row_iq4_kt_impl(const float * x, char * cy, int n_per_row,
                                     const float * quant_weights) {
    iq4kt_codebook_init();
    const IQKTCookedBook<kIQ4KT_GroupSize, kIQ4KT_NumBits> & ckA = g_iq4kt_codebook.a;
    const IQKTCookedBook<kIQ4KT_GroupSize, kIQ4KT_NumBits> & ckB = g_iq4kt_codebook.b;
    const float * cb_a = ckA.values.data();
    const float * cb_b = ckB.values.data();

    constexpr int kSuperBlockSize = QK_K;
    const int nblock = n_per_row / kSuperBlockSize;

    float * dptr = (float *)cy;
    block_iq4_kt * y = (block_iq4_kt *)(dptr + 1);

    // Compute weights (sigma2 + xb²) × imatrix when imatrix provided.
    std::vector<float> weights(n_per_row);
    {
        constexpr float kEps2 = 1e-14f;
        constexpr float kWeight = 1e-4f;
        constexpr float kSigmaScale = 2.0f;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xbl = x + ibl * kSuperBlockSize;
            float * wbl = weights.data() + ibl * kSuperBlockSize;
            float sumx2 = 0;
            for (int j = 0; j < kSuperBlockSize; ++j) sumx2 += xbl[j] * xbl[j];
            if (sumx2 < kEps2 * kSuperBlockSize) {
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = kWeight;
                continue;
            }
            const float sigma2 = kSigmaScale * sumx2 / kSuperBlockSize;
            if (quant_weights) {
                for (int ib = 0; ib < kIQ4KT_Nblock; ++ib) {
                    const float * qw = quant_weights + ibl * kSuperBlockSize + ib * kIQ4KT_BlockSize;
                    const float * xb = xbl + ib * kIQ4KT_BlockSize;
                    float * wb = wbl + ib * kIQ4KT_BlockSize;
                    for (int j = 0; j < kIQ4KT_BlockSize; ++j) {
                        wb[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
                    }
                }
            } else {
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = 0.25f * sigma2 + xbl[j] * xbl[j];
            }
        }
    }

    // Find row max abs.
    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) {
        amax_row = std::max(amax_row, std::abs(x[j]));
    }
    if (amax_row == 0.f) {
        dptr[0] = 0.f;
        std::memset(y, 0, (size_t)nblock * sizeof(block_iq4_kt));
        return;
    }

    std::vector<float> all_scales((size_t)nblock * kIQ4KT_Nblock);

    // Phase 1: per-sub-block, find best scale + codebook choice (A or B).
    float amax_scale = 0, max_scale = 0;
    int best_idx[kIQ4KT_Ng];
    float xaux[kIQ4KT_BlockSize];

    for (int ibl = 0; ibl < nblock; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq4_kt));
        const float * xbl = x + ibl * kSuperBlockSize;
        float * scales = all_scales.data() + (size_t)ibl * kIQ4KT_Nblock;

        for (int ib = 0; ib < kIQ4KT_Nblock; ++ib) {
            const float * weight = weights.data() + ibl * kSuperBlockSize + ib * kIQ4KT_BlockSize;
            float amax = 0;
            for (int j = 0; j < kIQ4KT_BlockSize; ++j) {
                xaux[j] = xbl[ib * kIQ4KT_BlockSize + j];
                amax = std::max(amax, std::abs(xaux[j]));
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            // Try a few starting scales with codebook A; pick best.  Mirrors ik_llama
            // logic but simplified (we do exhaustive cluster search per group, so only
            // a small handful of starting points needed).
            float best_score = -INFINITY;
            const float scale_0 = std::max(90.f, 124.f * amax / amax_row);
            uint8_t use_b[kIQ4KT_Ng];
            for (int g = 0; g < kIQ4KT_Ng; ++g) use_b[g] = 0;
            for (int sign = 0; sign < 2; ++sign) {
                const float d_init = (sign == 0 ? amax : -amax) / scale_0;
                for (int g = 0; g < kIQ4KT_Ng; ++g) {
                    best_idx[g] = iqkt_find_best_index<kIQ4KT_GroupSize, kIQ4KT_NumBits, false>(
                        xaux + g * kIQ4KT_GroupSize,
                        weight + g * kIQ4KT_GroupSize,
                        d_init, ckA);
                }
                const float d = iq4kt_find_best_scale(xaux, weight, best_idx, cb_a, cb_b, use_b);
                float sumqx = 0;
                for (int g = 0; g < kIQ4KT_Ng; ++g) {
                    const float * v  = cb_a + (size_t)best_idx[g] * kIQ4KT_GroupSize;
                    const float * xl = xaux + g * kIQ4KT_GroupSize;
                    const float * wl = weight + g * kIQ4KT_GroupSize;
                    for (int k = 0; k < kIQ4KT_GroupSize; ++k) sumqx += wl[k] * v[k] * xl[k];
                }
                const float score = sumqx * d;
                if (score > best_score) { best_score = score; scales[ib] = d; }
            }
            // Try codebook B with same approach.
            for (int sign = 0; sign < 2; ++sign) {
                const float d_init = (sign == 0 ? amax : -amax) / scale_0;
                for (int g = 0; g < kIQ4KT_Ng; ++g) {
                    best_idx[g] = iqkt_find_best_index<kIQ4KT_GroupSize, kIQ4KT_NumBits, false>(
                        xaux + g * kIQ4KT_GroupSize,
                        weight + g * kIQ4KT_GroupSize,
                        d_init, ckB);
                }
                for (int g = 0; g < kIQ4KT_Ng; ++g) use_b[g] = 1;
                const float d = iq4kt_find_best_scale(xaux, weight, best_idx, cb_a, cb_b, use_b);
                float sumqx = 0;
                for (int g = 0; g < kIQ4KT_Ng; ++g) {
                    const float * v  = cb_b + (size_t)best_idx[g] * kIQ4KT_GroupSize;
                    const float * xl = xaux + g * kIQ4KT_GroupSize;
                    const float * wl = weight + g * kIQ4KT_GroupSize;
                    for (int k = 0; k < kIQ4KT_GroupSize; ++k) sumqx += wl[k] * v[k] * xl[k];
                }
                const float score = sumqx * d;
                if (score > best_score) {
                    best_score = score;
                    scales[ib] = d;
                    y[ibl].qs[ib] = 1;  // mark "use codebook B" via shb[ib] & 1
                }
                for (int g = 0; g < kIQ4KT_Ng; ++g) use_b[g] = 0;
            }

            const float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) { amax_scale = abs_scale; max_scale = scales[ib]; }
        }
    }

    float d_row = -max_scale / 64.f;
    dptr[0] = d_row;
    if (d_row == 0.f) return;

    // Phase 2: per sub-block, encode the chosen scale + per-group indices.
    const float id = 1.f / d_row;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        uint32_t * shb = y[ibl].qs;
        uint8_t  * ql  = (uint8_t *)(shb + kIQ4KT_Nblock);   // 64 bytes
        uint8_t  * qh  = ql + kIQ4KT_NumGroups;               // 16 bytes
        std::memset(qh, 0, kIQ4KT_NumGroups / 2);
        const float * xbl = x + ibl * kSuperBlockSize;
        const float * scales = all_scales.data() + (size_t)ibl * kIQ4KT_Nblock;

        for (int ib = 0; ib < kIQ4KT_Nblock; ++ib) {
            const float * weight = weights.data() + ibl * kSuperBlockSize + ib * kIQ4KT_BlockSize;
            for (int j = 0; j < kIQ4KT_BlockSize; ++j) xaux[j] = xbl[ib * kIQ4KT_BlockSize + j];

            int ls = (int)nearbyintf(id * scales[ib]);
            if (ls < -64) ls = -64;
            if (ls >  63) ls =  63;
            // shb[ib] low byte: bit 0 = use_b (already set); bits 1..7 = ls + 64
            const uint32_t scale_byte = (uint32_t)((ls + 64) << 1) | (shb[ib] & 1);
            shb[ib] = (shb[ib] & ~0xffu) | scale_byte;

            const bool use_b = (shb[ib] & 1) != 0;
            const IQKTCookedBook<kIQ4KT_GroupSize, kIQ4KT_NumBits> & ck = use_b ? ckB : ckA;
            const float * cb = use_b ? cb_b : cb_a;
            const float dl = d_row * (float)ls;
            for (int g = 0; g < kIQ4KT_Ng; ++g) {
                best_idx[g] = iqkt_find_best_index<kIQ4KT_GroupSize, kIQ4KT_NumBits, false>(
                    xaux + g * kIQ4KT_GroupSize,
                    weight + g * kIQ4KT_GroupSize,
                    dl, ck);
            }

            for (int g = 0; g < kIQ4KT_Ng; ++g) {
                const int idx15 = best_idx[g] & 0x7fff;  // 15-bit index
                // Pack: 8 low bits in ql, 4 mid bits in qh, 3 high bits in shb (bits 8+3*g..10+3*g).
                shb[ib] |= ((uint32_t)((idx15 >> 12) & 7)) << (8 + 3 * g);
                ql[kIQ4KT_Ng * ib + g] = (uint8_t)(idx15 & 0xff);
                const int jj = kIQ4KT_Ng * ib + g;
                const int qh_byte = jj % (kIQ4KT_NumGroups / 2);
                const int qh_nibble = jj / (kIQ4KT_NumGroups / 2);
                qh[qh_byte] |= (uint8_t)(((idx15 >> 8) & 0xf) << (4 * qh_nibble));

                const float * v = cb + (size_t)best_idx[g] * kIQ4KT_GroupSize;
                const float * xl = xaux + g * kIQ4KT_GroupSize;
                const float * wl = weight + g * kIQ4KT_GroupSize;
                for (int k = 0; k < kIQ4KT_GroupSize; ++k) {
                    const float q = v[k] * (float)ls;
                    sumqx += wl[k] * xl[k] * q;
                    sumq2 += wl[k] * q * q;
                }
            }
        }
    }
    if (sumq2 > 0.f) dptr[0] = sumqx / sumq2;
}

}  // anonymous namespace

void dequantize_row_iq4_kt(const block_iq4_kt * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);
    const int nb = (int)(k / QK_K);

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint32_t * shb = x[ibl].qs;
        const uint8_t  * ql  = (const uint8_t *)(shb + kIQ4KT_Nblock);
        const uint8_t  * qh  = ql + kIQ4KT_NumGroups;
        for (int ib = 0; ib < kIQ4KT_Nblock; ++ib) {
            const int offset = (shb[ib] & 1) ? kIQ4KT_OffsetB : kIQ4KT_OffsetA;
            const int ls = (int)((shb[ib] & 0xff) >> 1) - 64;
            const float sl = d * (float)ls;
            for (int g = 0; g < kIQ4KT_Ng; ++g) {
                const int jj = kIQ4KT_Ng * ib + g;
                const int qh_byte = jj % (kIQ4KT_NumGroups / 2);
                const int qh_nibble = jj / (kIQ4KT_NumGroups / 2);
                const uint32_t idx = (uint32_t)ql[jj]
                                   | (((uint32_t)((qh[qh_byte] >> (4 * qh_nibble)) & 0xf)) << 8)
                                   | (((uint32_t)((shb[ib] >> (8 + 3 * g)) & 7)) << 12);
                iqkt_gen_group_int<kIQ4KT_GroupSize>(idx, offset, y);
                for (int kk = 0; kk < kIQ4KT_GroupSize; ++kk) y[kk] *= sl;
                y += kIQ4KT_GroupSize;
            }
        }
    }
}

size_t quantize_iq4_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ4_KT, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        const float * qw = imatrix;
        quantize_row_iq4_kt_impl(x, qrow, (int)n_per_row, qw);
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}

void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq4_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq4_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq4_kt_ref(x, (block_iq4_kt *)y, k);
}

// Scalar dot product via dequant-into-temp + multiply-accumulate.  Q8_K activation.
void ggml_vec_dot_iq4_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K   *)vy;
    const int nblock = n / QK_K;

    float sumf = 0;
    float gv[kIQ4KT_GroupSize];
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float db = d * y[ibl].d;
        const uint32_t * shb = x[ibl].qs;
        const uint8_t  * ql  = (const uint8_t *)(shb + kIQ4KT_Nblock);
        const uint8_t  * qh  = ql + kIQ4KT_NumGroups;
        const int8_t   * q8  = y[ibl].qs;
        for (int ib = 0; ib < kIQ4KT_Nblock; ++ib) {
            const int offset = (shb[ib] & 1) ? kIQ4KT_OffsetB : kIQ4KT_OffsetA;
            const int ls = (int)((shb[ib] & 0xff) >> 1) - 64;
            const float dl = db * (float)ls;
            for (int g = 0; g < kIQ4KT_Ng; ++g) {
                const int jj = kIQ4KT_Ng * ib + g;
                const int qh_byte = jj % (kIQ4KT_NumGroups / 2);
                const int qh_nibble = jj / (kIQ4KT_NumGroups / 2);
                const uint32_t idx = (uint32_t)ql[jj]
                                   | (((uint32_t)((qh[qh_byte] >> (4 * qh_nibble)) & 0xf)) << 8)
                                   | (((uint32_t)((shb[ib] >> (8 + 3 * g)) & 7)) << 12);
                iqkt_gen_group_int<kIQ4KT_GroupSize>(idx, offset, gv);
                for (int k = 0; k < kIQ4KT_GroupSize; ++k) {
                    sumf += dl * gv[k] * (float)q8[k];
                }
                q8 += kIQ4KT_GroupSize;
            }
        }
    }
    *s = sumf;
}

}  // extern "C"
