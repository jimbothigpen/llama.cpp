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

// =============================================================================
// IQ2_KT — 2.0 bpw trellis-coded (row_meta_size = 4: per-row float scale)
//
// Block layout: uint16_t qs[32] = 64 bytes; one 16-bit codebook index per group.
// No sub-block scale — all 32 groups in a superblock share the per-row float d.
// GROUP_SIZE=8, NUM_BITS=16, kNumVal=65536, kOffset=0 (single codebook).
// =============================================================================

namespace {

constexpr int kIQ2KT_GroupSize  = IQ2KTParams::kGroupSize;   // 8
constexpr int kIQ2KT_NumBits    = IQ2KTParams::kNumBits;     // 16
constexpr int kIQ2KT_NumVal     = IQ2KTParams::kNumVal;      // 65536
constexpr int kIQ2KT_NumGroups  = QK_K / kIQ2KT_GroupSize;  // 32
constexpr int kIQ2KT_Offset     = 0;

struct IQ2KT_Codebook {
    IQKTCookedBook<kIQ2KT_GroupSize, kIQ2KT_NumBits> cb;
    bool initialized = false;
};

static IQ2KT_Codebook g_iq2kt_codebook;
static std::once_flag g_iq2kt_init_once;

static void iq2kt_codebook_do_init() {
    // k=256: covers all base-3 "diagonal" bins for GS=8 — bin [2,...] is rank ~256
    // for entries near the lower bin-2 boundary (value ≈ 16), so k ≥ 256 ensures
    // both forward and reverse boundary crossings are registered.
    iqkt_cooked_book_init<kIQ2KT_GroupSize, kIQ2KT_NumBits, false>(
        g_iq2kt_codebook.cb, kIQ2KT_Offset, 256);
    g_iq2kt_codebook.initialized = true;
}

static inline void iq2kt_codebook_init() {
    std::call_once(g_iq2kt_init_once, iq2kt_codebook_do_init);
}

// Per-row IQ2_KT quantizer.
// No sub-block scales: all groups share a single per-row float d_row.
// Two-pass: estimate d_row via sumqx/sumq2, then assign final indices.
static void quantize_row_iq2_kt_impl(const float * x, char * cy, int n_per_row,
                                     const float * quant_weights) {
    iq2kt_codebook_init();
    const float * cb = g_iq2kt_codebook.cb.values.data();

    constexpr int kSuperBlockSize = QK_K;
    const int nblock = n_per_row / kSuperBlockSize;

    float * dptr = (float *)cy;
    block_iq2_kt * y = (block_iq2_kt *)(dptr + 1);

    // Compute importance weights (sigma² + x²) × imatrix when provided.
    std::vector<float> weights(n_per_row);
    {
        constexpr float kEps2       = 1e-14f;
        constexpr float kWeight     = 1e-4f;
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
                const float * qw = quant_weights + ibl * kSuperBlockSize;
                for (int j = 0; j < kSuperBlockSize; ++j) {
                    wbl[j] = qw[j] * sqrtf(sigma2 + xbl[j] * xbl[j]);
                }
            } else {
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = 0.25f * sigma2 + xbl[j] * xbl[j];
            }
        }
    }

    // Row max absolute value → initial scale estimate.
    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (amax_row == 0.f) {
        dptr[0] = 0.f;
        std::memset(y, 0, (size_t)nblock * sizeof(block_iq2_kt));
        return;
    }

    // Phase 1: estimate d_row via a single pass using d_init = amax/120.
    // The codebook range is approximately [-126, +126]; 120 is a conservative typical max.
    // sumqx/sumq2 WLS refinement handles the correct sign automatically.
    const float d_init = amax_row / 120.f;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float * xbl = x + ibl * kSuperBlockSize;
        const float * wbl = weights.data() + ibl * kSuperBlockSize;
        for (int g = 0; g < kIQ2KT_NumGroups; ++g) {
            const float * xg = xbl + g * kIQ2KT_GroupSize;
            const float * wg = wbl + g * kIQ2KT_GroupSize;
            const int idx = iqkt_find_best_index<kIQ2KT_GroupSize, kIQ2KT_NumBits, false>(
                xg, wg, d_init, g_iq2kt_codebook.cb);
            const float * v = cb + (size_t)idx * kIQ2KT_GroupSize;
            for (int k = 0; k < kIQ2KT_GroupSize; ++k) {
                sumqx += wg[k] * v[k] * xg[k];
                sumq2 += wg[k] * v[k] * v[k];
            }
        }
    }
    float d_row = sumq2 > 0.f ? sumqx / sumq2 : d_init;
    dptr[0] = d_row;
    if (d_row == 0.f) {
        std::memset(y, 0, (size_t)nblock * sizeof(block_iq2_kt));
        return;
    }

    // Phase 2: final pass — assign 16-bit codebook indices with refined d_row.
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float * xbl = x + ibl * kSuperBlockSize;
        const float * wbl = weights.data() + ibl * kSuperBlockSize;
        for (int g = 0; g < kIQ2KT_NumGroups; ++g) {
            const float * xg = xbl + g * kIQ2KT_GroupSize;
            const float * wg = wbl + g * kIQ2KT_GroupSize;
            const int idx = iqkt_find_best_index<kIQ2KT_GroupSize, kIQ2KT_NumBits, false>(
                xg, wg, d_row, g_iq2kt_codebook.cb);
            y[ibl].qs[g] = (uint16_t)(idx & 0xffff);
        }
    }

    // Phase 3: one final d_row refinement with committed indices.
    sumqx = 0; sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float * xbl = x + ibl * kSuperBlockSize;
        const float * wbl = weights.data() + ibl * kSuperBlockSize;
        for (int g = 0; g < kIQ2KT_NumGroups; ++g) {
            const float * xg = xbl + g * kIQ2KT_GroupSize;
            const float * wg = wbl + g * kIQ2KT_GroupSize;
            const float * v  = cb + (size_t)y[ibl].qs[g] * kIQ2KT_GroupSize;
            for (int k = 0; k < kIQ2KT_GroupSize; ++k) {
                sumqx += wg[k] * v[k] * xg[k];
                sumq2 += wg[k] * v[k] * v[k];
            }
        }
    }
    if (sumq2 > 0.f) dptr[0] = sumqx / sumq2;
}

}  // anonymous namespace (IQ2_KT)

// =============================================================================
// IQ3_KT — 3.0 bpw trellis-coded (row_meta_size = 4: per-row float scale)
//
// IQ3KTParams = IQKTParams<GROUP_SIZE=8, NUM_BITS=16, IS_ABS=false>.
// kNumVal = 65536.  Dual implicit codebooks (OffsetA=4096, OffsetB=36864), selected
// per-sub-block via shb[ib] bit 24.  Old files with bit 24 == 0 decode correctly (OffsetA).
//
// Block layout (96 bytes = 24 uint32_t per QK_K=256 elements):
//   qs[0..7]   (32 B): shb — bits[7:0] = scale byte (ls+128), bits[8+4g..11+4g] = idx[15:12],
//                            bit 24 = codebook selector (0=OffsetA, 1=OffsetB)
//   qs[8..15]  (32 B): ql  — 8 low bits per group, 4 groups per uint32_t (32 groups)
//   qs[16..19] (16 B): qh  — 4 mid bits per group, 2 groups per nibble (32 groups)
//   qs[20..23] (16 B): padding
// Index reconstruction: ql_byte | (qh_nibble << 8) | (sh_4bits << 12)
// =============================================================================

namespace {

constexpr int kIQ3KT_BlockSize  = 32;
constexpr int kIQ3KT_GroupSize  = IQ3KTParams::kGroupSize;   // 8
constexpr int kIQ3KT_NumBits    = IQ3KTParams::kNumBits;     // 16
constexpr int kIQ3KT_NumVal     = IQ3KTParams::kNumVal;      // 65536
constexpr int kIQ3KT_Ng         = kIQ3KT_BlockSize / kIQ3KT_GroupSize; // 4
constexpr int kIQ3KT_Nblock     = QK_K / kIQ3KT_BlockSize;             // 8
constexpr int kIQ3KT_NumGroups  = QK_K / kIQ3KT_GroupSize;             // 32
constexpr int kIQ3KT_OffsetA     = 4096;
constexpr int kIQ3KT_OffsetB     = 4096 + 32768;  // second implicit codebook
constexpr uint32_t kIQ3KT_SelBit = 1u << 24;       // shb[ib] bit 24 = use OffsetB
constexpr int kIQ3KT_NeighboursPB = 256;

struct IQ3KT_Codebook {
    IQKTCookedBook<kIQ3KT_GroupSize, kIQ3KT_NumBits> a;  // offset = kIQ3KT_OffsetA
    IQKTCookedBook<kIQ3KT_GroupSize, kIQ3KT_NumBits> b;  // offset = kIQ3KT_OffsetB
    bool initialized = false;
};

static IQ3KT_Codebook g_iq3kt_codebook;
static std::once_flag g_iq3kt_init_once;

static void iq3kt_codebook_do_init() {
    iqkt_cooked_book_init<kIQ3KT_GroupSize, kIQ3KT_NumBits, false>(
        g_iq3kt_codebook.a, kIQ3KT_OffsetA, kIQ3KT_NeighboursPB);
    iqkt_cooked_book_init<kIQ3KT_GroupSize, kIQ3KT_NumBits, false>(
        g_iq3kt_codebook.b, kIQ3KT_OffsetB, kIQ3KT_NeighboursPB);
    g_iq3kt_codebook.initialized = true;
}

static inline void iq3kt_codebook_init() {
    std::call_once(g_iq3kt_init_once, iq3kt_codebook_do_init);
}

// Find best per-sub-block scale d such that d * codebook[best_idx[g]] ≈ xb.
// IS_ABS=true: codebook values are non-negative; scale handles sign.
static float iq3kt_find_best_scale(const float * xb, const float * weight,
                                   const int * best_idx, const float * codebook) {
    float sumqx = 0, sumq2 = 0;
    for (int g = 0; g < kIQ3KT_Ng; ++g) {
        const float * v  = codebook + (size_t)best_idx[g] * kIQ3KT_GroupSize;
        const float * xl = xb + g * kIQ3KT_GroupSize;
        const float * wl = weight + g * kIQ3KT_GroupSize;
        for (int k = 0; k < kIQ3KT_GroupSize; ++k) {
            sumqx += wl[k] * v[k] * xl[k];
            sumq2 += wl[k] * v[k] * v[k];
        }
    }
    return sumq2 > 0.f ? sumqx / sumq2 : 0.f;
}

static void quantize_row_iq3_kt_impl(const float * x, char * cy, int n_per_row,
                                     const float * quant_weights) {
    iq3kt_codebook_init();
    const IQKTCookedBook<kIQ3KT_GroupSize, kIQ3KT_NumBits> & ckA = g_iq3kt_codebook.a;
    const IQKTCookedBook<kIQ3KT_GroupSize, kIQ3KT_NumBits> & ckB = g_iq3kt_codebook.b;
    const float * cb_a = ckA.values.data();
    const float * cb_b = ckB.values.data();

    constexpr int kSuperBlockSize = QK_K;
    const int nblock = n_per_row / kSuperBlockSize;

    float * dptr = (float *)cy;
    block_iq3_kt * y = (block_iq3_kt *)(dptr + 1);

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
                for (int ib = 0; ib < kIQ3KT_Nblock; ++ib) {
                    const float * qw = quant_weights + ibl * kSuperBlockSize + ib * kIQ3KT_BlockSize;
                    const float * xb = xbl + ib * kIQ3KT_BlockSize;
                    float * wb = wbl + ib * kIQ3KT_BlockSize;
                    for (int j = 0; j < kIQ3KT_BlockSize; ++j) {
                        wb[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
                    }
                }
            } else {
                for (int j = 0; j < kSuperBlockSize; ++j) wbl[j] = 0.25f * sigma2 + xbl[j] * xbl[j];
            }
        }
    }

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (amax_row == 0.f) {
        dptr[0] = 0.f;
        std::memset(y, 0, (size_t)nblock * sizeof(block_iq3_kt));
        return;
    }

    std::vector<float> all_scales((size_t)nblock * kIQ3KT_Nblock);

    float amax_scale = 0, max_scale = 0;
    int best_idx[kIQ3KT_Ng];
    float xaux[kIQ3KT_BlockSize];

    for (int ibl = 0; ibl < nblock; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq3_kt));
        const float * xbl = x + ibl * kSuperBlockSize;
        float * scales = all_scales.data() + (size_t)ibl * kIQ3KT_Nblock;

        for (int ib = 0; ib < kIQ3KT_Nblock; ++ib) {
            const float * weight = weights.data() + ibl * kSuperBlockSize + ib * kIQ3KT_BlockSize;
            float amax = 0;
            for (int j = 0; j < kIQ3KT_BlockSize; ++j) {
                xaux[j] = xbl[ib * kIQ3KT_BlockSize + j];
                amax = std::max(amax, std::abs(xaux[j]));
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            const float scale_0 = std::max(64.f, 96.f * amax / amax_row);
            float best_score = -INFINITY;
            // Codebook A trials
            for (int sign = 0; sign < 2; ++sign) {
                const float d_init = (sign == 0 ? amax : -amax) / scale_0;
                for (int g = 0; g < kIQ3KT_Ng; ++g) {
                    best_idx[g] = iqkt_find_best_index<kIQ3KT_GroupSize, kIQ3KT_NumBits, false>(
                        xaux + g * kIQ3KT_GroupSize,
                        weight + g * kIQ3KT_GroupSize,
                        d_init, ckA);
                }
                const float d = iq3kt_find_best_scale(xaux, weight, best_idx, cb_a);
                float sumqx = 0;
                for (int g = 0; g < kIQ3KT_Ng; ++g) {
                    const float * v  = cb_a + (size_t)best_idx[g] * kIQ3KT_GroupSize;
                    const float * xl = xaux + g * kIQ3KT_GroupSize;
                    const float * wl = weight + g * kIQ3KT_GroupSize;
                    for (int k = 0; k < kIQ3KT_GroupSize; ++k) sumqx += wl[k] * v[k] * xl[k];
                }
                const float score = sumqx * d;
                if (score > best_score) { best_score = score; scales[ib] = d; }
            }
            // Codebook B trials — set selector bit if B wins
            for (int sign = 0; sign < 2; ++sign) {
                const float d_init = (sign == 0 ? amax : -amax) / scale_0;
                for (int g = 0; g < kIQ3KT_Ng; ++g) {
                    best_idx[g] = iqkt_find_best_index<kIQ3KT_GroupSize, kIQ3KT_NumBits, false>(
                        xaux + g * kIQ3KT_GroupSize,
                        weight + g * kIQ3KT_GroupSize,
                        d_init, ckB);
                }
                const float d = iq3kt_find_best_scale(xaux, weight, best_idx, cb_b);
                float sumqx = 0;
                for (int g = 0; g < kIQ3KT_Ng; ++g) {
                    const float * v  = cb_b + (size_t)best_idx[g] * kIQ3KT_GroupSize;
                    const float * xl = xaux + g * kIQ3KT_GroupSize;
                    const float * wl = weight + g * kIQ3KT_GroupSize;
                    for (int k = 0; k < kIQ3KT_GroupSize; ++k) sumqx += wl[k] * v[k] * xl[k];
                }
                const float score = sumqx * d;
                if (score > best_score) {
                    best_score = score;
                    scales[ib] = d;
                    y[ibl].qs[ib] = kIQ3KT_SelBit;  // codebook B wins
                }
            }
            const float abs_scale = std::abs(scales[ib]);
            if (abs_scale > amax_scale) { amax_scale = abs_scale; max_scale = scales[ib]; }
        }
    }

    float d_row = -max_scale / 128.f;
    dptr[0] = d_row;
    if (d_row == 0.f) return;

    const float id = 1.f / d_row;
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        uint32_t * shb = y[ibl].qs;
        const float * xbl = x + ibl * kSuperBlockSize;
        const float * scales = all_scales.data() + (size_t)ibl * kIQ3KT_Nblock;

        for (int ib = 0; ib < kIQ3KT_Nblock; ++ib) {
            const float * weight = weights.data() + ibl * kSuperBlockSize + ib * kIQ3KT_BlockSize;
            for (int j = 0; j < kIQ3KT_BlockSize; ++j) xaux[j] = xbl[ib * kIQ3KT_BlockSize + j];

            int ls = (int)nearbyintf(id * scales[ib]);
            ls = std::max(-128, std::min(127, ls));
            // Preserve selector bit (bit 24) set in phase 1, write scale in bits [7:0]
            const bool use_b = (shb[ib] & kIQ3KT_SelBit) != 0;
            shb[ib] = ((uint32_t)(ls + 128) & 0xff) | (use_b ? kIQ3KT_SelBit : 0u);

            const IQKTCookedBook<kIQ3KT_GroupSize, kIQ3KT_NumBits> & ck2 = use_b ? ckB : ckA;
            const float * cb2 = use_b ? cb_b : cb_a;
            const float dl = d_row * (float)ls;
            for (int g = 0; g < kIQ3KT_Ng; ++g) {
                best_idx[g] = iqkt_find_best_index<kIQ3KT_GroupSize, kIQ3KT_NumBits, false>(
                    xaux + g * kIQ3KT_GroupSize,
                    weight + g * kIQ3KT_GroupSize,
                    dl, ck2);
            }

            for (int g = 0; g < kIQ3KT_Ng; ++g) {
                const int jj = kIQ3KT_Ng * ib + g;
                const int idx16 = best_idx[g] & 0xffff;
                // Low 8 bits in ql (qs[8..15])
                shb[kIQ3KT_Nblock + jj / 4] |= (uint32_t)(idx16 & 0xff) << ((jj % 4) * 8);
                // Mid 4 bits in qh (qs[16..19])
                const int qh_shift = ((jj / 2) & 3) * 8 + (jj % 2) * 4;
                shb[kIQ3KT_Nblock + kIQ3KT_NumGroups / 4 + jj / 8] |=
                    (uint32_t)((idx16 >> 8) & 0xf) << qh_shift;
                // High 4 bits in shb[ib] bits[8+4g..11+4g]
                shb[ib] |= (uint32_t)((idx16 >> 12) & 0xf) << (8 + 4 * g);

                const float * v = cb2 + (size_t)best_idx[g] * kIQ3KT_GroupSize;
                const float * xl = xaux + g * kIQ3KT_GroupSize;
                const float * wl = weight + g * kIQ3KT_GroupSize;
                for (int k = 0; k < kIQ3KT_GroupSize; ++k) {
                    const float q = v[k] * (float)ls;
                    sumqx += wl[k] * xl[k] * q;
                    sumq2 += wl[k] * q * q;
                }
            }
        }
    }
    if (sumq2 > 0.f) dptr[0] = sumqx / sumq2;
}

}  // anonymous namespace (IQ3_KT)

void dequantize_row_iq2_kt(const block_iq2_kt * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);
    const int nb = (int)(k / QK_K);

    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int g = 0; g < kIQ2KT_NumGroups; ++g) {
            const uint32_t idx = (uint32_t)x[ibl].qs[g];
            iqkt_gen_group_int<kIQ2KT_GroupSize>(idx, kIQ2KT_Offset, y);
            for (int k = 0; k < kIQ2KT_GroupSize; ++k) y[k] *= d;
            y += kIQ2KT_GroupSize;
        }
    }
}

size_t quantize_iq2_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ2_KT, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        const float * x = src + row * n_per_row;
        quantize_row_iq2_kt_impl(x, qrow, (int)n_per_row, imatrix);
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}

void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq2_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq2_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq2_kt_ref(x, (block_iq2_kt *)y, k);
}

void ggml_vec_dot_iq2_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq2_kt * x = (const block_iq2_kt *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K   *)vy;
    const int nblock = n / QK_K;

    float sumf = 0;
    float gv[kIQ2KT_GroupSize];
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float db = d * y[ibl].d;
        const int8_t * q8 = y[ibl].qs;
        for (int g = 0; g < kIQ2KT_NumGroups; ++g) {
            const uint32_t idx = (uint32_t)x[ibl].qs[g];
            iqkt_gen_group_int<kIQ2KT_GroupSize>(idx, kIQ2KT_Offset, gv);
            for (int k = 0; k < kIQ2KT_GroupSize; ++k) {
                sumf += db * gv[k] * (float)q8[k];
            }
            q8 += kIQ2KT_GroupSize;
        }
    }
    *s = sumf;
}


void dequantize_row_iq3_kt(const block_iq3_kt * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);
    const int nb = (int)(k / QK_K);

    for (int ibl = 0; ibl < nb; ++ibl) {
        const uint32_t * shb = x[ibl].qs;
        const uint8_t  * ql  = (const uint8_t *)(shb + kIQ3KT_Nblock);
        const uint32_t * qh_u32 = shb + kIQ3KT_Nblock + kIQ3KT_NumGroups / 4;
        for (int ib = 0; ib < kIQ3KT_Nblock; ++ib) {
            const int ls = (int)(shb[ib] & 0xff) - 128;
            const float sl = d * (float)ls;
            const int offset = (shb[ib] & kIQ3KT_SelBit) ? kIQ3KT_OffsetB : kIQ3KT_OffsetA;
            for (int g = 0; g < kIQ3KT_Ng; ++g) {
                const int jj = kIQ3KT_Ng * ib + g;
                const int qh_shift = ((jj / 2) & 3) * 8 + (jj % 2) * 4;
                const uint32_t qh_nibble = (qh_u32[jj / 8] >> qh_shift) & 0xfu;
                const uint32_t sh_4bits  = (shb[ib] >> (8 + 4 * g)) & 0xfu;
                const uint32_t idx = (uint32_t)ql[jj] | (qh_nibble << 8) | (sh_4bits << 12);
                iqkt_gen_group_int<kIQ3KT_GroupSize, false>(idx, offset, y);
                for (int kk = 0; kk < kIQ3KT_GroupSize; ++kk) y[kk] *= sl;
                y += kIQ3KT_GroupSize;
            }
        }
    }
}

size_t quantize_iq3_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ3_KT, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq3_kt_impl(src + row * n_per_row, qrow, (int)n_per_row, imatrix);
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}

void quantize_row_iq3_kt_ref(const float * GGML_RESTRICT x, block_iq3_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq3_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq3_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq3_kt_ref(x, (block_iq3_kt *)y, k);
}

void ggml_vec_dot_iq3_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;

    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);
    const block_q8_K   * y8 = (const block_q8_K   *)vy;
    const int nblock = n / QK_K;

    float sumf = 0;
    float gv[kIQ3KT_GroupSize];
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float db = d * y8[ibl].d;
        const uint32_t * shb = x[ibl].qs;
        const uint8_t  * ql  = (const uint8_t *)(shb + kIQ3KT_Nblock);
        const uint32_t * qh_u32 = shb + kIQ3KT_Nblock + kIQ3KT_NumGroups / 4;
        const int8_t   * q8  = y8[ibl].qs;
        for (int ib = 0; ib < kIQ3KT_Nblock; ++ib) {
            const int ls = (int)(shb[ib] & 0xff) - 128;
            const float dl = db * (float)ls;
            const int offset = (shb[ib] & kIQ3KT_SelBit) ? kIQ3KT_OffsetB : kIQ3KT_OffsetA;
            for (int g = 0; g < kIQ3KT_Ng; ++g) {
                const int jj = kIQ3KT_Ng * ib + g;
                const int qh_shift = ((jj / 2) & 3) * 8 + (jj % 2) * 4;
                const uint32_t qh_nibble = (qh_u32[jj / 8] >> qh_shift) & 0xfu;
                const uint32_t sh_4bits  = (shb[ib] >> (8 + 4 * g)) & 0xfu;
                const uint32_t idx = (uint32_t)ql[jj] | (qh_nibble << 8) | (sh_4bits << 12);
                iqkt_gen_group_int<kIQ3KT_GroupSize, false>(idx, offset, gv);
                for (int kk = 0; kk < kIQ3KT_GroupSize; ++kk) {
                    sumf += dl * gv[kk] * (float)q8[kk];
                }
                q8 += kIQ3KT_GroupSize;
            }
        }
    }
    *s = sumf;
}

// =============================================================================
// IQ1_KT — 1.75 bpw trellis-coded (row_meta_size = 4: per-row float scale)
//
// QuantizerIQKT<block=32, group=8, bits=13, is_abs=false, is_int=true>:
//   kBlockSize=32, kGroupSize=8, kNg=4, kNblock=8, kNumVal=8192, offset=4096.
// Block layout (block_iq1_kt, 56 B): sh[8] ql[32] qh[16].
//   idx[jj]  = ql[jj] | ((qh[jj%16] << (8-4*(jj/16))) & 0xf00)
//                     | ((sh[jj/4]  << (8-(jj%4)))     & 0x1000)
//   per-sub-block scale: dl = d_row * iq4k_values[sh[ib] & 0xf]
// =============================================================================

namespace {

constexpr int kIQ1KT_GroupSize = IQ1KTParams::kGroupSize;  // 8
constexpr int kIQ1KT_NumBits   = IQ1KTParams::kNumBits;    // 13
constexpr int kIQ1KT_NumVal    = IQ1KTParams::kNumVal;     // 8192
constexpr int kIQ1KT_BlockSize = 32;                       // sub-block (kBlockSize)
constexpr int kIQ1KT_Ng        = kIQ1KT_BlockSize / kIQ1KT_GroupSize;  // 4
constexpr int kIQ1KT_NBlock    = QK_K / kIQ1KT_BlockSize;  // 8
constexpr int kIQ1KT_NumGroups = QK_K / kIQ1KT_GroupSize;  // 32
constexpr int kIQ1KT_Offset    = 4096;

// iq4k_values mirror (first 16 entries) for the per-sub-block scale LUT.
// MUST match ggml-iqk-quants.c iq4k_values[0..15].
static const int8_t kIQ1KT_iq4k_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

static inline int iq1kt_best_index_iq4nl(float x) {
    if (x <= kIQ1KT_iq4k_values[0])  return 0;
    if (x >= kIQ1KT_iq4k_values[15]) return 15;
    int ml = 0, mu = 15;
    while (mu - ml > 1) {
        int mav = (ml + mu) / 2;
        if (x < kIQ1KT_iq4k_values[mav]) mu = mav; else ml = mav;
    }
    return x - kIQ1KT_iq4k_values[mu-1] < kIQ1KT_iq4k_values[mu] - x ? mu-1 : mu;
}

struct IQ1KT_Codebook {
    IQKTCookedBook<kIQ1KT_GroupSize, kIQ1KT_NumBits> cb;
    bool initialized = false;
};
static IQ1KT_Codebook g_iq1kt_codebook;
static std::once_flag g_iq1kt_init_once;

static void iq1kt_codebook_do_init() {
    // k_neighbours=256: matches IQ2_KT and IQ3_KT treatment (ripple from TODO 130).
    // IQ1_KT has kNumVal=8192 over 3^8=6561 bins (~1.25 entries/bin); k=256 ensures
    // boundary entries are not missed by the cluster-index directional bug (fixed in
    // iqkt_build_cluster_index Phase 2, d0773ae2d). ikllama originally used k=32.
    iqkt_cooked_book_init<kIQ1KT_GroupSize, kIQ1KT_NumBits, false>(
        g_iq1kt_codebook.cb, kIQ1KT_Offset, 256);
    g_iq1kt_codebook.initialized = true;
}
static inline void iq1kt_codebook_init() {
    std::call_once(g_iq1kt_init_once, iq1kt_codebook_do_init);
}

// Helper: pack a 13-bit index into ql/qh/sh for flat group jj (0..31).
static inline void iq1kt_pack_index(block_iq1_kt * b, int jj, uint32_t idx) {
    b->ql[jj] = (uint8_t)(idx & 0xff);                                   // 8 low
    const int qh_byte = jj % 16;
    const int qh_shift = (jj / 16) ? 4 : 0;                              // hi/lo nibble
    b->qh[qh_byte] |= (uint8_t)(((idx >> 8) & 0xf) << qh_shift);         // 4 mid
    b->sh[jj / 4]  |= (uint8_t)(((idx >> 12) & 0x1) << (4 + (jj % 4)));  // 13th bit
}

// Per-row IQ1_KT quantizer.  Two-stage: per-sub-block index+scale search, then a
// row-level WLS refinement of d_row that picks each sub-block's iq4k scale index.
static void quantize_row_iq1_kt_impl(const float * x, char * cy, int n_per_row,
                                     const float * quant_weights) {
    iq1kt_codebook_init();
    const float * cb = g_iq1kt_codebook.cb.values.data();

    const int nblock = n_per_row / QK_K;
    float * dptr = (float *)cy;
    block_iq1_kt * y = (block_iq1_kt *)(dptr + 1);

    std::vector<float> weights(n_per_row);
    std::vector<float> all_scales((size_t)nblock * kIQ1KT_NBlock, 0.f);
    std::vector<int>   all_idx((size_t)nblock * kIQ1KT_NumGroups, 0);

    // Importance weights: sigma^2 + x^2, scaled by imatrix when present.
    {
        constexpr float kEps2 = 1e-14f, kWeight = 1e-4f, kSigmaScale = 2.0f;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xbl = x + ibl * QK_K;
            float * wbl = weights.data() + ibl * QK_K;
            float sumx2 = 0;
            for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j] * xbl[j];
            if (sumx2 < kEps2 * QK_K) { for (int j = 0; j < QK_K; ++j) wbl[j] = kWeight; continue; }
            const float sigma2 = kSigmaScale * sumx2 / QK_K;
            if (quant_weights) {
                const float * qw = quant_weights + ibl * QK_K;
                for (int j = 0; j < QK_K; ++j) wbl[j] = qw[j] * sqrtf(sigma2 + xbl[j]*xbl[j]);
            } else {
                for (int j = 0; j < QK_K; ++j) wbl[j] = 0.25f * sigma2 + xbl[j]*xbl[j];
            }
        }
    }

    float amax_row = 0;
    for (int j = 0; j < n_per_row; ++j) amax_row = std::max(amax_row, std::abs(x[j]));
    if (amax_row == 0.f) {
        dptr[0] = 0.f;
        std::memset(y, 0, (size_t)nblock * sizeof(block_iq1_kt));
        return;
    }

    // Stage 1: per-sub-block best index set + a continuous scale estimate.
    float max_scale = 0.f, amax_scale = 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        std::memset(&y[ibl], 0, sizeof(block_iq1_kt));
        const float * xbl = x + ibl * QK_K;
        float * sc = all_scales.data() + (size_t)ibl * kIQ1KT_NBlock;
        for (int ib = 0; ib < kIQ1KT_NBlock; ++ib) {
            const float * xb = xbl + ib * kIQ1KT_BlockSize;
            const float * wb = weights.data() + ibl * QK_K + ib * kIQ1KT_BlockSize;
            float amax = 0;
            for (int j = 0; j < kIQ1KT_BlockSize; ++j) amax = std::max(amax, std::abs(xb[j]));
            if (amax < 1e-16f) { sc[ib] = 0.f; continue; }
            float scale_0 = std::max(90.f, 124.f * amax / amax_row);
            int * idxslot = &all_idx[(size_t)ibl * kIQ1KT_NumGroups + ib * kIQ1KT_Ng];

            // Match ikllama's Stage-1: try BOTH scale signs (+/- amax/scale_0) and a
            // second scale_0-8 pass, keeping whichever gives the best WLS score.
            // Searching only the positive sign (the previous port) biased every block
            // to a positive effective scale and produced a negative row d, diverging
            // from ground truth.  find_best (sumqx^2/sumq2) selects the better fit.
            int   cand[kIQ1KT_Ng];
            float best_d = 0.f, best_score = -1.f;
            auto try_scale = [&](float dtry) {
                if (dtry == 0.f) return;
                float sumqx = 0, sumq2 = 0;
                for (int ig = 0; ig < kIQ1KT_Ng; ++ig) {
                    const float * xg = xb + ig * kIQ1KT_GroupSize;
                    const float * wg = wb + ig * kIQ1KT_GroupSize;
                    cand[ig] = iqkt_find_best_index<kIQ1KT_GroupSize, kIQ1KT_NumBits, false>(
                        xg, wg, dtry, g_iq1kt_codebook.cb);
                    const float * v = cb + (size_t)cand[ig] * kIQ1KT_GroupSize;
                    for (int k = 0; k < kIQ1KT_GroupSize; ++k) {
                        sumqx += wg[k] * v[k] * xg[k];
                        sumq2 += wg[k] * v[k] * v[k];
                    }
                }
                if (sumq2 <= 0.f) return;
                const float score = sumqx * sumqx / sumq2;   // WLS fit quality
                if (score > best_score) {
                    best_score = score;
                    best_d     = sumqx / sumq2;
                    for (int ig = 0; ig < kIQ1KT_Ng; ++ig) idxslot[ig] = cand[ig];
                }
            };
            try_scale( amax / scale_0);
            try_scale(-amax / scale_0);
            scale_0 -= 8.f;
            if (scale_0 > 0.f) {
                try_scale( amax / scale_0);
                try_scale(-amax / scale_0);
            }
            sc[ib] = best_score > 0.f ? best_d : amax / std::max(90.f, 124.f * amax / amax_row);
            const float as = std::abs(sc[ib]);
            if (as > amax_scale) { amax_scale = as; max_scale = sc[ib]; }
        }
    }
    if (max_scale == 0.f) { dptr[0] = 0.f; return; }

    // Stage 2: pick row scale d via small grid search over the iq4k-quantized sub-scales.
    float d = max_scale / kIQ1KT_iq4k_values[0];
    float best = 0.f;
    for (int itry = -9; itry <= 9; ++itry) {
        const float id = (itry + kIQ1KT_iq4k_values[0]) / max_scale;
        float sumqx = 0, sumq2 = 0;
        for (int ibl = 0; ibl < nblock; ++ibl) {
            const float * xb = x + ibl * QK_K;
            const float * wb = weights.data() + ibl * QK_K;
            const float * sc = all_scales.data() + (size_t)ibl * kIQ1KT_NBlock;
            for (int ib = 0; ib < kIQ1KT_NBlock; ++ib) {
                const int ls = iq1kt_best_index_iq4nl(id * sc[ib]);
                const float dl = kIQ1KT_iq4k_values[ls];
                for (int ig = 0; ig < kIQ1KT_Ng; ++ig) {
                    const float * v = cb + (size_t)all_idx[(size_t)ibl*kIQ1KT_NumGroups + ib*kIQ1KT_Ng + ig] * kIQ1KT_GroupSize;
                    for (int k = 0; k < kIQ1KT_GroupSize; ++k) {
                        const int jj = ig * kIQ1KT_GroupSize + k;
                        const float q = dl * v[k];
                        sumqx += wb[jj] * xb[jj] * q;
                        sumq2 += wb[jj] * q * q;
                    }
                }
                xb += kIQ1KT_BlockSize; wb += kIQ1KT_BlockSize;
            }
        }
        if (sumq2 > 0 && sumqx*sumqx > best*sumq2) { d = sumqx / sumq2; best = d * sumqx; }
    }

    // Commit sub-block scale indices (low nibble of sh) using the chosen d.
    const float id = d != 0.f ? 1.f / d : 0.f;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float * sc = all_scales.data() + (size_t)ibl * kIQ1KT_NBlock;
        for (int ib = 0; ib < kIQ1KT_NBlock; ++ib) {
            y[ibl].sh[ib] = (uint8_t)iq1kt_best_index_iq4nl(id * sc[ib]);
        }
    }
    dptr[0] = d;
    if (d == 0.f) return;

    // Stage 3: final index assignment at the committed per-sub-block scale; pack bits.
    float sumqx = 0, sumq2 = 0;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float * xbl = x + ibl * QK_K;
        for (int ib = 0; ib < kIQ1KT_NBlock; ++ib) {
            const float * xb = xbl + ib * kIQ1KT_BlockSize;
            const float * wb = weights.data() + ibl * QK_K + ib * kIQ1KT_BlockSize;
            const int   lsv = kIQ1KT_iq4k_values[y[ibl].sh[ib] & 0xf];
            const float dl  = d * (float)lsv;
            for (int ig = 0; ig < kIQ1KT_Ng; ++ig) {
                const float * xg = xb + ig * kIQ1KT_GroupSize;
                const float * wg = wb + ig * kIQ1KT_GroupSize;
                const int gi = (dl != 0.f)
                    ? iqkt_find_best_index<kIQ1KT_GroupSize, kIQ1KT_NumBits, false>(xg, wg, dl, g_iq1kt_codebook.cb)
                    : 0;
                const int jj = ib * kIQ1KT_Ng + ig;     // flat group 0..31
                iq1kt_pack_index(&y[ibl], jj, (uint32_t)gi);
                const float * v = cb + (size_t)gi * kIQ1KT_GroupSize;
                for (int k = 0; k < kIQ1KT_GroupSize; ++k) {
                    const float q = (float)lsv * v[k];
                    sumqx += wg[k] * xg[k] * q;
                    sumq2 += wg[k] * q * q;
                }
            }
        }
    }
    if (sumq2 > 0) dptr[0] = (sumqx / sumq2) * 1.07f;   // ikllama final 1.07 fudge
}

}  // anonymous namespace (IQ1_KT)

void dequantize_row_iq1_kt(const block_iq1_kt * GGML_RESTRICT vx, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq1_kt * x = (const block_iq1_kt *)(dptr + 1);
    const int nb = (int)(k / QK_K);
    for (int ibl = 0; ibl < nb; ++ibl) {
        for (int ib = 0; ib < kIQ1KT_NBlock; ++ib) {
            const float sl = d * (float)kIQ1KT_iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < kIQ1KT_Ng; ++ig) {
                const int jj = ib * kIQ1KT_Ng + ig;     // 0..31
                uint32_t idx = (uint32_t)x[ibl].ql[jj]
                    | (((uint32_t)x[ibl].qh[jj % 16] << (8 - 4 * (jj / 16))) & 0xf00u)
                    | (((uint32_t)x[ibl].sh[jj / 4]  << (8 - (jj % 4)))      & 0x1000u);
                iqkt_gen_group_int<kIQ1KT_GroupSize, false>((int)idx, kIQ1KT_Offset, y);
                for (int kk = 0; kk < kIQ1KT_GroupSize; ++kk) y[kk] *= sl;
                y += kIQ1KT_GroupSize;
            }
        }
    }
}

size_t quantize_iq1_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst,
                       int64_t nrows, int64_t n_per_row, const float * imatrix) {
    assert(n_per_row % QK_K == 0);
    const size_t row_size = ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row);
    char * qrow = (char *)dst;
    for (int64_t row = 0; row < nrows; ++row) {
        quantize_row_iq1_kt_impl(src + row * n_per_row, qrow, (int)n_per_row, imatrix);
        qrow += row_size;
    }
    return (size_t)nrows * row_size;
}

void quantize_row_iq1_kt_ref(const float * GGML_RESTRICT x, block_iq1_kt * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    quantize_iq1_kt(x, (void *)y, 1, k, nullptr);
}

void quantize_row_iq1_kt(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_iq1_kt_ref(x, (block_iq1_kt *)y, k);
}

void ggml_vec_dot_iq1_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs,
                               const void * GGML_RESTRICT vx, size_t bx,
                               const void * GGML_RESTRICT vy, size_t by, int nrc) {
    assert(n % QK_K == 0);
    assert(nrc == 1);
    (void)nrc; (void)bx; (void)by; (void)bs;
    const float * dptr = (const float *)vx;
    const float d = dptr[0];
    const block_iq1_kt * x = (const block_iq1_kt *)(dptr + 1);
    const block_q8_K   * y = (const block_q8_K   *)vy;
    const int nblock = n / QK_K;
    float sumf = 0;
    float gv[kIQ1KT_GroupSize];
    for (int ibl = 0; ibl < nblock; ++ibl) {
        const float db = d * y[ibl].d;
        const int8_t * q8 = y[ibl].qs;
        for (int ib = 0; ib < kIQ1KT_NBlock; ++ib) {
            const float sl = (float)kIQ1KT_iq4k_values[x[ibl].sh[ib] & 0xf];
            for (int ig = 0; ig < kIQ1KT_Ng; ++ig) {
                const int jj = ib * kIQ1KT_Ng + ig;
                uint32_t idx = (uint32_t)x[ibl].ql[jj]
                    | (((uint32_t)x[ibl].qh[jj % 16] << (8 - 4 * (jj / 16))) & 0xf00u)
                    | (((uint32_t)x[ibl].sh[jj / 4]  << (8 - (jj % 4)))      & 0x1000u);
                iqkt_gen_group_int<kIQ1KT_GroupSize, false>((int)idx, kIQ1KT_Offset, gv);
                for (int kk = 0; kk < kIQ1KT_GroupSize; ++kk) sumf += db * sl * gv[kk] * (float)q8[kk];
                q8 += kIQ1KT_GroupSize;
            }
        }
    }
    *s = sumf;
}

}  // extern "C"
