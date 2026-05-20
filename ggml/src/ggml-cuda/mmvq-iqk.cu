// Standalone MMVQ kernels for yggdrasil IQK base weight types.
//
// MMVQ = matrix-vector multiplication with quantized weights, computed without
// first dequantizing the entire weight tensor to F16/F32.  Streams through the
// compressed weights, does dp4a-style int8 dot products against Q8_1-quantized
// activations, and accumulates into the output.  Important for token-generation
// throughput where the input is small (one token) and memory bandwidth on the
// dequantize step would otherwise dominate.
//
// Implements IQ4_K (137), IQ3_K (138), IQ2_K (139).
// Other types continue to use the cuBLAS-dequant fallback.
//
// Design: one warp per output row, 4 warps per CUDA block.  Each lane (thread)
// strides through the row's sub-blocks at stride 32, computes a partial sum
// via dp4a, then a warp-level reduction emits one output value per warp.
//
// Lifted from frankenturbo2/feature/turboquant-kv-cache (mmvq-iqk.cu),
// stripped to IQ4_K/IQ3_K/IQ2_K only.

#include "mmvq-iqk.cuh"
#include "convert.cuh"
#include "common.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"      // get_int_from_table_16
#include "turbo-quant.cuh"  // iq4k_values, iq3nl_values_dev, iq2nl_values_dev

#define MMVQ_IQK_NWARPS 4

// ============================================================================
// IQ4_K MMVQ kernel.  No row meta.  Per block_iq4_k:
//   half d, uint16 extra (2 codebook-shift bits per ib32),
//   scales_h[4] (2-bit high parts of 16 sub-block scales),
//   scales_l[8] (4-bit low parts), qs[128] (4-bit indices).
// Each ib32 covers 32 elements split into two 16-elem halves with separate
// scales (ls1, ls2) and separate codebook shifts (extra bits ib32*2, ib32*2+1).
// ============================================================================

static __global__ void mul_mat_vec_iq4_k_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const block_iq4_k * x = (const block_iq4_k *)((const char *)vx + (size_t)row * row_size_x);
    const block_q8_1  * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq4_k * bq4 = &x[blk];
        const block_q8_1  * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t  * q8 = (const int32_t *)bq8->qs;
        const uint16_t * q4 = (const uint16_t *)bq4->qs + 8 * ib32;
        const uint16_t  extra = bq4->extra >> (2 * ib32);

        const int8_t * values_low  = iq4k_values + ((extra & 1) << 4);
        const int8_t * values_high = iq4k_values + ((extra & 2) << 3);

        int sumi1 = 0, sumi2 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t aux32 = (uint32_t)q4[2*j + 0] | ((uint32_t)q4[2*j + 1] << 16);
            const int2 v_lo = get_int_from_table_16((int)aux32, values_low);
            const int2 v_hi = get_int_from_table_16((int)aux32, values_high);
            sumi1 = ggml_cuda_dp4a(v_lo.x, q8[j + 0], sumi1);
            sumi2 = ggml_cuda_dp4a(v_hi.y, q8[j + 4], sumi2);
        }

        const float    d   = __half2float(bq4->d) * __low2float(bq8->ds);
        const uint8_t  sh  = bq4->scales_h[ib32 / 2] >> (4 * (ib32 % 2));
        const int      ls1 = ((bq4->scales_l[ib32] & 0xf)  | ((sh << 4) & 0x30)) - 32;
        const int      ls2 = ((bq4->scales_l[ib32] >>   4) | ((sh << 2) & 0x30)) - 32;

        sumf += d * (float)(sumi1 * ls1 + sumi2 * ls2);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq4_k_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq4_k_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ3_K MMVQ kernel.  Per-half-block lane work (4 sub-blocks per lane).  Each
// half-block (128 elements) contributes 4 separate sumi values, each scaled by
// per-sub-block magnitude (4-bit, *2+1 odd-only, signed via scales_h bit).
// ============================================================================
// iq3nl_values_dev is defined in turbo-quant.cuh (included above).

static __global__ void mul_mat_vec_iq3_k_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const block_iq3_k * x = (const block_iq3_k *)((const char *)vx + (size_t)row * row_size_x);
    const block_q8_1  * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_halves = n_blocks * 2;

    float sumf = 0.0f;

    for (int hb_flat = lane; hb_flat < total_halves; hb_flat += 32) {
        const int blk   = hb_flat / 2;
        const int ib128 = hb_flat % 2;

        const block_iq3_k * bq3 = &x[blk];
        const uint16_t * ql_base = (const uint16_t *)bq3->qs + 16*ib128;
        const uint16_t * qh_base = (const uint16_t *)bq3->qh;
        const int        hshift  = 4*(1 - ib128);
        const float      d       = __half2float(bq3->d);

        // sumi_lo[k] accumulates over il8∈{0,1} (low 16 elem of each sub-block),
        // sumi_hi[k] over il8∈{2,3} (high 16 elem).  Each pair gets its own scale.
        int sumi_lo[4] = {0, 0, 0, 0};
        int sumi_hi[4] = {0, 0, 0, 0};

        for (int il8 = 0; il8 < 4; ++il8) {
            // IQ3_K codebook bits live in `extra` at position 2*ib32 + half,
            // where ib32 = 4*ib128 + sb_idx (0..3) and half = il8/2 (0=low,
            // 1=high). For all 4 sub-blocks in this half, the bits sit at
            // strides of 2 starting from (8*ib128 + half).
            // val1.lo = sb0, val2.lo = sb1, val1.hi = sb2, val2.hi = sb3.
            const int      half       = il8 / 2;
            const int      start_bit  = 8*ib128 + half;
            const uint32_t e_shifted  = (uint32_t)(bq3->extra >> start_bit);
            const uint32_t b0         = (e_shifted >> 0) & 1u;  // sub-block 0 (or 4 for ib128=1)
            const uint32_t b1         = (e_shifted >> 2) & 1u;  // sub-block 1 (or 5)
            const uint32_t b2         = (e_shifted >> 4) & 1u;  // sub-block 2 (or 6)
            const uint32_t b3         = (e_shifted >> 6) & 1u;  // sub-block 3 (or 7)
            const uint32_t e1 = (b0 * 0x08080808u) | (b2 * 0x80808080u);
            const uint32_t e2 = (b1 * 0x08080808u) | (b3 * 0x80808080u);

            int * sumi_dst = (il8 < 2) ? sumi_lo : sumi_hi;
            const block_q8_1 * bq8_b = &y[blk * (QK_K/QK8_1) + 4*ib128];

            for (int i = 0; i < 2; ++i) {
                const uint32_t vl = (uint32_t)ql_base[4*il8 + 2*i + 0] |
                                    ((uint32_t)ql_base[4*il8 + 2*i + 1] << 16);
                const uint32_t vh = (((uint32_t)qh_base[4*il8 + 2*i + 0] |
                                      ((uint32_t)qh_base[4*il8 + 2*i + 1] << 16)) << hshift);

                const uint32_t val1 = ((vl >> 0) & 0x33333333) | e1 | ((vh >> 2) & 0x04040404) | ((vh >> 0) & 0x40404040);
                const uint32_t val2 = ((vl >> 2) & 0x33333333) | e2 | ((vh >> 3) & 0x04040404) | ((vh >> 1) & 0x40404040);

                const int2 v1 = get_int_from_table_16((int)val1, iq3nl_values_dev);
                const int2 v2 = get_int_from_table_16((int)val2, iq3nl_values_dev);

                sumi_dst[0] = ggml_cuda_dp4a(v1.x, ((const int32_t *)bq8_b[0].qs)[2*il8 + i], sumi_dst[0]);
                sumi_dst[1] = ggml_cuda_dp4a(v2.x, ((const int32_t *)bq8_b[1].qs)[2*il8 + i], sumi_dst[1]);
                sumi_dst[2] = ggml_cuda_dp4a(v1.y, ((const int32_t *)bq8_b[2].qs)[2*il8 + i], sumi_dst[2]);
                sumi_dst[3] = ggml_cuda_dp4a(v2.y, ((const int32_t *)bq8_b[3].qs)[2*il8 + i], sumi_dst[3]);
            }
        }

        const uint16_t * sl16 = (const uint16_t *)bq3->scales_l + 2*ib128;
        const uint32_t  sl32 = (uint32_t)sl16[0] | ((uint32_t)sl16[1] << 16);
        const uint16_t  sh   = bq3->scales_h >> (8*ib128);

        // Per-sub-sub-block scales (4-bit magnitudes, *2+1 odd-only encoding).
        const uint32_t aux_lo = ((((sl32 >> 0) & 0x0f0f0f0f) << 1) | 0x01010101);
        const uint32_t aux_hi = ((((sl32 >> 4) & 0x0f0f0f0f) << 1) | 0x01010101);
        const int8_t * a8_lo = (const int8_t *)&aux_lo;
        const int8_t * a8_hi = (const int8_t *)&aux_hi;

        // Sign bits per sub-sub-block.
        const int sg0_lo = (sh & 0x01) ? -1 : 1;
        const int sg1_lo = (sh & 0x04) ? -1 : 1;
        const int sg2_lo = (sh & 0x10) ? -1 : 1;
        const int sg3_lo = (sh & 0x40) ? -1 : 1;
        const int sg0_hi = (sh & 0x02) ? -1 : 1;
        const int sg1_hi = (sh & 0x08) ? -1 : 1;
        const int sg2_hi = (sh & 0x20) ? -1 : 1;
        const int sg3_hi = (sh & 0x80) ? -1 : 1;

        const block_q8_1 * bq8_b = &y[blk * (QK_K/QK8_1) + 4*ib128];
        sumf += d * (__low2float(bq8_b[0].ds) * (a8_lo[0] * sg0_lo * (float)sumi_lo[0] + a8_hi[0] * sg0_hi * (float)sumi_hi[0]) +
                     __low2float(bq8_b[1].ds) * (a8_lo[1] * sg1_lo * (float)sumi_lo[1] + a8_hi[1] * sg1_hi * (float)sumi_hi[1]) +
                     __low2float(bq8_b[2].ds) * (a8_lo[2] * sg2_lo * (float)sumi_lo[2] + a8_hi[2] * sg2_hi * (float)sumi_hi[2]) +
                     __low2float(bq8_b[3].ds) * (a8_lo[3] * sg3_lo * (float)sumi_lo[3] + a8_hi[3] * sg3_hi * (float)sumi_hi[3]));
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq3_k_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq3_k_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ2_K MMVQ kernel.  Per block_iq2_k:
//   half d, uint16 extra (1 codebook-shift bit per 16-element sub-block),
//   scales[8] (two 4-bit signed-offset-by-8 scales per 32-elem ib32),
//   qs[64] (2-bit indices).  qs[0..31] is shared by ib32 ∈ {0..3}, qs[32..63]
//   by ib32 ∈ {4..7}; per-ib32 selection uses shift = 2*(ib32 & 3).  Within
//   each 32-byte stripe, bytes [0..15] are the low half (16 elements) and
//   [16..31] are the high half, with separate scales (ls_lo, ls_hi) and
//   independent codebook shifts (extra bits 2*ib32, 2*ib32+1).
// ============================================================================

// 4-entry codebook lookup.  idx_packed contains 4 byte-indices each in low 2
// bits; t is 4 codebook bytes packed as a uint32.  Returns the 4 selected
// bytes packed in a uint32.
static __device__ __forceinline__ uint32_t iq2k_lookup4(uint32_t idx_packed, uint32_t t) {
#if defined(GGML_USE_HIP)
    // Concat (0:t) → 8 bytes; per-byte selector picks bytes 0..3 (always in t).
    return __builtin_amdgcn_perm(0u, t, idx_packed);
#elif !defined(GGML_USE_MUSA)
    // Compress 4 byte-indices (low 2 bits each) into nibble selectors for __byte_perm.
    const uint32_t sel = (idx_packed & 0x00000003u)
                       | ((idx_packed & 0x00000300u) >>  4)
                       | ((idx_packed & 0x00030000u) >>  8)
                       | ((idx_packed & 0x03000000u) >> 12);
    return __byte_perm(t, 0u, sel);
#else
    // Generic fallback.
    const uint8_t * tb = (const uint8_t *)&t;
    const uint8_t i0 = (uint8_t)((idx_packed >>  0) & 3u);
    const uint8_t i1 = (uint8_t)((idx_packed >>  8) & 3u);
    const uint8_t i2 = (uint8_t)((idx_packed >> 16) & 3u);
    const uint8_t i3 = (uint8_t)((idx_packed >> 24) & 3u);
    return ((uint32_t)tb[i0] <<  0) | ((uint32_t)tb[i1] <<  8)
         | ((uint32_t)tb[i2] << 16) | ((uint32_t)tb[i3] << 24);
#endif
}

// iq2nl_values_dev is defined in turbo-quant.cuh (included above).

static __global__ void mul_mat_vec_iq2_k_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const block_iq2_k * x = (const block_iq2_k *)((const char *)vx + (size_t)row * row_size_x);
    const block_q8_1  * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq2_k * bq2 = &x[blk];
        const block_q8_1  * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int qs_off = 32 * (ib32 >> 2);
        const int shift  = 2 * (ib32 & 3);

        // Two halves of 16 elements each, 16 bytes apart, 4-byte aligned.
        const uint32_t * qs_lo = (const uint32_t *)(bq2->qs + qs_off);
        const uint32_t * qs_hi = qs_lo + 4;   // +16 bytes

        const uint8_t  scale_byte = bq2->scales[ib32];
        const int      ls_lo      = (int)(scale_byte & 0xf) - 8;
        const int      ls_hi      = (int)(scale_byte >>  4) - 8;

        const uint16_t extra      = bq2->extra >> (2 * ib32);
        const int8_t * vals_lo    = iq2nl_values_dev + ((extra & 1) << 2);
        const int8_t * vals_hi    = iq2nl_values_dev + ((extra & 2) << 1);
        const uint32_t t_lo       = *(const uint32_t *)vals_lo;
        const uint32_t t_hi       = *(const uint32_t *)vals_hi;

        const int32_t * q8 = (const int32_t *)bq8->qs;

        int sumi_lo = 0, sumi_hi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t idx_lo   = (qs_lo[j] >> shift) & 0x03030303u;
            const uint32_t idx_hi   = (qs_hi[j] >> shift) & 0x03030303u;
            const uint32_t codes_lo = iq2k_lookup4(idx_lo, t_lo);
            const uint32_t codes_hi = iq2k_lookup4(idx_hi, t_hi);
            sumi_lo = ggml_cuda_dp4a((int)codes_lo, q8[j + 0], sumi_lo);
            sumi_hi = ggml_cuda_dp4a((int)codes_hi, q8[j + 4], sumi_hi);
        }

        const float d = __half2float(bq2->d) * __low2float(bq8->ds);
        sumf += d * (float)(sumi_lo * ls_lo + sumi_hi * ls_hi);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq2_k_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq2_k_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// Public dispatch
// ============================================================================

bool ggml_cuda_iqk_mmvq_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ2_K:
            return true;
        default:
            return false;
    }
}

void ggml_cuda_mul_mat_iqk_mmvq(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1,
                                ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    cudaStream_t stream = ctx.stream();

    const int64_t ne00 = src0->ne[0];   // K
    const int64_t ne01 = src0->ne[1];   // N (output rows)
    const int64_t ne10 = src1->ne[0];   // K (must match ne00)
    const int64_t ne11 = src1->ne[1];   // batch
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];

    GGML_ASSERT(ne00 == ne10);
    GGML_ASSERT(ne12 == 1 && ne13 == 1 && "IQK MMVQ only handles 2D MUL_MAT for now");
    GGML_ASSERT(ne11 == 1 && "IQK MMVQ implements single-token decode path only");

    const size_t row_size_x = ggml_row_size(src0->type, ne00);

    // Quantize src1 (F32, single token) into block_q8_1.
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne10_padded * sizeof(block_q8_1) / QK8_1);
    quantize_row_q8_1_cuda((const float *)src1->data, /*ids=*/nullptr, src1_q8_1.get(), src0->type,
                           ne10, /*s11=*/ne10, /*s12=*/ne10, /*s13=*/ne10,
                           ne10_padded, /*ne11=*/1, /*ne12=*/1, /*ne13=*/1, stream);

    switch (src0->type) {
        case GGML_TYPE_IQ4_K:
            launch_mul_mat_vec_iq4_k_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ3_K:
            launch_mul_mat_vec_iq3_k_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ2_K:
            launch_mul_mat_vec_iq2_k_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        default:
            GGML_ASSERT(false && "ggml_cuda_iqk_mmvq_supported gate let through an unhandled type");
    }
}
