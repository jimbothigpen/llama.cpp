// Standalone MMVQ kernels for yggdrasil IQK weight types (lifted from ft2).
//
// MMVQ = matrix-vector multiplication with quantized weights, computed without
// first dequantizing the entire weight tensor to F16/F32.  Streams through the
// compressed weights, does dp4a-style int8 dot products against Q8_1-quantized
// activations, and accumulates into the output.  Important for token-generation
// throughput where the input is small (one token) and memory bandwidth on the
// dequantize step would otherwise dominate.
//
// Implements IQ4_KT (trellis), IQ4_KS, IQ4_KSS, IQ4_K, IQ3_K, IQ3_KS, IQ2_K.
// Other IQK types continue to use the cuBLAS-dequant fallback.
//
// Design: one warp per output row, 4 warps per CUDA block.  Each lane (thread)
// strides through the row's sub-blocks at stride 32, computes a partial sum
// via dp4a, then a warp-level reduction emits one output value per warp.

#include "mmvq-iqk.cuh"
#include "convert.cuh"
#include "common.cuh"
#include "quantize.cuh"
#include "vecdotq.cuh"      // get_int_from_table_16
#include "turbo-quant.cuh"  // iq4k_values

#define MMVQ_IQK_NWARPS 4

// ============================================================================
// IQ4_KT MMVQ kernel.  Mirrors ik_llama's vec_dot_iq4_kt_q8_1 but inlined into
// a per-row reduction kernel.
// ============================================================================

static __global__ void mul_mat_vec_iq4_kt_q8_1_kernel(
        const void * __restrict__ vx,        // weight tensor (per-row float scale + blocks)
        const void * __restrict__ vy,        // src1 quantized to block_q8_1
        float       * __restrict__ dst,      // output [nrows_x]
        const int    ncols_x,                // K dimension (input features)
        const int    nrows_x,                // N dimension (output rows)
        const size_t row_size_x) {           // bytes per row of weight (incl. row meta)

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    // Per-row weight pointer: 4 bytes float scale, then block_iq4_kt array.
    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq4_kt * x = (const block_iq4_kt *)(row_ptr + sizeof(float));

    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;   // 8 sub-blocks per QK_K=256 superblock

    float sumf = 0.0f;

    // Lane k handles sub-blocks { k, k+32, k+64, ... }.
    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq4_kt * bq4 = &x[blk];
        // Q8_1 has 32 elements per block, so QK_K/QK8_1 = 8 q8_1 blocks per superblock.
        const block_q8_1 * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t * q8 = (const int32_t *)bq8->qs;     // 8 int32 = 32 int8
        const int      ls = (int)((bq4->qs[ib32] & 0xff) >> 1);
        const float    dl = row_scale * (float)(ls - 64);
        const uint32_t idx0 = ((bq4->qs[ib32] & 1) << 15) + 4096;

        const uint8_t * ql = (const uint8_t *)(bq4->qs + 8);     // 64 bytes: low 8 bits per group
        const uint8_t * qh = ql + 64;                              // 16 bytes: 4 mid bits per group
        ql += 8 * ib32;
        qh += 8 * (ib32 % 4);
        const int shift1 = 8 - 4 * (ib32 / 4);

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const uint32_t sh = bq4->qs[ib32] >> (8 + 3 * j);
            uint32_t val = ql[j] + ((qh[j] << shift1) & 0xf00) + ((sh & 7) << 12) + idx0;
            int v4 = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                // dp4a(val & km, 0x01010101, -126) = -126 + sum of 4 bytes of (val & 0x3f3f3f3f).
                // Each byte is in [0, 63]; sum is in [0, 252]; result in [-126, 126].
                v4 |= (ggml_cuda_dp4a((int)(val & km), 0x01010101, -126) & 0xff) << (8 * k);
            }
            // v4 packs 4 int8 codebook values; q8[j] packs 4 int8 activations.
            sumi = ggml_cuda_dp4a(v4, q8[j], sumi);
        }

        sumf += dl * __low2float(bq8->ds) * (float)sumi;
    }

    // Warp-level reduction.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq4_kt_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq4_kt_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ3_KT MMVQ kernel.  Per-row layout: float row scale + N block_iq3_kt (96 B).
// IS_ABS=true single codebook, GROUP_SIZE=8, NUM_BITS=16, offset=4096.
// Block: qs[0..7]=shb (scale+4 high bits per group), qs[8..15]=ql, qs[16..19]=qh.
// ============================================================================

static __global__ void mul_mat_vec_iq3_kt_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq3_kt * x = (const block_iq3_kt *)(row_ptr + sizeof(float));

    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;   // 8 sub-blocks per QK_K=256 superblock

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;   // sub-block 0..7

        const block_iq3_kt * bq3 = &x[blk];
        const block_q8_1 * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t * q8 = (const int32_t *)bq8->qs;  // 8 int32 = 32 int8

        const int ls = (int)(bq3->qs[ib32] & 0xff) - 128;
        const float dl = row_scale * (float)ls;
        const uint32_t idx0 = (bq3->qs[ib32] >> 24) & 1u ? 4096u + 32768u : 4096u;

        // ql: 32 bytes (8 uint32_t) starting at qs[8]; 1 byte per group, 4 groups per u32
        const uint8_t * ql_base = (const uint8_t *)(bq3->qs + 8);
        // qh: 16 bytes (4 uint32_t) starting at qs[16]; 2 groups per byte (nibble each)
        const uint8_t * qh_base = (const uint8_t *)(bq3->qs + 16);

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {   // 4 groups per sub-block (Ng=4)
            const int jj = 4 * ib32 + j;
            const uint8_t ql_val   = ql_base[jj];
            const int qh_byte_idx  = jj >> 1;    // jj/2, range 0..15
            const int qh_nib_shift = (jj & 1) * 4;
            const uint8_t qh_nib   = (qh_base[qh_byte_idx] >> qh_nib_shift) & 0xf;
            const uint32_t sh_4bits = (bq3->qs[ib32] >> (8 + 4 * j)) & 0xf;
            uint32_t val = (uint32_t)ql_val | ((uint32_t)qh_nib << 8) | (sh_4bits << 12) | idx0;

            // First 4 elements of the 8-element group
            int v4a = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
                v4a |= (abs(sv) & 0xff) << (8 * k);
            }
            // Second 4 elements
            int v4b = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
                v4b |= (abs(sv) & 0xff) << (8 * k);
            }

            sumi = ggml_cuda_dp4a(v4a, q8[2 * j],     sumi);
            sumi = ggml_cuda_dp4a(v4b, q8[2 * j + 1], sumi);
        }

        sumf += dl * __low2float(bq8->ds) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq3_kt_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq3_kt_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ2_KT MMVQ kernel.  Per-row layout: float row scale + N block_iq2_kt (64 B:
// uint16_t qs[32], one 16-bit codebook index per GROUP_SIZE=8 group).  Trellis,
// NUM_BITS=16, offset=0, IS_ABS=false (signed outputs, NO abs).  There is no
// sub-block scale: the per-row float is the only scale (mirrors the CPU
// ggml_vec_dot_iq2_kt_q8_K, which uses iqkt_gen_group_int<8> signed values).
// ============================================================================
static __global__ void mul_mat_vec_iq2_kt_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {
    constexpr uint32_t ka = 0xCBAC1FED, km = 0x3f3f3f3f;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq2_kt * x = (const block_iq2_kt *)(row_ptr + sizeof(float));

    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;   // 8 q8_1 sub-blocks (32 elems) per QK_K superblock

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;   // sub-block 0..7

        const block_iq2_kt * bq2 = &x[blk];
        const block_q8_1 * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t * q8 = (const int32_t *)bq8->qs;  // 8 int32 = 32 int8

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {   // 4 groups of 8 = 32 elems per sub-block
            const int g = 4 * ib32 + j;             // flat group 0..31
            uint32_t val = (uint32_t)bq2->qs[g];    // offset 0

            // First 4 elements of the 8-element group
            int v4a = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
                v4a |= (sv & 0xff) << (8 * k);   // signed, NOT abs (IS_ABS=false)
            }
            // Second 4 elements
            int v4b = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
                v4b |= (sv & 0xff) << (8 * k);
            }

            sumi = ggml_cuda_dp4a(v4a, q8[2 * j],     sumi);
            sumi = ggml_cuda_dp4a(v4b, q8[2 * j + 1], sumi);
        }

        sumf += row_scale * __low2float(bq8->ds) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq2_kt_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq2_kt_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ1_KT MMVQ kernel.  Per-row layout: float row scale + N block_iq1_kt (56 B:
// sh[8] ql[32] qh[16]).  Trellis, GROUP_SIZE=8, NUM_BITS=13, offset=4096,
// IS_ABS=false (signed outputs, NO abs).  Per-sub-block scale = iq4k_values[sh&0xf].
//   idx[jj] = ql[jj] | ((qh[jj%16]<<(8-4*(jj/16)))&0xf00) | ((sh[jj/4]<<(8-(jj%4)))&0x1000)
// ============================================================================
static __global__ void mul_mat_vec_iq1_kt_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    constexpr uint32_t ka = 0xCBAC1FED;
    constexpr uint32_t km = 0x3f3f3f3f;

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq1_kt * x = (const block_iq1_kt *)(row_ptr + sizeof(float));

    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;   // 8 sub-blocks per QK_K=256 superblock

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;   // sub-block 0..7

        const block_iq1_kt * bq1 = &x[blk];
        const block_q8_1 * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t * q8 = (const int32_t *)bq8->qs;  // 8 int32 = 32 int8

        const int   sh_b = bq1->sh[ib32];
        const float dl   = row_scale * (float)iq4k_values[sh_b & 0xf];

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {   // 4 groups per sub-block (Ng=4)
            const int jj = 4 * ib32 + j;     // flat group 0..31
            const uint32_t idx = (uint32_t)bq1->ql[jj]
                | (((uint32_t)bq1->qh[jj % 16] << (8 - 4 * (jj / 16))) & 0xf00u)
                | (((uint32_t)bq1->sh[jj / 4]  << (8 - (jj % 4)))      & 0x1000u);
            uint32_t val = idx + 4096u;

            // First 4 elements of the 8-element group
            int v4a = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
                v4a |= (sv & 0xff) << (8 * k);   // signed, NOT abs (IS_ABS=false)
            }
            // Second 4 elements
            int v4b = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                val *= ka;
                const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
                v4b |= (sv & 0xff) << (8 * k);
            }

            sumi = ggml_cuda_dp4a(v4a, q8[2 * j],     sumi);
            sumi = ggml_cuda_dp4a(v4b, q8[2 * j + 1], sumi);
        }

        sumf += dl * __low2float(bq8->ds) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq1_kt_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq1_kt_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ4_KS MMVQ kernel.  Per-row layout: float row scale + N block_iq4_ks (136 B
// each: 8 scale bytes + 128 qs bytes).  Each sub-block (32 elements) has:
//   - scale byte: low bit selects iq4k_values vs +16-shifted variant; high 7
//     bits are signed-offset-127 dl multiplier.
//   - 16 qs bytes (32 4-bit indices), arranged so q4[0..3] decode to even-nibble
//     activations [4j..4j+3] and odd-nibble activations [4j+16..4j+19].
// ============================================================================

static __global__ void mul_mat_vec_iq4_ks_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq4_ks * x = (const block_iq4_ks *)(row_ptr + sizeof(float));

    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq4_ks * bq4 = &x[blk];
        const block_q8_1 * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t  * q8 = (const int32_t *)bq8->qs;
        const uint32_t * q4 = (const uint32_t *)bq4->qs + 4 * ib32;
        const uint8_t   sb = bq4->scales[ib32];
        const float    dl = row_scale * (float)((int)(sb & 254) - 127);
        const int8_t * values = iq4k_values + ((sb & 1) << 4);

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int2 v = get_int_from_table_16((int)q4[j], values);
            sumi = ggml_cuda_dp4a(v.x, q8[j + 0], sumi);
            sumi = ggml_cuda_dp4a(v.y, q8[j + 4], sumi);
        }

        sumf += dl * __low2float(bq8->ds) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq4_ks_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq4_ks_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ4_KSS MMVQ kernel.  Per-row layout: float row scale + N block_iq4_kss
// (128 B each, 32 uint32_t).  Each 32-elem sub-block is 4 uint32_t where:
//   bit 0 of each q4 carries 1 of 8 bits of the per-sub-block scale byte
//   bits 1..15 of each q4 hold even-parity scrambled 4-bit indices
//     (Gray-code self-inverse: aux32 = q & 0xfffefffe; aux32 ^= aux32 >> 1)
// ============================================================================

static __global__ void mul_mat_vec_iq4_kss_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq4_kss * x = (const block_iq4_kss *)(row_ptr + sizeof(float));

    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_subblocks = n_blocks * 8;

    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq4_kss * bq4 = &x[blk];
        const block_q8_1 * bq8 = &y[blk * (QK_K / QK8_1) + ib32];

        const int32_t  * q8 = (const int32_t *)bq8->qs;
        const uint32_t * q4 = (const uint32_t *)bq4->qs + 4 * ib32;

        // Reconstruct sub-block scale byte from bit0 of each of the 4 uint32s.
        uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2)
                     | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
        uint8_t  ls  = (s32 | (s32 >> 15)) & 0xff;
        const float    dl     = row_scale * (float)((int)(ls & 254) - 127);
        const int8_t * values = iq4k_values + ((ls & 1) << 4);

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            uint32_t aux32 = q4[j] & 0xfffefffe;
            aux32 ^= (aux32 >> 1);
            const int2 v = get_int_from_table_16((int)aux32, values);
            sumi = ggml_cuda_dp4a(v.x, q8[j + 0], sumi);
            sumi = ggml_cuda_dp4a(v.y, q8[j + 4], sumi);
        }

        sumf += dl * __low2float(bq8->ds) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq4_kss_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq4_kss_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

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

extern __constant__ int8_t iq3nl_values_dev[];

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

        // Sign bits per sub-sub-block.  ik_llama checks bit (1<<(2k)) of (sh>>il8/2).
        // For lo (il8/2=0): bits 0,2,4,6 of sh.  For hi (il8/2=1): bits 1,3,5,7 of sh.
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
// IQ3_KS MMVQ kernel.  Per-row half row_scale + N block_iq3_ks (102 B each:
// 2 extra + 4 scales + 64 qs + 32 qh).  Each ib32 sub-block has a single
// 5-bit signed scale (4-bit magnitude from scales[] + high bit from extra
// low byte) and a 1-bit codebook shift (extra high byte).
// Per-half-block lane work, mirroring IQ3_K layout.
// ============================================================================

static __global__ void mul_mat_vec_iq3_ks_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = __half2float(*(const __half *)row_ptr);
    const block_iq3_ks * x = (const block_iq3_ks *)(row_ptr + sizeof(__half));
    const block_q8_1   * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_halves = n_blocks * 2;

    float sumf = 0.0f;

    for (int hb_flat = lane; hb_flat < total_halves; hb_flat += 32) {
        const int blk   = hb_flat / 2;
        const int ib128 = hb_flat % 2;

        const block_iq3_ks * bq3 = &x[blk];
        const uint16_t * ql_base = (const uint16_t *)bq3->qs + 16*ib128;
        const uint16_t * qh_base = (const uint16_t *)bq3->qh;
        const int        hshift  = 4*ib128;
        const uint16_t   extra   = bq3->extra >> 4*ib128;

        const uint32_t extra_v   = (uint32_t)(extra >> 8) * 0x01010101;
        const uint32_t e1 = ((extra_v << 3) & 0x08080808) | ((extra_v << 5) & 0x80808080);
        const uint32_t e2 = ((extra_v << 2) & 0x08080808) | ((extra_v << 4) & 0x80808080);

        int sumi[4] = {0, 0, 0, 0};
        const block_q8_1 * bq8_b = &y[blk * (QK_K/QK8_1) + 4*ib128];

        for (int il8 = 0; il8 < 4; ++il8) {
            for (int i = 0; i < 2; ++i) {
                const uint32_t vl = (uint32_t)ql_base[4*il8 + 2*i + 0] |
                                    ((uint32_t)ql_base[4*il8 + 2*i + 1] << 16);
                const uint32_t vh = (((uint32_t)qh_base[4*il8 + 2*i + 0] |
                                      ((uint32_t)qh_base[4*il8 + 2*i + 1] << 16)) >> hshift);

                const uint32_t val1 = ((vl >> 0) & 0x33333333) | e1 | ((vh << 2) & 0x04040404) | ((vh << 4) & 0x40404040);
                const uint32_t val2 = ((vl >> 2) & 0x33333333) | e2 | ((vh << 1) & 0x04040404) | ((vh << 3) & 0x40404040);

                const int2 v1 = get_int_from_table_16((int)val1, iq3nl_values_dev);
                const int2 v2 = get_int_from_table_16((int)val2, iq3nl_values_dev);

                sumi[0] = ggml_cuda_dp4a(v1.x, ((const int32_t *)bq8_b[0].qs)[2*il8 + i], sumi[0]);
                sumi[1] = ggml_cuda_dp4a(v2.x, ((const int32_t *)bq8_b[1].qs)[2*il8 + i], sumi[1]);
                sumi[2] = ggml_cuda_dp4a(v1.y, ((const int32_t *)bq8_b[2].qs)[2*il8 + i], sumi[2]);
                sumi[3] = ggml_cuda_dp4a(v2.y, ((const int32_t *)bq8_b[3].qs)[2*il8 + i], sumi[3]);
            }
        }

        // Per-sub-block scales: 4-bit magnitude (offset −16) + +16 from extra low byte.
        const uint16_t * sl16 = (const uint16_t *)bq3->scales;
        const int32_t   aux32 = __vsub4((int)(((uint32_t)sl16[0] | ((uint32_t)sl16[1] << 16)) >> 4*ib128) & 0x0f0f0f0f, 0x10101010);
        const int8_t * a8 = (const int8_t *)&aux32;

        const float scale0 = (float)(a8[0] + ((extra << 4) & 0x10));
        const float scale1 = (float)(a8[1] + ((extra << 3) & 0x10));
        const float scale2 = (float)(a8[2] + ((extra << 2) & 0x10));
        const float scale3 = (float)(a8[3] + ((extra << 1) & 0x10));

        sumf += row_scale * (__low2float(bq8_b[0].ds) * scale0 * (float)sumi[0] +
                             __low2float(bq8_b[1].ds) * scale1 * (float)sumi[1] +
                             __low2float(bq8_b[2].ds) * scale2 * (float)sumi[2] +
                             __low2float(bq8_b[3].ds) * scale3 * (float)sumi[3]);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq3_ks_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq3_ks_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ2_KL MMVQ kernel.
// Row layout: [half row_scale][N × block_iq2_kl]
// block_iq2_kl: scales_h(2B) + scales_l[4](4B) + qs[64](64B) + qh[16](16B)
// QK_K=256 elements split into 4 ib64 groups of 64 elements each.
// Each ib64 group: 2 sub-blocks of 32 elements (scale ls1, ls2).
// Each lane processes one ib64 group across all blocks, strided by 32.
// ============================================================================

static __global__ void mul_mat_vec_iq2_kl_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float d = __half2float(*(const __half *)row_ptr);
    const block_iq2_kl * x = (const block_iq2_kl *)(row_ptr + sizeof(__half));
    const block_q8_1   * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_groups = n_blocks * (QK_K / 64);  // n_blocks * 4

    float sumf = 0.0f;

    for (int g_flat = lane; g_flat < total_groups; g_flat += WARP_SIZE) {
        const int blk  = g_flat / 4;
        const int ib64 = g_flat % 4;

        const block_iq2_kl * bq2 = &x[blk];

        // qs[16*ib64 .. 16*ib64+15] and qh[0..15] for this ib64 group.
        const uint16_t * ql = (const uint16_t *)bq2->qs + 8*ib64;
        const uint16_t * qh = (const uint16_t *)bq2->qh;

        const block_q8_1 * bq8_s1 = &y[blk * (QK_K/QK8_1) + 2*ib64 + 0];
        const block_q8_1 * bq8_s2 = &y[blk * (QK_K/QK8_1) + 2*ib64 + 1];
        const int32_t * q8s1 = (const int32_t *)bq8_s1->qs;
        const int32_t * q8s2 = (const int32_t *)bq8_s2->qs;

        int sumi1 = 0, sumi2 = 0;

        // Process 2 halves of 16 ql entries × 2 iterations = 32 pairs total.
        for (int il = 0; il < 2; ++il) {
            for (int i = 0; i < 2; ++i) {
                uint32_t vl = (uint32_t)ql[4*il + 2*i + 0] | ((uint32_t)ql[4*il + 2*i + 1] << 16);
                uint32_t vh = ((uint32_t)qh[4*il + 2*i + 0] | ((uint32_t)qh[4*il + 2*i + 1] << 16)) >> (2*ib64);

                // sub-block 0 (ib = 2*ib64): index = (qs & 0xf) | (qh_bit << 4)
                int32_t aux1 = (int32_t)((vl & 0x0f0f0f0fu) | ((vh << 4) & 0x10101010u));
                const uint8_t * a1 = (const uint8_t *)&aux1;
                int32_t v1 = (int32_t)((uint32_t)iq2kl_values[a1[0]] | ((uint32_t)iq2kl_values[a1[1]] << 16));
                int32_t v2 = (int32_t)((uint32_t)iq2kl_values[a1[2]] | ((uint32_t)iq2kl_values[a1[3]] << 16));
                sumi1 = ggml_cuda_dp4a(v1, q8s1[4*il + 2*i + 0], ggml_cuda_dp4a(v2, q8s1[4*il + 2*i + 1], sumi1));

                // sub-block 1 (ib = 2*ib64+1): index = (qs >> 4) | (qh_bit_next << 4)
                int32_t aux2 = (int32_t)(((vl >> 4) & 0x0f0f0f0fu) | ((vh << 3) & 0x10101010u));
                const uint8_t * a2 = (const uint8_t *)&aux2;
                int32_t v3 = (int32_t)((uint32_t)iq2kl_values[a2[0]] | ((uint32_t)iq2kl_values[a2[1]] << 16));
                int32_t v4 = (int32_t)((uint32_t)iq2kl_values[a2[2]] | ((uint32_t)iq2kl_values[a2[3]] << 16));
                sumi2 = ggml_cuda_dp4a(v3, q8s2[4*il + 2*i + 0], ggml_cuda_dp4a(v4, q8s2[4*il + 2*i + 1], sumi2));
            }
        }

        // Scale decode: 6-bit signed (4 low bits from scales_l + 2 high bits from scales_h) - 32
        const uint16_t sh = bq2->scales_h >> (4*ib64);
        const int ls1 = (int)(((bq2->scales_l[(2*ib64+0) % 4] >> (4*(ib64/2))) & 0xf) | ((sh << 4) & 0x30)) - 32;
        const int ls2 = (int)(((bq2->scales_l[(2*ib64+1) % 4] >> (4*(ib64/2))) & 0xf) | ((sh << 2) & 0x30)) - 32;

        sumf += d * (__low2float(bq8_s1->ds) * (float)ls1 * (float)sumi1 +
                     __low2float(bq8_s2->ds) * (float)ls2 * (float)sumi2);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq2_kl_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq2_kl_q8_1_kernel<<<grid, block, 0, stream>>>(
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
// bytes packed in a uint32 (byte k of out = t[byte k of idx_packed & 3]).
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

extern __constant__ int8_t iq2nl_values_dev[];

// ============================================================================
// IQ2_KS MMVQ kernel.
// Row layout: [half row_scale][N × block_iq2_ks (70 B: 2 extra + 4 scales + 64 qs)]
// QK_K=256 elements split into 2 ib128 halves; each half holds 4 sub-blocks of 32
// (scales d1..d4). Each lane processes one ib128 across all blocks, strided by 32.
// 4-entry codebook; per-sub-block 1-bit shift selects iq2nl_values_dev[+4].
// ============================================================================
static __global__ void mul_mat_vec_iq2_ks_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float d = __half2float(*(const __half *)row_ptr);
    const block_iq2_ks * x = (const block_iq2_ks *)(row_ptr + sizeof(__half));
    const block_q8_1   * y = (const block_q8_1 *)vy;

    const int n_blocks = ncols_x / QK_K;
    const int total_halves = n_blocks * (QK_K / 128);   // n_blocks * 2

    float sumf = 0.0f;

    for (int hb_flat = lane; hb_flat < total_halves; hb_flat += WARP_SIZE) {
        const int blk   = hb_flat / 2;
        const int ib128 = hb_flat % 2;

        const block_iq2_ks * bq2 = &x[blk];
        const uint8_t * qs = bq2->qs + 32*ib128;          // 32 bytes = this half's 4 sub-blocks
        const uint16_t  extra = bq2->extra >> (4*ib128);

        // Four q8_1 blocks for this 128-element half.
        const block_q8_1 * bq8 = &y[blk * (QK_K/QK8_1) + 4*ib128];
        const int32_t * q8_1 = (const int32_t *)bq8[0].qs;
        const int32_t * q8_2 = (const int32_t *)bq8[1].qs;
        const int32_t * q8_3 = (const int32_t *)bq8[2].qs;
        const int32_t * q8_4 = (const int32_t *)bq8[3].qs;

        // Per-sub-block scales (5-bit, offset −16).
        const int s1 = (int)(((bq2->scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
        const int s2 = (int)(((bq2->scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
        const int s3 = (int)(((bq2->scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
        const int s4 = (int)(((bq2->scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);

        const int8_t * v1tab = (extra & 1) ? iq2nl_values_dev + 4 : iq2nl_values_dev;
        const int8_t * v2tab = (extra & 2) ? iq2nl_values_dev + 4 : iq2nl_values_dev;
        const int8_t * v3tab = (extra & 4) ? iq2nl_values_dev + 4 : iq2nl_values_dev;
        const int8_t * v4tab = (extra & 8) ? iq2nl_values_dev + 4 : iq2nl_values_dev;

        int sumi1 = 0, sumi2 = 0, sumi3 = 0, sumi4 = 0;
        for (int g = 0; g < 8; ++g) {     // 8 groups of 4 bytes = 32 elements
            int32_t p1 = 0, p2 = 0, p3 = 0, p4 = 0;
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                const uint8_t qb = qs[4*g + b];
                p1 |= ((int32_t)(uint8_t)v1tab[(qb >> 0) & 3]) << (8*b);
                p2 |= ((int32_t)(uint8_t)v2tab[(qb >> 2) & 3]) << (8*b);
                p3 |= ((int32_t)(uint8_t)v3tab[(qb >> 4) & 3]) << (8*b);
                p4 |= ((int32_t)(uint8_t)v4tab[(qb >> 6) & 3]) << (8*b);
            }
            sumi1 = ggml_cuda_dp4a(p1, q8_1[g], sumi1);
            sumi2 = ggml_cuda_dp4a(p2, q8_2[g], sumi2);
            sumi3 = ggml_cuda_dp4a(p3, q8_3[g], sumi3);
            sumi4 = ggml_cuda_dp4a(p4, q8_4[g], sumi4);
        }

        sumf += d * (__low2float(bq8[0].ds) * (float)s1 * (float)sumi1 +
                     __low2float(bq8[1].ds) * (float)s2 * (float)sumi2 +
                     __low2float(bq8[2].ds) * (float)s3 * (float)sumi3 +
                     __low2float(bq8[3].ds) * (float)s4 * (float)sumi4);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }

    if (lane == 0) {
        dst[row] = sumf;
    }
}

static void launch_mul_mat_vec_iq2_ks_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq2_ks_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

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
// IQ5_K MMVQ kernel.  No row meta.  Per block_iq5_k:
//   half d, uint16 extra (1 codebook-shift bit per 16-element group),
//   scales_h[4] (2-bit high parts of 16 sub-block scales),
//   scales_l[8] (4-bit low parts), qs[128] (4 low bits of 5-bit index),
//   qh[32] (1 high bit per element, packed 8 per byte).
// Scale format identical to IQ4_K (6-bit: 2h+4l, offset -32).
// ============================================================================

extern __constant__ int8_t iq5nl_values_dev[];

static __global__ void mul_mat_vec_iq5_k_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;
    const int lane = threadIdx.x;

    const block_iq5_k * x = (const block_iq5_k *)((const char *)vx + (size_t)row * row_size_x);
    const block_q8_1  * y = (const block_q8_1 *)vy;

    const int total_subblocks = (ncols_x / QK_K) * 8;
    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq5_k * bq5 = &x[blk];
        const block_q8_1  * bq8 = &y[blk * (QK_K / QK8_1) + ib32];
        const int32_t * q8 = (const int32_t *)bq8->qs;

        // qs layout: qs[(ib32/2)*32 + e_local], low nibble for which_half=0, high for which_half=1.
        // qh layout: qh[e_local] >> ib32 & 1 (all 8 ib32 sub-blocks share qh[0..31]).
        const uint8_t * qs_base = bq5->qs + (ib32 / 2) * 32;
        const uint8_t * qh      = bq5->qh;   // no ib32 offset; bit index IS ib32
        const int       use_hi  = ib32 & 1;

        const uint16_t extra   = bq5->extra;
        const int8_t * vals_lo = iq5nl_values_dev + ((extra >> (2*ib32    )) & 1) * 32;
        const int8_t * vals_hi = iq5nl_values_dev + ((extra >> (2*ib32 + 1)) & 1) * 32;

        const uint8_t  sh  = bq5->scales_h[ib32 / 2] >> (4 * (ib32 % 2));
        const int      ls1 = ((bq5->scales_l[ib32] & 0xf)  | ((sh << 4) & 0x30)) - 32;
        const int      ls2 = ((bq5->scales_l[ib32] >>   4) | ((sh << 2) & 0x30)) - 32;

        int sumi1 = 0, sumi2 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e    = j * 4 + k;          // e_local 0..15
                const int qs4  = use_hi ? (qs_base[e] >> 4) : (qs_base[e] & 0xf);
                const int idx5 = qs4 | (((qh[e] >> ib32) & 1) << 4);
                v |= ((uint32_t)(uint8_t)vals_lo[idx5] << (k * 8));
            }
            sumi1 = ggml_cuda_dp4a(v, q8[j], sumi1);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e    = 16 + j * 4 + k;     // e_local 16..31
                const int qs4  = use_hi ? (qs_base[e] >> 4) : (qs_base[e] & 0xf);
                const int idx5 = qs4 | (((qh[e] >> ib32) & 1) << 4);
                v |= ((uint32_t)(uint8_t)vals_hi[idx5] << (k * 8));
            }
            sumi2 = ggml_cuda_dp4a(v, q8[j + 4], sumi2);
        }

        const float d = __half2float(bq5->d) * __low2float(bq8->ds);
        sumf += d * (float)(sumi1 * ls1 + sumi2 * ls2);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }
    if (lane == 0) dst[row] = sumf;
}

static void launch_mul_mat_vec_iq5_k_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq5_k_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ5_KS MMVQ kernel.  Per-row layout: float row scale + N block_iq5_ks (168 B:
// 8 scales + 128 qs + 32 qh).  Per sub-block (32 elems): scale byte (bit0 =
// codebook shift, bits1..7 = signed scale (s&254)-127); 16 qs bytes give the
// low nibble, qh gives the 5th index bit shift-selected by the ib64 group.
// Value codebook = iq5nl_values_dev (64 entries; +32 = shifted half).
// ============================================================================
static __global__ void mul_mat_vec_iq5_ks_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;
    const int lane = threadIdx.x;

    const char * row_ptr = (const char *)vx + (size_t)row * row_size_x;
    const float row_scale = *(const float *)row_ptr;
    const block_iq5_ks * x = (const block_iq5_ks *)(row_ptr + sizeof(float));
    const block_q8_1 * y = (const block_q8_1 *)vy;

    const int total_subblocks = (ncols_x / QK_K) * 8;   // 8 sub-blocks of 32 per superblock
    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;          // 0..7 sub-block within superblock
        const int ib64 = ib32 / 2;             // 0..3 group
        const int half = ib32 & 1;             // 0 => low nibble + qh bit (2*ib64), 1 => high nibble + qh bit (2*ib64+1)

        const block_iq5_ks * bq5 = &x[blk];
        const block_q8_1   * bq8 = &y[blk * (QK_K / QK8_1) + ib32];
        const int32_t * q8 = (const int32_t *)bq8->qs;     // 8 int32 = 32 int8

        const uint8_t sc = bq5->scales[ib32];
        const int     ls = (int)(sc & 254) - 127;
        const int8_t * vals = iq5nl_values_dev + ((sc & 1) << 5);

        // qs bytes for this sub-block: 32 bytes at qs + 32*ib64; this sub-block uses
        // nibble selected by `half`, qh bit at (2*ib64 + half).
        const uint8_t * qs = bq5->qs + 32*ib64;
        const uint8_t * qh = bq5->qh;          // 32 bytes, bit-selected
        const int qh_shift = 2*ib64 + half;

        int sumi = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e    = j*4 + k;       // 0..15 element within sub-block
                const int nib  = half ? (qs[e] >> 4) : (qs[e] & 0xf);
                const int idx5 = nib | (((qh[e] >> qh_shift) & 1) << 4);
                v |= ((uint32_t)(uint8_t)vals[idx5] << (k * 8));
            }
            sumi = ggml_cuda_dp4a(v, q8[j], sumi);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e    = 16 + j*4 + k;  // 16..31 element within sub-block
                const int nib  = half ? (qs[e] >> 4) : (qs[e] & 0xf);
                const int idx5 = nib | (((qh[e] >> qh_shift) & 1) << 4);
                v |= ((uint32_t)(uint8_t)vals[idx5] << (k * 8));
            }
            sumi = ggml_cuda_dp4a(v, q8[j + 4], sumi);
        }

        sumf += row_scale * (float)ls * __low2float(bq8->ds) * (float)sumi;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }
    if (lane == 0) dst[row] = sumf;
}

static void launch_mul_mat_vec_iq5_ks_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq5_ks_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// IQ6_K MMVQ kernel.  No row meta.  Per block_iq6_k:
//   half d, uint16 extra (1 codebook-shift bit per 16-element group),
//   scales[16] (direct int8 per 16-element group, no offset),
//   qs[128] (4 low bits of 6-bit index), qh[64] (2 high bits per element,
//   packed 4 per byte at (elem%4)*2 within qh[elem/4]).
// ============================================================================

extern __constant__ int8_t iq6nl_values_dev[];

static __global__ void mul_mat_vec_iq6_k_q8_1_kernel(
        const void * __restrict__ vx,
        const void * __restrict__ vy,
        float       * __restrict__ dst,
        const int    ncols_x,
        const int    nrows_x,
        const size_t row_size_x) {

    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= nrows_x) return;
    const int lane = threadIdx.x;

    const block_iq6_k * x = (const block_iq6_k *)((const char *)vx + (size_t)row * row_size_x);
    const block_q8_1  * y = (const block_q8_1 *)vy;

    const int total_subblocks = (ncols_x / QK_K) * 8;
    float sumf = 0.0f;

    for (int sb_flat = lane; sb_flat < total_subblocks; sb_flat += 32) {
        const int blk  = sb_flat / 8;
        const int ib32 = sb_flat % 8;

        const block_iq6_k * bq6 = &x[blk];
        const block_q8_1  * bq8 = &y[blk * (QK_K / QK8_1) + ib32];
        const int32_t * q8 = (const int32_t *)bq8->qs;

        // qs layout: qs[(ib32/2)*32 + e_local], low nibble for which_half=0, high for which_half=1.
        // qh layout: qh[(ib32/4)*32 + e_local] >> qh_shift, 2 bits per element.
        //   qh_shift cycles 0,2,4,6 as ib32=0,1,2,3,0,1,2,3 within each 4-group pair.
        const uint8_t * qs_base   = bq6->qs + (ib32 / 2) * 32;
        const uint8_t * qh_chunk  = bq6->qh + (ib32 / 4) * 32;
        const int       use_hi    = ib32 & 1;
        const int       qh_shift  = ((ib32 >> 1) & 1) * 4 + (ib32 & 1) * 2;

        const uint16_t extra   = bq6->extra;
        const int8_t * vals_lo = iq6nl_values_dev + ((extra >> (2*ib32    )) & 1) * 64;
        const int8_t * vals_hi = iq6nl_values_dev + ((extra >> (2*ib32 + 1)) & 1) * 64;

        const int ls1 = (int)bq6->scales[2*ib32    ];
        const int ls2 = (int)bq6->scales[2*ib32 + 1];

        int sumi1 = 0, sumi2 = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e    = j * 4 + k;           // e_local 0..15
                const int qs4  = use_hi ? (qs_base[e] >> 4) : (qs_base[e] & 0xf);
                const int q2   = (qh_chunk[e] >> qh_shift) & 0x03;
                v |= ((uint32_t)(uint8_t)vals_lo[qs4 | (q2 << 4)] << (k * 8));
            }
            sumi1 = ggml_cuda_dp4a(v, q8[j], sumi1);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int v = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int e    = 16 + j * 4 + k;      // e_local 16..31
                const int qs4  = use_hi ? (qs_base[e] >> 4) : (qs_base[e] & 0xf);
                const int q2   = (qh_chunk[e] >> qh_shift) & 0x03;
                v |= ((uint32_t)(uint8_t)vals_hi[qs4 | (q2 << 4)] << (k * 8));
            }
            sumi2 = ggml_cuda_dp4a(v, q8[j + 4], sumi2);
        }

        const float d = __half2float(bq6->d) * __low2float(bq8->ds);
        sumf += d * (float)(sumi1 * ls1 + sumi2 * ls2);
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sumf += __shfl_xor_sync(0xffffffff, sumf, offset, WARP_SIZE);
    }
    if (lane == 0) dst[row] = sumf;
}

static void launch_mul_mat_vec_iq6_k_q8_1(
        const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const size_t row_size_x,
        cudaStream_t stream) {
    GGML_ASSERT(ncols_x % QK_K == 0);
    const dim3 block(WARP_SIZE, MMVQ_IQK_NWARPS);
    const dim3 grid((nrows_x + MMVQ_IQK_NWARPS - 1) / MMVQ_IQK_NWARPS);
    mul_mat_vec_iq6_k_q8_1_kernel<<<grid, block, 0, stream>>>(
        vx, vy, dst, ncols_x, nrows_x, row_size_x);
}

// ============================================================================
// Public dispatch
// ============================================================================

bool ggml_cuda_iqk_mmvq_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_IQ1_KT:
        case GGML_TYPE_IQ2_KT:
        case GGML_TYPE_IQ3_KT:
        case GGML_TYPE_IQ4_KT:
        case GGML_TYPE_IQ4_KS:
        case GGML_TYPE_IQ5_KS:
        case GGML_TYPE_IQ4_KSS:
        case GGML_TYPE_IQ4_K:
        case GGML_TYPE_IQ3_K:
        case GGML_TYPE_IQ3_KS:
        case GGML_TYPE_IQ2_KS:
        case GGML_TYPE_IQ2_KL:
        case GGML_TYPE_IQ2_K:
        case GGML_TYPE_IQ5_K:
        case GGML_TYPE_IQ6_K:
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
    GGML_ASSERT(ne12 == 1 && ne13 == 1 && "IQK MMVQ only handles 2D MUL_MAT for now; multi-channel falls back to cuBLAS");
    GGML_ASSERT(ne11 == 1 && "IQK MMVQ implements TG (single-token) path only; multi-token falls back to cuBLAS");

    const size_t row_size_x = ggml_row_size(src0->type, ne00);

    // Quantize src1 (F32, single token) into block_q8_1.
    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(), ne10_padded * sizeof(block_q8_1) / QK8_1);
    quantize_row_q8_1_cuda((const float *)src1->data, /*ids=*/nullptr, src1_q8_1.get(), src0->type,
                           ne10, /*s11=*/ne10, /*s12=*/ne10, /*s13=*/ne10,
                           ne10_padded, /*ne11=*/1, /*ne12=*/1, /*ne13=*/1, stream);

    switch (src0->type) {
        case GGML_TYPE_IQ1_KT:
            launch_mul_mat_vec_iq1_kt_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ2_KT:
            launch_mul_mat_vec_iq2_kt_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ3_KT:
            launch_mul_mat_vec_iq3_kt_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ4_KT:
            launch_mul_mat_vec_iq4_kt_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ4_KS:
            launch_mul_mat_vec_iq4_ks_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ4_KSS:
            launch_mul_mat_vec_iq4_kss_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
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
        case GGML_TYPE_IQ3_KS:
            launch_mul_mat_vec_iq3_ks_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ2_KS:
            launch_mul_mat_vec_iq2_ks_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ5_KS:
            launch_mul_mat_vec_iq5_ks_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ2_KL:
            launch_mul_mat_vec_iq2_kl_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ2_K:
            launch_mul_mat_vec_iq2_k_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ5_K:
            launch_mul_mat_vec_iq5_k_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        case GGML_TYPE_IQ6_K:
            launch_mul_mat_vec_iq6_k_q8_1(
                src0->data, src1_q8_1.get(), (float *)dst->data,
                (int)ne00, (int)ne01, row_size_x, stream);
            break;
        default:
            GGML_ASSERT(false && "ggml_cuda_iqk_mmvq_supported gate let through an unhandled type");
    }
}
