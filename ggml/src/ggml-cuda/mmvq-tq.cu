/*
 * Fused mul_mat_vec for WHT4_0 / WHT3_0 weight types.
 *
 * V12: Single-phase fused kernel with shmem activation sharing.
 * All warps cooperatively rotate activation into shared memory,
 * then each warp processes one row reading from shmem (broadcast).
 *
 * Eliminates:
 *   - Global memory scratch buffer (no CUDA graph incompatibility)
 *   - Separate pre-rotation kernel launch
 *   - 2x activation bandwidth (was: write global + read global per row)
 *
 * V12 avoids the NR0 regression that killed V3/V6/V11 — the single
 * __syncthreads is OUTSIDE the dot product loop (between rotation and
 * mmvq phases), not inside it.
 *
 * Falls back to V8 two-phase if shmem exceeds 48 KB (ncols > 12288).
 *
 * Based on signalnine's V8 two-phase kernel (commit b107175).
 * Optimization by TheTom.
 */

#include "mmvq-tq.cuh"
#include "mmvq.cuh"   // MMVQ_MAX_BATCH_SIZE
#include "turbo-quant.cuh"

#define MMVQ_TQ_NWARPS 8

// ============================================================================
// V8 two-phase kernels (fallback for very large ncols that exceed shmem)
// ============================================================================

static __global__ void tq_prerotate_activation_v8(
        const float * __restrict__ src,
        float       * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = val;
}

static __global__ void mul_mat_vec_wht4_0_v8(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_WHT4_0;
    const block_wht4_0 * x_row = ((const block_wht4_0 *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_WHT4_0 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;

        sum += act * TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);

    if (lane == 0) dst[row] = sum;
}

static __device__ __forceinline__ uint8_t tq3_extract_index(const uint8_t * __restrict__ qs, int lane) {
    const int group = lane / 8;
    const int lane_in_group = lane % 8;
    const uint8_t * qp = qs + group * 3;
    const uint32_t packed = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
    return (packed >> (lane_in_group * 3)) & 7;
}

static __global__ void mul_mat_vec_wht3_0_v8(
        const void  * __restrict__ vx,
        const float * __restrict__ vy_rot,
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    const int row  = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_wht3_0 * x_row = ((const block_wht3_0 *) vx) + (int64_t)row * blocks_per_row;

    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = vy_rot[ib * QK_TQ3_0 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);

        sum += act * TQ3_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);

    if (lane == 0) dst[row] = sum;
}

// ============================================================================
// V12: Single-phase fused kernel — rotate in shmem, no global scratch
//
// All 8 warps cooperatively WHT-rotate activation into shared memory.
// Then each warp processes one row doing centroid×scale dot product
// reading activation from shmem (broadcast reads from L1).
//
// The key insight: the single __syncthreads is between the two phases
// (rotation vs dot product), NOT inside the inner dot product loop.
// This is why V3/V11 regressed (sync per block) but V12 should not.
// ============================================================================

static __global__ void mul_mat_vec_wht4_0_v12(
        const void  * __restrict__ vx,
        const float * __restrict__ vy,   // UNROTATED activation (raw src1)
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    extern __shared__ float s_act[];  // ncols_x floats

    const int lane    = threadIdx.x;  // 0-31
    const int warp_id = threadIdx.y;  // 0 to MMVQ_TQ_NWARPS-1
    const int blocks_per_row = ncols_x / QK_WHT4_0;

    // Phase 1: ALL warps cooperatively pre-rotate activation into shmem.
    // Each warp handles a strided subset of 32-element blocks.
    // 8 warps × 32 threads = 256 threads rotating in parallel.
    for (int ib = warp_id; ib < blocks_per_row; ib += MMVQ_TQ_NWARPS) {
        float val = vy[ib * 32 + lane];
        val *= TQ_WEIGHT_SIGNS[lane];

        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
            val = (lane & h) ? (o - val) : (val + o);
        }
        val *= 0.17677669529663688f;  // 1/sqrt(32)
        s_act[ib * 32 + lane] = val;
    }
    __syncthreads();  // ONE sync — between rotation and dot product, NOT in inner loop

    // Phase 2: Each warp processes one row using shmem activation (broadcast reads).
    const int row = blockIdx.x * MMVQ_TQ_NWARPS + warp_id;
    if (row >= nrows_x) return;

    const block_wht4_0 * x_row = ((const block_wht4_0 *) vx) + (int64_t)row * blocks_per_row;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = s_act[ib * 32 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        sum += act * TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);

    if (lane == 0) dst[row] = sum;
}

static __global__ void mul_mat_vec_wht3_0_v12(
        const void  * __restrict__ vx,
        const float * __restrict__ vy,   // UNROTATED activation (raw src1)
        float       * __restrict__ dst,
        const int ncols_x,
        const int nrows_x) {

    extern __shared__ float s_act[];

    const int lane    = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int blocks_per_row = ncols_x / QK_TQ3_0;

    // Phase 1: cooperative rotation into shmem
    for (int ib = warp_id; ib < blocks_per_row; ib += MMVQ_TQ_NWARPS) {
        float val = vy[ib * 32 + lane];
        val *= TQ_WEIGHT_SIGNS[lane];

        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
            val = (lane & h) ? (o - val) : (val + o);
        }
        val *= 0.17677669529663688f;
        s_act[ib * 32 + lane] = val;
    }
    __syncthreads();

    // Phase 2: mmvq from shmem
    const int row = blockIdx.x * MMVQ_TQ_NWARPS + warp_id;
    if (row >= nrows_x) return;

    const block_wht3_0 * x_row = ((const block_wht3_0 *) vx) + (int64_t)row * blocks_per_row;
    float sum = 0.0f;

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float act = s_act[ib * 32 + lane];
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);
        sum += act * TQ3_CENTROIDS_WEIGHT[idx] * d;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, offset, WARP_SIZE);

    if (lane == 0) dst[row] = sum;
}

// ============================================================================
// Dispatch — V12 shmem when it fits, V8 two-phase fallback
// ============================================================================

// ============================================================================
// Multi-token fast paths (ncols_dst in [1, MMVQ_MAX_BATCH_SIZE]).
//
// Ported from ft2's TQ4_1S/TQ3_1S machinery. The weight block is loaded ONCE
// per warp-lane and reused across all ncols_dst tokens, so small batches
// (speculative decode / short prefill) avoid the dequant-to-f16 + cuBLAS path.
//
//   - NVIDIA + WHT4_0 : int8 dp4a path (fixed int8 centroid LUT, q8_1 activation).
//   - AMD (all)       : scalar/half path (dp4a regresses on RDNA — ft2 finding).
//   - WHT3_0 (all)    : scalar/half path (no dp4a kernel — 3-bit unpack only).
//
// Activation is WHT-rotated once into a pooled scratch buffer (q8_1 for dp4a,
// half for scalar) and shared across all rows.
// ============================================================================

// Fixed int8 centroid table: centroid_i8[i] = round(TQ4_CENTROIDS_WEIGHT[i] * 127 / 2.733).
// Rescale factor to recover float centroids after the int8 dp4a accumulation.
static constexpr float TQ4_CENTROID_I8_RESCALE = 2.733f / 127.0f;

// Register-based centroid lookup: maps 4 qs bytes (1 uint32) to 2 packed 4× centroid_i8 for dp4a.
static __device__ __forceinline__ void tq4_cents8_reg(uint32_t four_bytes, int & c0, int & c1) {
    // Centroid i8 values packed into 4 registers (little-endian byte order):
    // [-127,-96,-75,-58] [-44,-31,-18,-6] [6,18,31,44] [58,75,96,127]
    constexpr uint32_t CR03 = 0xC6B5A081u;
    constexpr uint32_t CR47 = 0xFAEEE1D4u;
    constexpr uint32_t CR8B = 0x2C1F1206u;
    constexpr uint32_t CRCF = 0x7F604B3Au;

    const uint32_t lo = four_bytes & 0x0F0F0F0Fu;
    const uint32_t hi = (four_bytes >> 4) & 0x0F0F0F0Fu;

    const uint32_t sel0 = __byte_perm(lo, hi, 0x5140u);
    const uint32_t sel1 = __byte_perm(lo, hi, 0x7362u);

    {
        const uint32_t flo  = __byte_perm(CR03, CR47, sel0);
        const uint32_t fhi  = __byte_perm(CR8B, CRCF, sel0);
        const uint32_t msb  = (sel0 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c0 = (int)__byte_perm(flo, fhi, psel);
    }
    {
        const uint32_t flo  = __byte_perm(CR03, CR47, sel1);
        const uint32_t fhi  = __byte_perm(CR8B, CRCF, sel1);
        const uint32_t msb  = (sel1 >> 3) & 0x01010101u;
        const uint32_t psel = 0x03020100u | (msb << 2);
        c1 = (int)__byte_perm(flo, fhi, psel);
    }
}

// Pre-rotate activation to q8_1 (signs → Hadamard → 1/sqrt(32) → per-32 int8 quant).
static __global__ void tq_prerotate_q8_1(
        const float * __restrict__ src,
        block_q8_1  * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;

    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, WARP_SIZE));

    float sum = val;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, off, WARP_SIZE);

    const float d  = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].ds = make_half2(__float2half(d), __float2half(sum));
    }
}

// Pre-rotate activation to half (scalar path).
static __global__ void tq_prerotate_activation_half(
        const float * __restrict__ src,
        half        * __restrict__ dst,
        const int n_elements) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;
    const int offset = block_idx * 32 + lane;
    if (offset >= n_elements) return;

    float val = src[offset];
    val *= TQ_WEIGHT_SIGNS[lane];

    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;
    dst[offset] = __float2half(val);
}

// WHT4_0 dp4a multi-token kernel (NVIDIA). One block (32 elems) per lane, reused across tokens.
template <int ncols_dst>
static __global__ void mul_mat_wht4_0_dp4a_multi(
        const void       * __restrict__ vx,
        const block_q8_1 * __restrict__ vy_q8,
        float            * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_WHT4_0;
    const block_wht4_0 * x_row = ((const block_wht4_0 *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = lane; ib < blocks_per_row; ib += WARP_SIZE) {
        const block_wht4_0 * blk = &x_row[ib];
        const float fd0 = __half2float(blk->d0);
        const float fd1 = __half2float(blk->d1);

        const uint32_t * qs32 = (const uint32_t *)(blk->qs);
        const uint32_t w0 = qs32[0], w1 = qs32[1], w2 = qs32[2], w3 = qs32[3];

        int c0_0, c1_0, c0_1, c1_1, c0_2, c1_2, c0_3, c1_3;
        tq4_cents8_reg(w0, c0_0, c1_0);
        tq4_cents8_reg(w1, c0_1, c1_1);
        tq4_cents8_reg(w2, c0_2, c1_2);
        tq4_cents8_reg(w3, c0_3, c1_3);

        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const block_q8_1 * a_blk = &vy_q8[j * stride_col_y + ib];
            const float d_act = __low2float(a_blk->ds);
            const int * a_qs = (const int *)(a_blk->qs);

            const int s0 = ggml_cuda_dp4a(c0_0, a_qs[0], ggml_cuda_dp4a(c1_0, a_qs[1],
                           ggml_cuda_dp4a(c0_1, a_qs[2], ggml_cuda_dp4a(c1_1, a_qs[3], 0))));
            const int s1 = ggml_cuda_dp4a(c0_2, a_qs[4], ggml_cuda_dp4a(c1_2, a_qs[5],
                           ggml_cuda_dp4a(c0_3, a_qs[6], ggml_cuda_dp4a(c1_3, a_qs[7], 0))));

            sumf[j] += d_act * (fd0 * (float)s0 + fd1 * (float)s1);
        }
    }

    #pragma unroll
    for (int j = 0; j < ncols_dst; j++)
        sumf[j] *= TQ4_CENTROID_I8_RESCALE;

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset, WARP_SIZE);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// WHT4_0 scalar/half multi-token kernel (AMD fallback + general). One row per warp.
template <int ncols_dst>
static __global__ void mul_mat_wht4_0_scalar_multi(
        const void * __restrict__ vx,
        const half * __restrict__ vy_rot,
        float      * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    __shared__ float s_lut[16];
    if (threadIdx.y == 0 && threadIdx.x < 16) {
        s_lut[threadIdx.x] = TQ4_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_WHT4_0;
    const block_wht4_0 * x_row = ((const block_wht4_0 *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = (x_row[ib].qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
        const float w = s_lut[idx] * d;
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float act = __half2float(vy_rot[j * stride_col_y + ib * QK_WHT4_0 + lane]);
            sumf[j] += act * w;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset, WARP_SIZE);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

// WHT3_0 scalar/half multi-token kernel (all vendors). One row per warp.
template <int ncols_dst>
static __global__ void mul_mat_wht3_0_scalar_multi(
        const void * __restrict__ vx,
        const half * __restrict__ vy_rot,
        float      * __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int stride_col_y,
        const int stride_col_dst) {

    __shared__ float s_lut[8];
    if (threadIdx.y == 0 && threadIdx.x < 8) {
        s_lut[threadIdx.x] = TQ3_CENTROIDS_WEIGHT[threadIdx.x];
    }
    __syncthreads();

    const int row = blockIdx.x * MMVQ_TQ_NWARPS + threadIdx.y;
    if (row >= nrows_x) return;

    const int lane = threadIdx.x;
    const int blocks_per_row = ncols_x / QK_TQ3_0;
    const block_wht3_0 * x_row = ((const block_wht3_0 *) vx) + (int64_t)row * blocks_per_row;

    float sumf[ncols_dst] = {};

    for (int ib = 0; ib < blocks_per_row; ib++) {
        const float d = (lane < 16) ? __half2float(x_row[ib].d0) : __half2float(x_row[ib].d1);
        const uint8_t idx = tq3_extract_index(x_row[ib].qs, lane);
        const float w = s_lut[idx] * d;
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++) {
            const float act = __half2float(vy_rot[j * stride_col_y + ib * QK_TQ3_0 + lane]);
            sumf[j] += act * w;
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            sumf[j] += __shfl_xor_sync(0xffffffff, sumf[j], offset, WARP_SIZE);
    }

    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < ncols_dst; j++)
            dst[j * stride_col_dst + row] = sumf[j];
    }
}

template <int ncols_dst>
static void launch_wht4_0_dp4a_multi(
        const void * src0_d, const block_q8_1 * q8_buf, float * dst_d,
        int ncols_x, int nrows_x, int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    mul_mat_wht4_0_dp4a_multi<ncols_dst><<<grid, block, 0, stream>>>(
        src0_d, q8_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
}

template <int ncols_dst>
static void launch_wht_scalar_multi(
        bool is_wht4, const void * src0_d, const half * act_buf, float * dst_d,
        int ncols_x, int nrows_x, int stride_col_y, int stride_col_dst, cudaStream_t stream) {
    const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
    const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);
    if (is_wht4) {
        mul_mat_wht4_0_scalar_multi<ncols_dst><<<grid, block, 0, stream>>>(
            src0_d, act_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
    } else {
        mul_mat_wht3_0_scalar_multi<ncols_dst><<<grid, block, 0, stream>>>(
            src0_d, act_buf, dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst);
    }
}

// Multi-token dispatch (ncols_dst in [1, MMVQ_MAX_BATCH_SIZE]).
static void ggml_cuda_mul_mat_tq_multi(ggml_backend_cuda_context & ctx,
                                       const ggml_tensor * src0,
                                       const ggml_tensor * src1,
                                       ggml_tensor * dst) {
    const int ncols_x   = src0->ne[0];
    const int nrows_x   = src0->ne[1];
    const int ncols_dst = src1->ne[1];

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream  = ctx.stream();

    const int id = ggml_cuda_get_device();
    const int cc = ggml_cuda_info().devices[id].cc;
    const int n_total_elements = ncols_x * ncols_dst;
    const bool use_dp4a = !GGML_CUDA_CC_IS_AMD(cc) && src0->type == GGML_TYPE_WHT4_0;

    const int stride_col_dst = nrows_x;

    if (use_dp4a) {
        const int n_total_blocks = n_total_elements / 32;
        ggml_cuda_pool_alloc<block_q8_1> q8_1_buf(ctx.pool(id), n_total_blocks);
        {
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_total_blocks + wpb - 1) / wpb);
            tq_prerotate_q8_1<<<grid, block, 0, stream>>>(src1_d, q8_1_buf.get(), n_total_elements);
        }
        const int stride_col_y = ncols_x / 32;  // q8_1 blocks per column
        switch (ncols_dst) {
            case 1: launch_wht4_0_dp4a_multi<1>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 2: launch_wht4_0_dp4a_multi<2>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 3: launch_wht4_0_dp4a_multi<3>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 4: launch_wht4_0_dp4a_multi<4>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 5: launch_wht4_0_dp4a_multi<5>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 6: launch_wht4_0_dp4a_multi<6>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 7: launch_wht4_0_dp4a_multi<7>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 8: launch_wht4_0_dp4a_multi<8>(src0_d, q8_1_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
        }
    } else {
        const bool is_wht4 = (src0->type == GGML_TYPE_WHT4_0);
        ggml_cuda_pool_alloc<half> act_buf(ctx.pool(id), n_total_elements);
        {
            const int n_total_blocks = n_total_elements / 32;
            const int wpb = 4;
            const dim3 block(32, wpb);
            const dim3 grid((n_total_blocks + wpb - 1) / wpb);
            tq_prerotate_activation_half<<<grid, block, 0, stream>>>(src1_d, act_buf.get(), n_total_elements);
        }
        const int stride_col_y = ncols_x;  // half elements per column
        switch (ncols_dst) {
            case 1: launch_wht_scalar_multi<1>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 2: launch_wht_scalar_multi<2>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 3: launch_wht_scalar_multi<3>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 4: launch_wht_scalar_multi<4>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 5: launch_wht_scalar_multi<5>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 6: launch_wht_scalar_multi<6>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 7: launch_wht_scalar_multi<7>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
            case 8: launch_wht_scalar_multi<8>(is_wht4, src0_d, act_buf.get(), dst_d, ncols_x, nrows_x, stride_col_y, stride_col_dst, stream); break;
        }
    }
}

void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx,
                               const ggml_tensor * src0,
                               const ggml_tensor * src1,
                               ggml_tensor * dst) {
    GGML_ASSERT(src0->type == GGML_TYPE_WHT4_0 || src0->type == GGML_TYPE_WHT3_0);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->ne[1] >= 1 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE);
    GGML_ASSERT(src1->ne[2] == 1 && src1->ne[3] == 1);

    const int ncols_x = src0->ne[0];
    const int nrows_x = src0->ne[1];
    GGML_ASSERT(ncols_x % 32 == 0);

    const void  * src0_d = src0->data;
    const float * src1_d = (const float *) src1->data;
    float       * dst_d  = (float *) dst->data;
    cudaStream_t stream = ctx.stream();

    // Decode (ncols_dst == 1) and short multi-token batches both route to the fused
    // *_multi<ncols_dst> path: dp4a int8 on NVIDIA WHT4_0, scalar/half elsewhere. The weight
    // block is loaded once per warp-lane and reused across tokens, so single-token decode no
    // longer pays the slow fp32 mul_mat_vec_wht*_v12 cost.
    //
    // The legacy fp32 V12/V8 decode kernels are retired from the default path but kept reachable
    // via GGML_WHT_DECODE_V12=1 for single-binary A/B (throughput + PPL parity) validation.
    const int ncols_dst = src1->ne[1];
    static const bool use_legacy_v12_decode = (getenv("GGML_WHT_DECODE_V12") != nullptr);
    if (!(ncols_dst == 1 && use_legacy_v12_decode)) {
        ggml_cuda_mul_mat_tq_multi(ctx, src0, src1, dst);
        return;
    }

    const size_t shmem_needed = (size_t)ncols_x * sizeof(float);

    // V12: single kernel, activation in shmem (fits for all models up to ncols=12288)
    // V8 fallback: two-phase with global scratch (for hypothetical future huge models)
    if (shmem_needed <= 48 * 1024) {
        const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
        const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);

        if (src0->type == GGML_TYPE_WHT4_0) {
            mul_mat_vec_wht4_0_v12<<<grid, block, shmem_needed, stream>>>(src0_d, src1_d, dst_d, ncols_x, nrows_x);
        } else {
            mul_mat_vec_wht3_0_v12<<<grid, block, shmem_needed, stream>>>(src0_d, src1_d, dst_d, ncols_x, nrows_x);
        }
    } else {
        // V8 fallback: two-phase with global scratch buffer
        static float * d_act_buf = nullptr;
        static size_t  d_act_buf_size = 0;

        cudaStreamCaptureStatus capture_status;
        (void) cudaStreamIsCapturing(stream, &capture_status);

        if (capture_status != cudaStreamCaptureStatusNone) {
            GGML_ASSERT(d_act_buf != nullptr && d_act_buf_size >= shmem_needed &&
                         "TQ scratch buffer not pre-allocated before graph capture");
        } else {
            if (shmem_needed > d_act_buf_size) {
                if (d_act_buf) (void) cudaFree(d_act_buf);
                (void) cudaMalloc(&d_act_buf, shmem_needed);
                d_act_buf_size = shmem_needed;
            }
        }

        {
            const int n_blocks = ncols_x / 32;
            const dim3 rot_block(32, 4);
            const dim3 rot_grid((n_blocks + 3) / 4);
            tq_prerotate_activation_v8<<<rot_grid, rot_block, 0, stream>>>(src1_d, d_act_buf, ncols_x);
        }

        {
            const dim3 block(WARP_SIZE, MMVQ_TQ_NWARPS);
            const dim3 grid((nrows_x + MMVQ_TQ_NWARPS - 1) / MMVQ_TQ_NWARPS);

            if (src0->type == GGML_TYPE_WHT4_0) {
                mul_mat_vec_wht4_0_v8<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
            } else {
                mul_mat_vec_wht3_0_v8<<<grid, block, 0, stream>>>(src0_d, d_act_buf, dst_d, ncols_x, nrows_x);
            }
        }
    }
}

// ============================================================================
// Load-time conversion: WHT4_0 → q8_0
//
// Fused kernel: dequant WHT4_0 (centroid lookup + inverse WHT) → quantize q8_0.
// One warp (32 threads) per block of 32 elements.
// Used at model load to convert WHT4_0 weights to q8_0 in VRAM for dp4a decode.
// ============================================================================

static __global__ void k_convert_wht4_0_to_q8_0(
        const block_wht4_0 * __restrict__ src,
        block_q8_0         * __restrict__ dst,
        const int n_blocks) {

    const int block_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (block_idx >= n_blocks) return;

    const int lane = threadIdx.x;
    const block_wht4_0 * blk = &src[block_idx];

    // Step 1: Dequant — centroid lookup × half-block scale
    const float d_scale = (lane < 16) ? __half2float(blk->d0) : __half2float(blk->d1);
    const uint8_t idx = (blk->qs[lane / 2] >> ((lane & 1) * 4)) & 0xF;
    float val = TQ4_CENTROIDS_WEIGHT[idx] * d_scale;

    // Step 2: Inverse WHT via warp shuffle (same as dequant path)
    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        float o = __shfl_xor_sync(0xffffffff, val, h, WARP_SIZE);
        val = (lane & h) ? (o - val) : (val + o);
    }
    val *= 0.17677669529663688f;  // 1/sqrt(32)
    val *= TQ_WEIGHT_SIGNS[lane];

    // Step 3: Quantize to q8_0 — find block amax, compute scale, round
    float amax = fabsf(val);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, WARP_SIZE));

    const float d = amax / 127.0f;
    const float id = (d > 0.0f) ? 127.0f / amax : 0.0f;

    // Step 4: Write q8_0 block
    dst[block_idx].qs[lane] = (int8_t)roundf(val * id);
    if (lane == 0) {
        dst[block_idx].d = __float2half(d);
    }
}

void ggml_cuda_convert_wht4_0_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream) {
    GGML_ASSERT(n_elements % QK_WHT4_0 == 0);
    const int n_blocks = n_elements / QK_WHT4_0;

    const int wpb = 4;  // warps per CUDA block
    const dim3 block(32, wpb);
    const dim3 grid((n_blocks + wpb - 1) / wpb);

    k_convert_wht4_0_to_q8_0<<<grid, block, 0, stream>>>(
        (const block_wht4_0 *)src_tq4,
        (block_q8_0 *)dst_q8,
        n_blocks);
}
