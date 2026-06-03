#include "convert.cuh"
#include "dequantize.cuh"

#include <cstdint>

#define CUDA_Q8_0_NE_ALIGN 2048

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = 2 * (int64_t(blockDim.x)*blockIdx.x + threadIdx.x);

    if (i00 >= ne00) {
        return;
    }

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

            const int64_t ib = ibx0 + i00/qk; // block index
            const int64_t iqs = (i00%qk)/qr; // quant index
            const int64_t iybs = i00 - i00%qk; // y block start index
            const int64_t y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(vx, ib, iqs, v);

            const int64_t iy0 = (i0203*ne01 + i01)*ne00 + iybs + iqs;
            y[iy0 + 0]        = ggml_cuda_cast<dst_t>(v.x);
            y[iy0 + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template <bool need_check>
static __global__ void dequantize_block_q8_0_f16(const void * __restrict__ vx, half * __restrict__ y, const int64_t k) {
#if __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
    constexpr int nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE;

    const int64_t   i0 = CUDA_Q8_0_NE_ALIGN*blockIdx.x;
    const int * x0 = ((int *) vx) + blockIdx.x * nint;
    half2 * y2 = (half2 *) (y + i0);

    __shared__ int vals[nint];

#pragma unroll
    for (int ix0 = 0; ix0 < nint; ix0 += WARP_SIZE) {
        if (need_check && i0*sizeof(block_q8_0)/QK8_0 + sizeof(int)*(ix0 + threadIdx.x) >= k*sizeof(block_q8_0)/QK8_0) {
            break;
        }

        const int ix = ix0 + threadIdx.x;
        vals[ix] = x0[ix];
    }

    __syncthreads();

#pragma unroll
    for (int iy = 0; iy < CUDA_Q8_0_NE_ALIGN; iy += 2*WARP_SIZE) {
        if (need_check && i0 + iy + 2*threadIdx.x >= k) {
            return;
        }

        const half * b0 = ((const half  *) vals) + (sizeof(block_q8_0)/sizeof(half)) * ((iy + 2*threadIdx.x)/QK8_0);
        const half    d = *b0;
        const char2  qs = ((const char2 *) (b0 + 1))[threadIdx.x % (QK8_0/2)];

        y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y), __half2half2(d));
    }
#else
    GGML_UNUSED_VARS(vx, y, k);
    NO_DEVICE_CODE;
#endif // __CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL
}

template<typename dst_t>
static __global__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __global__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
}

template<typename dst_t>
static __global__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = threadIdx.x/4;
    const int64_t tid = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(threadIdx.x%4);
    const int64_t n = tid / 4;
    const int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
}

template<typename dst_t>
static __global__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
}

template<typename dst_t>
static __global__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * q2 = x[i].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq2_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    const float d = (float)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_xxs(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t  * q3 = x[i].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq3_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint8_t * qs = x[i].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    const float d = (float)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[i].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_s(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const float delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = (float)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq1_m(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq1_m * x = (const block_iq1_m  *) vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    const uint16_t * sc = (const uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const int64_t ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    const float d = (float)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    const float delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_nl(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_iq4_nl * x = (const block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = (float)x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_iq4_xs(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const int64_t i   = blockIdx.x;
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    const float d = (float)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
static __global__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int64_t i   = blockIdx.x;
    const block_mxfp4 * x = (const block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    const int64_t tid = threadIdx.x;
    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = ggml_cuda_e8m0_to_fp32(x[ib].e);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + 2*CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / (2*CUDA_DEQUANTIZE_BLOCK_SIZE), (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    dequantize_block<qk, qr, dequantize_kernel><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_cont_cuda(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k, cudaStream_t stream) {
    dequantize_block_cuda<qk, qr, dequantize_kernel, dst_t>(vx, y, k, 1, 1, 1, k/qk, k/qk, k/qk, stream);
}

static void dequantize_block_q8_0_f16_cuda(const void * __restrict__ vx, half * __restrict__ y, const int64_t k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_Q8_0_NE_ALIGN - 1) / CUDA_Q8_0_NE_ALIGN;
    if (k % CUDA_Q8_0_NE_ALIGN == 0) {
        const bool need_check = false;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    } else {
        const bool need_check = true;
        dequantize_block_q8_0_f16<need_check><<<num_blocks, WARP_SIZE, 0, stream>>>(vx, y, k);
    }
}

template<typename dst_t>
static void dequantize_row_q2_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q2_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q3_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q3_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q4_0_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_0<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_1_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb32 = k / 32;
    const int nb = (k + 255) / 256;
    dequantize_block_q4_1<<<nb, 32, 0, stream>>>(vx, y, nb32);
}

template<typename dst_t>
static void dequantize_row_q4_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q5_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q5_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_q6_K_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_q6_K<<<nb, 64, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq2_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq2_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_xxs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_xxs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq3_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq3_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_s_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_s<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_nl_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_nl<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq1_m_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = k / QK_K;
    dequantize_block_iq1_m<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_iq4_xs_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_xs<<<nb, 32, 0, stream>>>(vx, y);
}

template<typename dst_t>
static void dequantize_row_mxfp4_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_mxfp4<<<nb, 32, 0, stream>>>(vx, y);
}

template <typename dst_t>
static __global__ void dequantize_block_nvfp4(
        const void * __restrict__ vx,
        dst_t * __restrict__ yy,
        const int64_t ne) {
    const int64_t i = blockIdx.x;
    const int     tid = threadIdx.x;

    const int64_t base = i * QK_NVFP4;
    if (base >= ne) {
        return;
    }

    const block_nvfp4 * x = (const block_nvfp4 *) vx;
    const block_nvfp4 & xb = x[i];

    const int sub = tid / (QK_NVFP4_SUB / 2);
    const int j = tid % (QK_NVFP4_SUB / 2);

    const float d = ggml_cuda_ue4m3_to_fp32(xb.d[sub]);
    const uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    const int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    const int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_cuda_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}

template <typename dst_t>
static void dequantize_row_nvfp4_cuda(
        const void * vx,
        dst_t * y,
        const int64_t k,
        cudaStream_t stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    dequantize_block_nvfp4<<<nb, 32, 0, stream>>>(vx, y, k);
}
template <typename src_t, typename dst_t>
static __global__ void convert_unary(
        const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
        const int64_t ne0203, const uint3 ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    const int64_t i00 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i00 >= ne00) {
        return;
    }

    const src_t * x = (const src_t *) vx;

    for (int64_t i01 = blockIdx.y; i01 < ne01; i01 += gridDim.y) {
        for (int64_t i0203 = blockIdx.z; i0203 < ne0203; i0203 += gridDim.z) {
            const uint2 dm = fast_div_modulo((uint32_t)i0203, ne02);
            const int64_t i02 = dm.y;
            const int64_t i03 = dm.x;

            const int64_t ix = i03*s03 + i02*s02 + i01*s01 + i00;
            const int64_t iy = (i0203*ne01 + i01)*ne00 + i00;
            y[iy] = ggml_cuda_cast<dst_t>(x[ix]);
        }
    }
}

template <typename src_t, typename dst_t>
static void convert_unary_cuda(const void * vx, dst_t * y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t s01, const int64_t s02, const int64_t s03, cudaStream_t stream) {
    const int64_t ne0203 = ne02*ne03;
    const uint3 ne02_fdv = init_fastdiv_values(ne02);
    const dim3 num_blocks((ne00 + CUDA_DEQUANTIZE_BLOCK_SIZE - 1) / CUDA_DEQUANTIZE_BLOCK_SIZE, (int)std::min(ne01, (int64_t)65535), (int)std::min(ne0203, (int64_t)65535));
    convert_unary<src_t><<<num_blocks, CUDA_DEQUANTIZE_BLOCK_SIZE, 0, stream>>>
        (vx, y, ne00, ne01, ne0203, ne02_fdv, s01, s02, s03);
}

template <typename src_t, typename dst_t>
static void convert_unary_cont_cuda(const void * vx, dst_t * y, const int64_t k, cudaStream_t stream) {
    convert_unary_cuda<src_t>(vx, y, k, 1, 1, 1, k, k, k, stream);
}

// ============================================================================
// KS-family row-aware dequant kernels (Phase 5b-1b — lifted from ft2)
// These use per-row metadata (float/half row scale) and cannot use the
// standard dequantize_block_cuda<> template.
// ============================================================================

// IQ4_KS row-aware dequant: ported from ik_llama.cpp.  One CUDA block per QK_K-element
// quant block, 32 threads each producing 4×2 = 8 output elements.  Row scale is read at
// the start of each row (row_meta_size = 4 bytes per row).
template <typename dst_t>
static __global__ void dequantize_block_iq4_ks(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = *(const float *)cx;
    const block_iq4_ks * x = (const block_iq4_ks *)(cx + sizeof(float));
    const int64_t i = ii - (row * n_per_row) / QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 8;        // 0..3
    const int64_t ib  = tid % 8;        // 0..7
    dst_t * y = yy + ii * QK_K + 32 * ib + 4 * il;
    const uint8_t * q4 = x[i].qs + 16 * ib + 4 * il;
    const float d = scale * (float)((int)(x[i].scales[ib] & 254) - 127);
    const int8_t * values = iq4k_values + ((x[i].scales[ib] & 1) << 4);
    for (int j = 0; j < 4; ++j) {
        y[j +  0] = ggml_cuda_cast<dst_t>(d * (float)values[q4[j] & 0xf]);
        y[j + 16] = ggml_cuda_cast<dst_t>(d * (float)values[q4[j] >>  4]);
    }
}

template <typename dst_t>
static void dequantize_row_iq4_ks_cuda(const void * vx, dst_t * y,
                                       int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ4_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

// Public entry points for row-meta-aware dequant — called from cuBLAS dequant path
// in ggml-cuda.cu where we know nrows and n_per_row separately (the standard
// ggml_get_to_fp32_cuda dispatch only takes total k, which is insufficient for
// row_meta layouts).
void ggml_dequantize_iq4_ks_to_fp32_cuda(const void * vx, float * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq4_ks_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq4_ks_to_fp16_cuda(const void * vx, half * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq4_ks_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ5_KS row-aware dequant: ported from ik_llama.cpp #422.  One CUDA block per QK_K-element
// quant block, 32 threads each producing 4 outputs across 4 sub-blocks (ib64 layout).
// Row scale (float) at the start of each row (row_meta_size = 4).
template <typename dst_t>
static __global__ void dequantize_block_iq5_ks(const void * __restrict__ vx, dst_t * __restrict__ yy,
        int64_t n_per_row, int64_t row_size) {
    const int row   = blockIdx.x;
    const int block = blockIdx.y;
    const int tid   = threadIdx.x;
    const char * cx = (const char *)vx + (int64_t)row * row_size;
    const float d   = *(const float *)cx;
    const block_iq5_ks * x = (const block_iq5_ks *)(cx + sizeof(float));
    const block_iq5_ks * bq = &x[block];

    const int ib64 = tid / 8;   // 0..3
    const int il   = tid % 8;   // 0..7
    dst_t * y = yy + (int64_t)row * n_per_row + block * QK_K + 64*ib64 + 2*il;
    const float dl1 = d * (float)((int)(bq->scales[2*ib64+0] & 254) - 127);
    const float dl2 = d * (float)((int)(bq->scales[2*ib64+1] & 254) - 127);
    const uint8_t * qs = bq->qs + 32*ib64 + 2*il;
    const uint8_t * qh = bq->qh + 2*il;
    const int8_t * values1 = iq5nl_values_dev + ((bq->scales[2*ib64+0] & 1) << 5);
    const int8_t * values2 = iq5nl_values_dev + ((bq->scales[2*ib64+1] & 1) << 5);
    for (int j = 0; j < 2; ++j) {
        const uint8_t h1 = qh[j]    >> (2*ib64);
        const uint8_t h2 = qh[j+16] >> (2*ib64);
        y[j+ 0] = (dst_t)(dl1 * values1[(qs[j+ 0] & 0xf) | ((h1 & 1) << 4)]);
        y[j+16] = (dst_t)(dl1 * values1[(qs[j+16] & 0xf) | ((h2 & 1) << 4)]);
        y[j+32] = (dst_t)(dl2 * values2[(qs[j+ 0] >>  4) | ((h1 & 2) << 3)]);
        y[j+48] = (dst_t)(dl2 * values2[(qs[j+16] >>  4) | ((h2 & 2) << 3)]);
    }
}
template <typename dst_t>
static void dequantize_row_iq5_ks_cuda(const void * vx, dst_t * y,
        int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ5_KS, n_per_row);
    const int nb = (int)(n_per_row / QK_K);
    dim3 grid(nrows, nb);
    dequantize_block_iq5_ks<<<grid, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}
void ggml_dequantize_iq5_ks_to_fp32_cuda(const void * vx, float * y,
                                          int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq5_ks_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq5_ks_to_fp16_cuda(const void * vx, half * y,
                                          int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq5_ks_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ3_KS row-aware dequant (port of ik_llama.cpp's dequantize_block_iq3_ks).
// One CUDA block per QK_K elements, 32 threads each producing 8 elements.
template <typename dst_t>
static __global__ void dequantize_block_iq3_ks(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = __half2float(*(const half *)cx);
    const block_iq3_ks * x = (const block_iq3_ks *)(cx + sizeof(half));
    const int64_t i = ii - (row * n_per_row) / QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t is = tid / 16;        // 0 or 1: which half of the 256 elements
    const int64_t il = tid % 16;        // 0..15
    dst_t * y = yy + ii * QK_K + 128 * is + 2 * il;
    const uint8_t * qs = x[i].qs + 32 * is + 2 * il;
    const uint8_t * qh = x[i].qh + 2 * il;
    uint16_t extra = x[i].extra >> 4 * is;
    const float d0 = scale * (float)((int)(((x[i].scales[0] >> 4*is) & 0xf) | ((extra << 4) & 0x10)) - 16);
    const float d1 = scale * (float)((int)(((x[i].scales[1] >> 4*is) & 0xf) | ((extra << 3) & 0x10)) - 16);
    const float d2 = scale * (float)((int)(((x[i].scales[2] >> 4*is) & 0xf) | ((extra << 2) & 0x10)) - 16);
    const float d3 = scale * (float)((int)(((x[i].scales[3] >> 4*is) & 0xf) | ((extra << 1) & 0x10)) - 16);
    extra >>= 8;
    const int8_t * values0 = iq3nl_values_dev + ((extra & 1) << 3);
    const int8_t * values1 = iq3nl_values_dev + ((extra & 2) << 2);
    const int8_t * values2 = iq3nl_values_dev + ((extra & 4) << 1);
    const int8_t * values3 = iq3nl_values_dev + ((extra & 8) << 0);
    for (int j = 0; j < 2; ++j) {
        const uint8_t h = qh[j] >> 4*is;
        y[j +  0] = ggml_cuda_cast<dst_t>(d0 * (float)values0[((qs[j] >> 0) & 3) | ((h << 2) & 4)]);
        y[j + 32] = ggml_cuda_cast<dst_t>(d1 * (float)values1[((qs[j] >> 2) & 3) | ((h << 1) & 4)]);
        y[j + 64] = ggml_cuda_cast<dst_t>(d2 * (float)values2[((qs[j] >> 4) & 3) | ((h >> 0) & 4)]);
        y[j + 96] = ggml_cuda_cast<dst_t>(d3 * (float)values3[((qs[j] >> 6) & 3) | ((h >> 1) & 4)]);
    }
}

template <typename dst_t>
static void dequantize_row_iq3_ks_cuda(const void * vx, dst_t * y,
                                       int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ3_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq3_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

void ggml_dequantize_iq3_ks_to_fp32_cuda(const void * vx, float * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq3_ks_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq3_ks_to_fp16_cuda(const void * vx, half * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq3_ks_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ2_KS row-aware dequant (port of ik_llama.cpp's dequantize_block_iq2_ks).
// One CUDA block per QK_K elements, 32 threads each producing 8 elements.
template <typename dst_t>
static __global__ void dequantize_block_iq2_ks(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                               int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float d = __half2float(*(const half *)cx);
    const block_iq2_ks * x = (const block_iq2_ks *)(cx + sizeof(half));
    const int64_t i = ii - (row * n_per_row) / QK_K;

    const int tid = threadIdx.x;
    const int ib128 = tid / 16;   // 0 or 1
    const int il    = tid % 16;   // 0..15
    dst_t * y = yy + ii*QK_K + 128*ib128 + 2*il;
    const int16_t extra = x[i].extra >> 4*ib128;
    const float dl1 = d * (float)(((x[i].scales[2*ib128+0] & 0xf) | ((extra >> 4) & 0x10)) - 16);
    const float dl2 = d * (float)(((x[i].scales[2*ib128+0] >>  4) | ((extra >> 5) & 0x10)) - 16);
    const float dl3 = d * (float)(((x[i].scales[2*ib128+1] & 0xf) | ((extra >> 6) & 0x10)) - 16);
    const float dl4 = d * (float)(((x[i].scales[2*ib128+1] >>  4) | ((extra >> 7) & 0x10)) - 16);
    const uint8_t * qs = x[i].qs + 32*ib128 + 2*il;
    for (int j = 0; j < 2; ++j) {
        y[j +  0] = ggml_cuda_cast<dst_t>(dl1 * (float)iq2nl_values_dev[((qs[j] >> 0) & 0x03) + ((extra << 2) & 4)]);
        y[j + 32] = ggml_cuda_cast<dst_t>(dl2 * (float)iq2nl_values_dev[((qs[j] >> 2) & 0x03) + ((extra << 1) & 4)]);
        y[j + 64] = ggml_cuda_cast<dst_t>(dl3 * (float)iq2nl_values_dev[((qs[j] >> 4) & 0x03) + ((extra >> 0) & 4)]);
        y[j + 96] = ggml_cuda_cast<dst_t>(dl4 * (float)iq2nl_values_dev[((qs[j] >> 6) & 0x03) + ((extra >> 1) & 4)]);
    }
}

template <typename dst_t>
static void dequantize_row_iq2_ks_cuda(const void * vx, dst_t * y,
                                       int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ2_KS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq2_ks<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

void ggml_dequantize_iq2_ks_to_fp32_cuda(const void * vx, float * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq2_ks_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq2_ks_to_fp16_cuda(const void * vx, half * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq2_ks_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ4_KSS row-aware dequant (port of ik_llama.cpp's dequantize_block_iq4_kss).
// One CUDA block per QK_K elements, 32 threads each producing 4×2 output elements.
// Decode: scale byte is the LSBs of 8 uint16_t pairs; descramble via aux ^= (aux>>1).
template <typename dst_t>
static __global__ void dequantize_block_iq4_kss(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                 int64_t n_per_row, int64_t row_size) {
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = *(const float *)cx;
    const block_iq4_kss * x = (const block_iq4_kss *)(cx + sizeof(float));
    const int64_t i = ii - (row * n_per_row) / QK_K;

    const int64_t tid = threadIdx.x;
    const int64_t il  = tid / 8;     // 0..3
    const int64_t ib  = tid % 8;     // 0..7
    dst_t * y = yy + ii * QK_K + 32 * ib + 4 * il;
    const uint32_t * q4 = x[i].qs + 4 * ib;
    const uint32_t s32 = (q4[0] & 0x00010001) | ((q4[1] & 0x00010001) << 2)
                        | ((q4[2] & 0x00010001) << 4) | ((q4[3] & 0x00010001) << 6);
    const uint8_t ls = (uint8_t)((s32 | (s32 >> 15)) & 0xff);
    const float d = scale * (float)((int)(ls & 254) - 127);
    const int8_t * values = iq4k_values + ((ls & 1) << 4);

    uint32_t aux32_lo = q4[il] & 0xfffefffe;
    aux32_lo ^= (aux32_lo >> 1);
    const uint32_t aux32_hi = (aux32_lo >> 4) & 0x0f0f0f0f;
    aux32_lo &= 0x0f0f0f0f;
    const uint8_t * aux_lo = (const uint8_t *)&aux32_lo;
    const uint8_t * aux_hi = (const uint8_t *)&aux32_hi;
    for (int j = 0; j < 4; ++j) {
        y[j +  0] = ggml_cuda_cast<dst_t>(d * (float)values[aux_lo[j]]);
        y[j + 16] = ggml_cuda_cast<dst_t>(d * (float)values[aux_hi[j]]);
    }
}

template <typename dst_t>
static void dequantize_row_iq4_kss_cuda(const void * vx, dst_t * y,
                                        int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ4_KSS, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_kss<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

void ggml_dequantize_iq4_kss_to_fp32_cuda(const void * vx, float * y,
                                          int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq4_kss_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq4_kss_to_fp16_cuda(const void * vx, half * y,
                                          int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq4_kss_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ4_KT row-aware dequant.  No precomputed codebook; values regenerated from a
// 16-bit index via the deterministic `trellis_next_int` bit-mixing function.
// Each thread handles one group of 4 elements.
static __device__ __forceinline__ int iq4kt_trellis_next_int(uint32_t & val) {
    constexpr uint32_t ka = 0xCBAC1FED;
    val = ka * val;
    // ggml_cuda_dp4a(a, b, c) = c + sum_4(int8(a[i]) * int8(b[i]))
    // With b = 0x01010101 this sums the 4 bytes of (val & 0x3f3f3f3f), each in [0, 63].
    return ggml_cuda_dp4a(val & 0x3f3f3f3f, 0x01010101, -126);
}

template <typename dst_t>
static __global__ void dequantize_block_iq4_kt(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                int64_t n_per_row, int64_t row_size) {
    constexpr int kNumGroups = 64;   // QK_K / kGroupSize (256 / 4)
    constexpr int kNblock    = 8;    // QK_K / kBlockSize (256 / 32)

    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const float * dptr = (const float *)((const char *)vx + row * row_size);
    const float scale = dptr[0];
    const block_iq4_kt * x = (const block_iq4_kt *)(dptr + 1);
    const int64_t i = ii - (row * n_per_row) / QK_K;

    const int64_t tid = threadIdx.x;     // 0..31: one thread per group of 4 elements
    const int ib32 = tid / 4;            // sub-block 0..7
    const int ig   = tid % 4;            // group within sub-block 0..3 (group_size=4 → kNg=8 groups per 32-elem block; here we use 2 groups per thread? wait)

    // Actually IQ4_KT has 8 groups per 32-element block (since group_size=4, kNg=8),
    // and we have 64 groups per superblock.  With 32 threads, each thread handles 2 groups.
    // Map: thread t handles groups (t*2) and (t*2 + 1).
    const int64_t jj0 = tid * 2;          // 0, 2, 4, ..., 62
    const int64_t jj1 = tid * 2 + 1;
    const int ib = jj0 / 8;               // sub-block (0..7) for jj0
    const int g0 = jj0 % 8;               // group index within sub-block for jj0
    const int g1 = jj1 % 8;
    (void)ib32; (void)ig;

    const uint32_t * shb = x[i].qs;
    const uint8_t  * ql  = (const uint8_t *)(shb + kNblock);
    const uint8_t  * qh  = ql + kNumGroups;

    const int offset = (shb[ib] & 1) ? (4096 + 32768) : 4096;
    const int ls = (int)((shb[ib] & 0xff) >> 1) - 64;
    const float dl = scale * (float)ls;

    auto unpack_idx = [&](int jj, int g) -> uint32_t {
        const int qh_byte = jj % (kNumGroups / 2);
        const int qh_nibble = jj / (kNumGroups / 2);
        return ((uint32_t)ql[jj]
              | (((uint32_t)((qh[qh_byte] >> (4 * qh_nibble)) & 0xf)) << 8)
              | (((uint32_t)((shb[ib] >> (8 + 3 * g)) & 7)) << 12))
             + (uint32_t)offset;
    };

    uint32_t idx0 = unpack_idx(jj0, g0);
    uint32_t idx1 = unpack_idx(jj1, g1);

    dst_t * y = yy + ii * QK_K + 4 * jj0;  // jj0 covers 4 elements starting here
    for (int k = 0; k < 4; ++k) {
        y[k]     = ggml_cuda_cast<dst_t>(dl * (float)iq4kt_trellis_next_int(idx0));
    }
    for (int k = 0; k < 4; ++k) {
        y[k + 4] = ggml_cuda_cast<dst_t>(dl * (float)iq4kt_trellis_next_int(idx1));
    }
}

template <typename dst_t>
static void dequantize_row_iq4_kt_cuda(const void * vx, dst_t * y,
                                       int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ4_KT, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq4_kt<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

void ggml_dequantize_iq4_kt_to_fp32_cuda(const void * vx, float * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq4_kt_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq4_kt_to_fp16_cuda(const void * vx, half * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq4_kt_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ3_KT row-aware dequant.  No precomputed codebook; values regenerated via
// the deterministic 0xCBAC1FED bit-mixing hash.  IS_ABS=false (signed values);
// GROUP_SIZE=8; 32 groups per QK_K superblock.  One thread per group.
template <typename dst_t>
static __global__ void dequantize_block_iq3_kt(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                                int64_t n_per_row, int64_t row_size) {
    constexpr int kNblock    = 8;       // sub-blocks per QK_K (QK_K/32)
    constexpr int kNg        = 4;       // groups per sub-block (32/8)
    constexpr int kGroupSize = 8;       // elements per group
    constexpr int kNumGroups = 32;      // kNblock * kNg
    constexpr uint32_t ka    = 0xCBAC1FEDu;
    constexpr uint32_t km    = 0x3f3f3f3fu;
    constexpr uint32_t kOff  = 4096u;   // kIQ3KT_Offset

    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const float * dptr = (const float *)((const char *)vx + row * row_size);
    const float row_scale = dptr[0];
    const block_iq3_kt * x = (const block_iq3_kt *)(dptr + 1);
    const int64_t i = ii - (row * n_per_row) / QK_K;

    const int jj = threadIdx.x;         // flat group index 0..31
    const int ib = jj / kNg;            // sub-block 0..7
    const int ig = jj % kNg;            // group within sub-block 0..3

    const uint32_t * shb = x[i].qs;                                    // qs[0..7]
    const uint8_t  * ql  = (const uint8_t *)(shb + kNblock);           // qs[8..15]
    const uint8_t  * qh  = (const uint8_t *)(shb + kNblock + kNumGroups / 4); // qs[16..19]

    const int ls = (int)(shb[ib] & 0xff) - 128;
    const float dl = row_scale * (float)ls;

    const uint8_t  qh_nibble = (qh[jj / 2] >> ((jj & 1) * 4)) & 0xf;
    const uint32_t sh_4bits  = (shb[ib] >> (8 + 4 * ig)) & 0xfu;
    const uint32_t idx       = (uint32_t)ql[jj] | ((uint32_t)qh_nibble << 8) | (sh_4bits << 12);

    uint32_t val = idx + kOff;

    dst_t * y = yy + ii * QK_K + kGroupSize * jj;
    for (int k = 0; k < kGroupSize; ++k) {
        val = ka * val;
        const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);
        y[k] = ggml_cuda_cast<dst_t>(dl * (float)sv);
    }
}

template <typename dst_t>
static void dequantize_row_iq3_kt_cuda(const void * vx, dst_t * y,
                                       int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ3_KT, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq3_kt<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}

void ggml_dequantize_iq3_kt_to_fp32_cuda(const void * vx, float * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq3_kt_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq3_kt_to_fp16_cuda(const void * vx, half * y,
                                         int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq3_kt_cuda<half>(vx, y, nrows, n_per_row, stream);
}

// IQ1_KT row-aware dequant (trellis, IS_ABS=false; per-sub-block iq4k scale).
__device__ __forceinline__ const int8_t * iq1kt_iq4k_values() {
    // matches ggml-iqk-quants.c iq4k_values[0..15]
    static const int8_t v[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    return v;
}
template<typename dst_t>
static __global__ void dequantize_block_iq1_kt(const void * __restrict__ vx, dst_t * __restrict__ yy,
                                               int64_t n_per_row, int64_t row_size) {
    constexpr uint32_t ka = 0xCBAC1FEDu, km = 0x3f3f3f3fu, kOff = 4096u;
    const int64_t ii  = blockIdx.x;
    const int64_t row = (QK_K * ii) / n_per_row;
    const char * cx = (const char *)vx + row * row_size;
    const float scale = *(const float *)cx;
    const block_iq1_kt * x = (const block_iq1_kt *)(cx + sizeof(float));
    const int64_t i  = ii - (row * n_per_row) / QK_K;
    const int jj = threadIdx.x;                 // 0..31 flat group
    const uint32_t idx = (uint32_t)x[i].ql[jj]
        | (((uint32_t)x[i].qh[jj % 16] << (8 - 4 * (jj / 16))) & 0xf00u)
        | (((uint32_t)x[i].sh[jj / 4]  << (8 - (jj % 4)))      & 0x1000u);
    const float dl = scale * (float)iq1kt_iq4k_values()[x[i].sh[jj / 4] & 0xf];
    uint32_t val = idx + kOff;
    dst_t * y = yy + ii * QK_K + 8 * jj;
    for (int k = 0; k < 8; ++k) {
        val *= ka;
        const int sv = ggml_cuda_dp4a((int)(val & km), 0x01010101, -126);   // no abs (IS_ABS=false)
        y[k] = ggml_cuda_cast<dst_t>(dl * (float)sv);
    }
}
template<typename dst_t>
static void dequantize_row_iq1_kt_cuda(const void * vx, dst_t * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    const int64_t k = nrows * n_per_row;
    const int64_t row_size = ggml_row_size(GGML_TYPE_IQ1_KT, n_per_row);
    const int nb = (k + QK_K - 1) / QK_K;
    dequantize_block_iq1_kt<<<nb, 32, 0, stream>>>(vx, y, n_per_row, row_size);
}
void ggml_dequantize_iq1_kt_to_fp32_cuda(const void * vx, float * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq1_kt_cuda<float>(vx, y, nrows, n_per_row, stream);
}
void ggml_dequantize_iq1_kt_to_fp16_cuda(const void * vx, half * y, int64_t nrows, int64_t n_per_row, cudaStream_t stream) {
    dequantize_row_iq1_kt_cuda<half>(vx, y, nrows, n_per_row, stream);
}

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        default:
            return nullptr;
    }
}

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (fp16_available(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
                return dequantize_block_q8_0_f16_cuda;
            }
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_WHT4_0:
            return dequantize_block_cont_cuda<QK_WHT4_0, QR_WHT4_0, dequantize_wht4_0>;
        case GGML_TYPE_WHT3_0:
            return dequantize_block_cont_cuda<QK_TQ3_0, QR_WHT3_0, dequantize_wht3_0>;
        case GGML_TYPE_IQ4_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ4_K, dequantize_iq4_k>;
        case GGML_TYPE_IQ3_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ3_K, dequantize_iq3_k>;
        case GGML_TYPE_IQ2_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ2_K, dequantize_iq2_k>;
        case GGML_TYPE_IQ5_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ5_K, dequantize_iq5_k>;
        case GGML_TYPE_IQ6_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ6_K, dequantize_iq6_k>;
        case GGML_TYPE_F32:
            return convert_unary_cont_cuda<float>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_cont_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_row_q4_0_cuda;
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_cuda;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cont_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cont_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_cuda;
        case GGML_TYPE_Q3_K:
            return dequantize_row_q3_K_cuda;
        case GGML_TYPE_Q4_K:
            return dequantize_row_q4_K_cuda;
        case GGML_TYPE_Q5_K:
            return dequantize_row_q5_K_cuda;
        case GGML_TYPE_Q6_K:
            return dequantize_row_q6_K_cuda;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_cuda;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_cuda;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_cuda;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_cuda;
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_cuda;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_cuda;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_cuda;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_cuda;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_cuda;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_cuda;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_cuda;
        case GGML_TYPE_WHT4_0:
            return dequantize_block_cont_cuda<QK_WHT4_0, QR_WHT4_0, dequantize_wht4_0>;
        case GGML_TYPE_WHT3_0:
            return dequantize_block_cont_cuda<QK_TQ3_0, QR_WHT3_0, dequantize_wht3_0>;
        case GGML_TYPE_IQ4_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ4_K, dequantize_iq4_k>;
        case GGML_TYPE_IQ3_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ3_K, dequantize_iq3_k>;
        case GGML_TYPE_IQ2_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ2_K, dequantize_iq2_k>;
        case GGML_TYPE_IQ5_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ5_K, dequantize_iq5_k>;
        case GGML_TYPE_IQ6_K:
            return dequantize_block_cont_cuda<QK_K, QR_IQ6_K, dequantize_iq6_k>;
        case GGML_TYPE_F16:
            return convert_unary_cont_cuda<half>;
        case GGML_TYPE_BF16:
            return convert_unary_cont_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_WHT4_0:
            return dequantize_block_cuda<QK_WHT4_0, QR_WHT4_0, dequantize_wht4_0>;
        case GGML_TYPE_WHT3_0:
            return dequantize_block_cuda<QK_TQ3_0, QR_WHT3_0, dequantize_wht3_0>;
        case GGML_TYPE_IQ4_K:
            return dequantize_block_cuda<QK_K, QR_IQ4_K, dequantize_iq4_k>;
        case GGML_TYPE_IQ3_K:
            return dequantize_block_cuda<QK_K, QR_IQ3_K, dequantize_iq3_k>;
        case GGML_TYPE_IQ2_K:
            return dequantize_block_cuda<QK_K, QR_IQ2_K, dequantize_iq2_k>;
        case GGML_TYPE_IQ5_K:
            return dequantize_block_cuda<QK_K, QR_IQ5_K, dequantize_iq5_k>;
        case GGML_TYPE_IQ6_K:
            return dequantize_block_cuda<QK_K, QR_IQ6_K, dequantize_iq6_k>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_cuda<float, nv_bfloat16>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, nv_bfloat16>;
        default:
            return nullptr;
    }
}

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return convert_unary_cuda<half, float>;
        case GGML_TYPE_Q1_0:
            return dequantize_block_cuda<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_cuda<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_cuda<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_cuda<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_cuda<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_cuda<QK8_0, QR8_0, dequantize_q8_0>;
        case GGML_TYPE_WHT4_0:
            return dequantize_block_cuda<QK_WHT4_0, QR_WHT4_0, dequantize_wht4_0>;
        case GGML_TYPE_WHT3_0:
            return dequantize_block_cuda<QK_TQ3_0, QR_WHT3_0, dequantize_wht3_0>;
        case GGML_TYPE_IQ4_K:
            return dequantize_block_cuda<QK_K, QR_IQ4_K, dequantize_iq4_k>;
        case GGML_TYPE_IQ3_K:
            return dequantize_block_cuda<QK_K, QR_IQ3_K, dequantize_iq3_k>;
        case GGML_TYPE_IQ2_K:
            return dequantize_block_cuda<QK_K, QR_IQ2_K, dequantize_iq2_k>;
        case GGML_TYPE_IQ5_K:
            return dequantize_block_cuda<QK_K, QR_IQ5_K, dequantize_iq5_k>;
        case GGML_TYPE_IQ6_K:
            return dequantize_block_cuda<QK_K, QR_IQ6_K, dequantize_iq6_k>;
        case GGML_TYPE_BF16:
            return convert_unary_cuda<nv_bfloat16, float>;
        default:
            return nullptr;
    }
}
