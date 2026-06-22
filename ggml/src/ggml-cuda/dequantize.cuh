#include "common.cuh"
#include "turbo-quant.cuh"

static __device__ __forceinline__ void dequantize_q1_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q1_0 * x = (const block_q1_0 *) vx;

    const float d = x[ib].d;

    const int bit_index_0 = iqs;
    const int bit_index_1 = iqs + 1;

    const int byte_index_0 = bit_index_0 / 8;
    const int bit_offset_0 = bit_index_0 % 8;

    const int byte_index_1 = bit_index_1 / 8;
    const int bit_offset_1 = bit_index_1 % 8;

    // Extract bits: 1 = +d, 0 = -d (branchless)
    const int bit_0 = (x[ib].qs[byte_index_0] >> bit_offset_0) & 1;
    const int bit_1 = (x[ib].qs[byte_index_1] >> bit_offset_1) & 1;

    v.x = (2*bit_0 - 1) * d;
    v.y = (2*bit_1 - 1) * d;
}

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const float d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const float d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const float2 dm = __half22float2(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

    v.x = (v.x * dm.x) + dm.y;
    v.y = (v.y * dm.x) + dm.y;
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

// WHT4_0: 4-bit weight type with inverse WHT, block size 32, dual half-block scales
// Cold path only (convert.cu) — dequants full block, applies inverse RHT, returns pair
static __device__ __forceinline__ void dequantize_wht4_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_wht4_0 * x = (const block_wht4_0 *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    // Dequant full block (centroid lookup + scale)
    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = (x[ib].qs[j / 2] >> ((j & 1) * 4)) & 0xF;
        float d = (j < 16) ? d0 : d1;
        buf[j] = TQ4_CENTROIDS_WEIGHT[idx] * d;
    }

    // Inverse RHT: WHT butterfly then normalize+unsign
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j] = a + b; buf[j + step] = a - b;
            }
        }
    }
    const float inv_sqrt32 = 0.17677669529663688f;
    for (int j = 0; j < 32; j++) buf[j] *= inv_sqrt32 * TQ_WEIGHT_SIGNS[j];

    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// WHT3_0: 3-bit weight type with inverse WHT, block size 32, dual half-block scales
// 3-bit packing: 4 groups of 8 indices in 3 bytes each (24 bits = 8 * 3-bit)
static __device__ __forceinline__ void dequantize_wht3_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_wht3_0 * x = (const block_wht3_0 *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    // Unpack all 32 3-bit indices (4 groups of 8 in 3 bytes)
    float buf[32];
    for (int g = 0; g < 4; g++) {
        const uint8_t * qp = x[ib].qs + g * 3;
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
            buf[j] = TQ3_CENTROIDS_WEIGHT[idx[i]] * d;
        }
    }

    // Inverse RHT: WHT butterfly then normalize+unsign
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j] = a + b; buf[j + step] = a - b;
            }
        }
    }
    const float inv_sqrt32 = 0.17677669529663688f;
    for (int j = 0; j < 32; j++) buf[j] *= inv_sqrt32 * TQ_WEIGHT_SIGNS[j];

    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// Shared inverse-RHT + sign restore for the wider WHT dequant paths. Operates
// in-place on a 32-element block (centroid*scale values), matching the CPU
// tq3_0_rht_inverse exactly.
static __device__ __forceinline__ void wht_inverse_rht_32(float * buf) {
    for (int step = 1; step < 32; step <<= 1) {
        for (int i = 0; i < 32; i += step << 1) {
            for (int j = i; j < i + step; j++) {
                float a = buf[j], b = buf[j + step];
                buf[j] = a + b; buf[j + step] = a - b;
            }
        }
    }
    const float inv_sqrt32 = 0.17677669529663688f;
    for (int j = 0; j < 32; j++) buf[j] *= inv_sqrt32 * TQ_WEIGHT_SIGNS[j];
}

// WHT5_0: 5-bit weight type, block size 32, dual half-block scales.
// Packing: 4 groups of 8 indices, each group's 40 bits in 5 little-endian bytes.
static __device__ __forceinline__ void dequantize_wht5_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_wht5_0 * x = (const block_wht5_0 *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    float buf[32];
    for (int g = 0; g < 4; g++) {
        const uint8_t * qp = x[ib].qs + g * 5;
        uint64_t acc = 0;
        for (int b = 0; b < 5; b++) acc |= (uint64_t)qp[b] << (8 * b);
        for (int i = 0; i < 8; i++) {
            int j = g * 8 + i;
            uint8_t idx = (acc >> (5 * i)) & 0x1F;
            float d = (j < 16) ? d0 : d1;
            buf[j] = WHT5_CENTROIDS_WEIGHT[idx] * d;
        }
    }
    wht_inverse_rht_32(buf);
    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// WHT6_0: 6-bit weight type, block size 32, dual half-block scales.
// Packing: 8 groups of 4 indices, each group's 24 bits in 3 little-endian bytes.
static __device__ __forceinline__ void dequantize_wht6_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_wht6_0 * x = (const block_wht6_0 *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    float buf[32];
    for (int g = 0; g < 8; g++) {
        const uint8_t * qp = x[ib].qs + g * 3;
        uint32_t acc = (uint32_t)qp[0] | ((uint32_t)qp[1] << 8) | ((uint32_t)qp[2] << 16);
        for (int i = 0; i < 4; i++) {
            int j = g * 4 + i;
            uint8_t idx = (acc >> (6 * i)) & 0x3F;
            float d = (j < 16) ? d0 : d1;
            buf[j] = WHT6_CENTROIDS_WEIGHT[idx] * d;
        }
    }
    wht_inverse_rht_32(buf);
    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// WHT8_0: 8-bit weight type, block size 32, dual half-block scales (1 index/byte).
static __device__ __forceinline__ void dequantize_wht8_0(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_wht8_0 * x = (const block_wht8_0 *) vx;
    const float d0 = __half2float(x[ib].d0);
    const float d1 = __half2float(x[ib].d1);

    float buf[32];
    for (int j = 0; j < 32; j++) {
        uint8_t idx = x[ib].qs[j];
        float d = (j < 16) ? d0 : d1;
        buf[j] = WHT8_CENTROIDS_WEIGHT[idx] * d;
    }
    wht_inverse_rht_32(buf);
    v.x = buf[iqs];
    v.y = buf[iqs + 1];
}

// IQ4_K: 256-element superblock, 8 sub-blocks of 32 elements.
// Each sub-block has a 6-bit scale (split into 4 low + 2 high bits) and uses
// either iq4k_values or shifted iq4k_values+16 based on the `extra` bitfield.
static __device__ __forceinline__ void dequantize_iq4_k(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq4_k * x = (const block_iq4_k *) vx + ib;
    const float d = __half2float(x->d);

    const int sub_ib = iqs >> 5;       // sub-block index 0..7
    const int pos    = iqs & 31;        // position within sub-block 0..30 (even)

    const uint8_t sh = x->scales_h[sub_ib >> 1] >> (4 * (sub_ib & 1));
    const int dl1_int = ((x->scales_l[sub_ib] & 0xf) | ((sh << 4) & 0x30)) - 32;
    const int dl2_int = ((x->scales_l[sub_ib] >> 4) | ((sh << 2) & 0x30)) - 32;
    const float dl = (pos < 16) ? d * (float)dl1_int : d * (float)dl2_int;

    const int extra_bits = (int)(x->extra >> (2 * sub_ib));
    const int extra_bit  = (pos < 16) ? (extra_bits & 1) : ((extra_bits >> 1) & 1);
    const int8_t * values = iq4k_values + (extra_bit ? 16 : 0);

    const uint8_t * qs_base = x->qs + sub_ib * 16;
    const int qs_off = (pos < 16) ? pos : (pos - 16);
    const uint8_t qs0 = qs_base[qs_off];
    const uint8_t qs1 = qs_base[qs_off + 1];

    if (pos < 16) {
        v.x = dl * (float)values[qs0 & 0xf];
        v.y = dl * (float)values[qs1 & 0xf];
    } else {
        v.x = dl * (float)values[qs0 >> 4];
        v.y = dl * (float)values[qs1 >> 4];
    }
}

// IQ3_K: 256-element block, 8 sub-blocks of 32. Per 16-element scale group with sign bit.
static __device__ __forceinline__ void dequantize_iq3_k(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq3_k * x = (const block_iq3_k *) vx + ib;
    const float d = __half2float(x->d);

    const int ib32 = iqs >> 5;
    const int pos  = iqs & 31;
    const int half = pos >> 4;
    const int sl = (half == 0) ? (x->scales_l[ib32] & 0xf) : (x->scales_l[ib32] >> 4);
    const int sign_bit = (x->scales_h >> (2*ib32 + half)) & 1;
    const int ls = (2*sl + 1) * (sign_bit ? -1 : 1);
    const float dl = d * (float)ls;

    const int extra_bit = (x->extra >> (2*ib32 + half)) & 1;
    const int8_t * values = iq3nl_values_dev + (extra_bit ? 8 : 0);

    const int qs_block_off = 32 * (ib32 / 4);
    const int qs_base_off  = qs_block_off + (half ? 16 : 0);
    const int qh_block_off = 32 * (ib32 / 8);
    const int qh_base_off  = qh_block_off + (half ? 16 : 0);
    const int j  = pos & 15;
    const int shift_l = 2 * (ib32 & 3);
    const int shift_h = ib32 & 7;

    const uint8_t qs0 = x->qs[qs_base_off + j];
    const uint8_t qs1 = x->qs[qs_base_off + j + 1];
    const uint8_t qh0 = x->qh[qh_base_off + j];
    const uint8_t qh1 = x->qh[qh_base_off + j + 1];

    const int idx0 = ((qs0 >> shift_l) & 3) | (((qh0 >> shift_h) & 1) << 2);
    const int idx1 = ((qs1 >> shift_l) & 3) | (((qh1 >> shift_h) & 1) << 2);

    v.x = dl * (float)values[idx0];
    v.y = dl * (float)values[idx1];
}

// IQ2_K: 256-element block, 8 sub-blocks of 32, two 4-bit signed-offset-by-8 scales per
// sub-block (one per 16-element half).
static __device__ __forceinline__ void dequantize_iq2_k(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq2_k * x = (const block_iq2_k *) vx + ib;
    const float d = __half2float(x->d);

    const int ib32 = iqs >> 5;
    const int pos  = iqs & 31;
    const int half = pos >> 4;

    const int sl = half == 0 ? (x->scales[ib32] & 0xf) : (x->scales[ib32] >> 4);
    const int ls = sl - 8;
    const float dl = d * (float)ls;

    const int extra_bit = (x->extra >> (2*ib32 + half)) & 1;
    const int8_t * values = iq2nl_values_dev + (extra_bit ? 4 : 0);

    const int qs_block_off = 32 * (ib32 / 4);
    const int qs_base_off  = qs_block_off + (half ? 16 : 0);
    const int j  = pos & 15;
    const int shift = 2 * (ib32 & 3);

    const uint8_t qs0 = x->qs[qs_base_off + j];
    const uint8_t qs1 = x->qs[qs_base_off + j + 1];

    const int idx0 = (qs0 >> shift) & 3;
    const int idx1 = (qs1 >> shift) & 3;

    v.x = dl * (float)values[idx0];
    v.y = dl * (float)values[idx1];
}

// IQ5_K: 256-element superblock, 4 sub-blocks of 64, four 6-bit signed scales per sub-block.
// 5-bit index = 4 low bits (qs nibble) | 1 high bit (qh). extra bit selects shifted (+32) codebook.
static __device__ __forceinline__ void dequantize_iq5_k(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq5_k * x = (const block_iq5_k *) vx + ib;
    const float d = __half2float(x->d);

    const int ib64   = iqs >> 6;          // 0..3: 64-element group
    const int quarter = (iqs >> 4) & 3;   // 0..3: 16-element quarter within group
    const int j      = iqs & 15;          // 0..14 (even): position within quarter

    const uint8_t sl0 = x->scales_l[2*ib64+0];
    const uint8_t sl1 = x->scales_l[2*ib64+1];
    const uint8_t sh  = x->scales_h[ib64];
    const int dl_int = (quarter == 0) ? (int)((sl0 & 0xf) | ((sh << 4) & 0x30)) - 32 :
                       (quarter == 1) ? (int)((sl0 >>  4) | ((sh << 2) & 0x30)) - 32 :
                       (quarter == 2) ? (int)((sl1 & 0xf) | ((sh >> 0) & 0x30)) - 32 :
                                        (int)((sl1 >>  4) | ((sh >> 2) & 0x30)) - 32;
    const float dl = d * (float)dl_int;

    const int extra_bit = (x->extra >> (4*ib64 + quarter)) & 1;
    const int8_t * values = iq5nl_values_dev + (extra_bit ? 32 : 0);

    const int use_upper = quarter >> 1;          // 0=lower nibble, 1=upper nibble
    const int qh_shift  = iqs >> 5;              // 0..7: all ib64 groups share qh[0..31], diff bits
    const int qs_base   = 32*ib64 + (quarter & 1)*16 + j;
    const int qh_base   =           (quarter & 1)*16 + j; // all ib64 share qh[0..31]

    const uint8_t qs0 = x->qs[qs_base];
    const uint8_t qs1 = x->qs[qs_base + 1];
    const uint8_t qh0 = x->qh[qh_base];
    const uint8_t qh1 = x->qh[qh_base + 1];

    const int idx0 = (use_upper ? (qs0 >> 4) : (qs0 & 0xf)) | (((qh0 >> qh_shift) & 1) << 4);
    const int idx1 = (use_upper ? (qs1 >> 4) : (qs1 & 0xf)) | (((qh1 >> qh_shift) & 1) << 4);

    v.x = dl * (float)values[idx0];
    v.y = dl * (float)values[idx1];
}

// IQ6_K: 256-element superblock, 4 sub-blocks of 64, four direct int8 scales per sub-block.
// 6-bit index = 4 low bits (qs nibble) | 2 high bits (qh). extra bit selects shifted (+64) codebook.
static __device__ __forceinline__ void dequantize_iq6_k(const void * vx, const int64_t ib, const int iqs, float2 & v) {
    const block_iq6_k * x = (const block_iq6_k *) vx + ib;
    const float d = __half2float(x->d);

    const int ib64   = iqs >> 6;          // 0..3: 64-element group
    const int quarter = (iqs >> 4) & 3;   // 0..3: 16-element quarter within group
    const int j      = iqs & 15;          // 0..14 (even): position within quarter

    const float dl = d * (float)x->scales[4*ib64 + quarter];

    const int extra_bit = (x->extra >> (4*ib64 + quarter)) & 1;
    const int8_t * values = iq6nl_values_dev + (extra_bit ? 64 : 0);

    const int use_upper = quarter >> 1;          // 0=lower nibble, 1=upper nibble
    const int qh_base   = (ib64 >> 1)*32 + (quarter & 1)*16 + j;
    const int qh_shift  = (ib64 & 1)*4;          // 0 for ib64 0,2; 4 for ib64 1,3
    const int qs_base   = 32*ib64 + (quarter & 1)*16 + j;

    const uint8_t qs0 = x->qs[qs_base];
    const uint8_t qs1 = x->qs[qs_base + 1];
    const uint8_t qh0 = x->qh[qh_base];
    const uint8_t qh1 = x->qh[qh_base + 1];

    const int idx0 = use_upper ?
        ((qs0 >> 4) | (((qh0 >> qh_shift) & 0x0c) << 2)) :
        ((qs0 & 0xf) | (((qh0 >> qh_shift) & 0x03) << 4));
    const int idx1 = use_upper ?
        ((qs1 >> 4) | (((qh1 >> qh_shift) & 0x0c) << 2)) :
        ((qs1 & 0xf) | (((qh1 >> qh_shift) & 0x03) << 4));

    v.x = dl * (float)values[idx0];
    v.y = dl * (float)values[idx1];
}
