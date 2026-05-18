// Asymmetric K/V flash attention: aliased SSBO views of bindings 1 (K) and 2 (V)
// covering every supported FA element type, plus an uber dequantize4() that
// switches on FaTypeK / FaTypeV. After spec-constant specialization the driver
// folds away every path except the one matching the K/V type for this pipeline.
//
// Included by flash_attn.comp and flash_attn_cm1.comp. Not included by
// flash_attn_cm2.comp, which has its own buffer_reference-based decode path.
//
// We use macros (rather than per-quant decode functions taking a struct) on
// purpose: the FA shaders don't enable GL_EXT_shader_explicit_arithmetic_types_float16
// when FLOAT16 isn't defined, which makes float16-containing struct values
// illegal to return from / pass to functions. Macros expand inline where the
// float16 stays in storage and is converted to FLOAT_TYPE at use.

// F32 is fed as a vec4 "block" (4 floats), matching what dequant_funcs_cm2.glsl
// does for F32 in the cm2 shader. FaBlockBytesK/V == 16 for F32.
layout (binding = 1) readonly buffer K_PACKED_F32  { vec4 data[]; }                k_packed_f32;
layout (binding = 2) readonly buffer V_PACKED_F32  { vec4 data[]; }                v_packed_f32;

layout (binding = 1) readonly buffer K_PACKED_Q4_0 { block_q4_0_packed16 data[]; } k_packed_q4_0;
layout (binding = 2) readonly buffer V_PACKED_Q4_0 { block_q4_0_packed16 data[]; } v_packed_q4_0;
layout (binding = 1) readonly buffer K_PACKED_Q4_1 { block_q4_1_packed16 data[]; } k_packed_q4_1;
layout (binding = 2) readonly buffer V_PACKED_Q4_1 { block_q4_1_packed16 data[]; } v_packed_q4_1;
layout (binding = 1) readonly buffer K_PACKED_Q5_0 { block_q5_0_packed16 data[]; } k_packed_q5_0;
layout (binding = 2) readonly buffer V_PACKED_Q5_0 { block_q5_0_packed16 data[]; } v_packed_q5_0;
layout (binding = 1) readonly buffer K_PACKED_Q5_1 { block_q5_1_packed16 data[]; } k_packed_q5_1;
layout (binding = 2) readonly buffer V_PACKED_Q5_1 { block_q5_1_packed16 data[]; } v_packed_q5_1;
layout (binding = 1) readonly buffer K_PACKED_Q8_0 { block_q8_0_packed16 data[]; } k_packed_q8_0;
layout (binding = 2) readonly buffer V_PACKED_Q8_0 { block_q8_0_packed16 data[]; } v_packed_q8_0;

// turboq2_0 / turboq3_0 / turboq4_0 use struct bindings (block_turboq{2,3,4}_0) rather than
// packed16 views because their 34/50/68-byte blocks don't fit a uniform 16/32-bit
// interleave. Same applies to turboq{2,3}_tcq (36/52-byte blocks).
layout (binding = 1) readonly buffer K_PACKED_TURBOQ2_0 { block_turboq2_0 data[]; } k_packed_turboq2_0;
layout (binding = 2) readonly buffer V_PACKED_TURBOQ2_0 { block_turboq2_0 data[]; } v_packed_turboq2_0;
layout (binding = 1) readonly buffer K_PACKED_TURBOQ3_0 { block_turboq3_0 data[]; } k_packed_turboq3_0;
layout (binding = 2) readonly buffer V_PACKED_TURBOQ3_0 { block_turboq3_0 data[]; } v_packed_turboq3_0;
layout (binding = 1) readonly buffer K_PACKED_TURBOQ4_0 { block_turboq4_0 data[]; } k_packed_turboq4_0;
layout (binding = 2) readonly buffer V_PACKED_TURBOQ4_0 { block_turboq4_0 data[]; } v_packed_turboq4_0;

layout (binding = 1) readonly buffer K_PACKED_TURBOQ2_TCQ { block_turboq2_tcq data[]; } k_packed_turboq2_tcq;
layout (binding = 2) readonly buffer V_PACKED_TURBOQ2_TCQ { block_turboq2_tcq data[]; } v_packed_turboq2_tcq;
layout (binding = 1) readonly buffer K_PACKED_TURBOQ3_TCQ { block_turboq3_tcq data[]; } k_packed_turboq3_tcq;
layout (binding = 2) readonly buffer V_PACKED_TURBOQ3_TCQ { block_turboq3_tcq data[]; } v_packed_turboq3_tcq;

// Q4_1 and Q5_1 packed32 views: aliased to the same memory as the packed16
// views, used by the MMQ K-side hot path for fast 4-uint loads.
layout (binding = 1) readonly buffer K_PACKED_Q4_1_P32 { block_q4_1_packed32 data[]; } k_packed_q4_1_p32;
layout (binding = 1) readonly buffer K_PACKED_Q5_1_P32 { block_q5_1_packed32 data[]; } k_packed_q5_1_p32;

// Per-quant decode bodies are expanded once for the K view set and once for
// the V view set. The macros take the buffer name as a parameter.
#define FA_DEQUANT4_F32(BUF) \
    return FLOAT_TYPEV4(BUF.data[a_offset + ib]);

#define FA_DEQUANT4_Q4_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles - FLOAT_TYPE(8.0f));                  \
}

#define FA_DEQUANT4_Q4_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * nibbles                                        \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q5_0(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = uint(BUF.data[a_offset + ib].qh[0])                                                 \
            | (uint(BUF.data[a_offset + ib].qh[1]) << 16);                                        \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb - FLOAT_TYPE(16.0f));            \
}

#define FA_DEQUANT4_Q5_1(BUF) {                                                                   \
    uint vui_lo = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 0]);                          \
    uint vui_hi = uint(BUF.data[a_offset + ib].qs[(iqs & 0xF) / 2 + 1]);                          \
    uint shift = (iqs & 0x10) >> 2;                                                               \
    vui_lo >>= shift;                                                                             \
    vui_hi >>= shift;                                                                             \
    uint qh = BUF.data[a_offset + ib].qh;                                                         \
    FLOAT_TYPEV4 hb = FLOAT_TYPEV4((qh >> iqs)       & 1, (qh >> (iqs + 1)) & 1,                  \
                                   (qh >> (iqs + 2)) & 1, (qh >> (iqs + 3)) & 1)                  \
                      * FLOAT_TYPE(16.0f);                                                        \
    FLOAT_TYPEV4 nibbles = FLOAT_TYPEV4(vui_lo & 0xF, (vui_lo >> 8) & 0xF,                        \
                                        vui_hi & 0xF, (vui_hi >> 8) & 0xF);                       \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * (nibbles + hb)                                 \
         + FLOAT_TYPE(BUF.data[a_offset + ib].m);                                                 \
}

#define FA_DEQUANT4_Q8_0(BUF) {                                                                   \
    const i8vec2 v0 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2    ])).xy;               \
    const i8vec2 v1 = unpack8(int32_t(BUF.data[a_offset + ib].qs[iqs / 2 + 1])).xy;               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].d) * FLOAT_TYPEV4(v0.x, v0.y, v1.x, v1.y);          \
}

#include "turboq_centroids.glsl"

#define NEEDS_TCQ2_CB
#define NEEDS_TCQ3_CB
#include "tcq_codebook.glsl"

#define FA_DEQUANT4_TURBOQ2_0(BUF) {                                                              \
    const uint qb0 = uint(BUF.data[a_offset + ib].qs[(iqs    ) / 4]);                             \
    const uint l0 = (qb0 >> (((iqs    ) % 4) * 2u)) & 0x3u;                                       \
    const uint l1 = (qb0 >> (((iqs + 1) % 4) * 2u)) & 0x3u;                                       \
    const uint l2 = (qb0 >> (((iqs + 2) % 4) * 2u)) & 0x3u;                                       \
    const uint l3 = (qb0 >> (((iqs + 3) % 4) * 2u)) & 0x3u;                                       \
    FLOAT_TYPEV4 c = FLOAT_TYPEV4(TURBOQ2_CENTROIDS[l0], TURBOQ2_CENTROIDS[l1],                   \
                                  TURBOQ2_CENTROIDS[l2], TURBOQ2_CENTROIDS[l3]);                  \
    return FLOAT_TYPE(BUF.data[a_offset + ib].norm) * c;                                          \
}

#define FA_DEQUANT4_TURBOQ3_0(BUF) {                                                              \
    const uint qb0 = uint(BUF.data[a_offset + ib].qs[(iqs    ) / 4]);                             \
    const uint qb1 = uint(BUF.data[a_offset + ib].qs[(iqs + 4) / 4]);                             \
    const uint sb  = uint(BUF.data[a_offset + ib].signs[iqs / 8]);                                \
    const uint l0 = (qb0 >> (((iqs    ) % 4) * 2u)) & 0x3u;                                       \
    const uint l1 = (qb0 >> (((iqs + 1) % 4) * 2u)) & 0x3u;                                       \
    const uint l2 = (qb0 >> (((iqs + 2) % 4) * 2u)) & 0x3u;                                       \
    const uint l3 = (qb0 >> (((iqs + 3) % 4) * 2u)) & 0x3u;                                       \
    const uint h0 = (sb >> ((iqs    ) % 8)) & 0x1u;                                               \
    const uint h1 = (sb >> ((iqs + 1) % 8)) & 0x1u;                                               \
    const uint h2 = (sb >> ((iqs + 2) % 8)) & 0x1u;                                               \
    const uint h3 = (sb >> ((iqs + 3) % 8)) & 0x1u;                                               \
    FLOAT_TYPEV4 c = FLOAT_TYPEV4(TURBOQ3_CENTROIDS[l0 | (h0 << 2)],                              \
                                  TURBOQ3_CENTROIDS[l1 | (h1 << 2)],                              \
                                  TURBOQ3_CENTROIDS[l2 | (h2 << 2)],                              \
                                  TURBOQ3_CENTROIDS[l3 | (h3 << 2)]);                             \
    return FLOAT_TYPE(BUF.data[a_offset + ib].norm) * c;                                          \
}

// TCQ: 16-bit sliding-window bit extraction over qs[]. The trailing pad byte
// in block_turboq{2,3}_tcq makes `qs[byte_idx + 1]` safe on the last symbol.
// V-side decode-time alpha is hardcoded 1.0f for L2 (no-op); a tunable
// TCQ{2,3}_FA_V_ALPHA can be split into K/V macro variants in a later session
// if calibration shows benefit.
#define TCQ2_TCQ_STATE(BUF, J) (                                                                  \
    ((uint(BUF.data[a_offset + ib].qs[((J) * 2u) >> 3u])                                          \
    | (uint(BUF.data[a_offset + ib].qs[(((J) * 2u) >> 3u) + 1]) << 8))                            \
    >> (((J) * 2u) & 7u)) & 0xFFu)

#define TCQ3_TCQ_STATE(BUF, J) (                                                                  \
    ((uint(BUF.data[a_offset + ib].qs[((J) * 3u) >> 3u])                                          \
    | (uint(BUF.data[a_offset + ib].qs[(((J) * 3u) >> 3u) + 1]) << 8))                            \
    >> (((J) * 3u) & 7u)) & 0x1FFu)

#define FA_DEQUANT4_TURBOQ2_TCQ(BUF) {                                                            \
    const float nm = float(BUF.data[a_offset + ib].norm);                                         \
    FLOAT_TYPEV4 c = FLOAT_TYPEV4(TCQ2_CB[TCQ2_TCQ_STATE(BUF, iqs    )],                          \
                                  TCQ2_CB[TCQ2_TCQ_STATE(BUF, iqs + 1)],                          \
                                  TCQ2_CB[TCQ2_TCQ_STATE(BUF, iqs + 2)],                          \
                                  TCQ2_CB[TCQ2_TCQ_STATE(BUF, iqs + 3)]);                         \
    return FLOAT_TYPE(nm) * c;                                                                    \
}

#define FA_DEQUANT4_TURBOQ3_TCQ(BUF) {                                                            \
    const float nm = float(BUF.data[a_offset + ib].norm);                                         \
    FLOAT_TYPEV4 c = FLOAT_TYPEV4(TCQ3_CB[TCQ3_TCQ_STATE(BUF, iqs    )],                          \
                                  TCQ3_CB[TCQ3_TCQ_STATE(BUF, iqs + 1)],                          \
                                  TCQ3_CB[TCQ3_TCQ_STATE(BUF, iqs + 2)],                          \
                                  TCQ3_CB[TCQ3_TCQ_STATE(BUF, iqs + 3)]);                         \
    return FLOAT_TYPE(nm) * c;                                                                    \
}

// PolarQuant 4-bit centroids (Lloyd-Max for Gaussian). 16 levels.
// Matches CENTROIDS_4BIT in ggml-turbo-quant.c dequantize_row_turboq4_0.
// iqs is always a multiple of 4 (coord = 4*d), so qs[iqs/2] and qs[iqs/2+1]
// cover exactly the 4 nibble-packed elements iqs..iqs+3.
const float T4C[16] = float[16](
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
);

#define FA_DEQUANT4_TURBOQ4_0(BUF) {                                                              \
    const uint qb0 = uint(BUF.data[a_offset + ib].qs[iqs / 2    ]);                               \
    const uint qb1 = uint(BUF.data[a_offset + ib].qs[iqs / 2 + 1]);                               \
    return FLOAT_TYPE(BUF.data[a_offset + ib].norm) *                                             \
           FLOAT_TYPEV4(T4C[(qb0)      & 0xFu], T4C[(qb0 >> 4) & 0xFu],                          \
                        T4C[(qb1)      & 0xFu], T4C[(qb1 >> 4) & 0xFu]);                          \
}

FLOAT_TYPEV4 dequantize4(uint ib, uint iqs, uint a_offset, uint binding_idx) {
    if (binding_idx == BINDING_IDX_K) {
        switch (FaTypeK) {
            case FA_TYPE_F32:      FA_DEQUANT4_F32     (k_packed_f32)
            case FA_TYPE_Q4_0:     FA_DEQUANT4_Q4_0    (k_packed_q4_0)
            case FA_TYPE_Q4_1:     FA_DEQUANT4_Q4_1    (k_packed_q4_1)
            case FA_TYPE_Q5_0:     FA_DEQUANT4_Q5_0    (k_packed_q5_0)
            case FA_TYPE_Q5_1:     FA_DEQUANT4_Q5_1    (k_packed_q5_1)
            case FA_TYPE_Q8_0:     FA_DEQUANT4_Q8_0    (k_packed_q8_0)
            case FA_TYPE_TURBOQ2_0: FA_DEQUANT4_TURBOQ2_0(k_packed_turboq2_0)
            case FA_TYPE_TURBOQ3_0: FA_DEQUANT4_TURBOQ3_0(k_packed_turboq3_0)
            case FA_TYPE_TURBOQ4_0: FA_DEQUANT4_TURBOQ4_0(k_packed_turboq4_0)
            case FA_TYPE_TURBOQ2_TCQ: FA_DEQUANT4_TURBOQ2_TCQ(k_packed_turboq2_tcq)
            case FA_TYPE_TURBOQ3_TCQ: FA_DEQUANT4_TURBOQ3_TCQ(k_packed_turboq3_tcq)
        }
    } else {
        switch (FaTypeV) {
            case FA_TYPE_F32:      FA_DEQUANT4_F32     (v_packed_f32)
            case FA_TYPE_Q4_0:     FA_DEQUANT4_Q4_0    (v_packed_q4_0)
            case FA_TYPE_Q4_1:     FA_DEQUANT4_Q4_1    (v_packed_q4_1)
            case FA_TYPE_Q5_0:     FA_DEQUANT4_Q5_0    (v_packed_q5_0)
            case FA_TYPE_Q5_1:     FA_DEQUANT4_Q5_1    (v_packed_q5_1)
            case FA_TYPE_Q8_0:     FA_DEQUANT4_Q8_0    (v_packed_q8_0)
            case FA_TYPE_TURBOQ2_0: FA_DEQUANT4_TURBOQ2_0(v_packed_turboq2_0)
            case FA_TYPE_TURBOQ3_0: FA_DEQUANT4_TURBOQ3_0(v_packed_turboq3_0)
            case FA_TYPE_TURBOQ4_0: FA_DEQUANT4_TURBOQ4_0(v_packed_turboq4_0)
            case FA_TYPE_TURBOQ2_TCQ: FA_DEQUANT4_TURBOQ2_TCQ(v_packed_turboq2_tcq)
            case FA_TYPE_TURBOQ3_TCQ: FA_DEQUANT4_TURBOQ3_TCQ(v_packed_turboq3_tcq)
        }
    }
    return FLOAT_TYPEV4(0);
}
