void load_a_to_shmem(const uint pos_a, const uint row, const uint col, const uint idx_m, const uint block, const uint end_k) {
#if defined(DATA_A_F32) || defined(DATA_A_F16)
#if LOAD_VEC_A == 8
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
                FLOAT_TYPEV8 aa = FLOAT_TYPEV8(data_a[idx]);
                buf_a[buf_idx    ] = aa[0].xy;
                buf_a[buf_idx + 1] = aa[0].zw;
                buf_a[buf_idx + 2] = aa[1].xy;
                buf_a[buf_idx + 3] = aa[1].zw;
                return;
            }
#elif LOAD_VEC_A == 4
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
                FLOAT_TYPEV4 aa = FLOAT_TYPEV4(data_a[idx]);
                buf_a[buf_idx    ] = aa.xy;
                buf_a[buf_idx + 1] = aa.zw;
                return;
            }
#endif
            const uint idx = pos_a + col * p.stride_a + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPEV2(data_a_scalar[idx],
                                              data_a_scalar[idx + 1]);
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPEV2(data_a_scalar[idx], 0.0f);
            } else {
                buf_a[buf_idx] = FLOAT_TYPEV2(0.0f);
            }
#elif defined(DATA_A_BF16)
#if LOAD_VEC_A == 4
            if (ALIGNED != 0) {
                const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;
                FLOAT_TYPEV4 aa = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_a[idx]));
                buf_a[buf_idx    ] = aa.xy;
                buf_a[buf_idx + 1] = aa.zw;
                return;
            }
#endif
            const uint idx = pos_a + col * p.stride_a + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_m < p.M && block + row * 2 + 1 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_a_scalar[idx]),
                                              TO_FLOAT_TYPE(data_a_scalar[idx + 1]));
            } else if (idx_m < p.M && block + row * 2 < end_k) {
                buf_a[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_a_scalar[idx]), 0.0f);
            } else {
                buf_a[buf_idx] = FLOAT_TYPEV2(0.0f);
            }
#elif defined(DATA_A_Q4_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 4;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const float d = float(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[2*iqs]) | (uint(data_a_packed16[ib].qs[2*iqs + 1]) << 16);
            const vec4 v0 = (vec4(unpack8(vui & 0x0F0F0F0F)) - 8.0f) * d;
            const vec4 v1 = (vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) - 8.0f) * d;

            buf_a[buf_idx    ] = FLOAT_TYPEV2(v0.xy);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v0.zw);
            buf_a[buf_idx + 8] = FLOAT_TYPEV2(v1.xy);
            buf_a[buf_idx + 9] = FLOAT_TYPEV2(v1.zw);
#elif defined(DATA_A_Q4_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 4;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const vec2 dm = vec2(data_a_packed32[ib].dm);
            const uint vui = data_a_packed32[ib].qs[iqs];
            const vec4 v0 = vec4(unpack8(vui & 0x0F0F0F0F)) * dm.x + dm.y;
            const vec4 v1 = vec4(unpack8((vui >> 4) & 0x0F0F0F0F)) * dm.x + dm.y;

            buf_a[buf_idx     ] = FLOAT_TYPEV2(v0.xy);
            buf_a[buf_idx + 1 ] = FLOAT_TYPEV2(v0.zw);
            buf_a[buf_idx + 8 ] = FLOAT_TYPEV2(v1.xy);
            buf_a[buf_idx + 9 ] = FLOAT_TYPEV2(v1.zw);
#elif defined(DATA_A_Q5_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 4;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const uint uint_qh = uint(data_a_packed16[ib].qh[1]) << 16 | uint(data_a_packed16[ib].qh[0]);
            const ivec2 qh0 = ivec2(((uint_qh >> 2*iqs) << 4) & 0x10, (uint_qh >> (2*iqs + 12)) & 0x10);
            const ivec2 qh1 = ivec2(((uint_qh >> (2*iqs + 1)) << 4) & 0x10, (uint_qh >> (2*iqs + 13)) & 0x10);

            const uint vui = uint(data_a_packed16[ib].qs[iqs]);
            const vec4 v = (vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, (vui >> 12) | qh1.y) - 16.0f) * d;

            buf_a[buf_idx    ] = FLOAT_TYPEV2(v.xz);
            buf_a[buf_idx + 8] = FLOAT_TYPEV2(v.yw);
#elif defined(DATA_A_Q5_1)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 4;

            const uint ib = idx / 4;
            const uint iqs = idx & 0x03;

            const vec2 dm = vec2(data_a_packed32[ib].dm);
            const uint uint_qh = data_a_packed32[ib].qh;
            const uvec2 qh0 = uvec2(((uint_qh >> 4*iqs) << 4) & 0x10, (uint_qh >> (4*iqs + 12)) & 0x10);
            const uvec2 qh1 = uvec2(((uint_qh >> (4*iqs + 1)) << 4) & 0x10, (uint_qh >> (4*iqs + 13)) & 0x10);
            const uvec2 qh2 = uvec2(((uint_qh >> (4*iqs + 2)) << 4) & 0x10, (uint_qh >> (4*iqs + 14)) & 0x10);
            const uvec2 qh3 = uvec2(((uint_qh >> (4*iqs + 3)) << 4) & 0x10, (uint_qh >> (4*iqs + 15)) & 0x10);

            const uint vui = data_a_packed32[ib].qs[iqs];
            const vec4 v0 = vec4((vui & 0xF) | qh0.x, ((vui >> 4) & 0xF) | qh0.y, ((vui >> 8) & 0xF) | qh1.x, ((vui >> 12) & 0xF) | qh1.y) * dm.x + dm.y;
            const vec4 v1 = vec4(((vui >> 16) & 0xF) | qh2.x, ((vui >> 20) & 0xF) | qh2.y, ((vui >> 24) & 0xF) | qh3.x, ((vui >> 28) & 0xF) | qh3.y) * dm.x + dm.y;

            buf_a[buf_idx    ] = FLOAT_TYPEV2(v0.xz);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v1.xz);
            buf_a[buf_idx + 8] = FLOAT_TYPEV2(v0.yw);
            buf_a[buf_idx + 9] = FLOAT_TYPEV2(v1.yw);
#elif defined(DATA_A_Q8_0)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const float d = float(data_a_packed16[ib].d);
            const i8vec2 v0 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs])).xy; // vec4 used due to #12147
            const i8vec2 v1 = unpack8(int32_t(data_a_packed16[ib].qs[2*iqs + 1])).xy;
            const vec4 v = vec4(v0.x, v0.y, v1.x, v1.y) * d;

            buf_a[buf_idx    ] = FLOAT_TYPEV2(v.xy);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v.zw);
#elif defined(DATA_A_Q1_0) || defined(DATA_A_Q1_0_G128)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 16;
            const uint iqs = idx & 0xfu;

            const float d = float(data_a[ib].d);
            const uint bits = uint(data_a[ib].qs[iqs]);

            buf_a[buf_idx    ] = FLOAT_TYPEV2((bits & 0x01u) != 0u ? d : -d, (bits & 0x02u) != 0u ? d : -d);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2((bits & 0x04u) != 0u ? d : -d, (bits & 0x08u) != 0u ? d : -d);
            buf_a[buf_idx + 2] = FLOAT_TYPEV2((bits & 0x10u) != 0u ? d : -d, (bits & 0x20u) != 0u ? d : -d);
            buf_a[buf_idx + 3] = FLOAT_TYPEV2((bits & 0x40u) != 0u ? d : -d, (bits & 0x80u) != 0u ? d : -d);
#elif defined(DATA_A_Q2_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;                          // 4 values per idx
            const uint iqs = (idx % 64) * 2;                   // 0,2,4..126

            const uint qsi = (iqs / 64) * 16 + (iqs % 16);     // 0..15
            const uint scalesi = iqs / 8;                      // 0..15
            const uint qsshift = ((iqs % 64) / 16) * 2;        // 0,2,4,6

            const vec4 qs = vec4(unpack8((data_a_packed32[ib].qs[qsi / 2] >> qsshift) & 0x03030303));
            const uint scales = data_a[ib].scales[scalesi];
            const vec2 dm = vec2(data_a[ib].dm);

            const vec4 v = dm.x * float(scales & 0xF) * qs - dm.y * float(scales >> 4);

            buf_a[buf_idx    ] = FLOAT_TYPEV2(v.xy);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v.zw);
#elif defined(DATA_A_Q3_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                   // 2 values per idx
            const uint iqs = idx % 128;                  // 0..127

            const uint n = iqs / 64;                     // 0,1
            const uint qsi = n * 32 + (iqs % 16) * 2;    // 0,2,4..62
            const uint hmi =          (iqs % 16) * 2;    // 0,2,4..30
            const uint j = (iqs % 64) / 4;               // 0..3
            const uint is = iqs / 8;                     // 0..15
            const uint halfsplit = ((iqs % 64) / 16);    // 0,1,2,3
            const uint qsshift = halfsplit * 2;          // 0,2,4,6

            const int8_t us = int8_t(((data_a[ib].scales[is % 8] >> (4 * int(is / 8))) & 0xF)
                                  | (((data_a[ib].scales[8 + (is % 4)] >> (2 * int(is / 4))) & 3) << 4));
            const float dl = float(data_a[ib].d) * float(us - 32);

            const vec2 qs = vec2(unpack8((uint(data_a_packed16[ib].qs[qsi / 2]) >> qsshift) & 0x0303).xy);
            const vec2 hm = vec2(unpack8(((uint(data_a_packed16[ib].hmask[hmi / 2]) >> (4 * n + halfsplit)) & 0x0101 ^ 0x0101) << 2).xy);

            buf_a[buf_idx] = FLOAT_TYPEV2(dl * (qs.x - hm.x),
                                          dl * (qs.y - hm.y));
#elif defined(DATA_A_IQ2_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib  = idx / 128;                      // 2 values per idx
            const uint iqs = idx % 128;                      // 0..127 (element-pair index)

            const uint ib32     = iqs / 16;                  // 0..7
            const uint jel      = (iqs % 16) * 2;             // 0,2,..30 (low element of the pair)
            const uint half_idx = jel >> 4;                   // 0,1
            const uint shift    = (ib32 & 3) << 1;            // 0,2,4,6

            const uint qs_byte_idx = (ib32 >> 2) * 32 + jel;  // even, 0..62
            const uint qs16 = uint(data_a_packed16[ib].qs[qs_byte_idx / 2]);
            const uint nibble0 = (qs16 >> shift) & 3;
            const uint nibble1 = (qs16 >> (shift + 8)) & 3;

            const uint sc_byte = uint(data_a[ib].scales[ib32]);
            const int  sc_nib  = int((sc_byte >> (half_idx * 4)) & 0xf);
            const float dl = float(data_a[ib].d) * float(sc_nib - 8);

            const uint extra = uint(data_a[ib].extra);
            const uint codebook_off = ((extra >> (2 * ib32 + half_idx)) & 1) << 2;

            const int iq2nl_values_const[8] = int[8](
                -31, -13,  1, 17,
                -26,  -8,  6, 22
            );

            const vec2 v = dl * vec2(iq2nl_values_const[codebook_off + nibble0],
                                     iq2nl_values_const[codebook_off + nibble1]);
            buf_a[buf_idx] = FLOAT_TYPEV2(v.x, v.y);
#elif defined(DATA_A_IQ3_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib  = idx / 128;                      // 2 values per idx
            const uint iqs = idx % 128;                      // 0..127 (element-pair index)

            const uint ib32     = iqs / 16;                  // 0..7
            const uint jel      = (iqs % 16) * 2;             // 0,2,..30 (low element of the pair)
            const uint half_idx = jel >> 4;                   // 0,1
            const uint shift_l  = (ib32 & 3) << 1;            // 0,2,4,6
            const uint shift_h  = ib32;                       // 0..7

            const uint qs_byte_idx = (ib32 >> 2) * 32 + jel;  // even, 0..62
            const uint qs16 = uint(data_a_packed16[ib].qs[qs_byte_idx / 2]);
            const uint qs_lo = qs16 & 0xFF;
            const uint qs_hi = (qs16 >> 8) & 0xFF;

            const uint qh16 = uint(data_a_packed16[ib].qh[jel / 2]);
            const uint qh_lo = qh16 & 0xFF;
            const uint qh_hi = (qh16 >> 8) & 0xFF;

            const uint val_idx0 = ((qs_lo >> shift_l) & 3) | (((qh_lo >> shift_h) & 1) << 2);
            const uint val_idx1 = ((qs_hi >> shift_l) & 3) | (((qh_hi >> shift_h) & 1) << 2);

            const uint sl_byte   = uint(data_a[ib].scales_l[ib32]);
            const int  magnitude = int((sl_byte >> (4 * half_idx)) & 0xf);
            const uint sh        = uint(data_a[ib].scales_h);
            const int  sh_bit    = int((sh >> (2 * ib32 + half_idx)) & 1);
            const int  ls        = (2 * magnitude + 1) * (sh_bit != 0 ? -1 : 1);
            const float dl = float(data_a[ib].d) * float(ls);

            const uint extra = uint(data_a[ib].extra);
            const uint codebook_off = ((extra >> (2 * ib32 + half_idx)) & 1) << 3;

            const int iq3nl_values_const[16] = int[16](
                -63, -40, -23, -10, 1, 13, 28,  47,
                -59, -36, -19,  -6, 5, 17, 32,  51
            );

            const vec2 v = dl * vec2(iq3nl_values_const[codebook_off + val_idx0],
                                     iq3nl_values_const[codebook_off + val_idx1]);
            buf_a[buf_idx] = FLOAT_TYPEV2(v.x, v.y);
#elif defined(DATA_A_IQ4_K)
            const int iq4k_values_const[32] = int[32](
                -127, -104, -83, -65, -49, -35, -22, -10,    1,  13,  25,  38,  53,  69,  89, 113,
                -123, -100, -79, -61, -45, -31, -18,  -6,    5,  17,  29,  42,  57,  73,  93, 117
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib  = idx / 128;                      // 2 values per idx
            const uint iqs = idx % 128;                      // 0..127 (element-pair index)

            const uint ib32      = iqs / 16;                 // 0..7 (32-elem sub-block)
            const uint jel       = (iqs % 16) * 2;           // 0,2,..30 (low elem of pair within ib32)
            const uint half_idx  = jel >> 4;                 // 0,1
            const uint j_in_half = jel & 0xf;                // 0,2,..14 (low elem of pair within half)

            const uint byte_idx0 = ib32 * 16 + j_in_half;
            const uint byte_idx1 = byte_idx0 + 1;
            const uint nibble0 = (uint(data_a[ib].qs[byte_idx0]) >> (4 * half_idx)) & 0xf;
            const uint nibble1 = (uint(data_a[ib].qs[byte_idx1]) >> (4 * half_idx)) & 0xf;

            const uint sh = (uint(data_a[ib].scales_h[ib32 >> 1]) >> (4 * (ib32 & 1))) & 0xf;
            const uint sl = uint(data_a[ib].scales_l[ib32]);
            const uint scale_low  = (half_idx == 0) ? (sl & 0xf) : (sl >> 4);
            const uint scale_high = (half_idx == 0) ? ((sh << 4) & 0x30) : ((sh << 2) & 0x30);
            const int  ls = int(scale_low | scale_high) - 32;
            const float dl = float(data_a[ib].d) * float(ls);

            const uint codebook_off = ((uint(data_a[ib].extra) >> (2 * ib32 + half_idx)) & 1) << 4;

            const vec2 v = dl * vec2(iq4k_values_const[codebook_off + nibble0],
                                     iq4k_values_const[codebook_off + nibble1]);
            buf_a[buf_idx] = FLOAT_TYPEV2(v.x, v.y);
#elif defined(DATA_A_IQ5_K)
            const int iq5k_values_const[64] = int[64](
                -126,-114,-103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6,
                  -1,   5,  11,  17,  23,  29,  36,  43,  51,  59,  68,  77,  87,  97, 109, 121,
                -124,-112,-101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10,  -4,
                   1,   7,  13,  19,  25,  31,  38,  45,  53,  61,  70,  79,  89,  99, 111, 123
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib  = idx / 128;                      // 2 values per idx
            const uint iqs = idx % 128;                      // 0..127

            const uint k0        = iqs * 2;                   // 0,2,..254 (low elem of the pair)
            const uint ib64      = k0 >> 6;                   // 0..3
            const uint pos       = k0 & 63;
            const uint pos_quad  = pos >> 4;                  // 0..3
            const uint j0        = pos & 0xf;                 // 0,2,..14

            const uint sl_idx = 2 * ib64 + (pos_quad >> 1);
            const uint sl_nib = pos_quad & 1;
            const uint low4   = (uint(data_a[ib].scales_l[sl_idx]) >> (4 * sl_nib)) & 0xf;
            const uint high2  = ((uint(data_a[ib].scales_h[ib64]) >> (2 * pos_quad)) & 0x3) << 4;
            const float dl = float(data_a[ib].d) * float(int(low4 | high2) - 32);

            const uint qs_idx0 = 32 * ib64 + (pos_quad & 1) * 16 + j0;
            const uint qs_idx1 = qs_idx0 + 1;
            const uint nib_sh  = (pos_quad >> 1) * 4;
            const uint nibble0 = (uint(data_a[ib].qs[qs_idx0]) >> nib_sh) & 0xf;
            const uint nibble1 = (uint(data_a[ib].qs[qs_idx1]) >> nib_sh) & 0xf;

            const uint qh_idx0  = (pos_quad & 1) * 16 + j0;
            const uint qh_idx1  = qh_idx0 + 1;
            const uint qh_shift = 2 * ib64 + (pos_quad >> 1);
            const uint qh_bit0  = (uint(data_a[ib].qh[qh_idx0]) >> qh_shift) & 1;
            const uint qh_bit1  = (uint(data_a[ib].qh[qh_idx1]) >> qh_shift) & 1;

            const uint idx5_0 = nibble0 | (qh_bit0 << 4);
            const uint idx5_1 = nibble1 | (qh_bit1 << 4);

            const uint codebook_off = ((uint(data_a[ib].extra) >> (4 * ib64 + pos_quad)) & 1) << 5;

            const vec2 v = dl * vec2(iq5k_values_const[codebook_off + idx5_0],
                                     iq5k_values_const[codebook_off + idx5_1]);
            buf_a[buf_idx] = FLOAT_TYPEV2(v.x, v.y);
#elif defined(DATA_A_IQ6_K)
            const int iq6k_values_const[128] = int[128](
                -127, -121, -115, -109, -104,  -98,  -93,  -88,  -84,  -79,  -74,  -70,  -66,  -62,  -58,  -54,
                 -51,  -47,  -44,  -40,  -37,  -34,  -31,  -28,  -25,  -22,  -19,  -16,  -13,  -11,   -8,   -5,
                  -2,    0,    3,    6,    9,   12,   14,   17,   20,   23,   27,   30,   33,   36,   40,   44,
                  47,   51,   55,   59,   63,   68,   72,   77,   82,   87,   92,   98,  103,  109,  115,  121,
                -126, -120, -114, -108, -103,  -97,  -92,  -87,  -83,  -78,  -73,  -69,  -65,  -61,  -57,  -53,
                 -50,  -46,  -43,  -39,  -36,  -33,  -30,  -27,  -24,  -21,  -18,  -15,  -12,  -10,   -7,   -4,
                  -1,    1,    4,    7,   10,   13,   15,   18,   21,   24,   28,   31,   34,   37,   41,   45,
                  48,   52,   56,   60,   64,   69,   73,   78,   83,   88,   93,   99,  104,  110,  116,  122
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib  = idx / 128;                      // 2 values per idx
            const uint iqs = idx % 128;                      // 0..127

            const uint k0        = iqs * 2;                   // 0,2,..254
            const uint g         = k0 >> 4;                   // 0..15 (16-elem group)
            const uint j0        = k0 & 0xf;                  // 0,2,..14
            const uint ib64      = g >> 2;                    // 0..3
            const uint pos_quad  = g & 3;                     // 0..3

            const float dl = float(data_a[ib].d) * float(int(data_a[ib].scales[g]));

            const uint qs_idx0 = 32 * ib64 + (pos_quad & 1) * 16 + j0;
            const uint qs_idx1 = qs_idx0 + 1;
            const uint nib_sh  = (pos_quad >> 1) * 4;
            const uint nibble0 = (uint(data_a[ib].qs[qs_idx0]) >> nib_sh) & 0xf;
            const uint nibble1 = (uint(data_a[ib].qs[qs_idx1]) >> nib_sh) & 0xf;

            const uint qh_idx0  = (ib64 >> 1) * 32 + (pos_quad & 1) * 16 + j0;
            const uint qh_idx1  = qh_idx0 + 1;
            const uint qh_shift = (ib64 & 1) * 4 + (pos_quad >> 1) * 2;
            const uint high2_0  = ((uint(data_a[ib].qh[qh_idx0]) >> qh_shift) & 0x3) << 4;
            const uint high2_1  = ((uint(data_a[ib].qh[qh_idx1]) >> qh_shift) & 0x3) << 4;

            const uint idx6_0 = nibble0 | high2_0;
            const uint idx6_1 = nibble1 | high2_1;

            const uint codebook_off = ((uint(data_a[ib].extra) >> g) & 1) << 6;

            const vec2 v = dl * vec2(iq6k_values_const[codebook_off + idx6_0],
                                     iq6k_values_const[codebook_off + idx6_1]);
            buf_a[buf_idx] = FLOAT_TYPEV2(v.x, v.y);
#elif defined(DATA_A_IQ4_KS)
            const int iq4k_values_const[32] = int[32](
                -127, -104, -83, -65, -49, -35, -22, -10,    1,  13,  25,  38,  53,  69,  89, 113,
                -123, -100, -79, -61, -45, -31, -18,  -6,    5,  17,  29,  42,  57,  73,  93, 117
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint elem_idx = idx * 4;
            const uint row_a = elem_idx / p.stride_a;
            const uint k_in_row = elem_idx - row_a * p.stride_a;

            const uint nb = p.stride_a / 256;
            const uint row_size_u32 = 1 + nb * 34;

            const uint ib_in_row = k_in_row / 256;
            const uint k_in_block = k_in_row % 256;

            const uint row_u32 = row_a * row_size_u32;
            const uint block_u32 = row_u32 + 1 + ib_in_row * 34;

            const float d_row = uintBitsToFloat(data_a[row_u32]);

            const uint sub_block = k_in_block >> 5;
            const uint sc_word = (sub_block < 4) ? data_a[block_u32 + 0] : data_a[block_u32 + 1];
            const uint sc_byte = (sc_word >> ((sub_block & 3) * 8)) & 0xff;

            const float dl = d_row * float(int(sc_byte & 254) - 127);
            const uint codebook_off = (sc_byte & 1) << 4;

            const uint qs_byte_idx_base = sub_block * 16 + (k_in_block & 15);
            const uint qs_u32 = data_a[block_u32 + 2 + (qs_byte_idx_base >> 2)];

            const bool high_nibble = (k_in_block & 16) != 0;

            FLOAT_TYPEV4 v;
            [[unroll]] for (uint r = 0; r < 4; ++r) {
                const uint qs_byte = (qs_u32 >> (r * 8)) & 0xff;
                const uint nibble = high_nibble ? (qs_byte >> 4) : (qs_byte & 0xf);
                const int val = iq4k_values_const[codebook_off + nibble];
                v[r] = FLOAT_TYPE(dl * float(val));
            }

            buf_a[buf_idx + 0] = FLOAT_TYPEV2(v[0], v[1]);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v[2], v[3]);
#elif defined(DATA_A_IQ5_KS)
            const int iq5nl_values_const[64] = int[64](
                -126,-114,-103, -92, -83, -74, -65, -57, -50, -43, -36, -30, -24, -18, -12, -6,
                  -1,   5,  11,  17,  23,  29,  36,  43,  51,  59,  68,  77,  87,  97, 109, 121,
                -124,-112,-101, -90, -81, -72, -63, -55, -48, -41, -34, -28, -22, -16, -10,  -4,
                   1,   7,  13,  19,  25,  31,  38,  45,  53,  61,  70,  79,  89,  99, 111, 123
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint elem_idx = idx * 4;
            const uint row_a = elem_idx / p.stride_a;
            const uint k_in_row = elem_idx - row_a * p.stride_a;

            const uint nb = p.stride_a / 256;
            const uint row_size_u32 = 1 + nb * 42;

            const uint ib_in_row = k_in_row / 256;
            const uint k_in_block = k_in_row % 256;

            const uint row_u32 = row_a * row_size_u32;
            const uint block_u32 = row_u32 + 1 + ib_in_row * 42;

            const float d_row = uintBitsToFloat(data_a[row_u32]);

            const uint ib = k_in_block >> 5;
            const uint j  = k_in_block & 31;

            const uint sc_word = (ib < 4) ? data_a[block_u32 + 0] : data_a[block_u32 + 1];
            const uint sc_byte = (sc_word >> ((ib & 3) * 8)) & 0xff;

            const float dl = d_row * float(int(sc_byte & 254) - 127);
            const uint codebook_off = (sc_byte & 1) << 5;

            const uint qs_byte_idx = 32 * (ib >> 1) + j;
            const uint qs_u32 = data_a[block_u32 + 2 + (qs_byte_idx >> 2)];
            const uint qh_u32 = data_a[block_u32 + 34 + (j >> 2)];

            const bool high_nibble = (ib & 1) != 0;

            FLOAT_TYPEV4 v;
            [[unroll]] for (uint r = 0; r < 4; ++r) {
                const uint qs_byte = (qs_u32 >> (r * 8)) & 0xff;
                const uint qh_byte = (qh_u32 >> (r * 8)) & 0xff;
                const uint nibble = high_nibble ? (qs_byte >> 4) : (qs_byte & 0xf);
                const uint qhbit = (qh_byte >> ib) & 1;
                const uint idx5 = nibble | (qhbit << 4);
                const int val = iq5nl_values_const[codebook_off + idx5];
                v[r] = FLOAT_TYPE(dl * float(val));
            }

            buf_a[buf_idx + 0] = FLOAT_TYPEV2(v[0], v[1]);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v[2], v[3]);
#elif defined(DATA_A_IQ2_KS)
            const int iq2nl_values_const[8] = int[8](
                -31, -13,  1, 17,
                -26,  -8,  6, 22
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint elem0 = idx * 2;
            const uint row_a = elem0 / p.stride_a;
            const uint k_in_row = elem0 - row_a * p.stride_a;

            const uint nb = p.stride_a / 256;
            const uint row_size_u16 = 1 + nb * 35;

            const uint ib_in_row = k_in_row / 256;
            const uint k_in_block = k_in_row % 256;

            const uint row_u16 = row_a * row_size_u16;
            const float d_row = float(unpackHalf2x16(uint(data_a[row_u16])).x);

            const uint block_u16 = row_u16 + 1 + ib_in_row * 35;

            const uint extra = uint(data_a[block_u16]);

            const uint sb = k_in_block / 32;
            const uint j  = k_in_block % 32;

            const uint ib64 = sb / 2;
            const uint sc_u16 = uint(data_a[block_u16 + 1 + ib64 / 2]);
            const uint sc_byte_val = (sc_u16 >> ((ib64 & 1) * 8)) & 0xff;
            const uint nibble_val = (sb & 1) == 0
                ? (sc_byte_val & 0xf)
                : ((sc_byte_val >> 4) & 0xf);

            const int ls = int(nibble_val) | (int((extra >> (8 + sb)) & 1) << 4);
            const float dl = d_row * float(ls - 16);

            const uint values_off = ((extra >> sb) & 1) << 2;

            const uint qs_base  = 32 * (sb / 4);
            const uint qs_shift = 2 * (sb % 4);

            const uint qs_byte_idx0 = qs_base + j;
            const uint qs_byte_idx1 = qs_base + j + 1;
            const uint qs_byte0 = (uint(data_a[block_u16 + 3 + qs_byte_idx0 / 2]) >> ((qs_byte_idx0 & 1) * 8)) & 0xff;
            const uint qs_byte1 = (uint(data_a[block_u16 + 3 + qs_byte_idx1 / 2]) >> ((qs_byte_idx1 & 1) * 8)) & 0xff;

            const uint nibble0 = (qs_byte0 >> qs_shift) & 3;
            const uint nibble1 = (qs_byte1 >> qs_shift) & 3;

            const vec2 v = dl * vec2(iq2nl_values_const[values_off + nibble0],
                                     iq2nl_values_const[values_off + nibble1]);
            buf_a[buf_idx] = FLOAT_TYPEV2(v.x, v.y);
#elif defined(DATA_A_IQ3_KS)
            const int iq3nl_values_const[16] = int[16](
                -63, -40, -23, -10,   1,  13,  28,  47,
                -59, -36, -19,  -6,   5,  17,  32,  51
            );

            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint elem_idx = idx * 4;
            const uint row_a = elem_idx / p.stride_a;
            const uint k_in_row = elem_idx - row_a * p.stride_a;

            const uint nb = p.stride_a / 256;
            const uint row_size_u16 = 1 + nb * 51;

            const uint ib_in_row = k_in_row / 256;
            const uint k_in_block = k_in_row % 256;

            const uint row_u16 = row_a * row_size_u16;
            const float d_row = float(unpackHalf2x16(uint(data_a[row_u16])).x);

            const uint block_u16 = row_u16 + 1 + ib_in_row * 51;

            const uint extra = uint(data_a[block_u16]);

            const uint i128  = k_in_block >> 7;
            const uint ib     = (k_in_block >> 5) & 3;
            const uint j_base = k_in_block & 31;

            const uint sc_u16  = uint(data_a[block_u16 + 1 + (ib >> 1)]);
            const uint sc_byte = (sc_u16 >> ((ib & 1) * 8)) & 0xff;
            const uint nibble  = (i128 == 0) ? (sc_byte & 0xf) : (sc_byte >> 4);
            const int  ls      = int(nibble) | (int((extra >> (ib + 4 * i128)) & 1) << 4);
            const float dl     = d_row * float(ls - 16);

            const uint shift      = (extra >> (8 + 4 * i128 + ib)) & 1;
            const uint values_off = shift << 3;

            FLOAT_TYPEV4 v;
            [[unroll]] for (uint r = 0; r < 4; ++r) {
                const uint j = j_base + r;

                const uint qs_byte_idx = i128 * 32 + j;
                const uint qs_byte = (uint(data_a[block_u16 + 3  + qs_byte_idx / 2]) >> ((qs_byte_idx & 1) * 8)) & 0xff;
                const uint qh_byte = (uint(data_a[block_u16 + 35 + j           / 2]) >> ((j           & 1) * 8)) & 0xff;

                const uint idxq = ((qs_byte >> (2 * ib)) & 3) | (((qh_byte >> (4 * i128 + ib)) & 1) << 2);
                const int  val  = iq3nl_values_const[values_off + idxq];
                v[r] = FLOAT_TYPE(dl * float(val));
            }

            buf_a[buf_idx + 0] = FLOAT_TYPEV2(v[0], v[1]);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v[2], v[3]);
#elif defined(DATA_A_IQ2_KT)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint elem_idx = idx * 8;
            const uint row_a = elem_idx / p.stride_a;
            const uint k_in_row = elem_idx - row_a * p.stride_a;

            const uint nb = p.stride_a / 256;
            const uint row_size_u32 = 1 + nb * 16;  // 1 float row scale + 16 u32 per block

            const uint ib_in_row = k_in_row / 256;
            const uint k_in_block = k_in_row % 256;
            const uint jj = k_in_block / 8; // group index 0..31

            const uint row_u32 = row_a * row_size_u32;
            const uint block_u32 = row_u32 + 1 + ib_in_row * 16;
            const float d = uintBitsToFloat(data_a[row_u32]);

            const uint word = data_a[block_u32 + (jj >> 1)];
            const uint idx_val = (jj & 1u) == 0u ? (word & 0xffffu) : (word >> 16u);

            uint x = idx_val;
            float gv[8];
            [[unroll]] for (int r = 0; r < 8; r++) {
                x = 0xCBAC1FEDu * x;
                const uint s = x & 0x3f3f3f3fu;
                const int sum = int(s & 0xffu) + int((s >> 8) & 0xffu) + int((s >> 16) & 0xffu) + int((s >> 24) & 0xffu);
                gv[r] = float(sum - 126) * d;
            }
            buf_a[buf_idx    ] = FLOAT_TYPEV2(gv[0], gv[1]);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(gv[2], gv[3]);
            buf_a[buf_idx + 2] = FLOAT_TYPEV2(gv[4], gv[5]);
            buf_a[buf_idx + 3] = FLOAT_TYPEV2(gv[6], gv[7]);
#elif defined(DATA_A_IQ4_KT)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint elem_idx = idx * 4;
            const uint row_a = elem_idx / p.stride_a;
            const uint k_in_row = elem_idx - row_a * p.stride_a;

            const uint nb = p.stride_a / 256;
            const uint row_size_u32 = 1 + nb * 32;  // 1 float row scale + 32 u32 per block

            const uint ib_in_row = k_in_row / 256;
            const uint k_in_block = k_in_row % 256;
            const uint jj = k_in_block / 4;  // group index 0..63
            const uint ib = jj >> 3;         // sub-block 0..7
            const uint g  = jj & 7u;         // group within sub-block 0..7

            const uint row_u32 = row_a * row_size_u32;
            const uint block_u32 = row_u32 + 1 + ib_in_row * 32;
            const float d = uintBitsToFloat(data_a[row_u32]);

            const uint shb    = data_a[block_u32 + ib];
            const uint offset = ((shb & 1u) != 0u) ? 36864u : 4096u;
            const int  ls     = int((shb & 0xffu) >> 1) - 64;
            const float sl    = d * float(ls);

            const uint ql_word = data_a[block_u32 + 8u + (jj >> 2)];
            const uint ql_byte = (ql_word >> ((jj & 3u) * 8u)) & 0xffu;

            const uint qh_byte_idx = jj & 31u;
            const uint qh_nibble   = jj >> 5;
            const uint qh_word     = data_a[block_u32 + 24u + (qh_byte_idx >> 2)];
            const uint qh_byte_val = (qh_word >> ((qh_byte_idx & 3u) * 8u)) & 0xffu;
            const uint qh_nib_val  = (qh_byte_val >> (4u * qh_nibble)) & 0xfu;

            const uint shb_3bits = (shb >> (8u + 3u * g)) & 7u;
            const uint idx_val   = ql_byte | (qh_nib_val << 8u) | (shb_3bits << 12u);

            uint x = idx_val + offset;
            float gv[4];
            [[unroll]] for (int r = 0; r < 4; r++) {
                x = 0xCBAC1FEDu * x;
                const uint s = x & 0x3f3f3f3fu;
                const int sum = int(s & 0xffu) + int((s >> 8) & 0xffu) + int((s >> 16) & 0xffu) + int((s >> 24) & 0xffu);
                gv[r] = float(sum - 126) * sl;
            }
            buf_a[buf_idx    ] = FLOAT_TYPEV2(gv[0], gv[1]);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(gv[2], gv[3]);
#elif defined(DATA_A_Q4_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;                  // 4 values per idx
            const uint iqs = (idx % 64) * 2;           // 0,2,4..126

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126

            const vec2 loadd = vec2(data_a[ib].dm);

            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
                                       data_a_packed32[ib].scales[1],
                                       data_a_packed32[ib].scales[2]);
            const uint scalesoffs = (is & 3) * 8;

            const uint scidx0 = (is < 4) ? 0 : 2;
            const uint scidxshift0 = scalesoffs;
            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
            const uint mbidx0 = (is < 4) ? 1 : 2;
            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            const vec4 q = vec4(unpack8((data_a_packed32[ib].qs[qsi / 4] >> (b * 4)) & 0x0F0F0F0F));

            buf_a[buf_idx    ] = FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m));
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m));
#elif defined(DATA_A_Q5_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;                  // 4 values per idx
            const uint iqs = (idx % 64) * 2;           // 0,2,4..126

            const uint n = iqs / 32;                   // 0,1,2,3
            const uint b = (iqs % 32) / 16;            // 0,1
            const uint is = 2 * n + b;                 // 0..7
            const uint qsi = n * 32 + (iqs % 16) * 2;  // 0,2,4..126
            const uint qhi = (iqs % 16) * 2;           // 0,2,4..30

            const vec2 loadd = vec2(data_a[ib].dm);

            const uvec3 scales = uvec3(data_a_packed32[ib].scales[0],
                                       data_a_packed32[ib].scales[1],
                                       data_a_packed32[ib].scales[2]);
            const uint scalesoffs = (is & 3) * 8;

            const uint scidx0 = (is < 4) ? 0 : 2;
            const uint scidxshift0 = scalesoffs;
            const uint scidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;
            const uint mbidx0 = (is < 4) ? 1 : 2;
            const uint mbidxshift0 = (is < 4) ? scalesoffs : scalesoffs + 4;
            const uint mbidxshift1 = (is < 4) ? scalesoffs : scalesoffs + 2;

            const uint8_t sc    = uint8_t(((scales[scidx0] >> scidxshift0) & 0xF) | ((scales[0] >> scidxshift1) & 0x30));
            const uint8_t mbyte = uint8_t(((scales[mbidx0] >> mbidxshift0) & 0xF) | ((scales[1] >> mbidxshift1) & 0x30));

            const float d = loadd.x * sc;
            const float m = -loadd.y * mbyte;

            const uint qs = (data_a_packed32[ib].qs[qsi / 4] >> (b * 4)) & 0x0F0F0F0F;
            const uint qh = ((data_a_packed32[ib].qh[qhi / 4] >> (iqs / 16)) & 0x01010101) << 4;
            const vec4 q = vec4(unpack8(qs | qh));

            buf_a[buf_idx    ] = FLOAT_TYPEV2(fma(d, q.x, m), fma(d, q.y, m));
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(fma(d, q.z, m), fma(d, q.w, m));
#elif defined(DATA_A_Q6_K)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 128;                  // 2 values per idx
            const uint iqs = idx % 128;                 // 0..127

            const uint n = iqs / 64;                    // 0,1
            const uint b = ((iqs % 64) / 32) * 4;       // 0,4
            const uint is_b = (iqs % 16) / 8;           // 0,1
            const uint qhshift = ((iqs % 64) / 16) * 2; // 0,2,4,6
            const uint is = 8 * n + qhshift + is_b;     // 0..15
            const uint qsi = n * 32 + (iqs % 32);       // 0..63
            const uint qhi = n * 16 + (iqs % 16);       // 0..31

            const float dscale = float(data_a[ib].d) * float(data_a[ib].scales[is]);

            const uint ql = (uint(data_a_packed16[ib].ql[qsi]) >> b) & 0x0F0F;
            const uint qh = (uint(data_a_packed16[ib].qh[qhi]) >> qhshift) & 0x0303;
            const vec2 q = (vec2(unpack8(ql | (qh << 4)).xy) - 32) * dscale;

            buf_a[buf_idx] = FLOAT_TYPEV2(q.x, q.y);
#elif defined(DATA_A_IQ1_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;                  // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 32;

            const float d = float(data_a[ib].d);
            const uint qh = data_a[ib].qh[ib32];
            const uint qs = data_a[ib].qs[ib8];
            const float dl = d * (2 * bitfieldExtract(qh, 12, 3) + 1);
            const float delta = ((qh & 0x8000) != 0) ? -IQ1S_DELTA : IQ1S_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | (bitfieldExtract(qh, 3 * int(ib8 & 3), 3) << 8)]);

            [[unroll]] for (int k = 0; k < 4; ++k) {
                buf_a[buf_idx + k] = FLOAT_TYPEV2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                  dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta));
            }
#elif defined(DATA_A_IQ1_M)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32;
            const uint ib16 = ib8 / 2;

            const uint16_t[4] scales = data_a[ib].scales;
            const u16vec4 s = u16vec4(scales[0], scales[1], scales[2], scales[3]) >> 12;
            const float d = float(unpackHalf2x16(s.x | (s.y << 4) | (s.z << 8) | (s.w << 12)).x);
            const uint sc = scales[ib8 / 8];
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib16] >> (4 * (ib8 & 1));
            const float dl = d * (2 * bitfieldExtract(sc, 3 * int(ib16 & 3), 3) + 1);
            const float delta = ((qh & 8) != 0) ? -IQ1M_DELTA : IQ1M_DELTA;
            const int16_t grid = int16_t(iq1s_grid[qs | ((qh & 7) << 8)]);

            [[unroll]] for (int k = 0; k < 4; ++k) {
                buf_a[buf_idx + k] = FLOAT_TYPEV2(dl * (bitfieldExtract(grid, 4 * k    , 2) + delta),
                                                  dl * (bitfieldExtract(grid, 4 * k + 2, 2) + delta));
            }
#elif defined(DATA_A_IQ2_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;                 // 8 values per idx
            const uint ib32 = (idx % 32) / 4;         // 0..7
            const uint ib8 = idx % 4;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[8 * ib32 + ib8];
            const uint signs = pack32(u8vec4(
                data_a[ib].qs[8*ib32 + 4],
                data_a[ib].qs[8*ib32 + 5],
                data_a[ib].qs[8*ib32 + 6],
                data_a[ib].qs[8*ib32 + 7]
            ));
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + (signs >> 28)));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * int(ib8), 7);
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xxs_grid[qs];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            buf_a[buf_idx    ] = db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                   (sign &   2) != 0 ? -grid0.y : grid0.y);
            buf_a[buf_idx + 1] = db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                   (sign &   8) != 0 ? -grid0.w : grid0.w);
            buf_a[buf_idx + 2] = db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                   (sign &  32) != 0 ? -grid1.y : grid1.y);
            buf_a[buf_idx + 3] = db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                   (sign & 128) != 0 ? -grid1.w : grid1.w);
#elif defined(DATA_A_IQ2_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;            // 8 values per idx
            const uint ib32 = (idx % 32) / 4;    // 0..7
            const uint ib8 = idx % 4;            // 0..3

            const float d = float(data_a[ib].d);
            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uint qs = data_a[ib].qs[4 * ib32 + ib8];
            const uint sign7 = qs >> 9;
            const uint sign = sign7 | (bitCount(sign7) << 7);
            const uvec2 grid = iq2xs_grid[qs & 511];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            buf_a[buf_idx    ] = db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                   (sign &   2) != 0 ? -grid0.y : grid0.y);
            buf_a[buf_idx + 1] = db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                   (sign &   8) != 0 ? -grid0.w : grid0.w);
            buf_a[buf_idx + 2] = db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                   (sign &  32) != 0 ? -grid1.y : grid1.y);
            buf_a[buf_idx + 3] = db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                   (sign & 128) != 0 ? -grid1.w : grid1.w);
#elif defined(DATA_A_IQ2_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 32;  // 8 values per idx
            const uint ib8 = idx % 32; // 0..31
            const uint ib32 = ib8 / 4; // 0..7

            const uint scale = (data_a[ib].scales[ib32] >> (2 * (ib8 & 2))) & 0xf;
            const uint qs = data_a[ib].qs[ib8];
            const uint qh = data_a[ib].qh[ib32];
            const uint qhshift = 2 * (ib8 % 4);
            const uint sign = data_a[ib].qs[QUANT_K / 8 + ib8];

            const float d = float(data_a[ib].d);
            const FLOAT_TYPE db = FLOAT_TYPE(d * 0.25 * (0.5 + scale));
            const uvec2 grid = iq2s_grid[qs | ((qh << (8 - qhshift)) & 0x300)];
            const vec4 grid0 = vec4(unpack8(grid.x));
            const vec4 grid1 = vec4(unpack8(grid.y));

            buf_a[buf_idx    ] = db * FLOAT_TYPEV2((sign &   1) != 0 ? -grid0.x : grid0.x,
                                                   (sign &   2) != 0 ? -grid0.y : grid0.y);
            buf_a[buf_idx + 1] = db * FLOAT_TYPEV2((sign &   4) != 0 ? -grid0.z : grid0.z,
                                                   (sign &   8) != 0 ? -grid0.w : grid0.w);
            buf_a[buf_idx + 2] = db * FLOAT_TYPEV2((sign &  16) != 0 ? -grid1.x : grid1.x,
                                                   (sign &  32) != 0 ? -grid1.y : grid1.y);
            buf_a[buf_idx + 3] = db * FLOAT_TYPEV2((sign &  64) != 0 ? -grid1.z : grid1.z,
                                                   (sign & 128) != 0 ? -grid1.w : grid1.w);
#elif defined(DATA_A_IQ3_XXS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint is = QUANT_K / 4 + 4 * (iqs / 8); // 8 values

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint signs = pack32(u16vec2(
                data_a_packed16[ib].qs[is/2],
                data_a_packed16[ib].qs[is/2+1]
            ));
            const float db = d * 0.5 * (0.5 + (signs >> 28));
            const uint32_t sign7 = bitfieldExtract(signs, 7 * (int(iqs / 2) % 4), 7);
            const uint sign = (sign7 | (bitCount(sign7) << 7)) >> (4 * (idx % 2));
            const uint grid = iq3xxs_grid[qs];
            const vec4 v = db * vec4(unpack8(grid));

            buf_a[buf_idx    ] = FLOAT_TYPEV2((sign &   1) != 0 ? -v.x : v.x,
                                              (sign &   2) != 0 ? -v.y : v.y);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2((sign &   4) != 0 ? -v.z : v.z,
                                              (sign &   8) != 0 ? -v.w : v.w);
#elif defined(DATA_A_IQ3_S)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;            // 4 values per idx
            const uint iqs = idx % 64;           // 0..63
            const uint iqh = iqs / 8;

            const float d = float(data_a[ib].d);
            const uint qs = data_a[ib].qs[iqs];
            const uint qh = data_a[ib].qh[iqh];
            const int8_t sign = int8_t(data_a[ib].signs[iqs / 2] >> (4 * (idx % 2)));
            const uint scale = data_a[ib].scales[iqs / 16];
            const i8vec2 sign01 = i8vec2(1 - (2 & i8vec2(sign << 1, sign)));
            const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));
            const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)];
            const vec4 v = db * vec4(unpack8(grid));

            buf_a[buf_idx    ] = FLOAT_TYPEV2((sign &   1) != 0 ? -v.x : v.x,
                                              (sign &   2) != 0 ? -v.y : v.y);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2((sign &   4) != 0 ? -v.z : v.z,
                                              (sign &   8) != 0 ? -v.w : v.w);
#elif defined(DATA_A_IQ4_XS)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 2;

            const uint ib = idx / 64;            // 4 values per idx
            const uint ib32 = (idx % 64) / 8;    // 0..7
            const uint iq = 4 * ib32 + (idx % 4);

            const uint sl = (data_a[ib].scales_l[ib32/2] >> (4 * (ib32 & 1))) & 0xF;
            const uint sh = ((data_a[ib].scales_h) >> (2 * ib32)) & 3;
            const uint qshift = idx & 4;
            u8vec4 qs = unpack8((uint(data_a_packed32[ib].qs[iq]) >> qshift) & 0x0F0F0F0F);

            const float d = float(data_a[ib].d);
            const vec4 v = d * float(int(sl | (sh << 4)) - 32) * vec4(kvalues_iq4nl[qs.x], kvalues_iq4nl[qs.y], kvalues_iq4nl[qs.z], kvalues_iq4nl[qs.w]);

            buf_a[buf_idx    ] = FLOAT_TYPEV2(v.xy);
            buf_a[buf_idx + 1] = FLOAT_TYPEV2(v.zw);
#elif defined(DATA_A_IQ4_NL)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 4;

            const uint ib = idx / 8;
            const uint iqs = idx & 0x07;

            const FLOAT_TYPE d = FLOAT_TYPE(data_a_packed16[ib].d);
            const uint vui = uint(data_a_packed16[ib].qs[iqs]);

            buf_a[buf_idx    ] = d * FLOAT_TYPEV2(kvalues_iq4nl[vui & 0xF],
                                                  kvalues_iq4nl[bitfieldExtract(vui, 8, 4)]);
            buf_a[buf_idx + 8] = d * FLOAT_TYPEV2(kvalues_iq4nl[bitfieldExtract(vui, 4, 4)],
                                                  kvalues_iq4nl[vui >> 12]);
#elif defined(DATA_A_MXFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_A / 4;

            const uint ib = idx / 8;
            const uint iqs = (idx & 0x07) * 2;

            const float d = e8m0_to_fp32(data_a[ib].e) * 0.5;
            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

            buf_a[buf_idx    ] = FLOAT_TYPEV2(kvalues_mxfp4[vui  & 0xF] * d,
                                              kvalues_mxfp4[vui2 & 0xF] * d);
            buf_a[buf_idx + 8] = FLOAT_TYPEV2(kvalues_mxfp4[vui  >>  4] * d,
                                              kvalues_mxfp4[vui2 >>  4] * d);
#elif defined(DATA_A_NVFP4)
            const uint idx = pos_a + col * p.stride_a / LOAD_VEC_A + row;
            // lo and hi nibbles are 8 elements apart, which doesn't quite line up with
            // how the thread mapping and buf_idx calculation works for other types.
            const uint buf_idx = col * SHMEM_STRIDE + (row & 3) + (row & ~3) * 2;

            const uint ib = idx / 16u;
            const uint sub = (idx & 0xC) >> 2;
            const uint iqs = (idx & 0xF) * 2;
            const float d = ue4m3_to_fp32(data_a[ib].d[sub]) * 0.5;
            const uint vui = uint(data_a[ib].qs[iqs]);
            const uint vui2 = uint(data_a[ib].qs[iqs+1]);

            buf_a[buf_idx    ] = FLOAT_TYPEV2(kvalues_mxfp4[vui  & 0xF] * d,
                                              kvalues_mxfp4[vui2 & 0xF] * d);
            buf_a[buf_idx + 4] = FLOAT_TYPEV2(kvalues_mxfp4[vui  >>  4] * d,
                                              kvalues_mxfp4[vui2 >>  4] * d);
#endif
}

#if !defined(MUL_MAT_ID)
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint idx_n, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            if (ALIGNED != 0) {
                // Not supported for b_type bf16 because bf16mat2x4 does not exist
                const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
                buf_b[buf_idx + 0] = bb[0].xy;
                buf_b[buf_idx + 1] = bb[0].zw;
                buf_b[buf_idx + 2] = bb[1].xy;
                buf_b[buf_idx + 3] = bb[1].zw;
                return;
            }
#elif LOAD_VEC_B == 4
            if (ALIGNED != 0) {
                const uint idx = pos_b + col * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
                buf_b[buf_idx + 0] = bb.xy;
                buf_b[buf_idx + 1] = bb.zw;
                return;
            }
#endif
            const uint idx = pos_b + col * p.stride_b + row * 2;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (idx_n < p.N && block + row * 2 + 1 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                              TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
            } else if (idx_n < p.N && block + row * 2 < end_k) {
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
            }
}
#else
void load_b_to_shmem(const uint pos_b, const uint row, const uint col, const uint ic, const uint _ne1, const uint block, const uint end_k) {
#if LOAD_VEC_B == 8
            if (ALIGNED != 0) {
                // Not supported for b_type bf16 because bf16mat2x4 does not exist
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
                FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx]);
                buf_b[buf_idx + 0] = bb[0].xy;
                buf_b[buf_idx + 1] = bb[0].zw;
                buf_b[buf_idx + 2] = bb[1].xy;
                buf_b[buf_idx + 3] = bb[1].zw;
                return;
            }
#elif LOAD_VEC_B == 4
            if (ALIGNED != 0) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b / LOAD_VEC_B + (row_idx.x % p.ne11) * p.stride_b / LOAD_VEC_B + row;
                const uint buf_idx = col * SHMEM_STRIDE + row * LOAD_VEC_B / 2;
#if defined(DATA_B_BF16)
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(TO_FLOAT_TYPE(data_b[idx]));
#else
                FLOAT_TYPEV4 bb = FLOAT_TYPEV4(data_b[idx]);
#endif
                buf_b[buf_idx + 0] = bb.xy;
                buf_b[buf_idx + 1] = bb.zw;
                return;
            }
#endif
            const uint row_i = ic * BN + col;
            const uint buf_idx = col * SHMEM_STRIDE + row;
            if (row_i < _ne1 && block + row * 2 + 1 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]),
                                              TO_FLOAT_TYPE(data_b_scalar[idx + 1]));
            } else if (row_i < _ne1 && block + row * 2 < end_k) {
                const u16vec2 row_idx = row_ids[col];
                const uint idx = pos_b + row_idx.y * p.batch_stride_b + (row_idx.x % p.ne11) * p.stride_b + row * 2;
                buf_b[buf_idx] = FLOAT_TYPEV2(TO_FLOAT_TYPE(data_b_scalar[idx]), 0.0f);
            } else {
                buf_b[buf_idx] = FLOAT_TYPEV2(0.0f);
            }
}
#endif
