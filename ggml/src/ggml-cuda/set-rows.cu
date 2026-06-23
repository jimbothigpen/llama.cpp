#include "set-rows.cuh"
#include "cpy-utils.cuh"
#include "turbo-quant.cuh"
#include <cstring>
#include <cerrno>
#include <cstdlib>

// One-shot loader for TURBO_TCQ_ALPHA (K) and TURBO_TCQ_ALPHA_V (V) env vars.
// Updates the d_tcq_norm_alpha{,_v} __constant__ symbols in turbo-quant.cuh.
// If both env vars are unset, keep the compiled-in defaults (K=1.1, V=1.3).
// If TURBO_TCQ_ALPHA is set alone, V tracks K (backwards-compat).
// If TURBO_TCQ_ALPHA_V is set, it sets V independently of K.
// Called on first TURBOQ{3,2}_TCQ SET_ROWS dispatch; idempotent thereafter.
static void load_tcq_norm_alpha() {
    static bool loaded = false;
    if (loaded) return;
    loaded = true;
    const char * s  = getenv("TURBO_TCQ_ALPHA");
    const char * sv = getenv("TURBO_TCQ_ALPHA_V");
    if (!s && !sv) return; // use compiled-in defaults
    float alpha_k = 1.0f; // matches d_tcq_norm_alpha compiled-in default
    bool k_set = false;
    if (s) {
        char * end;
        errno = 0;
        float a = strtof(s, &end);
        if (end == s || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid TURBO_TCQ_ALPHA='%s'\n", s);
        } else {
            alpha_k = a;
            k_set = true;
            (void) cudaMemcpyToSymbol(d_tcq_norm_alpha, &alpha_k, sizeof(float));
        }
    }
    if (sv) {
        char * end;
        errno = 0;
        float a = strtof(sv, &end);
        if (end == sv || errno != 0 || a <= 0.0f || a >= 10.0f) {
            fprintf(stderr, "TCQ: invalid TURBO_TCQ_ALPHA_V='%s'\n", sv);
        } else {
            (void) cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &a, sizeof(float));
            fprintf(stderr, "TCQ: norm alpha K=%.3f V=%.3f\n", alpha_k, a);
            return;
        }
    }
    // TURBO_TCQ_ALPHA set without TURBO_TCQ_ALPHA_V: V matches K (backwards-compat).
    if (k_set) {
        (void) cudaMemcpyToSymbol(d_tcq_norm_alpha_v, &alpha_k, sizeof(float));
        fprintf(stderr, "TCQ: norm alpha K=V=%.3f\n", alpha_k);
    }
}

// TCQ error dump (port of buun 764c686b0). Opt-in via TURBO_TCQ_DUMP_ERRORS=N;
// dumps the first N groups' post-FWHT values + Viterbi output symbols to a
// binary file at flush time (atexit). Output path defaults to "tcq_errors.bin"
// in the current working directory; override with TURBO_TCQ_DUMP_PATH. The
// /tmp default from buun's commit is avoided here because /tmp may be a
// RAM-backed tmpfs and large dumps would OOM the ramdisk.
static int       tcq_dump_n        = 0;
static float   * tcq_dump_x_host   = nullptr;
static uint8_t * tcq_dump_out_host = nullptr;
static float   * tcq_dump_x_dev    = nullptr;
static uint8_t * tcq_dump_out_dev  = nullptr;
static const char * tcq_dump_path  = nullptr;

static void tcq_error_dump_flush() {
    if (tcq_dump_n == 0) return;
    (void) cudaMemcpy(tcq_dump_x_host,   tcq_dump_x_dev,   tcq_dump_n * 128 * sizeof(float),   cudaMemcpyDeviceToHost);
    (void) cudaMemcpy(tcq_dump_out_host, tcq_dump_out_dev, tcq_dump_n * 128 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    const char * path = tcq_dump_path ? tcq_dump_path : "tcq_errors.bin";
    FILE * f = fopen(path, "wb");
    if (f) {
        int32_t header[1] = { tcq_dump_n };
        fwrite(header,           sizeof(int32_t), 1,                  f);
        fwrite(tcq_dump_x_host,  sizeof(float),   tcq_dump_n * 128,   f);
        fwrite(tcq_dump_out_host, sizeof(uint8_t), tcq_dump_n * 128,  f);
        fclose(f);
        fprintf(stderr, "TCQ: dumped %d groups to %s\n", tcq_dump_n, path);
    } else {
        fprintf(stderr, "TCQ: failed to open dump path '%s' for write\n", path);
    }
    (void) cudaFree(tcq_dump_x_dev);
    (void) cudaFree(tcq_dump_out_dev);
    free(tcq_dump_x_host);
    free(tcq_dump_out_host);
}

static void init_tcq_error_dump() {
    static bool loaded = false;
    if (loaded) return;
    loaded = true;
    const char * s = getenv("TURBO_TCQ_DUMP_ERRORS");
    if (!s) return;
    int n = atoi(s);
    if (n <= 0 || n > 100000) return;
    tcq_dump_n        = n;
    tcq_dump_path     = getenv("TURBO_TCQ_DUMP_PATH"); // nullptr → fallback in flush
    tcq_dump_x_host   = (float   *)malloc(n * 128 * sizeof(float));
    tcq_dump_out_host = (uint8_t *)malloc(n * 128 * sizeof(uint8_t));
    (void) cudaMalloc(&tcq_dump_x_dev,   n * 128 * sizeof(float));
    (void) cudaMalloc(&tcq_dump_out_dev, n * 128 * sizeof(uint8_t));
    (void) cudaMemcpyToSymbol(d_tcq_dump_x_buf,   &tcq_dump_x_dev,   sizeof(float   *));
    (void) cudaMemcpyToSymbol(d_tcq_dump_out_buf, &tcq_dump_out_dev, sizeof(uint8_t *));
    (void) cudaMemcpyToSymbol(d_tcq_dump_max,     &n,                sizeof(int));
    atexit(tcq_error_dump_flush);
    fprintf(stderr, "TCQ: will dump errors for first %d groups to %s\n",
            n, tcq_dump_path ? tcq_dump_path : "tcq_errors.bin (cwd)");
}

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}
// ---- TurboQuant3 set_rows: GROUP_SIZE-element groups with WHT rotation + norm correction ----
//
// Templated on GROUP_SIZE (128 or 64).
// Parallel kernel: one CUDA block per group, GROUP_SIZE threads per block.
// Thread j handles element j within the group.
//
// Steps (all parallel):
//   1. Load element j from global memory
//   2. Parallel L2 norm (warp reduce + inter-warp via shared memory)
//   3. Normalize
//   4. Forward WHT (log2(GROUP_SIZE) butterfly stages, shared memory)
//   5. Quantize element j to 3-bit centroid index
//   6. Pack qs (warp shuffle) and signs (__ballot_sync) into turboq3 block, no atomics
//   7. Parallel reconstruction norm (same pattern as step 2)
//   8. Write corrected norm (one thread per sub-block)

template <typename idx_t, int GROUP_SIZE>
__launch_bounds__(128)  // max of 128 or 64
static __global__ void k_set_rows_turboq3(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    static_assert(GROUP_SIZE == 128 || GROUP_SIZE == 64, "GROUP_SIZE must be 128 or 64");

    // blockIdx.x = flat group index; threadIdx.x = element within group (0..GROUP_SIZE-1)
    const int j = threadIdx.x;

    // Decode blockIdx.x → (i_grp, i01, i02, i03)
    constexpr int blocks_per_group = GROUP_SIZE / QK_TURBOQ3;
    const int64_t n_groups_per_row = ne00 / GROUP_SIZE;
    const int64_t g = blockIdx.x;
    const int64_t i_grp = g % n_groups_per_row;
    int64_t       tmp   = g / n_groups_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq3_0 * dst_row_ptr = (block_turboq3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turboq3_0 * blk_base    = dst_row_ptr + i_grp * blocks_per_group;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[GROUP_SIZE];
    x[j] = src_row[i_grp * GROUP_SIZE + j];
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = GROUP_SIZE / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 → butterfly → signs2, normalized) ----
    if (GROUP_SIZE == 128) {
        x[j] *= TURBO_WHT_SIGNS1[j];
    } else {
        x[j] *= TURBO_WHT_SIGNS1_64[j];
    }
    __syncthreads();

#define WHT_STAGE_SHARED(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    // Butterfly stages: loop from h=1 to h<GROUP_SIZE, doubling each time
    WHT_STAGE_SHARED(1)
    WHT_STAGE_SHARED(2)
    WHT_STAGE_SHARED(4)
    WHT_STAGE_SHARED(8)
    WHT_STAGE_SHARED(16)
    WHT_STAGE_SHARED(32)
    if (GROUP_SIZE == 128) { WHT_STAGE_SHARED(64) }
#undef WHT_STAGE_SHARED

    constexpr float inv_sqrt_group = (GROUP_SIZE == 128) ? 0.08838834764831845f : 0.125f;
    if (GROUP_SIZE == 128) {
        x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2[j];
    } else {
        x[j] = x[j] * inv_sqrt_group * TURBO_WHT_SIGNS2_64[j];
    }
    __syncthreads();

    // ---- Step 5: Quantize element j ----
    const float rv = x[j];
    const uint8_t idx = turbo_nearest_centroid_3bit(rv);

    // ---- Step 6: Pack qs and signs (warp-cooperative, no atomics) ----
    // Each warp handles 32 elements. With QK_TURBOQ3 > WARP_SIZE, multiple warps
    // share one block and write to different byte offsets within it.
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;
    const int elem_in_block = j % QK_TURBOQ3;
    block_turboq3_0 * blk = blk_base + (j / QK_TURBOQ3);

    // Pack qs: 4 elements per byte, 2 bits each.
    // All 4 threads in a qs-group gather their low2 bits via shuffle.
    const int qs_byte_idx = elem_in_block / 4;
    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_low2, (lane & ~3) + k, WARP_SIZE);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[qs_byte_idx] = qs_byte;

    // Pack signs: 8 elements per byte, 1 bit each.  __ballot_sync across warp.
    // Ballot is per-warp (32 bits); extract local byte, write to global position in block.
    const uint32_t ballot = __ballot_sync(0xffffffffffffffffull, (idx >> 2) & 1);
    const int local_signs_byte = lane / 8;             // byte within 32-bit ballot (0..3)
    const int global_signs_byte = elem_in_block / 8;   // byte within block's signs array
    const uint8_t signs_byte = (uint8_t)((ballot >> (local_signs_byte * 8)) & 0xFF);
    if (lane % 8 == 0) blk->signs[global_signs_byte] = signs_byte;

    // ---- Step 7: Reconstruction norm (parallel, same pattern as step 2) ----
    const float c = TURBO_CENTROIDS_3BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm (one per turboq3 block) ----
    if (elem_in_block == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

// ---- TurboQuant3 tail kernel: straight 3-bit quantize without WHT rotation ----
//
// For head dims not divisible by 128 (e.g. 576 = 4*128 + 64), the remainder
// elements can't use the 128-element WHT. They are quantised directly into
// standard turboq3 blocks.  Q is also NOT rotated for these positions (the graph
// guards on ne[0] % 128), so <Q_tail, K_tail> stays in the original space.
//
// One CUDA block per row, with tail_size threads (must be multiple of 32).

template <typename idx_t>
static __global__ void k_set_rows_turboq3_tail(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq3_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const int tail_size) {

    const int j = threadIdx.x;  // 0 .. tail_size-1

    // Decode blockIdx.x → (i01, i02, i03)
    int64_t tmp = blockIdx.x;
    const int64_t i01 = tmp % ne01; tmp /= ne01;
    const int64_t i02 = tmp % ne12;
    const int64_t i03 = tmp / ne12;

    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;
    const int64_t i12 = i02;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq3_0 * dst_row_ptr = (block_turboq3_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    // Tail starts after all full 128-element groups
    const int64_t n_full = ne00 / QK_TURBOQ3_GROUP;
    const int64_t tail_start = n_full * QK_TURBOQ3_GROUP;
    block_turboq3_0 * blk_base = dst_row_ptr + n_full * (QK_TURBOQ3_GROUP / QK_TURBOQ3);

    // ---- Load ----
    const float val = src_row[tail_start + j];

    // ---- L2 norm over the tail group (warp reduce + inter-warp) ----
    const int n_warps = tail_size / WARP_SIZE;
    const int warp_id = j / WARP_SIZE;
    const int lane    = j % WARP_SIZE;

    __shared__ float warp_accum[4];  // max 3 warps (tail ≤ 96)
    float v2 = val * val;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (lane == 0) warp_accum[warp_id] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Normalize (no WHT!) ----
    const float rv = val * inv_norm;

    // ---- Quantize ----
    const uint8_t idx = turbo_nearest_centroid_3bit(rv);

    // ---- Pack qs and signs (same warp-cooperative logic) ----
    block_turboq3_0 * blk = blk_base + warp_id;

    const uint8_t my_low2 = idx & 0x3;
    uint8_t qs_byte = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        uint8_t contrib = __shfl_sync(0xffffffff, my_low2, (lane & ~3) + k, WARP_SIZE);
        qs_byte |= contrib << (k * 2);
    }
    if (lane % 4 == 0) blk->qs[lane / 4] = qs_byte;

    const uint32_t ballot = __ballot_sync(0xffffffffffffffffull, (idx >> 2) & 1);
    const int signs_byte_idx = lane / 8;
    const uint8_t signs_byte = (uint8_t)((ballot >> (signs_byte_idx * 8)) & 0xFF);
    if (lane % 8 == 0) blk->signs[signs_byte_idx] = signs_byte;

    // ---- Reconstruction norm ----
    const float c = TURBO_CENTROIDS_3BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (lane == 0) warp_accum[warp_id] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    if (lane == 0) blk->norm = __float2half(corrected_norm);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turboq3(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ3 == 0);  // must be block-aligned (32)

    cudaStream_t stream = ctx.stream();

    // Read WHT group size from op_params (set by llama-kv-cache.cpp based on head_dim).
    // Default to 128 if not set (backward compat with head_dim=128 models).
    int group_size = 128;
    memcpy(&group_size, dst->op_params, sizeof(int));
    if (group_size != 64 && group_size != 128) group_size = 128;
    GGML_ASSERT(ne00 % group_size == 0);

    const int64_t n_full_groups   = ne00 / group_size;
    const int     tail_size       = (int)(ne00 % group_size);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    // Launch 1: full groups with WHT rotation
    if (n_full_groups > 0) {
        const int64_t ne_total = n_full_groups * ne01 * ne02 * ne03;
        if (group_size == 128) {
            k_set_rows_turboq3<idx_t, 128><<<(int)ne_total, 128, 0, stream>>>(
                src0_d, src1_d, (block_turboq3_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        } else {
            k_set_rows_turboq3<idx_t, 64><<<(int)ne_total, 64, 0, stream>>>(
                src0_d, src1_d, (block_turboq3_0 *)dst->data,
                ne00, ne01, ne10, ne11, ne12, ne13,
                s01, s02, s03, s10, s11, s12,
                nb1, nb2, nb3);
        }
    }

    // Launch 2: tail elements (no WHT, straight quantize)
    // Not needed for 64-aligned dims but kept for potential future use
    if (tail_size > 0) {
        GGML_ASSERT(tail_size % QK_TURBOQ3 == 0);  // tail must be block-aligned
        const int64_t n_rows = ne01 * ne02 * ne03;
        k_set_rows_turboq3_tail<idx_t><<<(int)n_rows, tail_size, 0, stream>>>(
            src0_d, src1_d, (block_turboq3_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3, tail_size);
    }
}

// ---- TurboQuant4 set_rows: 128-element groups with WHT rotation + 4-bit quantization ----
//
// turboq4 block size IS the WHT group size (128), so 1 CUDA block = 1 turboq4 block.
// 128 threads per block, thread j handles element j.
// 4-bit centroids (16 values), nibble packed: qs[j/2] |= (idx & 0xF) << ((j%2)*4)

template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turboq2(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq2_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    // blockIdx.x = flat block index; threadIdx.x = element within block (0..127)
    const int j = threadIdx.x;

    // Decode blockIdx.x → (i_blk, i01, i02, i03)
    const int64_t n_blocks_per_row = ne00 / QK_TURBOQ2;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq2_0 * dst_row_ptr = (block_turboq2_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turboq2_0 * blk = dst_row_ptr + i_blk;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBOQ2 + j];
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = 128 / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 → butterfly → signs2, normalized) ----
    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T2(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T2(1)
    WHT_STAGE_SHARED_T2(2)
    WHT_STAGE_SHARED_T2(4)
    WHT_STAGE_SHARED_T2(8)
    WHT_STAGE_SHARED_T2(16)
    WHT_STAGE_SHARED_T2(32)
    WHT_STAGE_SHARED_T2(64)
#undef WHT_STAGE_SHARED_T2

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Step 5: Quantize element j to 2-bit centroid ----
    const float rv = x[j];
    const uint8_t idx = turbo_nearest_centroid_2bit(rv);

    // ---- Step 6: Pack qs (2-bit packed, 4 indices per byte) ----
    // Thread 0 of each group-of-4 reads idx for j..j+3 and packs them.
    const int lane = j % WARP_SIZE;
    const uint8_t my_idx = idx & 0x3;
    uint8_t partner1 = __shfl_sync(0xffffffff, my_idx, lane ^ 1, WARP_SIZE);
    uint8_t partner2 = __shfl_sync(0xffffffff, my_idx, lane ^ 2, WARP_SIZE);
    uint8_t partner3 = __shfl_sync(0xffffffff, my_idx, lane ^ 3, WARP_SIZE);
    if (j % 4 == 0) {
        // lane ^ 1 swaps lane within pair; lane ^ 2 swaps the pair; together cover j..j+3.
        const uint8_t i0 = my_idx;
        const uint8_t i1 = partner1;
        const uint8_t i2 = partner2;
        const uint8_t i3 = partner3;
        const uint8_t qs_byte = i0 | (i1 << 2) | (i2 << 4) | (i3 << 6);
        blk->qs[j / 4] = qs_byte;
    }

    // ---- Step 7: Reconstruction norm (parallel) ----
    const float c = TURBO_CENTROIDS_2BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm (one thread) ----
    if (j == 0) {
        blk->norm = __float2half(corrected_norm);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turboq2(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ2 == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBOQ2;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (n_blocks > 0) {
        const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;
        k_set_rows_turboq2<idx_t><<<(int)ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turboq2_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
    }
}


template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turboq4(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq4_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    // blockIdx.x = flat block index; threadIdx.x = element within block (0..127)
    const int j = threadIdx.x;

    // Decode blockIdx.x → (i_blk, i01, i02, i03)
    const int64_t n_blocks_per_row = ne00 / QK_TURBOQ4;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq4_0 * dst_row_ptr = (block_turboq4_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turboq4_0 * blk = dst_row_ptr + i_blk;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBOQ4 + j];
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = 128 / WARP_SIZE;  // = 4
    __shared__ float warp_accum[n_warps];
    float v = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm  = sqrtf(s_norm_sq);
    const float inv_norm  = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 → butterfly → signs2, normalized) ----
    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T4(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T4(1)
    WHT_STAGE_SHARED_T4(2)
    WHT_STAGE_SHARED_T4(4)
    WHT_STAGE_SHARED_T4(8)
    WHT_STAGE_SHARED_T4(16)
    WHT_STAGE_SHARED_T4(32)
    WHT_STAGE_SHARED_T4(64)
#undef WHT_STAGE_SHARED_T4

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Step 5: Quantize element j to 4-bit centroid ----
    const float rv = x[j];
    const uint8_t idx = turbo_nearest_centroid_4bit(rv);

    // ---- Step 6: Pack qs (nibble packed, warp-cooperative) ----
    // 2 elements per byte, 4 bits each.
    // Thread pairs (j, j+1) share a qs byte.
    const int lane = j % WARP_SIZE;
    const uint8_t my_nibble = idx & 0xF;
    uint8_t qs_byte = 0;
    // Gather nibble from partner thread
    uint8_t partner_nibble = __shfl_sync(0xffffffff, my_nibble, lane ^ 1, WARP_SIZE);
    if (j % 2 == 0) {
        qs_byte = my_nibble | (partner_nibble << 4);
        blk->qs[j / 2] = qs_byte;
    }

    // ---- Step 7: Reconstruction norm (parallel) ----
    const float c = TURBO_CENTROIDS_4BIT[idx];
    float rc = c * c;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        rc += __shfl_xor_sync(0xffffffff, rc, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = rc;
    __syncthreads();

    __shared__ float s_recon_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_recon_sq = total;
    }
    __syncthreads();
    const float recon_norm     = sqrtf(s_recon_sq);
    const float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;

    // ---- Step 8: Write corrected norm and zero rnorm (one thread) ----
    if (j == 0) {
        blk->norm  = __float2half(corrected_norm);
        blk->rnorm = __float2half(0.0f);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turboq4(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ4 == 0);  // must be block-aligned (128)

    cudaStream_t stream = ctx.stream();

    // turboq4 block size = WHT group size = 128, always
    const int64_t n_blocks = ne00 / QK_TURBOQ4;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (n_blocks > 0) {
        const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;
        k_set_rows_turboq4<idx_t><<<(int)ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turboq4_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
    }
}

// TURBOQ8 SET_ROWS encode: 8-bit uniform-grid + per-block absmax.
// Steps 1-4 (load, parallel L2 norm, normalize, forward WHT) are
// identical to turboq4. Steps 5-7 diverge: uniform 256-level grid in the rotated domain
// (no Lloyd-Max codebook, no reconstruction-norm correction). 1 block = 1 turboq8 block (128).
template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turboq8(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq8_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    const int j = threadIdx.x;

    // Decode blockIdx.x → (i_blk, i01, i02, i03)
    const int64_t n_blocks_per_row = ne00 / QK_TURBOQ8;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq8_0 * dst_row_ptr = (block_turboq8_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turboq8_0 * blk = dst_row_ptr + i_blk;

    // ---- Step 1: Load element j (coalesced) ----
    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBOQ8 + j];
    __syncthreads();

    // ---- Step 2: Parallel L2 norm ----
    constexpr int n_warps = 128 / WARP_SIZE;  // = 4
    __shared__ float warp_accum[n_warps];
    float v  = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    // ---- Step 3: Normalize ----
    x[j] *= inv_norm;
    __syncthreads();

    // ---- Step 4: Forward WHT (signs1 → butterfly → signs2, normalized) ----
    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T8(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T8(1)
    WHT_STAGE_SHARED_T8(2)
    WHT_STAGE_SHARED_T8(4)
    WHT_STAGE_SHARED_T8(8)
    WHT_STAGE_SHARED_T8(16)
    WHT_STAGE_SHARED_T8(32)
    WHT_STAGE_SHARED_T8(64)
#undef WHT_STAGE_SHARED_T8

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    // ---- Step 5: Per-block absmax (parallel reduction over the rotated values) ----
    float av = fabsf(x[j]);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, offset, WARP_SIZE));
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = av;
    __syncthreads();

    __shared__ float s_absmax;
    if (j == 0) {
        float m = 0.0f;
        for (int w = 0; w < n_warps; w++) m = fmaxf(m, warp_accum[w]);
        s_absmax = m;
    }
    __syncthreads();
    const float scale     = s_absmax > 1e-10f ? s_absmax : 1e-10f;
    const float inv_scale = 1.0f / scale;

    // ---- Step 6: Quantize element j to the uniform 256-level grid (direct byte store) ----
    int idx = (int)lrintf(x[j] * inv_scale * 127.5f + 127.5f);
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    blk->qs[j] = (uint8_t)idx;

    // ---- Step 7: Store norm (folds in the absmax scale) ----
    if (j == 0) {
        blk->norm = __float2half(grp_norm * scale);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turboq8(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ8 == 0);  // must be block-aligned (128)

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBOQ8;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);


    if (n_blocks > 0) {
        const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;
        k_set_rows_turboq8<idx_t><<<(int)ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turboq8_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
    }
}

// TURBOQ5 SET_ROWS encode: 5-bit uniform-grid + per-block absmax (invented ygg TODO 250).
// Steps 1-5 identical to turboq8. Step 6 packs a 5-bit index split (low nibble in qs, high bit
// in qh). Because adjacent threads share a packed byte, indices are staged in shared memory and
// a disjoint subset of threads writes each output byte (no atomics needed). 1 block = 128 values.
template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turboq5(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq5_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    const int j = threadIdx.x;

    const int64_t n_blocks_per_row = ne00 / QK_TURBOQ5;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq5_0 * dst_row_ptr = (block_turboq5_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turboq5_0 * blk = dst_row_ptr + i_blk;

    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBOQ5 + j];
    __syncthreads();

    constexpr int n_warps = 128 / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v  = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    x[j] *= inv_norm;
    __syncthreads();

    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T5(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T5(1)
    WHT_STAGE_SHARED_T5(2)
    WHT_STAGE_SHARED_T5(4)
    WHT_STAGE_SHARED_T5(8)
    WHT_STAGE_SHARED_T5(16)
    WHT_STAGE_SHARED_T5(32)
    WHT_STAGE_SHARED_T5(64)
#undef WHT_STAGE_SHARED_T5

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    float av = fabsf(x[j]);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, offset, WARP_SIZE));
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = av;
    __syncthreads();

    __shared__ float s_absmax;
    if (j == 0) {
        float m = 0.0f;
        for (int w = 0; w < n_warps; w++) m = fmaxf(m, warp_accum[w]);
        s_absmax = m;
    }
    __syncthreads();
    const float scale     = s_absmax > 1e-10f ? s_absmax : 1e-10f;
    const float inv_scale = 1.0f / scale;

    // ---- Step 6: Quantize to uniform 32-level grid, stage idx, then pack qs/qh ----
    __shared__ unsigned char idx_sh[128];
    int idx = (int)lrintf(x[j] * inv_scale * 15.5f + 15.5f);
    idx = idx < 0 ? 0 : (idx > 31 ? 31 : idx);
    idx_sh[j] = (unsigned char)idx;
    __syncthreads();

    if (j < QK_TURBOQ5 / 2) {       // 64 threads: low nibble pack
        blk->qs[j] = (unsigned char)((idx_sh[2*j] & 0xF) | ((idx_sh[2*j + 1] & 0xF) << 4));
    }
    if (j < QK_TURBOQ5 / 8) {       // 16 threads: 1 high bit × 8 per byte
        unsigned char b = 0;
        for (int k = 0; k < 8; k++) b |= (unsigned char)(((idx_sh[8*j + k] >> 4) & 0x1) << k);
        blk->qh[j] = b;
    }

    if (j == 0) {
        blk->norm = __float2half(grp_norm * scale);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turboq5(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ5 == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBOQ5;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (n_blocks > 0) {
        const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;
        k_set_rows_turboq5<idx_t><<<(int)ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turboq5_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
    }
}

// TURBOQ6 SET_ROWS encode: 6-bit uniform-grid + per-block absmax (invented ygg TODO 250).
// Step 6 packs a 6-bit index split (low nibble in qs, high 2 bits in qh, 4 per byte).
template <typename idx_t>
__launch_bounds__(128)
static __global__ void k_set_rows_turboq6(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_turboq6_0 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3) {

    const int j = threadIdx.x;

    const int64_t n_blocks_per_row = ne00 / QK_TURBOQ6;
    const int64_t g = blockIdx.x;
    const int64_t i_blk = g % n_blocks_per_row;
    int64_t       tmp   = g / n_blocks_per_row;
    const int64_t i01   = tmp % ne01;
    tmp                 = tmp / ne01;
    const int64_t i02   = tmp % ne12;
    const int64_t i03   = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_turboq6_0 * dst_row_ptr = (block_turboq6_0 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);
    block_turboq6_0 * blk = dst_row_ptr + i_blk;

    __shared__ float x[128];
    x[j] = src_row[i_blk * QK_TURBOQ6 + j];
    __syncthreads();

    constexpr int n_warps = 128 / WARP_SIZE;
    __shared__ float warp_accum[n_warps];
    float v  = x[j];
    float v2 = v * v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        v2 += __shfl_xor_sync(0xffffffff, v2, offset, WARP_SIZE);
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = v2;
    __syncthreads();

    __shared__ float s_norm_sq;
    if (j == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += warp_accum[w];
        s_norm_sq = total;
    }
    __syncthreads();
    const float grp_norm = sqrtf(s_norm_sq);
    const float inv_norm = (grp_norm > 1e-10f) ? 1.0f / grp_norm : 0.0f;

    x[j] *= inv_norm;
    __syncthreads();

    x[j] *= TURBO_WHT_SIGNS1[j];
    __syncthreads();

#define WHT_STAGE_SHARED_T6(h) \
    if (j % (2*(h)) < (h)) { float a = x[j], b = x[j+(h)]; x[j] = a+b; x[j+(h)] = a-b; } \
    __syncthreads();

    WHT_STAGE_SHARED_T6(1)
    WHT_STAGE_SHARED_T6(2)
    WHT_STAGE_SHARED_T6(4)
    WHT_STAGE_SHARED_T6(8)
    WHT_STAGE_SHARED_T6(16)
    WHT_STAGE_SHARED_T6(32)
    WHT_STAGE_SHARED_T6(64)
#undef WHT_STAGE_SHARED_T6

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    x[j] = x[j] * inv_sqrt_128 * TURBO_WHT_SIGNS2[j];
    __syncthreads();

    float av = fabsf(x[j]);
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        av = fmaxf(av, __shfl_xor_sync(0xffffffff, av, offset, WARP_SIZE));
    if (j % WARP_SIZE == 0)
        warp_accum[j / WARP_SIZE] = av;
    __syncthreads();

    __shared__ float s_absmax;
    if (j == 0) {
        float m = 0.0f;
        for (int w = 0; w < n_warps; w++) m = fmaxf(m, warp_accum[w]);
        s_absmax = m;
    }
    __syncthreads();
    const float scale     = s_absmax > 1e-10f ? s_absmax : 1e-10f;
    const float inv_scale = 1.0f / scale;

    // ---- Step 6: Quantize to uniform 64-level grid, stage idx, then pack qs/qh ----
    __shared__ unsigned char idx_sh[128];
    int idx = (int)lrintf(x[j] * inv_scale * 31.5f + 31.5f);
    idx = idx < 0 ? 0 : (idx > 63 ? 63 : idx);
    idx_sh[j] = (unsigned char)idx;
    __syncthreads();

    if (j < QK_TURBOQ6 / 2) {       // 64 threads: low nibble pack
        blk->qs[j] = (unsigned char)((idx_sh[2*j] & 0xF) | ((idx_sh[2*j + 1] & 0xF) << 4));
    }
    if (j < QK_TURBOQ6 / 4) {       // 32 threads: 2 high bits × 4 per byte
        unsigned char b = 0;
        for (int k = 0; k < 4; k++) b |= (unsigned char)(((idx_sh[4*j + k] >> 4) & 0x3) << (k*2));
        blk->qh[j] = b;
    }

    if (j == 0) {
        blk->norm = __float2half(grp_norm * scale);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
}

template<typename idx_t>
static void set_rows_cuda_turboq6(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ6 == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t n_blocks = ne00 / QK_TURBOQ6;

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (n_blocks > 0) {
        const int64_t ne_total = n_blocks * ne01 * ne02 * ne03;
        k_set_rows_turboq6<idx_t><<<(int)ne_total, 128, 0, stream>>>(
            src0_d, src1_d, (block_turboq6_0 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3);
    }
}



// =====================================================================================
// TCQ KV cache SET_ROWS Viterbi encode kernels — Phase 3a port from buun master
// k_set_rows_turboq3_tcq: 512-thread block (one per trellis state), k=3 L=9
// k_set_rows_turboq2_tcq: 256-thread block (one per trellis state), k=2 L=8
// Codebooks live in turbo-quant.cuh; GET_ROWS dequant also lives there.
// =====================================================================================

// Global backtrace buffer for TCQ Viterbi (replaces 32KB shared/block + 16KB for 2-bit).
// Sized to ne_total_groups * 128 * 64 (compressed: 64 low-state groups per step,
// same layout for both turboq3 and turboq2). Grown on demand. Devices that opt in
// to per-block shared-memory backtrace bypass this buffer.
static uint8_t * tcq_bt_buf = nullptr;
static int64_t   tcq_bt_buf_bytes = 0;

static void ensure_tcq_bt_buf(int64_t bytes_needed) {
    if (bytes_needed <= tcq_bt_buf_bytes) return;
    if (tcq_bt_buf) (void) cudaFree(tcq_bt_buf);
    (void) cudaMalloc(&tcq_bt_buf, bytes_needed);
    tcq_bt_buf_bytes = bytes_needed;
}

// TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis
// 512 threads per block (one per trellis state), one block per 128-element group
// Double-buffered cost arrays + global memory backtrace (128 syncs/group, was 384)
template<typename idx_t>
static __global__ void __launch_bounds__(512, 1) k_set_rows_turboq3_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turboq3_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf,
        const int use_shared_bt,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int is_k,
        const int64_t s1,  const int64_t s2,  const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int64_t group = blockIdx.x;
    if (group >= ne_total_groups) return;

    const int sid = threadIdx.x; // state index 0..511

    // Compute source and destination pointers (same index math as turbo3)
    const int64_t i_base = group * QK_TURBOQ3_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turboq3_tcq * dst_blk = (block_turboq3_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                                  + (i00 / QK_TURBOQ3_TCQ);

    // Shared memory layout (~5KB, was ~35KB before global-bt + double-buffer port):
    // x[128]     : rotated+normalized input (also reused as outputs[] after Viterbi)
    // cost[512]  : path costs buffer A (also reused as reduction scratch)
    // cost_b[512]: path costs buffer B (double-buffering eliminates 2/3 of syncs)
    // Backtrace: one predecessor byte for each of the 64 low-state groups per
    // step. The predecessor is independent of the output bits in sid[8:6], so
    // storing 128×64 bytes is equivalent to the older 128×512 layout. The
    // backtrace lives in dynamic shared memory when the device opts in, else
    // in bt_buf in global memory (still 128×64 bytes per block, byte-packed).
    extern __shared__ uint8_t bt_shared[];
    __shared__ float x[128];
    __shared__ float cost[512];
    __shared__ float cost_b[512];
    __shared__ int   warp_min_idx[16];
    __shared__ float warp_min_cost[16];
    __shared__ float pred_min_cost[64];
    // phase 3a #21: see buun 12a648efc cuda: streamline tcq final state selection
    // (removed __shared__ uint8_t pred_min_p[64] — dead-store; bt[t*64+sid] already
    // holds the same value used by backtrack)
    __shared__ int   shared_initial_state;
    // Dedicated shared buffer for the Viterbi-backtrack output bytes. Previously
    // aliased onto x[] via (uint8_t *)x, but writing uint8_t into a float-typed
    // shared array is a strict-aliasing violation: under HIP/ROCm the compiler
    // can hoist cross-thread reads of outputs[] above the __syncthreads() that
    // follows the sid==0 backtrack write, so sids 1..48 in the parallel bitpack
    // observed stale (non-winning) symbol bytes — root cause of the Phase 3a #20
    // +12.7% PPL regression (session-65-resume-cell-c-ppl bisect, 2026-05-17).
    __shared__ uint8_t s_outputs[128];

    // Parallel pre-Viterbi: load (threads 0-127 each grab one element)
    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    // Parallel norm reduction: tree-reduce x[i]^2 via cost[0..511] → cost[0]
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 256; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  8, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  4, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  2, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  1, WARP_SIZE);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    // Normalize (parallel)
    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // Parallel FWHT: signs1 → 7-stage butterfly → scale + signs2.
    // The first five stages are contained within each 32-lane warp, so use
    // warp shuffles and only synchronize for the two cross-warp stages.
    if (sid < 128) {
        float v = x[sid] * TURBO_WHT_SIGNS1[sid];
        const int lane = sid & 31;
        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, v, h, WARP_SIZE);
            v = (lane & h) ? (other - v) : (v + other);
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        float a = x[j], b = x[j + 32];
        x[j] = a + b; x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        float a = x[sid], b = x[sid + 64];
        x[sid] = a + b; x[sid + 64] = a - b;
    }
    __syncthreads();
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * TURBO_WHT_SIGNS2[sid];
    __syncthreads();

    // Post-rotation extraction (if enabled)
    // (yggdrasil: TURBO_EXTRACT data-collection elided)

    // Stash norm in cost[0] for post-Viterbi (will be re-zeroed by the Viterbi init below)
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Initialize Viterbi: free initial state (all states equally viable)
    // Double-buffered cost (1 sync/step, was 3); byte-packed bt in shared or global memory.
    uint8_t * bt = use_shared_bt ? bt_shared : bt_buf + (int64_t)blockIdx.x * (128 * 64);
    cost[sid] = 0.0f;
    __syncthreads();

    // Forward pass: 128 time steps, fully parallel across 512 states
    for (int t = 0; t < 128; t++) {
        // Double-buffer: even steps read cost/write cost_b, odd steps read cost_b/write cost.
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis: ns = (prev >> 3) | (out << 6). The best
        // predecessor depends only on sid's low 6 bits, so compute those 64
        // minima once instead of repeating the same 8-way scan for each out.
        if (sid < 64) {
            const int base_prev = sid << 3;
            float best = cost_rd[base_prev];
            int   best_p = 0;
            #pragma unroll
            for (int p = 1; p < 8; p++) {
                float c = cost_rd[base_prev | p];
                if (c < best) {
                    best = c;
                    best_p = p;
                }
            }
            pred_min_cost[sid] = best;
            // phase 3a #21: see buun 12a648efc — dead-store removal (pred_min_p was never read)
            bt[t * 64 + sid]   = (uint8_t) best_p;
        }
        __syncthreads();

        const int pred_idx = sid & 0x3F;
        float dist = xt - d_turboq3_tcq_codebook[sid];
        dist = dist * dist;

        cost_wr[sid] = pred_min_cost[pred_idx] + dist;
        __syncthreads();
    }
    // After 128 steps (even count): final costs are in cost[] (step 127 is odd → cost_wr=cost)

    // Parallel find-min: warp-level argmin over 512 costs, then reduce 16 warps
    {
        float my_cost = cost[sid];
        int   my_idx  = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, my_cost, offset, WARP_SIZE);
            int   other_idx  = __shfl_xor_sync(0xFFFFFFFF, my_idx,  offset, WARP_SIZE);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32]  = my_idx;
        }
    }
    __syncthreads();
    // H1 fix: revert warp-shuffle parallel argmin to pre-#21 serial loop.
    // HIP/RDNA3 wave32 __shfl_down_sync semantics produced +0.20% chunk-1 PPL drift
    // vs the pre-#21 serial reduction; intentional divergence from buun 12a648efc
    // to preserve fidelity with mainline llama.cpp.
    if (sid == 0) {
        float best     = warp_min_cost[0];
        int   best_idx = warp_min_idx[0];
        for (int w = 1; w < 16; w++) {
            if (warp_min_cost[w] < best) { best = warp_min_cost[w]; best_idx = warp_min_idx[w]; }
        }
        shared_initial_state = best_idx; // temporarily: best final state (becomes initial after backtrack)
    }
    __syncthreads();

    // TCQ error dump (port of buun 764c686b0): save post-FWHT x[] before backtrack overwrites it.
    if (d_tcq_dump_max > 0 && group < d_tcq_dump_max && sid < 128)
        d_tcq_dump_x_buf[group * 128 + sid] = x[sid];

    // Thread 0: backtrack (inherently sequential — each step depends on the next)
    // Reads byte-packed bt from global memory (no nibble unpack). Writes the
    // winning-path output bytes into __shared__ s_outputs[] (type-clean; see
    // declaration comment above).
    uint8_t * outputs = s_outputs;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6); // output = top 3 bits (right-shift trellis)
            int p = bt[t * 64 + (state & 0x3F)];
            state = ((state & 0x3F) << 3) | p; // reconstruct predecessor
        }
        // After full backtrack, 'state' is the initial state chosen by Viterbi
        shared_initial_state = state;
    }
    __syncthreads();

    // TCQ error dump: save Viterbi output symbols.
    if (d_tcq_dump_max > 0 && group < d_tcq_dump_max && sid < 128)
        d_tcq_dump_out_buf[group * 128 + sid] = outputs[sid];

    // Parallel recon norm: for t >= 2, state_t = out[t-2] | (out[t-1]<<3) | (out[t]<<6)
    // For t < 2: sequential chain from initial_state (k=3, 3 shifts of 3 = 9 bits clears prefix)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 2) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++) {
                cur_state = (cur_state >> 3) | (((int)outputs[t]) << 6);
            }
        } else {
            cur_state = ((int)outputs[sid - 2] & 0x7)
                      | (((int)outputs[sid - 1] & 0x7) << 3)
                      | (((int)outputs[sid]     & 0x7) << 6);
        }
        float c = d_turboq3_tcq_codebook[cur_state];
        my_recon_sq = c * c;
    }
    // Tree-reduce recon_norm_sq via cost[], warp-shuffle the final 5 strides
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 256; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  8, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  4, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  2, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  1, WARP_SIZE);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Parallel bitpack: qs stores 6 initial-state bits followed by 128 3-bit
    // output symbols. Each byte is independent, so avoid the old serial OR loop.
    if (sid < 49) {
        const int init_bits = (shared_initial_state >> 3) & 0x3F;
        uint8_t packed = 0;
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            const int pos = sid * 8 + bit;
            int v = 0;
            if (pos < 6) {
                v = (init_bits >> pos) & 1;
            } else {
                const int sym_bit_pos = pos - 6;
                const int sym_idx     = sym_bit_pos / 3;
                if (sym_idx < 128) {
                    v = (outputs[sym_idx] >> (sym_bit_pos % 3)) & 1;
                }
            }
            packed |= (uint8_t)(v << bit);
        }
        dst_blk->qs[sid] = packed;
    }
    if (sid == 0) {
        dst_blk->norm = __float2half(corrected_norm);
    }
}

// 2-bit TCQ SET_ROWS encode: Viterbi optimal path with right-shift trellis (k=2, L=8)
// Double-buffered cost arrays + global memory backtrace (128 syncs/group, was 384)
template<typename idx_t>
static __global__ void __launch_bounds__(256, 1) k_set_rows_turboq2_tcq(
        const float * __restrict__ src0, const idx_t * __restrict__ src1,
        block_turboq2_tcq * __restrict__ dst, const int64_t ne_total_groups,
        uint8_t * __restrict__ bt_buf,
        const int use_shared_bt,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t s10, const int64_t s11, const int64_t s12,
        const int iq_is_k,
        const int64_t s1, const int64_t s2, const int64_t s3,
        const uint3 ne00_fd, const uint3 ne01_fd, const uint3 ne02_fd,
        const uint3 ne11_fd, const uint3 ne12_fd) {

    const int grp = blockIdx.x;
    if (grp >= ne_total_groups) return;
    const int sid = threadIdx.x; // 0..255 = trellis state

    // Compute source and destination pointers (all threads, used by thread 0)
    const int64_t i_base = int64_t(grp) * QK_TURBOQ2_TCQ;
    uint32_t tmp = (uint32_t)i_base; uint2 div_mod;
    div_mod = fast_div_modulo(tmp, ne00_fd); const int64_t i00 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne01_fd); const int64_t i01 = div_mod.y; tmp = div_mod.x;
    div_mod = fast_div_modulo(tmp, ne02_fd); const int64_t i02 = div_mod.y; const int64_t i03 = div_mod.x;
    const int64_t i12 = fastmodulo((uint32_t)i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t)i02, ne11_fd);
    const int64_t dst_row = *(src1 + i01*s10 + i11*s11 + i12*s12);
    const float * grp_src = src0 + i01*s01 + i02*s02 + i03*s03 + i00;
    block_turboq2_tcq * dst_blk = (block_turboq2_tcq *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3)
                               + (i00 / QK_TURBOQ2_TCQ);

    // Shared memory layout:
    // x[128]     : rotated+normalized input (also reused as scratch during reductions)
    // cost[256]  : path costs buffer A (also reused as norm-reduction scratch)
    // cost_b[256]: path costs buffer B (double-buffering eliminates 2/3 of syncs)
    // Backtrace: one predecessor byte for each of the 64 low-state groups per
    // step (compressed from the old 128x256 layout — the predecessor depends
    // only on sid's low 6 bits, never on the output bits in sid[7:6]). The
    // backtrace lives in dynamic shared memory when the device opts in, else
    // in bt_buf in global memory (still 128x64 bytes per block, byte-packed).
    extern __shared__ uint8_t bt_shared[];
    __shared__ float x[128];
    __shared__ float cost[256];
    __shared__ float cost_b[256];
    __shared__ int   warp_min_idx[8];
    __shared__ float warp_min_cost[8];
    __shared__ float pred_min_cost[64];
    __shared__ int   shared_initial_state;
    // Dedicated shared buffer for the Viterbi-backtrack output bytes. Previously
    // aliased onto x[] via (uint8_t *)x, but writing uint8_t into a float-typed
    // shared array is a strict-aliasing violation: under HIP/ROCm the compiler
    // can hoist cross-thread reads of outputs[] above the __syncthreads() that
    // follows the sid==0 backtrack write. The parallel bitpack introduced by
    // this port reads outputs[sym_idx] from sids 0..32, which would hit the
    // same hazard turboq3 hit in Phase 3a #20 (+12.7% PPL regression). Apply
    // the same fix preemptively, as forecast in commit 70b3dd57c.
    __shared__ uint8_t s_outputs[128];

    // Parallel pre-Viterbi: load (threads 0-127)
    if (sid < 128) x[sid] = grp_src[sid];
    __syncthreads();

    // Parallel norm reduction: tree-reduce x[i]^2 via cost[0..255] → cost[0]
    cost[sid] = (sid < 128) ? x[sid] * x[sid] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  8, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  4, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  2, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  1, WARP_SIZE);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float grp_norm = sqrtf(cost[0]);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    // Normalize (parallel)
    if (sid < 128) x[sid] *= inv_norm;
    __syncthreads();

    // Parallel FWHT: signs1 → 7-stage butterfly → scale + signs2.
    // The first five stages run inside each warp via __shfl_xor_sync; the
    // last two stages span warps so they fall back to shared memory.
    if (sid < 128) {
        float v = x[sid] * TURBO_WHT_SIGNS1[sid];
        const int lane = sid & 31;
        #pragma unroll
        for (int h = 1; h < 32; h <<= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFF, v, h, WARP_SIZE);
            v = (lane & h) ? (other - v) : (v + other);
        }
        x[sid] = v;
    }
    __syncthreads();
    if (sid < 64) {
        const int j = ((sid >> 5) << 6) + (sid & 31);
        float a = x[j], b = x[j + 32];
        x[j] = a + b; x[j + 32] = a - b;
    }
    __syncthreads();
    if (sid < 64) {
        float a = x[sid], b = x[sid + 64];
        x[sid] = a + b; x[sid + 64] = a - b;
    }
    __syncthreads();
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    if (sid < 128) x[sid] *= inv_sqrt_128 * TURBO_WHT_SIGNS2[sid];
    __syncthreads();

    // Post-rotation extraction (if enabled)
    // (yggdrasil: TURBO_EXTRACT data-collection elided)

    // Stash norm in cost[0] for post-Viterbi
    if (sid == 0) cost[0] = grp_norm;
    __syncthreads();

    float saved_norm = cost[0];

    // Initialize Viterbi: free initial state (all 256 states equally viable)
    // Double-buffered cost (1 sync/step, was 3); byte-packed bt in shared or global memory.
    uint8_t * bt = use_shared_bt ? bt_shared : bt_buf + (int64_t)blockIdx.x * (128 * 64);
    cost[sid] = 0.0f;
    __syncthreads();

    // Forward pass: 128 time steps, parallel across 256 states
    for (int t = 0; t < 128; t++) {
        float * cost_rd = (t & 1) ? cost_b : cost;
        float * cost_wr = (t & 1) ? cost   : cost_b;

        float xt = x[t];

        // Right-shift trellis (k=2, L=8): ns = (prev >> 2) | (out << 6). The
        // best predecessor depends only on sid's low 6 bits, so compute those
        // 64 minima once instead of repeating the same 4-way scan per output.
        if (sid < 64) {
            const int base_prev = sid << 2;
            float best = cost_rd[base_prev];
            int   best_p = 0;
            #pragma unroll
            for (int p = 1; p < 4; p++) {
                float c = cost_rd[base_prev | p];
                if (c < best) {
                    best = c;
                    best_p = p;
                }
            }
            pred_min_cost[sid] = best;
            bt[t * 64 + sid]   = (uint8_t) best_p;
        }
        __syncthreads();

        const int pred_idx = sid & 0x3F;
        float dist = xt - d_turboq2_tcq_codebook[sid];
        dist = dist * dist;

        cost_wr[sid] = pred_min_cost[pred_idx] + dist;
        __syncthreads();
    }
    // After 128 steps (even count): final costs are in cost[] (step 127 is odd → cost_wr=cost)

    // Parallel find-min: warp-level argmin over 256 costs, then reduce 8 warps
    {
        float my_cost = cost[sid];
        int   my_idx  = sid;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_xor_sync(0xFFFFFFFF, my_cost, offset, WARP_SIZE);
            int   other_idx  = __shfl_xor_sync(0xFFFFFFFF, my_idx,  offset, WARP_SIZE);
            if (other_cost < my_cost) { my_cost = other_cost; my_idx = other_idx; }
        }
        if (sid % 32 == 0) {
            warp_min_cost[sid / 32] = my_cost;
            warp_min_idx[sid / 32]  = my_idx;
        }
    }
    __syncthreads();
    // Reduce 8 warp minima via a single-warp shuffle (32 lanes) instead of a
    // serial single-thread loop. Upper 24 lanes seed FLT_MAX so they never win.
    if (sid < 32) {
        float best     = (sid < 8) ? warp_min_cost[sid] : 3.4028234663852886e38f;
        int   best_idx = (sid < 8) ? warp_min_idx[sid]  : 0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_cost = __shfl_down_sync(0xFFFFFFFF, best,     offset, WARP_SIZE);
            int   other_idx  = __shfl_down_sync(0xFFFFFFFF, best_idx, offset, WARP_SIZE);
            if (other_cost < best) { best = other_cost; best_idx = other_idx; }
        }
        if (sid == 0) {
            shared_initial_state = best_idx; // temporarily: best final state (becomes initial after backtrack)
        }
    }
    __syncthreads();

    // TCQ error dump (port of buun 764c686b0): save post-FWHT x[] before backtrack overwrites it.
    if (d_tcq_dump_max > 0 && grp < d_tcq_dump_max && sid < 128)
        d_tcq_dump_x_buf[grp * 128 + sid] = x[sid];

    // Thread 0: backtrack (inherently sequential, reads byte-packed bt from
    // shared or global memory). Writes the winning-path output bytes into
    // __shared__ s_outputs[] (type-clean; see declaration comment above).
    uint8_t * outputs = s_outputs;
    if (sid == 0) {
        int state = shared_initial_state;
        for (int t = 127; t >= 0; t--) {
            outputs[t] = (uint8_t)(state >> 6); // output = top 2 bits (k=2)
            int p = bt[t * 64 + (state & 0x3F)];
            state = ((state & 0x3F) << 2) | p; // reconstruct predecessor
        }
        shared_initial_state = state;
    }
    __syncthreads();

    // TCQ error dump: save Viterbi output symbols.
    if (d_tcq_dump_max > 0 && grp < d_tcq_dump_max && sid < 128)
        d_tcq_dump_out_buf[grp * 128 + sid] = outputs[sid];

    // Parallel recon norm: for t >= 3, state_t = out[t-3] | (out[t-2]<<2) | (out[t-1]<<4) | (out[t]<<6)
    // For t < 3: sequential chain from initial_state (k=2, 4 shifts of 2 = 8 bits clears prefix)
    float my_recon_sq = 0.0f;
    if (sid < 128) {
        int cur_state;
        if (sid < 3) {
            cur_state = shared_initial_state;
            for (int t = 0; t <= sid; t++) {
                cur_state = (cur_state >> 2) | (((int)outputs[t]) << 6);
            }
        } else {
            cur_state = ((int)outputs[sid - 3] & 0x3)
                      | (((int)outputs[sid - 2] & 0x3) << 2)
                      | (((int)outputs[sid - 1] & 0x3) << 4)
                      | (((int)outputs[sid]     & 0x3) << 6);
        }
        float c = d_turboq2_tcq_codebook[cur_state];
        my_recon_sq = c * c;
    }
    // Tree-reduce recon_norm_sq via cost[], warp-shuffle the final 5 strides
    cost[sid] = my_recon_sq;
    __syncthreads();
    for (int stride = 128; stride >= 32; stride >>= 1) {
        if (sid < stride) cost[sid] += cost[sid + stride];
        __syncthreads();
    }
    if (sid < 32) {
        float v = cost[sid];
        v += __shfl_down_sync(0xFFFFFFFF, v, 16, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  8, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  4, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  2, WARP_SIZE);
        v += __shfl_down_sync(0xFFFFFFFF, v,  1, WARP_SIZE);
        if (sid == 0) cost[0] = v;
    }
    __syncthreads();
    float recon_norm = sqrtf(cost[0]);
    float corrected_norm = (recon_norm > 1e-10f) ? saved_norm / recon_norm : saved_norm;
    corrected_norm *= iq_is_k ? d_tcq_norm_alpha : d_tcq_norm_alpha_v;

    // Parallel bitpack: qs stores 6 initial-state bits followed by 128 two-bit
    // output symbols. Each byte is independent (the 2-bit symbols never cross
    // byte boundaries after the 6-bit prefix), so 33 threads can each pack one
    // byte without atomics. Cross-thread reads of outputs[sym_idx] are safe
    // because outputs aliases the type-clean __shared__ uint8_t s_outputs[]
    // (see declaration comment).
    if (sid < 33) {
        const int init_bits = (shared_initial_state >> 2) & 0x3F;
        uint8_t packed = 0;
        #pragma unroll
        for (int bit = 0; bit < 8; bit++) {
            const int pos = sid * 8 + bit;
            int v = 0;
            if (pos < 6) {
                v = (init_bits >> pos) & 1;
            } else {
                const int sym_bit_pos = pos - 6;
                const int sym_idx = sym_bit_pos / 2;
                if (sym_idx < 128) {
                    v = (outputs[sym_idx] >> (sym_bit_pos % 2)) & 1;
                }
            }
            packed |= (uint8_t)(v << bit);
        }
        dst_blk->qs[sid] = packed;
    }
    if (sid == 0) {
        dst_blk->norm = __float2half(corrected_norm);
    }
}


template <typename idx_t>
static void set_rows_cuda_turboq3_tcq(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ3_TCQ == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t s01_f = nb01/sizeof(float);
    const int64_t s02_f = nb02/sizeof(float);
    const int64_t s03_f = nb03/sizeof(float);
    const int64_t s10_i = nb10/sizeof(idx_t);
    const int64_t s11_i = nb11/sizeof(idx_t);
    const int64_t s12_i = nb12/sizeof(idx_t);

    // Resolve TURBO_TCQ_ALPHA{,_V} env vars (one-shot; updates __constants__).
    load_tcq_norm_alpha();
    // Resolve TURBO_TCQ_DUMP_ERRORS env var (one-shot; allocates dump buffers).
    init_tcq_error_dump();

    // Detect K vs V cache by dst tensor name (llama-kv-cache convention: cache_k_l%d / cache_v_l%d).
    // TODO(yggdrasil): if upstream cache view naming changes, this detection breaks.
    const int is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;

    const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBOQ3_TCQ;
    if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        // One-shot probe: on CUDA, opt in to shared-memory backtrace if the device
        // exposes enough opt-in shared memory per block (and the env knob allows).
        // HIP/MUSA paths skip the probe and always use the global bt_buf branch.
        static int  tcq3_use_shared_bt = 0;
        static bool tcq3_bt_checked    = false;
        constexpr int tcq3_bt_shared_bytes = 128 * 64;
        if (!tcq3_bt_checked) {
            tcq3_bt_checked = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
            if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                int max_shared_optin = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                if (max_shared_optin >= tcq3_bt_shared_bytes) {
                    CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turboq3_tcq<idx_t>, tcq3_bt_shared_bytes);
                    tcq3_use_shared_bt = 1;
                    fprintf(stderr, "TCQ encode: using shared-memory backtrace (%d bytes/block)\n", tcq3_bt_shared_bytes);
                } else {
                    fprintf(stderr, "TCQ encode: shared-memory backtrace unavailable, only %d bytes/block are available\n", max_shared_optin);
                }
            }
#endif
        }
        if (!tcq3_use_shared_bt) {
            ensure_tcq_bt_buf(ne_total_groups * 128 * 64);
        }
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
        const int shared_bytes = tcq3_use_shared_bt ? tcq3_bt_shared_bytes : 0;
        k_set_rows_turboq3_tcq<idx_t><<<(int)ne_total_groups, 512, shared_bytes, stream>>>(
            src0_d, src1_d, (block_turboq3_tcq *)dst->data,
            ne_total_groups, tcq_bt_buf, tcq3_use_shared_bt,
            ne00, ne01, ne02, ne10, ne11, ne12, ne13,
            s01_f, s02_f, s03_f, s10_i, s11_i, s12_i,
            is_k,
            nb1, nb2, nb3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename idx_t>
static void set_rows_cuda_turboq2_tcq(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_TURBOQ2_TCQ == 0);

    cudaStream_t stream = ctx.stream();

    const int64_t s01_f = nb01/sizeof(float);
    const int64_t s02_f = nb02/sizeof(float);
    const int64_t s03_f = nb03/sizeof(float);
    const int64_t s10_i = nb10/sizeof(idx_t);
    const int64_t s11_i = nb11/sizeof(idx_t);
    const int64_t s12_i = nb12/sizeof(idx_t);

    // Resolve TURBO_TCQ_ALPHA{,_V} env vars (one-shot; updates __constants__).
    load_tcq_norm_alpha();
    // Resolve TURBO_TCQ_DUMP_ERRORS env var (one-shot; allocates dump buffers).
    init_tcq_error_dump();

    // Detect K vs V cache by dst tensor name (llama-kv-cache convention: cache_k_l%d / cache_v_l%d).
    // TODO(yggdrasil): if upstream cache view naming changes, this detection breaks.
    const int iq_is_k = (strncmp(dst->name, "cache_k_", 8) == 0) ? 1 : 0;

    const int64_t ne_total_groups = (ne00 * ne01 * ne02 * ne03) / QK_TURBOQ2_TCQ;
    if (ne_total_groups > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        // One-shot probe: on CUDA, opt in to shared-memory backtrace if the device
        // exposes enough opt-in shared memory per block (and the env knob allows).
        // HIP/MUSA paths skip the probe and always use the global bt_buf branch.
        static int  tcq2_use_shared_bt = 0;
        static bool tcq2_bt_checked    = false;
        constexpr int tcq2_bt_shared_bytes = 128 * 64;
        if (!tcq2_bt_checked) {
            tcq2_bt_checked = true;
#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA)
            const char * tcq_shared_bt_env = getenv("TURBO_TCQ_SHARED_BT");
            if (!tcq_shared_bt_env || atoi(tcq_shared_bt_env) != 0) {
                int max_shared_optin = 0;
                CUDA_CHECK(cudaDeviceGetAttribute(&max_shared_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, ctx.device));
                if (max_shared_optin >= tcq2_bt_shared_bytes) {
                    CUDA_SET_SHARED_MEMORY_LIMIT(k_set_rows_turboq2_tcq<idx_t>, tcq2_bt_shared_bytes);
                    tcq2_use_shared_bt = 1;
                    fprintf(stderr, "TCQ2 encode: using shared-memory backtrace (%d bytes/block)\n", tcq2_bt_shared_bytes);
                } else {
                    fprintf(stderr, "TCQ2 encode: shared-memory backtrace unavailable, only %d bytes/block are available\n", max_shared_optin);
                }
            }
#endif
        }
        if (!tcq2_use_shared_bt) {
            ensure_tcq_bt_buf(ne_total_groups * 128 * 64);
        }
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);
        const int shared_bytes = tcq2_use_shared_bt ? tcq2_bt_shared_bytes : 0;
        k_set_rows_turboq2_tcq<idx_t><<<(int)ne_total_groups, 256, shared_bytes, stream>>>(
            src0_d, src1_d, (block_turboq2_tcq *)dst->data,
            ne_total_groups, tcq_bt_buf, tcq2_use_shared_bt,
            ne00, ne01, ne02, ne10, ne11, ne12, ne13,
            s01_f, s02_f, s03_f, s10_i, s11_i, s12_i,
            iq_is_k,
            nb1, nb2, nb3,
            ne00_fd, ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}


// OSCAR_WHT_FULL_DIM: full head dimension for WHT (D=256 for OScaR target models with head_dim=256).
// ne00 is the combined GQA dim = head_dim * n_kv_heads. We process groups of OSCAR_WHT_FULL_DIM
// elements (= one head for models with head_dim=256). For head_dim=128 models where ne00%256≠0,
// we fall back to QK_OSCAR_INT2=128-pt WHT (full-dim for D=128, identical to Phase 1 behavior).
// §-FLAG: models with head_dim=128 AND n_kv_heads such that ne00%256==0 would use 256-pt WHT
// incorrectly (cross-head boundary); requires a head_dim parameter for full correctness (deferred).
constexpr int OSCAR_WHT_FULL_DIM = 2 * QK_OSCAR_INT2; // 256

// k_set_rows_oscar_int2: OScaR 2-bit KV encode: full-dim FHT + per-subblock min-max uniform INT2.
// blockDim.x (= wht_group) threads per block; one block per WHT group within the combined GQA row.
// For wht_group=256 (head_dim=256): single 256-pt WHT across one head's elements, 2 sub-blocks.
// For wht_group=128 (fallback): 128-pt WHT per sub-block (identical to Phase 1 for head_dim=128).
// Parseval check: H_D is normalized (1/sqrt(D) scale) so H_D^T * H_D = I (orthonormal).
template<typename idx_t>
static __global__ void k_set_rows_oscar_int2(
        const float * __restrict__ src0,
        const idx_t * __restrict__ src1,
        block_kv_oscar_int2 * __restrict__ dst,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne10,
        const int64_t ne11,
        const int64_t ne12,
        const int64_t ne13,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t s10,
        const int64_t s11,
        const int64_t s12,
        const int64_t s1,
        const int64_t s2,
        const int64_t s3,
        const bool apply_wht) {

    const int j = threadIdx.x; // 0..wht_group-1
    const int D = (int)blockDim.x; // = wht_group (256 or 128)

    // Decode blockIdx.x → (i_group within row, i01, i02, i03)
    const int64_t n_groups_per_row = ne00 / D;
    const int64_t g       = blockIdx.x;
    const int64_t i_group = g % n_groups_per_row;
    int64_t       tmp     = g / n_groups_per_row;
    const int64_t i01     = tmp % ne01;
    tmp                   = tmp / ne01;
    const int64_t i02     = tmp % ne12;
    const int64_t i03     = tmp / ne12;

    const int64_t i12 = i02;
    const int64_t i11 = i01 % ne11;
    const int64_t i10 = i01;

    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    const float * src_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_kv_oscar_int2 * dst_row_ptr = (block_kv_oscar_int2 *)((char *)dst + dst_row*s1 + i02*s2 + i03*s3);

    // Step 1: Load D elements for this group into shared memory (sized for max D=256)
    __shared__ float x[256];
    x[j] = src_row[i_group * D + j]; // i_group * D = offset into combined GQA row
    __syncthreads();

    // Step 2: Full-dim D-pt WHT (no sign matrices — OScaR uses standard Hadamard).
    // For D=128: 7 stages (h=1..64). For D=256: 8 stages (h=1..128).
    // Each butterfly: thread (j/h)%2==0 mixes x[j] and x[j+h].
    // V cache (apply_wht=false) is plain INT2 — skip WHT+normalize entirely; K cache
    // (apply_wht=true) is WHT-rotated. Mirrors the CPU/Vulkan op_params[0] gate
    // (CPU fix c52607674c: gate WHT on op_param flag, not destination type). apply_wht is
    // block-uniform (all threads share blockDim/op_params), so the inner __syncthreads()
    // are reached uniformly; the trailing barrier stays unconditional.
    if (apply_wht) {
        for (int h = 1; h < D; h <<= 1) {
            if ((j / h) % 2 == 0) {
                float a = x[j], b = x[j + h];
                x[j]     = a + b;
                x[j + h] = a - b;
            }
            __syncthreads();
        }

        // Normalize: 1/sqrt(D) so H_D is orthonormal (H_D^T * H_D = I).
        x[j] *= rsqrtf((float)D);
    }
    __syncthreads();

    // Steps 3-6: Per-sub-block min/max reduction and quantize.
    // Sub-block ib covers elements [ib*QK_OSCAR_INT2 .. (ib+1)*QK_OSCAR_INT2).
    const int ib = j / QK_OSCAR_INT2; // sub-block index (0 for D=128, 0 or 1 for D=256)
    const int jb = j % QK_OSCAR_INT2; // position within sub-block (0..127)
    const int n_warps_per_sb = QK_OSCAR_INT2 / WARP_SIZE; // warps per sub-block

    // warp_min/max indexed by global warp id (j / WARP_SIZE); sized for max 8 warps (D=256, WARP_SIZE=32)
    __shared__ float warp_min[256 / WARP_SIZE];
    __shared__ float warp_max[256 / WARP_SIZE];
    __shared__ float s_min[2]; // per-sub-block global min/max
    __shared__ float s_max[2];

    float v = x[j];
    float vmin = v, vmax = v;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, WARP_SIZE));
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, WARP_SIZE));
    }
    if (j % WARP_SIZE == 0) {
        warp_min[j / WARP_SIZE] = vmin;
        warp_max[j / WARP_SIZE] = vmax;
    }
    __syncthreads();

    // First thread in each sub-block reduces across that sub-block's warps
    if (jb == 0) {
        const int w_start = ib * n_warps_per_sb;
        float gmin = warp_min[w_start], gmax = warp_max[w_start];
        for (int w = 1; w < n_warps_per_sb; w++) {
            gmin = fminf(gmin, warp_min[w_start + w]);
            gmax = fmaxf(gmax, warp_max[w_start + w]);
        }
        s_min[ib] = gmin;
        s_max[ib] = gmax;
    }
    __syncthreads();

    const float bmin  = s_min[ib];
    const float range = s_max[ib] - bmin;
    const float bd    = (range > 1e-10f) ? range / 3.0f : 1.0f;
    const float inv_d = 1.0f / bd;

    // Step 4: Quantize element j to 2-bit
    const int q = min(3, max(0, (int)(__float2int_rn((x[j] - bmin) * inv_d))));

    // Step 5: Pack 4 elements per byte using warp shuffle (within sub-block, lane = jb % WARP_SIZE)
    // Output block index: i_group * (D/QK_OSCAR_INT2) + ib (stride by blocks-per-group per head)
    const int blk_base = (int)i_group * (D / QK_OSCAR_INT2); // base block offset for this group
    const uint8_t my_q = (uint8_t)(q & 0x3);
    const int lane = jb % WARP_SIZE;
    const uint8_t q1 = __shfl_sync(0xffffffff, my_q, lane ^ 1, WARP_SIZE);
    const uint8_t q2 = __shfl_sync(0xffffffff, my_q, lane ^ 2, WARP_SIZE);
    const uint8_t q3 = __shfl_sync(0xffffffff, my_q, lane ^ 3, WARP_SIZE);
    if (jb % 4 == 0) {
        dst_row_ptr[blk_base + ib].qs[jb / 4] = my_q | (q1 << 2) | (q2 << 4) | (q3 << 6);
    }

    // Step 6: Write per-sub-block d and m (first thread in each sub-block)
    if (jb == 0) {
        dst_row_ptr[blk_base + ib].d = __float2half(bd);
        dst_row_ptr[blk_base + ib].m = __float2half(bmin);
    }

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne13);
    GGML_UNUSED(v);
}

template<typename idx_t>
static void set_rows_cuda_oscar_int2(
        ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0,
        const ggml_tensor * src1,
        ggml_tensor * dst) {

    const float * src0_d = (const float *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS
    GGML_ASSERT(ne00 % QK_OSCAR_INT2 == 0); // each QK_OSCAR_INT2 elements produce one quantization block

    cudaStream_t stream = ctx.stream();

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);

    if (ne01 > 0) {
        // wht_group: full-dim WHT size per head (256 if ne00 divisible by OSCAR_WHT_FULL_DIM,
        // else QK_OSCAR_INT2=128 for D=128 head-dim models).
        // Shared memory is sized for OSCAR_WHT_FULL_DIM (max 256), so wht_group <= 256.
        const int64_t wht_group = (ne00 % OSCAR_WHT_FULL_DIM == 0) ? OSCAR_WHT_FULL_DIM : QK_OSCAR_INT2;
        const int64_t n_groups  = ne00 / wht_group; // groups per row (= n_kv_heads for head_dim=256)
        const int64_t ne_total  = n_groups * ne01 * ne02 * ne03;
        k_set_rows_oscar_int2<idx_t><<<(int)ne_total, (int)wht_group, 0, stream>>>(
            src0_d, src1_d, (block_kv_oscar_int2 *)dst->data,
            ne00, ne01, ne10, ne11, ne12, ne13,
            s01, s02, s03, s10, s11, s12,
            nb1, nb2, nb3,
            true);
    }
}


template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_TURBOQ2_0) {
        set_rows_cuda_turboq2<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ3_0) {
        set_rows_cuda_turboq3<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ4_0) {
        set_rows_cuda_turboq4<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ8_0) {
        set_rows_cuda_turboq8<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ5_0) {
        set_rows_cuda_turboq5<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ6_0) {
        set_rows_cuda_turboq6<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ3_TCQ) {
        set_rows_cuda_turboq3_tcq<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_TURBOQ2_TCQ) {
        set_rows_cuda_turboq2_tcq<idx_t>(ctx, src0, src1, dst);
    } else if (dst->type == GGML_TYPE_KV_OSCAR_INT2) {
        set_rows_cuda_oscar_int2<idx_t>(ctx, src0, src1, dst);
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}


void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
    }
}
