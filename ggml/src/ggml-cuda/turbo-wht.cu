#include "common.cuh"

static __device__ float turbo_sign(int i) {
    return ((((unsigned) i * 0x9E3779B9u) >> 31) & 1) ? -1.0f : 1.0f;
}

static __global__ void turbo_wht_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_total,
        const int direction) {
    extern __shared__ float smem[];

    const int64_t group_id = blockIdx.x;
    const int tid = threadIdx.x;
    const int64_t base = group_id * 128;

    if (base + tid >= n_total) {
        return;
    }

    float val = src[base + tid];
    if (direction == 0) {
        val *= turbo_sign(tid);
    }
    smem[tid] = val;
    __syncthreads();

    for (int step = 1; step < 128; step <<= 1) {
        const int partner = tid ^ step;
        const float other = smem[partner];
        __syncthreads();
        smem[tid] = (tid & step) ? (other - val) : (other + val);
        val = smem[tid];
        __syncthreads();
    }

    float out = val * (1.0f / sqrtf(128.0f));
    if (direction != 0) {
        out *= turbo_sign(tid);
    }
    dst[base + tid] = out;
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const float * src_d = (const float *) src->data;
    float * dst_d = (float *) dst->data;

    int32_t params[1];
    memcpy(params, dst->op_params, sizeof(params));
    const int direction = params[0];

    const int64_t n_total = ggml_nelements(src);
    GGML_ASSERT(n_total % 128 == 0);
    const int64_t n_groups = n_total / 128;

    turbo_wht_kernel<<<n_groups, 128, 128 * sizeof(float), ctx.stream()>>>(
            src_d, dst_d, n_total, direction);
}
