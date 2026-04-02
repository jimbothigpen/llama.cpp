#include "common.cuh"

static __device__ __constant__ float turbo_wht_s1[128] = {-1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1};
static __device__ __constant__ float turbo_wht_s2[128] = {1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1};
static __device__ __constant__ float turbo_wht_s1_v[128] = {1,-1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,-1,-1,1,-1,1,-1,-1,-1,1,-1,-1,1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,1,1,-1,-1,-1,1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,-1,1,-1,1,-1,-1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,-1,1,1,-1,-1,-1,-1,1,1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,1,1,-1};
static __device__ __constant__ float turbo_wht_s2_v[128] = {-1,1,1,-1,1,-1,-1,-1,1,-1,1,1,1,1,1,1,1,1,1,1,-1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,1,-1,-1,-1,1,-1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,1,1,1,-1,1,-1,1,1,-1,1,1,1,-1,1,1,1,1,-1,1,-1,-1,1,-1,-1,-1,-1,-1,1,-1,-1,1,1,1,-1,1,-1,-1,1,-1,1,-1,1,-1,-1,-1,-1,1};

static __global__ void turbo_wht_kernel(
        const float * __restrict__ src,
        float * __restrict__ dst,
        const int64_t n_total,
        const int direction) {
    const int64_t group_idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t n_groups = n_total / 128;
    if (group_idx >= n_groups) {
        return;
    }

    const float * in = src + group_idx * 128;
    float * out = dst + group_idx * 128;

    float x[128];
    const float * s_first = (direction == 0) ? turbo_wht_s1 : turbo_wht_s2_v;
    const float * s_second = (direction == 0) ? turbo_wht_s2 : turbo_wht_s1_v;

    for (int i = 0; i < 128; ++i) {
        x[i] = in[i] * s_first[i];
    }

    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += h * 2) {
            for (int j = i; j < i + h; ++j) {
                const float a = x[j];
                const float b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }

    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) {
        out[i] = x[i] * inv_sqrt_128 * s_second[i];
    }
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src = dst->src[0];
    GGML_ASSERT(src->type == GGML_TYPE_F32);

    const float * src_d = (const float *) src->data;
    float * dst_d = (float *) dst->data;

    int32_t direction = 0;
    memcpy(&direction, dst->op_params, sizeof(direction));

    const int64_t n_total = ggml_nelements(src);
    GGML_ASSERT(n_total % 128 == 0);
    const int64_t n_groups = n_total / 128;

    const int threads = 256;
    const int blocks = (n_groups + threads - 1) / threads;

    turbo_wht_kernel<<<blocks, threads, 0, ctx.stream()>>>(
            src_d, dst_d, n_total, direction);
}
