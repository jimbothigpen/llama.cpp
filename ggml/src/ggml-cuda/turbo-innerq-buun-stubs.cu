// FA-side InnerQ host wrappers and buun-calibration no-ops.
//
// turbo_innerq_init_fattn / turbo_innerq_update_fattn_scales write to
// d_innerq_channel_scale_inv_fattn, a __device__ symbol defined in fattn.cu
// (Phase X-3). Cross-TU device-symbol access requires -fgpu-rdc / CUDA RDC.
// Until Phase X-3 enables RDC and defines the symbol, these functions are
// no-ops guarded by GGML_INNERQ_FA_DEVICE_SCALES. Phase X-3 must define that
// macro (e.g. in common.cuh) and confirm RDC is enabled in CMake before
// removing the guard.
//
// turbo_q_calibrate_init / turbo_q_calibrate_finalize are env-gated Q²
// statistics paths from buun (TURBO_Q_CALIBRATE=1). Not exposed in ft2;
// kept as no-ops here to document buun attribution.

#include "common.cuh"
#include "turbo-innerq.cuh"

#ifdef GGML_INNERQ_FA_DEVICE_SCALES
// Requires: -fgpu-rdc in CMake + d_innerq_channel_scale_inv_fattn defined
// in fattn.cu. Enable in Phase X-3.
extern __device__ float d_innerq_channel_scale_inv_fattn[128];

void turbo_innerq_init_fattn() {
    float ones[128];
    for (int i = 0; i < 128; i++) ones[i] = 1.0f;
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (int id = 0; id < device_count; id++) {
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, ones, sizeof(ones)));
    }
    CUDA_CHECK(cudaSetDevice(cur_device));
}

void turbo_innerq_update_fattn_scales(const float * scale_inv) {
    int cur_device;
    CUDA_CHECK(cudaGetDevice(&cur_device));
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    for (int id = 0; id < device_count; id++) {
        CUDA_CHECK(cudaSetDevice(id));
        CUDA_CHECK(cudaMemcpyToSymbol(d_innerq_channel_scale_inv_fattn, scale_inv, 128 * sizeof(float)));
    }
    CUDA_CHECK(cudaSetDevice(cur_device));
}
#else
// Phase X-3 stub: d_innerq_channel_scale_inv_fattn not yet defined in fattn.cu.
// FA decode uses identity scales (correct behaviour pre-finalization).
void turbo_innerq_init_fattn() {}
void turbo_innerq_update_fattn_scales(const float * scale_inv) { (void)scale_inv; }
#endif  // GGML_INNERQ_FA_DEVICE_SCALES

// buun calibration no-ops (buun's TURBO_Q_CALIBRATE path is vestigial in ft2)
void turbo_q_calibrate_init() {}
void turbo_q_calibrate_finalize() {}
