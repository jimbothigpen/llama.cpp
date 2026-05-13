#pragma once

#include "common.cuh"

void ggml_cuda_mul_mat_vec_tq(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst);

// Load-time conversion: WHT4_0 → q8_0 in VRAM (dequant + requantize)
void ggml_cuda_convert_wht4_0_to_q8_0(const void * src_tq4, void * dst_q8, int64_t n_elements, cudaStream_t stream);
