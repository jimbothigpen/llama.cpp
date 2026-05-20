#pragma once
#include "common.cuh"

// Returns true if a dedicated MMVQ kernel is implemented for src0->type.
// When false, the caller should fall through to the cuBLAS-dequant path.
bool ggml_cuda_iqk_mmvq_supported(enum ggml_type type);

// Standalone MMVQ for yggdrasil IQK base weight types.
// Implements IQ4_K (137), IQ3_K (138), IQ2_K (139).
// Fast single-token decode path; falls back to cuBLAS-dequant for multi-token.
void ggml_cuda_mul_mat_iqk_mmvq(ggml_backend_cuda_context & ctx,
                                const ggml_tensor * src0,
                                const ggml_tensor * src1,
                                ggml_tensor * dst);
