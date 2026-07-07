#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

// block_rocmfp4 and block_rocmfp4_fast structs are defined canonically in
// ggml/src/ggml-common.h (Phase 1 import — struct layout centralised to avoid
// duplicate-typedef conflicts with ggml.c / ggml-cpu.c).  We pull in
// ggml-common.h so a C TU that includes only this header still sees them.
#ifndef GGML_COMMON_DECL
#ifndef GGML_COMMON_DECL_C
#define GGML_COMMON_DECL_C
#endif
#include "../src/ggml-common.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define QK_ROCMFP4 32
#define QR_ROCMFP4 2
#define QI_ROCMFP4 (QK_ROCMFP4 / (4 * QR_ROCMFP4))
#define QS_ROCMFP4 32

GGML_API void   rocmfp4_quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_rocmfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfp4_dequantize_row_q4_0(const block_rocmfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfp4_quantize_row_q4_0_fast_ref(const float * GGML_RESTRICT x, block_rocmfp4_fast * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfp4_dequantize_row_q4_0_fast(const block_rocmfp4_fast * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void   rocmfp4_quantize_row_q4_0(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfp4_quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void   rocmfp4_quantize_row_q4_0_fast(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfp4_quantize_q4_0_fast(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API bool   rocmfp4_validate_row_data(const void * data, size_t nbytes);
GGML_API bool   rocmfp4_validate_row_data_fast(const void * data, size_t nbytes);

GGML_API void rocmfp4_vec_dot_q4_0_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);
GGML_API void rocmfp4_vec_dot_q4_0_fast_q8_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

#ifdef __cplusplus
}
#endif
