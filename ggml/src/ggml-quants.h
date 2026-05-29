#pragma once

#define GGML_COMMON_DECL_C
#include "ggml-common.h"

#include "ggml.h"

// GGML internal header

#ifdef __cplusplus
extern "C" {
#endif

// NOTE: these functions are defined as GGML_API because they used by the CPU backend

// Quantization
GGML_API void quantize_row_q1_0_ref(const float * GGML_RESTRICT x, block_q1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q1_0_g128_ref(const float * GGML_RESTRICT x, block_q1_0_g128 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_0_ref(const float * GGML_RESTRICT x, block_q4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_1_ref(const float * GGML_RESTRICT x, block_q4_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_0_ref(const float * GGML_RESTRICT x, block_q5_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_1_ref(const float * GGML_RESTRICT x, block_q5_1 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_0_ref(const float * GGML_RESTRICT x, block_q8_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_1_ref(const float * GGML_RESTRICT x, block_q8_1 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_mxfp4_ref(const float * GGML_RESTRICT x, block_mxfp4 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_nvfp4_ref(const float * GGML_RESTRICT x, block_nvfp4 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_q2_K_ref(const float * GGML_RESTRICT x, block_q2_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q3_K_ref(const float * GGML_RESTRICT x, block_q3_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q4_K_ref(const float * GGML_RESTRICT x, block_q4_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q5_K_ref(const float * GGML_RESTRICT x, block_q5_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q6_K_ref(const float * GGML_RESTRICT x, block_q6_K * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_q8_K_ref(const float * GGML_RESTRICT x, block_q8_K * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_tq1_0_ref(const float * GGML_RESTRICT x, block_tq1_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_tq2_0_ref(const float * GGML_RESTRICT x, block_tq2_0 * GGML_RESTRICT y, int64_t k);

GGML_API void quantize_row_iq3_xxs_ref(const float * GGML_RESTRICT x, block_iq3_xxs * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_nl_ref (const float * GGML_RESTRICT x, block_iq4_nl  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_xs_ref (const float * GGML_RESTRICT x, block_iq4_xs  * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_s_ref  (const float * GGML_RESTRICT x, block_iq3_s   * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_s_ref  (const float * GGML_RESTRICT x, block_iq2_s   * GGML_RESTRICT y, int64_t k);

// Dequantization
GGML_API void dequantize_row_q1_0(const block_q1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q1_0_g128(const block_q1_0_g128 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_0(const block_q4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_1(const block_q4_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_0(const block_q5_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_1(const block_q5_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
//GGML_API void dequantize_row_q8_1(const block_q8_1 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_mxfp4(const block_mxfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_nvfp4(const block_nvfp4 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q3_K(const block_q3_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q4_K(const block_q4_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q5_K(const block_q5_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q6_K(const block_q6_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_q8_K(const block_q8_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_tq1_0(const block_tq1_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_tq2_0(const block_tq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

GGML_API void dequantize_row_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_xs (const block_iq2_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_s  (const block_iq2_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_xxs(const block_iq3_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_s  (const block_iq1_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq1_m  (const block_iq1_m   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_nl (const block_iq4_nl  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_xs (const block_iq4_xs  * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_s  (const block_iq3_s   * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);

// Quantization utilizing an importance matrix (a.k.a. "Activation aWare Quantization")
GGML_API size_t quantize_iq2_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq2_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_xxs(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq1_m  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_nl (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq4_xs (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_iq3_s  (const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_tq1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_tq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_q2_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q3_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q6_K(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q1_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q1_0_g128(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q4_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q5_1(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_q8_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API size_t quantize_mxfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_nvfp4(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// TurboQuant KV cache compression (arXiv 2504.19874)
GGML_API void quantize_row_turboq2_0_ref(const float * GGML_RESTRICT x, block_turboq2_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_turboq3_0_ref(const float * GGML_RESTRICT x, block_turboq3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_turboq4_0_ref(const float * GGML_RESTRICT x, block_turboq4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turboq2_0(const block_turboq2_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turboq3_0(const block_turboq3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turboq4_0(const block_turboq4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_turboq2_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_turboq3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API size_t quantize_turboq4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// TurboQuant TCQ KV cache (Trellis-Coded Quantization) — source: buun master TURBO[23]_TCQ
GGML_API void quantize_row_turboq3_tcq_ref(const float * GGML_RESTRICT x, block_turboq3_tcq * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turboq3_tcq(const block_turboq3_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_turboq3_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void quantize_row_turboq2_tcq_ref(const float * GGML_RESTRICT x, block_turboq2_tcq * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_turboq2_tcq(const block_turboq2_tcq * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_turboq2_tcq(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// OScaR KV INT2: FHT + per-block min-max uniform 2-bit quantization — Phase 1 CUDA prototype (arXiv:2605.19660)
GGML_API void quantize_row_kv_oscar_int2_ref(const float * GGML_RESTRICT x, block_kv_oscar_int2 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_kv_oscar_int2(const block_kv_oscar_int2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_kv_oscar_int2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// WHT3_0: WHT-rotated 3-bit weight quantization (8-level Lloyd-Max)
GGML_API void quantize_row_wht3_0_ref(const float * GGML_RESTRICT x, block_wht3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_wht3_0(const block_wht3_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_wht3_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// WHT4_0: WHT-rotated 4-bit weight quantization (16-level Lloyd-Max)
GGML_API void quantize_row_wht4_0_ref(const float * GGML_RESTRICT x, block_wht4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_wht4_0(const block_wht4_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_wht4_0(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

// IQ4_K: ik_llama 4-bit imatrix-aware weight quant (4.50 bpw) — source: ik_llama
GGML_API void quantize_row_iq4_k_ref(const float * GGML_RESTRICT x, block_iq4_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_k    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_k  (const block_iq4_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq4_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// IQ3_K: ik_llama 3-bit imatrix-aware weight quant (3.4375 bpw) — source: ik_llama
GGML_API void quantize_row_iq3_k_ref(const float * GGML_RESTRICT x, block_iq3_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_k    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_k  (const block_iq3_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq3_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// IQ2_K: ik_llama 2-bit imatrix-aware weight quant (2.375 bpw) — source: ik_llama
GGML_API void quantize_row_iq2_k_ref(const float * GGML_RESTRICT x, block_iq2_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_k    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_k  (const block_iq2_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq2_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// IQ5_K: ik_llama 5-bit imatrix-aware weight quant (5.50 bpw) — source: ikllama/main
GGML_API void quantize_row_iq5_k_ref(const float * GGML_RESTRICT x, block_iq5_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq5_k    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq5_k  (const block_iq5_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq5_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq5_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// IQ6_K: ik_llama 6-bit imatrix-aware weight quant (6.625 bpw) — source: ikllama/ik/iq6_k
GGML_API void quantize_row_iq6_k_ref(const float * GGML_RESTRICT x, block_iq6_k * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq6_k    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq6_k  (const block_iq6_k * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq6_k(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq6_k_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// Phase 5b-1b: row-meta KS family weight quants
GGML_API void quantize_row_iq4_ks_ref(const float * GGML_RESTRICT x, block_iq4_ks * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_ks    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_ks  (const block_iq4_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq4_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void quantize_row_iq3_ks_ref(const float * GGML_RESTRICT x, block_iq3_ks * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq3_ks    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq3_ks  (const block_iq3_ks * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq3_ks(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq3_ks_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void quantize_row_iq4_kss_ref(const float * GGML_RESTRICT x, block_iq4_kss * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_kss    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_kss  (const block_iq4_kss * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_kss(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq4_kss_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void quantize_row_iq4_kt_ref(const float * GGML_RESTRICT x, block_iq4_kt * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq4_kt    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq4_kt  (const block_iq4_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq4_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq4_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void quantize_row_iq2_kt_ref(const float * GGML_RESTRICT x, block_iq2_kt * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_kt    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_kt  (const block_iq2_kt * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_kt(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq2_kt_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void quantize_row_iq2_kl_ref(const float * GGML_RESTRICT x, block_iq2_kl * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iq2_kl    (const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iq2_kl  (const block_iq2_kl * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iq2_kl(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_iq2_kl_q8_K(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// RotorQuant KV cache — PlanarQuant 3-bit (Givens-rotation, 8 centroids, sign-mag split)
GGML_API void quantize_row_planar3_0_ref(const float * GGML_RESTRICT x, block_planar3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_planar3_0    (const float * GGML_RESTRICT x, void            * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_planar3_0  (const void  * GGML_RESTRICT x, float           * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_planar3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_rq_planar3_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// RotorQuant KV cache — PlanarQuant 4-bit (Givens-rotation, 16 centroids, nibble-packed)
GGML_API void quantize_row_planar4_0_ref(const float * GGML_RESTRICT x, block_planar4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_planar4_0    (const float * GGML_RESTRICT x, void            * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_planar4_0  (const void  * GGML_RESTRICT x, float           * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_planar4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_rq_planar4_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// RotorQuant KV cache — IsoQuant 3-bit (Hadamard/isometric rotation, 8 centroids, sign-mag split)
GGML_API void quantize_row_iso3_0_ref(const float * GGML_RESTRICT x, block_iso3_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iso3_0    (const float * GGML_RESTRICT x, void         * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iso3_0  (const void  * GGML_RESTRICT x, float        * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iso3_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_rq_iso3_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

// RotorQuant KV cache — IsoQuant 4-bit (Hadamard/isometric rotation, 16 centroids, nibble-packed)
GGML_API void quantize_row_iso4_0_ref(const float * GGML_RESTRICT x, block_iso4_0 * GGML_RESTRICT y, int64_t k);
GGML_API void quantize_row_iso4_0    (const float * GGML_RESTRICT x, void         * GGML_RESTRICT y, int64_t k);
GGML_API void dequantize_row_iso4_0  (const void  * GGML_RESTRICT x, float        * GGML_RESTRICT y, int64_t k);
GGML_API size_t quantize_iso4_0(const float * src, void * dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
GGML_API void ggml_vec_dot_rq_iso4_0(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, size_t bx, const void * GGML_RESTRICT vy, size_t by, int nrc);

GGML_API void iq2xs_init_impl(enum ggml_type type);
GGML_API void iq2xs_free_impl(enum ggml_type type);
GGML_API void iq3xs_init_impl(int grid_size);
GGML_API void iq3xs_free_impl(int grid_size);

#ifdef __cplusplus
}
#endif
