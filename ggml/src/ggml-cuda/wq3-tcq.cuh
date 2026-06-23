#pragma once
#include "common.cuh"

// WQ3_TCQ weight quantization: 3-bit trellis-coded quantization with FWHT rotation.
// Weights are stored in the FWHT-rotated domain; dequant applies inverse FWHT to
// recover original-domain values before matmul.

// Pre-quantized int8 codebook + global scale for the MMQ path.
// Defined in template-instances/mmq-instance-wq3_tcq.cu (same TU as
// load_tiles_wq3_tcq). Populated via ggml_cuda_wq3_tcq_mmq_set_codebook_i8,
// which is called from ggml_cuda_set_wq3_tcq_codebook in wq3-tcq.cu.
void ggml_cuda_wq3_tcq_mmq_set_codebook_i8(const int8_t * h_codebook_i8, float scale);

// Initialize codebook + FWHT signs. Must be called before any WQ3_TCQ operation.
void ggml_cuda_set_wq3_tcq_codebook(const float * codebook, int n_entries);
void ggml_cuda_set_wq3_tcq_signs(uint32_t sign_seed);
void ggml_cuda_set_wq3_tcq_signs_direct(const float * s1, const float * s2, int n);

// Dequant launchers (to_fp16_cuda_t / to_fp32_cuda_t compatible).
void dequantize_wq3_tcq_to_fp16(const void * vx, half * y, int64_t k, cudaStream_t stream);
void dequantize_wq3_tcq_to_fp32(const void * vx, float * y, int64_t k, cudaStream_t stream);

// Fused FWHT-rotate + q8_1_mmq quantize (replaces quantize_mmq_q8_1<D4> for
// WQ3_TCQ src0). Output layout matches MMQ expectations: one block_q8_1_mmq
// at ib = group_idx * n_tokens + tok_idx (outer = feature group, inner = token).
//   xrot_q8_1 sized n_tokens * (ncols/128) * sizeof(block_q8_1_mmq)
//   x_stride_tok is per-token float stride of the input activations.
void ggml_cuda_wq3_tcq_rotate_quantize_q8_1_mmq(
    const float * x,
    void        * xrot_q8_1,
    int ncols,
    int n_tokens,
    int64_t x_stride_tok,
    cudaStream_t stream);

// ── Native decode path (batch=1 mmvq) ────────────────────────────────────────
// From-scratch WQ3_TCQ decode kernel set. No q8_1 activation quantization, no
// MMVQ template scaffolding — a pair of fused FWHT-rotate + trellis-decode
// GEMV kernels specialized for the 52-byte block_turboq3_tcq layout.

// Rotate one fp32 activation vector of length N via the same pipeline the
// weights are stored under:  xrot[i] = signs2[i] · FWHT(signs1·x)[i] / √128.
// N must be a multiple of 128.
void ggml_cuda_wq3_tcq_rotate_activation_fp32(
    const float * x,
    float * xrot,
    int N,
    cudaStream_t stream);

struct ggml_cuda_wq3_tcq_decoded_cache;

// Native decode GEMV launcher. Pool-allocates xrot internally, runs rotate,
// runs trellis-decode + fp32 MAD GEMV (TILE_M register tiling; dispatch
// picks TILE_M=8 for FFN-sized nrows, TILE_M=4 otherwise).
void ggml_cuda_wq3_tcq_mmvq_native(
    ggml_cuda_pool & pool,
    const void * vx, const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache,
    const float * y, float * dst,
    int ncols, int nrows, cudaStream_t stream);

void ggml_cuda_wq3_tcq_mmvq_fused_gate_up_glu(
    ggml_cuda_pool & pool,
    const void * vx_up, const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache_up,
    const void * vx_gate, const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache_gate,
    const float * y_up, const float * y_gate, float * dst,
    int ncols, int nrows, ggml_glu_op glu_op, cudaStream_t stream);

ggml_cuda_wq3_tcq_decoded_cache * ggml_cuda_wq3_tcq_decoded_cache_try_create(
    const char * tensor_name,
    const void * vx,
    size_t nbytes,
    cudaStream_t stream);

void ggml_cuda_wq3_tcq_decoded_cache_free(ggml_cuda_wq3_tcq_decoded_cache * cache);

struct ggml_cuda_wq3_tcq_profile_scope {
    cudaEvent_t start = nullptr;
    cudaEvent_t stop  = nullptr;
    const char * tensor_name = nullptr;
    int ncols = 0;
    int nrows = 0;
    bool active = false;
};

ggml_cuda_wq3_tcq_profile_scope ggml_cuda_wq3_tcq_profile_begin(
    const char * tensor_name,
    int ncols,
    int nrows,
    cudaStream_t stream);

void ggml_cuda_wq3_tcq_profile_end(
    ggml_cuda_wq3_tcq_profile_scope & scope,
    cudaStream_t stream);
