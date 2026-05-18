#pragma once
#include "common.cuh"

// Callback type for external sparse attention kernel.
// Q/K/V/O are nv_bfloat16 (BF16), contiguous [B, S, H, D] pFlash row-major.
// Returns 0 on success.
typedef int (*ggml_cuda_sparse_attn_fn_t)(
    const void * Q, const void * K, const void * V, void * O,
    int batch, int seq_len, int n_q_heads, int n_k_heads, int head_dim,
    float scale, float alpha);

// Register the external sparse attention kernel (call once at init).
void ggml_cuda_flash_attn_sparse_set_kernel(ggml_cuda_sparse_attn_fn_t fn);

void ggml_cuda_flash_attn_sparse(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
