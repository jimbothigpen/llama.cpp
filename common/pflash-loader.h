#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;

enum pflash_norm_type {
    PFLASH_NORM_RMS,   // standard RMSNorm (Llama, Qwen, Mistral)
    PFLASH_NORM_LAYER, // LayerNorm with (1+weight) scale (Gemma3)
};

enum pflash_ffn_act {
    PFLASH_FFN_SILU, // SwiGLU (Llama, Qwen, Mistral)
    PFLASH_FFN_GELU, // GeGLU (Gemma3)
};

struct pflash_model {
    // architecture
    int n_layers   = 0;
    int n_embd     = 0;
    int n_heads    = 0;
    int n_kv_heads = 0;
    int d_head     = 0;
    int n_ff       = 0;
    int n_vocab    = 0;
    float rope_freq_base = 0.0f;
    int   rope_type      = 0;

    // arch feature flags (set by pflash_model_load)
    bool             has_qk_norm   = false;          // per-head QK norms (Qwen3, Qwen3.5)
    bool             has_attn_bias = false;           // QKV projection biases (Qwen2, Qwen2.5)
    float            norm_eps      = 1e-6f;
    pflash_norm_type norm_type     = PFLASH_NORM_RMS;
    pflash_ffn_act   ffn_act       = PFLASH_FFN_SILU;

    // per-layer weights
    struct layer {
        ggml_tensor * attn_norm;     // rms_norm weight
        ggml_tensor * wq;            // q_proj
        ggml_tensor * wk;            // k_proj
        ggml_tensor * wv;            // v_proj
        ggml_tensor * wo;            // o_proj
        ggml_tensor * q_norm;        // q_norm (nullable; Qwen3/Qwen3.5 only)
        ggml_tensor * k_norm;        // k_norm (nullable; Qwen3/Qwen3.5 only)
        ggml_tensor * attn_q_b;      // q_proj bias (nullable; Qwen2/Qwen2.5)
        ggml_tensor * attn_k_b;      // k_proj bias (nullable; Qwen2/Qwen2.5)
        ggml_tensor * attn_v_b;      // v_proj bias (nullable; Qwen2/Qwen2.5)
        ggml_tensor * ffn_norm;      // post_attention_layernorm
        ggml_tensor * ffn_gate;      // gate_proj
        ggml_tensor * ffn_up;        // up_proj
        ggml_tensor * ffn_down;      // down_proj
    };

    std::vector<layer> layers;

    // global weights
    ggml_tensor * tok_embd  = nullptr;
    ggml_tensor * output_norm = nullptr;
    ggml_tensor * output    = nullptr; // lm_head (may alias tok_embd)

    // ggml state
    ggml_context * ctx_ggml = nullptr;
    ggml_backend_buffer * buf_gpu = nullptr;

    // mmap state
    void * mmap_addr = nullptr;
    size_t mmap_size = 0;
    int    mmap_fd   = -1;
};

// Load a scorer GGUF. Supports Qwen3, Qwen3.5, Llama, Qwen2/Qwen2.5, Mistral3/4, Gemma3.
// For hybrid SSM+attention models (qwen35), only full-attention layers are loaded.
// Returns 0 on success, -1 on error.
int pflash_model_load(pflash_model & model, const std::string & gguf_path, int gpu_device = 0);

// Free all GPU buffers and host memory.
void pflash_model_free(pflash_model & model);
