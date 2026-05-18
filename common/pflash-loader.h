#pragma once

#include <string>
#include <vector>
#include <cstdint>

struct ggml_context;
struct ggml_tensor;
struct ggml_backend_buffer;

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

    // per-layer weights
    struct layer {
        ggml_tensor * attn_norm;     // rms_norm weight
        ggml_tensor * wq;            // q_proj
        ggml_tensor * wk;            // k_proj
        ggml_tensor * wv;            // v_proj
        ggml_tensor * wo;            // o_proj
        ggml_tensor * q_norm;        // q_norm (Qwen3 has per-head QK norm)
        ggml_tensor * k_norm;        // k_norm
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

// Load a Qwen3-0.6B GGUF onto the specified CUDA device.
// Returns 0 on success, -1 on error.
int pflash_model_load(pflash_model & model, const std::string & gguf_path, int gpu_device = 0);

// Free all GPU buffers and host memory.
void pflash_model_free(pflash_model & model);
