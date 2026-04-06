#pragma once

#include "common.cuh"

#include <sstream>
#include <string>
#include <mutex>
#include <unordered_set>
#include <vector>

// Prompt-fast CUDA repack for TQ3_4S:
// keep the 3-bit payload compact, but widen the per-8 scale codes into final fp16 deltas
// once at tensor upload time so prompt kernels stop re-decoding them every launch.

typedef struct {
    ggml_half d[4];
    uint8_t   qs[QK_TQ3_0 * 3 / 8];
} block_tq3_4s_rp;
static_assert(sizeof(block_tq3_4s_rp) == 4 * sizeof(ggml_half) + QK_TQ3_0 * 3 / 8, "wrong tq3_4s_rp block size/padding");

typedef struct {
    uint8_t g[4][4]; // [d_byte, qs0, qs1, qs2] for each 8-value group
} block_tq3_4s_p4;
static_assert(sizeof(block_tq3_4s_p4) == sizeof(block_tq3_4s), "wrong tq3_4s_p4 block size/padding");

static constexpr uint64_t GGML_CUDA_TQ3_4S_REPACK_MAGIC = 0x543351345350524Bull; // "TQ34SPRK"
static constexpr uint64_t GGML_CUDA_TQ3_4S_PACK4_MAGIC  = 0x5433513450344750ull; // "TQ34P4GP"

struct ggml_cuda_tq3_4s_repack_extra {
    uint64_t magic;
};

static inline bool ggml_cuda_tq3_4s_repack_enabled() {
    static bool enabled = getenv("GGML_CUDA_TQ3_4S_REPACK") != nullptr;
    return enabled;
}

static inline bool ggml_cuda_tq3_4s_pack4_enabled() {
    static bool enabled = getenv("GGML_CUDA_TQ3_4S_PACK4") != nullptr;
    return enabled;
}

static inline const std::vector<std::string> & ggml_cuda_tq3_4s_repack_match_list() {
    static const std::vector<std::string> patterns = [] {
        const char * env = getenv("GGML_CUDA_TQ3_4S_REPACK_MATCH");
        const char * raw = (env != nullptr && env[0] != '\0')
            ? env
            : "ffn_down.weight,ffn_gate.weight,ffn_up.weight,attn_q.weight,attn_k.weight,attn_v.weight,attn_output.weight";

        std::vector<std::string> out;
        std::stringstream ss(raw);
        std::string item;
        while (std::getline(ss, item, ',')) {
            auto start = item.find_first_not_of(" \t");
            if (start == std::string::npos) {
                continue;
            }
            auto end = item.find_last_not_of(" \t");
            out.emplace_back(item.substr(start, end - start + 1));
        }
        return out;
    }();
    return patterns;
}

static inline size_t ggml_cuda_tq3_4s_repack_extra_budget_bytes() {
    static const size_t budget = [] {
        const char * env = getenv("GGML_CUDA_TQ3_4S_REPACK_EXTRA_MIB");
        if (env == nullptr || env[0] == '\0') {
            return size_t(1024) * 1024 * 1024;
        }
        char * end = nullptr;
        const long long mib = strtoll(env, &end, 10);
        if (end == env || mib <= 0) {
            return size_t(1024) * 1024 * 1024;
        }
        return size_t(mib) * 1024 * 1024;
    }();
    return budget;
}

static inline bool ggml_cuda_tq3_4s_repack_candidate(const ggml_tensor * tensor) {
    return tensor != nullptr
        && tensor->type == GGML_TYPE_TQ3_4S
        && tensor->view_src == nullptr
        && ggml_is_quantized(tensor->type)
        && ggml_n_dims(tensor) >= 2;
}

static inline bool ggml_cuda_tq3_4s_repack_tensor_match(const ggml_tensor * tensor) {
    const std::string name = ggml_get_name(tensor);
    for (const auto & pattern : ggml_cuda_tq3_4s_repack_match_list()) {
        if (!pattern.empty() && name.find(pattern) != std::string::npos) {
            return true;
        }
    }
    return false;
}

static inline float ggml_cuda_tq3_4s_prompt_delta_host(uint8_t byte) {
    if (byte == 0) {
        return 0.0f;
    }
    const int exp = (byte >> 5) - 9;
    const float mantissa = 1.0f + float(byte & 31) / 32.0f;
    return ldexpf(mantissa, exp) * (2.1519f / 127.0f);
}

static inline size_t ggml_cuda_tq3_4s_repack_row_size(int64_t ne0) {
    GGML_ASSERT(ne0 % QK_TQ3_0 == 0);
    return size_t(ne0 / QK_TQ3_0) * sizeof(block_tq3_4s_rp);
}

static inline size_t ggml_cuda_tq3_4s_repack_alloc_size(const ggml_tensor * tensor) {
    const int64_t padded_ne0 = GGML_PAD(tensor->ne[0], MATRIX_ROW_PADDING);
    return ggml_cuda_tq3_4s_repack_row_size(padded_ne0) * ggml_nrows(tensor);
}

static inline size_t ggml_cuda_tq3_4s_plain_alloc_size(const ggml_tensor * tensor) {
    size_t size = ggml_nbytes(tensor);
    const int64_t ne0 = tensor->ne[0];
    if (ggml_is_quantized(tensor->type) && ne0 % MATRIX_ROW_PADDING != 0) {
        size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
    }
    return size;
}

static inline size_t ggml_cuda_tq3_4s_repack_extra_bytes(const ggml_tensor * tensor) {
    const size_t repacked = ggml_cuda_tq3_4s_repack_alloc_size(tensor);
    const size_t plain = ggml_cuda_tq3_4s_plain_alloc_size(tensor);
    return repacked > plain ? repacked - plain : 0;
}

static inline bool ggml_cuda_tq3_4s_repack_selected(const ggml_tensor * tensor) {
    static std::mutex mutex;
    static size_t repack_extra_used = 0;
    static std::unordered_set<std::string> selected;
    static std::unordered_set<std::string> rejected;
    static std::unordered_set<std::string> logged;

    if (!ggml_cuda_tq3_4s_repack_enabled() || !ggml_cuda_tq3_4s_repack_candidate(tensor) || !ggml_cuda_tq3_4s_repack_tensor_match(tensor)) {
        return false;
    }

    const std::string name = ggml_get_name(tensor);
    std::lock_guard<std::mutex> lock(mutex);

    if (selected.find(name) != selected.end()) {
        return true;
    }
    if (rejected.find(name) != rejected.end()) {
        return false;
    }

    const size_t extra_bytes = ggml_cuda_tq3_4s_repack_extra_bytes(tensor);
    const size_t extra_budget = ggml_cuda_tq3_4s_repack_extra_budget_bytes();
    if (repack_extra_used + extra_bytes > extra_budget) {
        rejected.insert(name);
        return false;
    }

    selected.insert(name);
    repack_extra_used += extra_bytes;
    if (logged.insert(name).second) {
        GGML_LOG_INFO("%s: repack enabled for %s (+%.2f MiB, total %.2f / %.2f MiB)\n",
            __func__, name.c_str(),
            extra_bytes / 1024.0 / 1024.0,
            repack_extra_used / 1024.0 / 1024.0,
            extra_budget / 1024.0 / 1024.0);
    }
    return true;
}

static inline bool ggml_cuda_tq3_4s_pack4_selected(const ggml_tensor * tensor) {
    return ggml_cuda_tq3_4s_pack4_enabled()
        && ggml_cuda_tq3_4s_repack_candidate(tensor)
        && ggml_cuda_tq3_4s_repack_tensor_match(tensor);
}
