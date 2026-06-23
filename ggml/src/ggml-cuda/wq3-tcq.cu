// WQ3_TCQ weight dequantization and native matmul kernels.
//
// Weight tensors are stored in the FWHT-rotated domain (same block layout as
// turbo3_tcq KV cache: 52 bytes per 128 elements). Dequant decodes the trellis
// bitstream, looks up a 1024-entry codebook, then applies the inverse FWHT
// rotation (signs2 → butterfly → signs1 → normalize) to recover original-domain
// weight values.
//
// Two kernel paths:
//   1. Dequant-to-fp16/fp32: for cuBLAS fallback (prefill, batch > 1)
//   2. Native mmvq: fused dequant + dot product (decode, batch = 1)

#include "wq3-tcq.cuh"
#include "common.cuh"
#include "mmq.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <strings.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ── Constant memory ──────────────────────────────────────────────────────────

static __constant__ float d_wq3_tcq_codebook[1024];
static bool d_wq3_tcq_codebook_loaded = false;

// Pre-quantized int8 codebook for the MMQ path lives in mmq-instance-wq3_tcq.cu
// (the TU where load_tiles_wq3_tcq is instantiated). This TU pushes to it via
// ggml_cuda_wq3_tcq_mmq_set_codebook_i8 declared in wq3-tcq.cuh. Kept separate
// because without -rdc=true cross-TU __constant__ references don't link.
//
// The native decode path (k_wq3_tcq_mmvq_v2 in this TU) needs the same
// int8 codebook. We keep a TU-local copy here and populate both in lockstep
// from ggml_cuda_set_wq3_tcq_codebook — single call site, no drift risk.
// Cost: 1 KB duplicated constant memory; budget is 64 KB on sm_86.
alignas(16) static __constant__ int8_t s_codebook_i8_native[1024];
static __constant__ float  s_codebook_scale_native;
static __constant__ float  s_proc_qscale_native;

static void set_codebook_native(const int8_t * h_codebook_i8, float scale) {
    CUDA_CHECK(cudaMemcpyToSymbol(s_codebook_i8_native, h_codebook_i8, 1024 * sizeof(int8_t)));
    CUDA_CHECK(cudaMemcpyToSymbol(s_codebook_scale_native, &scale, sizeof(float)));
}

// FWHT rotation signs. Generated from seeds via torch.Generator on the training
// host. Default: seed1=42 (signs1), seed2=seed1+1042=1084 (signs2).
// These are uploaded at model load time by ggml_cuda_set_wq3_tcq_signs().
static __constant__ float d_wq3_tcq_signs1[128];
static __constant__ float d_wq3_tcq_signs2[128];
static bool d_wq3_tcq_signs_loaded = false;

namespace {

enum wq3_tcq_codebook_mode {
    WQ3_TCQ_CODEBOOK_LUT = 0,
    WQ3_TCQ_CODEBOOK_PROC_MURMUR = 1,
};

static wq3_tcq_codebook_mode g_wq3_tcq_codebook_mode = WQ3_TCQ_CODEBOOK_LUT;

static constexpr uint32_t WQ3_TCQ_PROC_MURMUR_MUL  = 0xCBAC1FEDu;
static constexpr uint32_t WQ3_TCQ_PROC_MURMUR_ADD  = 0x6789ABCDu;
static constexpr uint32_t WQ3_TCQ_PROC_MURMUR_MIX1 = 0x85EBCA6Bu;
static constexpr uint32_t WQ3_TCQ_PROC_MURMUR_MIX2 = 0xC2B2AE35u;
static constexpr float    WQ3_TCQ_PROC_TARGET_RMS  = 0.08838834764831845f; // 1 / sqrt(128)

static inline double wq3_tcq_norm_ppf_host(double p) {
    static constexpr double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00,
    };
    static constexpr double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01,
    };
    static constexpr double c[] = {
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00,  2.938163982698783e+00,
    };
    static constexpr double d[] = {
         7.784695709041462e-03,  3.224671290700398e-01,
         2.445134137142996e+00,  3.754408661907416e+00,
    };

    const double plow = 0.02425;
    const double phigh = 1.0 - plow;

    if (p < plow) {
        const double q = sqrt(-2.0 * log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
    if (p > phigh) {
        const double q = sqrt(-2.0 * log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }

    const double q = p - 0.5;
    const double r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
}

static inline float wq3_tcq_generate_murmur_gauss_raw_host(uint32_t state) {
    uint32_t x = state * WQ3_TCQ_PROC_MURMUR_MUL + WQ3_TCQ_PROC_MURMUR_ADD;
    x ^= x >> 16;
    x *= WQ3_TCQ_PROC_MURMUR_MIX1;
    x ^= x >> 13;
    x *= WQ3_TCQ_PROC_MURMUR_MIX2;
    x ^= x >> 16;
    const double u = ((double)x + 0.5) * (1.0 / 4294967296.0);
    return (float)wq3_tcq_norm_ppf_host(u);
}

static bool wq3_tcq_detect_murmur_gauss(
        const float * codebook,
        int n_entries,
        const int8_t * h_codebook_i8,
        float s_cb,
        float * proc_qscale_out) {
    if (n_entries != 1024) {
        return false;
    }

    double sumsq = 0.0;
    for (int i = 0; i < 1024; ++i) {
        const double raw = (double)wq3_tcq_generate_murmur_gauss_raw_host((uint32_t)i);
        sumsq += raw * raw;
    }
    const float raw_rms = (float)sqrt(sumsq / 1024.0);
    if (!(raw_rms > 0.0f)) {
        return false;
    }

    const float proc_norm = WQ3_TCQ_PROC_TARGET_RMS / raw_rms;
    const float proc_qscale = proc_norm / s_cb;

    int mismatches = 0;
    double mse = 0.0;
    for (int i = 0; i < 1024; ++i) {
        const float raw = wq3_tcq_generate_murmur_gauss_raw_host((uint32_t)i);
        int q = (int)lrintf(raw * proc_qscale);
        if (q > 127)  q = 127;
        if (q < -127) q = -127;
        if ((int8_t)q != h_codebook_i8[i]) {
            ++mismatches;
        }
        const float want = raw * proc_norm;
        const float diff = want - codebook[i];
        mse += (double)diff * (double)diff;
    }

    const double rmse = sqrt(mse / 1024.0);
    if (mismatches == 0 || (mismatches <= 4 && rmse < 1e-4)) {
        *proc_qscale_out = proc_qscale;
        return true;
    }
    return false;
}

static bool wq3_tcq_use_proc_u16_layout() {
    static const bool use_u16 = []() {
        const char * env = getenv("GGML_CUDA_WQ3_PROC_LAYOUT");
        if (env == nullptr || env[0] == '\0') {
            return true;
        }
        return strcmp(env, "u16") == 0 || strcmp(env, "v2") == 0 || strcmp(env, "aligned") == 0;
    }();
    return use_u16;
}

static int wq3_tcq_cached_block_rows_override() {
    static const int block_rows = []() {
        const char * env = getenv("GGML_CUDA_WQ3_CACHE_BLOCK_ROWS");
        if (env == nullptr || env[0] == '\0') {
            return 0;
        }
        const int value = atoi(env);
        return value == 2 || value == 4 || value == 8 || value == 16 || value == 32 || value == 64 ? value : 0;
    }();
    return block_rows;
}

struct block_wq3_tcq_i8_cache {
    uint16_t norm;
    uint16_t pad;
    int8_t   qs[128];
};
static_assert(sizeof(block_wq3_tcq_i8_cache) == 132, "unexpected WQ3 TCQ decoded cache block size");

} // namespace

struct ggml_cuda_wq3_tcq_decoded_cache {
    block_wq3_tcq_i8_cache * data = nullptr;
    size_t nblocks = 0;
    size_t bytes = 0;
};

namespace {

static bool wq3_tcq_decoded_cache_enabled_for(const char * tensor_name) {
    const char * env = getenv("GGML_CUDA_WQ3_DECODE_CACHE");
    if (env == nullptr || env[0] == '\0' || strcmp(env, "0") == 0) {
        return false;
    }
    if (strcmp(env, "1") == 0 || strcasecmp(env, "all") == 0) {
        return true;
    }

    const bool wants_ffn_up   = strstr(env, "ffn_up")   != nullptr || strstr(env, "up")   != nullptr;
    const bool wants_ffn_gate = strstr(env, "ffn_gate") != nullptr || strstr(env, "gate") != nullptr;
    const bool wants_ffn_down = strstr(env, "ffn_down") != nullptr || strstr(env, "down") != nullptr;

    if ((wants_ffn_up   && strstr(tensor_name, "ffn_up")   != nullptr) ||
        (wants_ffn_gate && strstr(tensor_name, "ffn_gate") != nullptr) ||
        (wants_ffn_down && strstr(tensor_name, "ffn_down") != nullptr)) {
        return true;
    }

    std::string token;
    for (const char * p = env; ; ++p) {
        const char c = *p;
        if (c == ',' || c == ':' || c == ';' || c == ' ' || c == '\t' || c == '\0') {
            const std::string exact_suffix = "." + token + ".weight";
            if (!token.empty() &&
                    token != "up" && token != "gate" && token != "down" &&
                    token != "ffn_up" && token != "ffn_gate" && token != "ffn_down" &&
                    strstr(tensor_name, exact_suffix.c_str()) != nullptr) {
                return true;
            }
            token.clear();
            if (c == '\0') {
                break;
            }
        } else {
            token.push_back(c);
        }
    }

    return false;
}

static __global__ void k_wq3_tcq_build_decoded_i8_cache(
        const block_turboq3_tcq * __restrict__ src,
        block_wq3_tcq_i8_cache * __restrict__ dst,
        int nblocks) {
    const int ib = blockIdx.x;
    const int tid = threadIdx.x;
    if (ib >= nblocks || tid >= 128) {
        return;
    }

    const block_turboq3_tcq * s = src + ib;
    block_wq3_tcq_i8_cache * d = dst + ib;
    if (tid == 0) {
        d->norm = reinterpret_cast<const uint16_t *>(&s->norm)[0];
        d->pad = 0;
    }

    const uint8_t * q = s->qs;
    const int byte_idx = (tid * 3) >> 3;
    const int bit_off  = (tid * 3) & 7;
    const uint32_t raw = (uint32_t)q[byte_idx]
                       | ((uint32_t)q[byte_idx + 1] << 8)
                       | ((uint32_t)q[byte_idx + 2] << 16);
    const int state = (raw >> bit_off) & 0x3FF;
    d->qs[tid] = s_codebook_i8_native[state];
}

} // namespace

ggml_cuda_wq3_tcq_decoded_cache * ggml_cuda_wq3_tcq_decoded_cache_try_create(
        const char * tensor_name,
        const void * vx,
        size_t nbytes,
        cudaStream_t stream) {
    if (tensor_name == nullptr || !wq3_tcq_decoded_cache_enabled_for(tensor_name)) {
        return nullptr;
    }
    if (!d_wq3_tcq_codebook_loaded || g_wq3_tcq_codebook_mode != WQ3_TCQ_CODEBOOK_LUT) {
        return nullptr;
    }
    if (nbytes % sizeof(block_turboq3_tcq) != 0) {
        return nullptr;
    }

    const size_t nblocks = nbytes / sizeof(block_turboq3_tcq);
    if (nblocks == 0 || nblocks > (size_t)INT_MAX) {
        return nullptr;
    }

    auto * cache = new ggml_cuda_wq3_tcq_decoded_cache{};
    cache->nblocks = nblocks;
    cache->bytes = nblocks * sizeof(block_wq3_tcq_i8_cache);
    CUDA_CHECK(cudaMalloc((void **)&cache->data, cache->bytes));
    k_wq3_tcq_build_decoded_i8_cache<<<(int)nblocks, 128, 0, stream>>>(
        (const block_turboq3_tcq *)vx, cache->data, (int)nblocks);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    fprintf(stderr, "WQ3_TCQ: decoded i8 cache enabled for %s (%.2f MiB)\n",
            tensor_name, (double)cache->bytes / (1024.0 * 1024.0));
    return cache;
}

void ggml_cuda_wq3_tcq_decoded_cache_free(ggml_cuda_wq3_tcq_decoded_cache * cache) {
    if (cache == nullptr) {
        return;
    }
    if (cache->data != nullptr) {
        CUDA_CHECK(cudaFree(cache->data));
    }
    delete cache;
}

namespace {

struct wq3_tcq_profile_stat {
    int ncols = 0;
    int nrows = 0;
    uint64_t calls = 0;
    double total_ms = 0.0;
};

static bool wq3_tcq_profile_enabled() {
    static const bool enabled = []() {
        const char * env = getenv("GGML_CUDA_WQ3_PROFILE");
        return env != nullptr && env[0] != '\0' && strcmp(env, "0") != 0;
    }();
    return enabled;
}

static std::mutex & wq3_tcq_profile_mutex() {
    static std::mutex * m = new std::mutex();
    return *m;
}

static std::unordered_map<std::string, wq3_tcq_profile_stat> & wq3_tcq_profile_stats() {
    static auto * stats = new std::unordered_map<std::string, wq3_tcq_profile_stat>();
    return *stats;
}

static void wq3_tcq_profile_dump() {
    if (!wq3_tcq_profile_enabled()) {
        return;
    }

    try {
        std::vector<std::pair<std::string, wq3_tcq_profile_stat>> rows;
        {
            std::lock_guard<std::mutex> lock(wq3_tcq_profile_mutex());
            rows.reserve(wq3_tcq_profile_stats().size());
            for (const auto & kv : wq3_tcq_profile_stats()) {
                rows.push_back(kv);
            }
        }

        std::sort(rows.begin(), rows.end(), [](const auto & a, const auto & b) {
            return a.second.total_ms > b.second.total_ms;
        });

        double total_ms = 0.0;
        for (const auto & row : rows) {
            total_ms += row.second.total_ms;
        }

        fprintf(stderr, "\nWQ3_TCQ decode profile (%zu tensors, total %.3f ms)\n",
                rows.size(), total_ms);
        fprintf(stderr, "%10s  %10s  %8s  %10s  %9s  %s\n",
                "total_ms", "avg_ms", "calls", "nrows", "ncols", "tensor");

        for (const auto & row : rows) {
            const auto & stat = row.second;
            const double avg_ms = stat.calls > 0 ? stat.total_ms / (double)stat.calls : 0.0;
            fprintf(stderr, "%10.3f  %10.3f  %8llu  %10d  %9d  %s\n",
                    stat.total_ms, avg_ms, (unsigned long long)stat.calls,
                    stat.nrows, stat.ncols, row.first.c_str());
        }
    } catch (const std::exception & ex) {
        fprintf(stderr, "\nWQ3_TCQ decode profile dump failed: %s\n", ex.what());
    } catch (...) {
        fprintf(stderr, "\nWQ3_TCQ decode profile dump failed: unknown exception\n");
    }
}

static void wq3_tcq_profile_register_atexit() {
    static const bool registered = []() {
        atexit(wq3_tcq_profile_dump);
        return true;
    }();
    GGML_UNUSED(registered);
}

} // namespace

// Fallback hardcoded signs for seed1=42, seed2=1084 (PyTorch 2.x MT19937).
// If the GGUF sign_seed matches, we use these directly.
static const float h_wq3_tcq_signs1_seed42[128] = {
    -1, 1,-1,-1,-1, 1,-1,-1,-1, 1,-1,-1,-1,-1, 1,-1,
     1, 1, 1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1,
     1, 1, 1,-1, 1,-1,-1,-1,-1,-1, 1, 1, 1, 1, 1,-1,
     1, 1,-1, 1,-1, 1,-1, 1, 1,-1,-1,-1,-1,-1,-1,-1,
    -1, 1, 1,-1, 1, 1, 1, 1,-1, 1,-1, 1, 1, 1,-1, 1,
    -1, 1,-1, 1,-1,-1, 1,-1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1,-1,-1, 1, 1, 1, 1, 1, 1, 1, 1,-1, 1,-1,
     1, 1,-1, 1,-1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,
};

static const float h_wq3_tcq_signs2_seed1084[128] = {
    -1, 1, 1,-1, 1,-1, 1,-1,-1, 1, 1,-1,-1, 1,-1,-1,
    -1, 1, 1,-1,-1, 1,-1, 1, 1, 1,-1,-1,-1, 1, 1, 1,
     1, 1,-1,-1, 1,-1, 1,-1,-1, 1, 1,-1, 1, 1, 1, 1,
    -1, 1, 1, 1, 1, 1, 1, 1, 1,-1,-1, 1,-1,-1,-1, 1,
    -1,-1, 1,-1,-1,-1,-1, 1,-1,-1,-1,-1,-1,-1, 1, 1,
     1, 1,-1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1, 1, 1, 1,
    -1, 1, 1, 1, 1,-1, 1, 1,-1,-1, 1, 1, 1, 1,-1, 1,
    -1,-1, 1, 1,-1,-1,-1, 1,-1,-1, 1,-1,-1,-1, 1,-1,
};

// ── Initialization ───────────────────────────────────────────────────────────

void ggml_cuda_set_wq3_tcq_codebook(const float * codebook, int n_entries) {
    GGML_ASSERT(n_entries <= 1024);
    CUDA_CHECK(cudaMemcpyToSymbol(d_wq3_tcq_codebook, codebook, n_entries * sizeof(float)));

    // Pre-quantize to int8 for the MMQ path. Global amax-based scale gives
    // ~0.4% max per-entry relative error with 1024 entries over a typical
    // trained codebook.
    float amax = 0.0f;
    for (int i = 0; i < n_entries; i++) {
        const float a = fabsf(codebook[i]);
        if (a > amax) amax = a;
    }
    const float s_cb = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv_s_cb = 1.0f / s_cb;

    int8_t h_codebook_i8[1024];
    for (int i = 0; i < 1024; i++) {
        if (i < n_entries) {
            int q = (int)lrintf(codebook[i] * inv_s_cb);
            if (q >  127) q =  127;
            if (q < -127) q = -127;
            h_codebook_i8[i] = (int8_t)q;
        } else {
            h_codebook_i8[i] = 0;
        }
    }
    ggml_cuda_wq3_tcq_mmq_set_codebook_i8(h_codebook_i8, s_cb);
    set_codebook_native(h_codebook_i8, s_cb);

    float proc_qscale = 0.0f;
    if (wq3_tcq_detect_murmur_gauss(codebook, n_entries, h_codebook_i8, s_cb, &proc_qscale)) {
        CUDA_CHECK(cudaMemcpyToSymbol(s_proc_qscale_native, &proc_qscale, sizeof(float)));
        g_wq3_tcq_codebook_mode = WQ3_TCQ_CODEBOOK_PROC_MURMUR;
        fprintf(stderr, "WQ3_TCQ: detected procedural codebook variant murmur_gauss (qscale=%.6g, layout=%s)\n",
                proc_qscale, wq3_tcq_use_proc_u16_layout() ? "u16" : "byte");
    } else {
        const float zero = 0.0f;
        CUDA_CHECK(cudaMemcpyToSymbol(s_proc_qscale_native, &zero, sizeof(float)));
        g_wq3_tcq_codebook_mode = WQ3_TCQ_CODEBOOK_LUT;
    }

    d_wq3_tcq_codebook_loaded = true;
    fprintf(stderr, "WQ3_TCQ: loaded %d-entry weight codebook to device (int8 scale=%.6g, amax=%.6g)\n",
            n_entries, s_cb, amax);
}

ggml_cuda_wq3_tcq_profile_scope ggml_cuda_wq3_tcq_profile_begin(
        const char * tensor_name,
        int ncols,
        int nrows,
        cudaStream_t stream) {
    ggml_cuda_wq3_tcq_profile_scope scope;
    if (!wq3_tcq_profile_enabled()) {
        return scope;
    }

    wq3_tcq_profile_register_atexit();
    scope.tensor_name = tensor_name;
    scope.ncols = ncols;
    scope.nrows = nrows;
    scope.active = true;
    CUDA_CHECK(cudaEventCreateWithFlags(&scope.start, cudaEventDefault));
    CUDA_CHECK(cudaEventCreateWithFlags(&scope.stop, cudaEventDefault));
    CUDA_CHECK(cudaEventRecord(scope.start, stream));
    return scope;
}

void ggml_cuda_wq3_tcq_profile_end(
        ggml_cuda_wq3_tcq_profile_scope & scope,
        cudaStream_t stream) {
    if (!scope.active) {
        return;
    }

    CUDA_CHECK(cudaEventRecord(scope.stop, stream));
    CUDA_CHECK(cudaEventSynchronize(scope.stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, scope.start, scope.stop));

    {
        std::lock_guard<std::mutex> lock(wq3_tcq_profile_mutex());
        auto & stat = wq3_tcq_profile_stats()[scope.tensor_name ? scope.tensor_name : "<unnamed>"];
        stat.ncols = scope.ncols;
        stat.nrows = scope.nrows;
        stat.calls += 1;
        stat.total_ms += elapsed_ms;
    }

    CUDA_CHECK(cudaEventDestroy(scope.start));
    CUDA_CHECK(cudaEventDestroy(scope.stop));
    scope = {};
}

void ggml_cuda_set_wq3_tcq_signs(uint32_t sign_seed) {
    if (sign_seed == 42) {
        // Use hardcoded signs for the standard seed pair (42, 1084)
        CUDA_CHECK(cudaMemcpyToSymbol(d_wq3_tcq_signs1, h_wq3_tcq_signs1_seed42, 128 * sizeof(float)));
        CUDA_CHECK(cudaMemcpyToSymbol(d_wq3_tcq_signs2, h_wq3_tcq_signs2_seed1084, 128 * sizeof(float)));
    } else {
        GGML_ABORT("WQ3_TCQ: unsupported sign_seed %u (only seed=42 is hardcoded; "
                    "add runtime generation or hardcode new seeds)", sign_seed);
    }
    d_wq3_tcq_signs_loaded = true;
    fprintf(stderr, "WQ3_TCQ: loaded FWHT signs for seed=%u\n", sign_seed);
}

void ggml_cuda_set_wq3_tcq_signs_direct(const float * s1, const float * s2, int n) {
    GGML_ASSERT(n == 128);
    CUDA_CHECK(cudaMemcpyToSymbol(d_wq3_tcq_signs1, s1, 128 * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_wq3_tcq_signs2, s2, 128 * sizeof(float)));
    d_wq3_tcq_signs_loaded = true;
    fprintf(stderr, "WQ3_TCQ: loaded FWHT signs from GGUF vectors\n");
}

// ── FWHT butterfly (128-point, 128 threads) ─────────────────────────────────

// Identical to fwht128_butterfly_inplace in fattn.cu but in this compilation
// unit so constant memory resolves correctly.
static __device__ __forceinline__
float wq3_fwht128(float val, float * smem) {
    const int tid = threadIdx.x & 127;

    // Intra-warp butterfly: strides 1, 2, 4, 8, 16 via __shfl_xor_sync
    #pragma unroll
    for (int h = 1; h <= 16; h *= 2) {
        const float other = __shfl_xor_sync(0xFFFFFFFF, val, h, WARP_SIZE);
        val = (tid & h) ? (other - val) : (val + other);
    }

    // Cross-warp butterfly: stride 32 (smem)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 32) ? (smem[tid - 32] - val) : (val + smem[tid + 32]);
    __syncthreads();

    // Cross-warp butterfly: stride 64 (smem)
    smem[tid] = val;
    __syncthreads();
    val = (tid & 64) ? (smem[tid - 64] - val) : (val + smem[tid + 64]);
    __syncthreads();

    return val;
}

// Pack two adjacent fp16 values into a half2 store (halves global memory writes).
static __device__ __forceinline__
void wq3_store_half2(float val, half * dst) {
    const int tid = threadIdx.x & 127;
    const float neighbor = __shfl_xor_sync(0xFFFFFFFF, val, 1, WARP_SIZE);
    if ((tid & 1) == 0) {
        *((half2 *)(dst + tid)) = __floats2half2_rn(val, neighbor);
    }
}

// ── Decode all 128 elements from trellis bitstream ──────────────────────────
//
// Weight TCQ uses L=10, K=3.  The bitstream layout is:
//   bits [0,7)  : prefix  = init_state >> K  (upper L-K bits of initial state)
//   bits [7+t*3, 7+t*3+3): stored[t] = state[t] >> (L-K) = top 3 bits of state[t]
//
// The trellis transition is: state[t] = ((state[t-1] << K) | transition[t]) & mask_L
// where stored[t] = state[t] >> (L-K).
//
// Because L=10 is NOT a multiple of K=3, a simple 10-bit sliding window does NOT
// reconstruct the trellis state.  Instead, thread 0 sequentially reconstructs all
// 128 states and writes codebook values to shared memory.  The FWHT butterfly that
// follows already uses shared memory, so this adds no extra sync cost.

// Sliding window decode: read L=10 contiguous bits at position tid*3.
// Must read 3 bytes because when bit_off >= 7 the 10-bit state spans into
// the third byte.  qs[49] aliases the pad byte (always 0) so it is safe.
static __device__ __forceinline__
float wq3_decode_element(const uint8_t * qs, int tid) {
    const int bit_pos = tid * 3;
    const int byte_idx = bit_pos >> 3;
    const int bit_off  = bit_pos & 7;
    const uint32_t raw = (uint32_t)qs[byte_idx]
                       | ((uint32_t)qs[byte_idx + 1] << 8)
                       | ((uint32_t)qs[byte_idx + 2] << 16);
    const int state = (raw >> bit_off) & 0x3FF;
    return d_wq3_tcq_codebook[state];
}

// ── Dequant kernels ──────────────────────────────────────────────────────────

static __global__ void k_wq3_tcq_dequant_f32(
        const void * __restrict__ vx,
        float * __restrict__ y,
        const int64_t k) {
    const int group_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const block_turboq3_tcq * blk = ((const block_turboq3_tcq *)vx) + group_idx;
    const float norm = __half2float(blk->norm);

    __shared__ float smem[128];

    float val = wq3_fwht128(wq3_decode_element(blk->qs, tid) * d_wq3_tcq_signs2[tid], smem);
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val *= inv_sqrt_128 * d_wq3_tcq_signs1[tid] * norm;

    const int64_t out_idx = (int64_t)group_idx * 128 + tid;
    if (out_idx < k) {
        y[out_idx] = val;
    }
}

static __global__ void k_wq3_tcq_dequant_f16_fast(
        const void * __restrict__ vx,
        half * __restrict__ y,
        const int64_t k) {
    const int group_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const block_turboq3_tcq * blk = ((const block_turboq3_tcq *)vx) + group_idx;
    const float norm = __half2float(blk->norm);

    __shared__ float smem[128];

    float val = wq3_fwht128(wq3_decode_element(blk->qs, tid) * d_wq3_tcq_signs2[tid], smem);
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    val *= inv_sqrt_128 * d_wq3_tcq_signs1[tid] * norm;

    wq3_store_half2(val, y + (int64_t)group_idx * 128);
}

// ── Dequant launchers (to_fp16_cuda_t / to_fp32_cuda_t interface) ────────────

// Registered as ggml_get_to_fp16_cuda(WQ3_TCQ). Emits original-domain fp16
// weights (full inverse FWHT) so any non-MMQ caller sees the true weight
// tensor. The old rotated-domain variant was deleted in M6 — the Stage B
// cuBLAS prefill path that required rotated-domain fp16 is gone.
void dequantize_wq3_tcq_to_fp16(const void * vx, half * y, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % 128 == 0);
    const int n_groups = (int)(k / 128);
    k_wq3_tcq_dequant_f16_fast<<<n_groups, 128, 0, stream>>>(vx, y, k);
}

void dequantize_wq3_tcq_to_fp32(const void * vx, float * y, int64_t k, cudaStream_t stream) {
    GGML_ASSERT(k % 128 == 0);
    const int n_groups = (int)(k / 128);
    k_wq3_tcq_dequant_f32<<<n_groups, 128, 0, stream>>>(vx, y, k);
}

// ── Native decode path kernels ───────────────────────────────────────────────
//
// From-scratch WQ3_TCQ-specific batch=1 decode kernel set. Two stages:
//   1. k_wq3_tcq_rotate_activation_fp32: FWHT-rotate the input x into xrot.
//   2. k_wq3_tcq_mmvq_v2:               per-row trellis-decode + fp32 MAD
//      against xrot. Weight dequant and the row-parallel dot product are
//      fused; the FWHT on the weight side is absorbed by the x-side
//      pre-rotation, so no per-row FWHT is needed.
//
// Rotate-activation: one 128-elem group per block.
//   xrot[i] = signs2[i] * FWHT(signs1 * x)[i] / sqrt(128)
static __global__ void k_wq3_tcq_rotate_activation_fp32(
        const float * __restrict__ x,
        float * __restrict__ xrot,
        const int N) {
    const int group_idx = blockIdx.x;
    const int tid       = threadIdx.x;
    const int idx       = group_idx * 128 + tid;
    if (idx >= N) return;

    __shared__ float smem[128];

    float v = x[idx] * d_wq3_tcq_signs1[tid];
    v = wq3_fwht128(v, smem);
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    v *= inv_sqrt_128 * d_wq3_tcq_signs2[tid];

    xrot[idx] = v;
}

// v2 MMVQ decode: one warp per output row, BLOCK_ROWS warps per CTA. Each lane
// decodes 4 contiguous elements of a 128-element block by sliding a 10-bit
// window over the 390-bit qs stream at bit offset lane*12.
//
// Weight fetch: block spans 52 bytes = 13 × uint32. Lanes 0..12 each issue one
// 4-byte LDG at blk+4*lane, covering (qs_u16[2L-1], qs_u16[2L]); lane 0's low
// 16 bits carry the fp16 norm. Each lane __shfl_syncs the two source lanes
// that hold its window's u16 halves (derived from bit_base / 16).
//
// qs[49] is the always-zero pad byte (exporter invariant), so the tail lane's
// u16 window at bits 384..391 has zero high bits — no out-of-bounds read. Max
// source lane used is 12.
template <int BLOCK_ROWS>
static __global__ void k_wq3_tcq_mmvq_v2(
        const void  * __restrict__ vx,
        const float * __restrict__ xrot,
        float       * __restrict__ dst,
        const int ncols,
        const int nrows) {
    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int row  = blockIdx.x * BLOCK_ROWS + warp;

    const int n_groups = ncols / 128;
    const block_turboq3_tcq * src_row =
        ((const block_turboq3_tcq *)vx) + (int64_t)row * n_groups;

    // 1 KiB codebook staged in smem once per CTA. Both source and destination
    // are 16-byte aligned (alignas(16) on the __constant__ symbol), so widen
    // to int4 stores: 64 total instead of 1024.
    __shared__ alignas(16) int8_t cb_smem[1024];
    const int4 *       cb_src = reinterpret_cast<const int4 *>(s_codebook_i8_native);
    int4       * const cb_dst = reinterpret_cast<int4 *>(cb_smem);
    for (int e = tid; e < 64; e += 32 * BLOCK_ROWS) {
        cb_dst[e] = cb_src[e];
    }
    __syncthreads();

    const int bit_base  = lane * 12;
    const int w_idx_lo  = bit_base >> 4;
    const int w_idx_hi  = w_idx_lo + 1;
    const int shift     = bit_base & 15;
    const float cb_scale = s_codebook_scale_native;

    // 4-byte-aligned pair-load layout. Block is 52 B = 13 × uint32 (4-aligned).
    // qs[0] sits at blk+2, so qs-based uint32s would be 2-byte-misaligned;
    // index uint32s from the block base instead.
    //
    //   lane 0:    blk[0..3]    = (norm_fp16, qs_u16[0])
    //   lane L≥1:  blk[4L..4L+3] = (qs_u16[2L-1], qs_u16[2L])
    //   lane 12:   blk[48..51]  = (qs_u16[23], qs_u16[24])
    //   lane 13..31: idle (never shfled from)
    //
    // qs_u16[K] lives at lane ((K+1)>>1), bit ((K+1)&1)<<4.
    const int pair_src_lo = (w_idx_lo + 1) >> 1;
    const int pair_bit_lo = ((w_idx_lo + 1) & 1) << 4;
    const int pair_src_hi = (w_idx_hi + 1) >> 1;
    const int pair_bit_hi = ((w_idx_hi + 1) & 1) << 4;
    float sum = 0.0f;
    #pragma unroll 4
    for (int g = 0; g < n_groups; ++g) {
        const block_turboq3_tcq * blk = src_row + g;

        const uint32_t * blk_u32 = reinterpret_cast<const uint32_t *>(blk);
        const uint32_t own_pair = (lane < 13) ? blk_u32[lane] : 0u;

        const uint16_t norm_bits = (uint16_t)(__shfl_sync(0xFFFFFFFFu, own_pair, 0, WARP_SIZE) & 0xFFFFu);
        const float    norm      = __half2float(__ushort_as_half(norm_bits)) * cb_scale;

        const uint32_t p_lo = __shfl_sync(0xFFFFFFFFu, own_pair, pair_src_lo, WARP_SIZE);
        const uint32_t p_hi = __shfl_sync(0xFFFFFFFFu, own_pair, pair_src_hi, WARP_SIZE);
        const uint32_t w_lo = (p_lo >> pair_bit_lo) & 0xFFFFu;
        const uint32_t w_hi = (p_hi >> pair_bit_hi) & 0xFFFFu;
        const uint32_t merged = (w_hi << 16) | w_lo;

        const int state0 = (merged >> (shift + 0)) & 0x3FF;
        const int state1 = (merged >> (shift + 3)) & 0x3FF;
        const int state2 = (merged >> (shift + 6)) & 0x3FF;
        const int state3 = (merged >> (shift + 9)) & 0x3FF;

        const int c0 = (int)cb_smem[state0];
        const int c1 = (int)cb_smem[state1];
        const int c2 = (int)cb_smem[state2];
        const int c3 = (int)cb_smem[state3];

        // Direct coalesced float4 load; all BLOCK_ROWS warps read the same
        // 512 B slice — later warps hit L1 but still spend their own LSU slots.
        const float4 xv = *reinterpret_cast<const float4 *>(xrot + g * 128 + lane * 4);

        float acc = (float)c0 * xv.x;
        acc = fmaf((float)c1, xv.y, acc);
        acc = fmaf((float)c2, xv.z, acc);
        acc = fmaf((float)c3, xv.w, acc);
        sum = fmaf(acc, norm, sum);
    }

    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        dst[row] = sum;
    }
}

// Experimental row-pair variant: one warp accumulates 2 output rows while
// reusing the same xrot float4 load for both. This trades extra registers for
// fewer xrot LSU ops per row and leaves all decode math bit-identical.
template <int BLOCK_ROWS>
static __global__ void k_wq3_tcq_mmvq_v2_rowpair(
        const void  * __restrict__ vx,
        const float * __restrict__ xrot,
        float       * __restrict__ dst,
        const int ncols,
        const int nrows) {
    static_assert((BLOCK_ROWS % 2) == 0, "rowpair kernel requires even BLOCK_ROWS");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int row0 = blockIdx.x * BLOCK_ROWS + warp * 2 + 0;
    const int row1 = row0 + 1;

    const int n_groups = ncols / 128;
    const block_turboq3_tcq * src_row0 =
        ((const block_turboq3_tcq *)vx) + (int64_t)row0 * n_groups;
    const block_turboq3_tcq * src_row1 =
        ((const block_turboq3_tcq *)vx) + (int64_t)row1 * n_groups;

    __shared__ alignas(16) int8_t cb_smem[1024];
    const int4 *       cb_src = reinterpret_cast<const int4 *>(s_codebook_i8_native);
    int4       * const cb_dst = reinterpret_cast<int4 *>(cb_smem);
    for (int e = tid; e < 64; e += 32 * (BLOCK_ROWS / 2)) {
        cb_dst[e] = cb_src[e];
    }
    __syncthreads();

    const int bit_base  = lane * 12;
    const int w_idx_lo  = bit_base >> 4;
    const int w_idx_hi  = w_idx_lo + 1;
    const int shift     = bit_base & 15;
    const float cb_scale = s_codebook_scale_native;

    const int pair_src_lo = (w_idx_lo + 1) >> 1;
    const int pair_bit_lo = ((w_idx_lo + 1) & 1) << 4;
    const int pair_src_hi = (w_idx_hi + 1) >> 1;
    const int pair_bit_hi = ((w_idx_hi + 1) & 1) << 4;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    #pragma unroll 4
    for (int g = 0; g < n_groups; ++g) {
        const block_turboq3_tcq * blk0 = src_row0 + g;
        const block_turboq3_tcq * blk1 = src_row1 + g;

        const uint32_t * blk0_u32 = reinterpret_cast<const uint32_t *>(blk0);
        const uint32_t * blk1_u32 = reinterpret_cast<const uint32_t *>(blk1);
        const uint32_t own_pair0 = (lane < 13) ? blk0_u32[lane] : 0u;
        const uint32_t own_pair1 = (lane < 13) ? blk1_u32[lane] : 0u;

        const uint16_t norm_bits0 = (uint16_t)(__shfl_sync(0xFFFFFFFFu, own_pair0, 0, WARP_SIZE) & 0xFFFFu);
        const uint16_t norm_bits1 = (uint16_t)(__shfl_sync(0xFFFFFFFFu, own_pair1, 0, WARP_SIZE) & 0xFFFFu);
        const float    norm0      = __half2float(__ushort_as_half(norm_bits0)) * cb_scale;
        const float    norm1      = __half2float(__ushort_as_half(norm_bits1)) * cb_scale;

        const uint32_t p0_lo = __shfl_sync(0xFFFFFFFFu, own_pair0, pair_src_lo, WARP_SIZE);
        const uint32_t p0_hi = __shfl_sync(0xFFFFFFFFu, own_pair0, pair_src_hi, WARP_SIZE);
        const uint32_t p1_lo = __shfl_sync(0xFFFFFFFFu, own_pair1, pair_src_lo, WARP_SIZE);
        const uint32_t p1_hi = __shfl_sync(0xFFFFFFFFu, own_pair1, pair_src_hi, WARP_SIZE);
        const uint32_t w0_lo = (p0_lo >> pair_bit_lo) & 0xFFFFu;
        const uint32_t w0_hi = (p0_hi >> pair_bit_hi) & 0xFFFFu;
        const uint32_t w1_lo = (p1_lo >> pair_bit_lo) & 0xFFFFu;
        const uint32_t w1_hi = (p1_hi >> pair_bit_hi) & 0xFFFFu;
        const uint32_t merged0 = (w0_hi << 16) | w0_lo;
        const uint32_t merged1 = (w1_hi << 16) | w1_lo;

        const int state00 = (merged0 >> (shift + 0)) & 0x3FF;
        const int state01 = (merged0 >> (shift + 3)) & 0x3FF;
        const int state02 = (merged0 >> (shift + 6)) & 0x3FF;
        const int state03 = (merged0 >> (shift + 9)) & 0x3FF;
        const int state10 = (merged1 >> (shift + 0)) & 0x3FF;
        const int state11 = (merged1 >> (shift + 3)) & 0x3FF;
        const int state12 = (merged1 >> (shift + 6)) & 0x3FF;
        const int state13 = (merged1 >> (shift + 9)) & 0x3FF;

        const int c00 = (int)cb_smem[state00];
        const int c01 = (int)cb_smem[state01];
        const int c02 = (int)cb_smem[state02];
        const int c03 = (int)cb_smem[state03];
        const int c10 = (int)cb_smem[state10];
        const int c11 = (int)cb_smem[state11];
        const int c12 = (int)cb_smem[state12];
        const int c13 = (int)cb_smem[state13];

        const float4 xv = *reinterpret_cast<const float4 *>(xrot + g * 128 + lane * 4);

        float acc0 = (float)c00 * xv.x;
        acc0 = fmaf((float)c01, xv.y, acc0);
        acc0 = fmaf((float)c02, xv.z, acc0);
        acc0 = fmaf((float)c03, xv.w, acc0);
        sum0 = fmaf(acc0, norm0, sum0);

        float acc1 = (float)c10 * xv.x;
        acc1 = fmaf((float)c11, xv.y, acc1);
        acc1 = fmaf((float)c12, xv.z, acc1);
        acc1 = fmaf((float)c13, xv.w, acc1);
        sum1 = fmaf(acc1, norm1, sum1);
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    if (lane == 0) {
        dst[row0] = sum0;
        dst[row1] = sum1;
    }
}

// Experimental int8 activation path: consume xrot quantized to q8_1_mmq and
// use dp4a for the 4-way inner product. This changes activation numerics, so it
// is speed-only until sanity/PPL gates are run.
template <int BLOCK_ROWS>
static __global__ void k_wq3_tcq_mmvq_q8_1_rowpair(
        const void  * __restrict__ vx,
        const void  * __restrict__ vxrot_q8_1,
        float       * __restrict__ dst,
        const int ncols,
        const int nrows) {
    static_assert((BLOCK_ROWS % 2) == 0, "rowpair kernel requires even BLOCK_ROWS");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int row0 = blockIdx.x * BLOCK_ROWS + warp * 2 + 0;
    const int row1 = row0 + 1;

    const int n_groups = ncols / 128;
    const block_turboq3_tcq * src_row0 =
        ((const block_turboq3_tcq *)vx) + (int64_t)row0 * n_groups;
    const block_turboq3_tcq * src_row1 =
        ((const block_turboq3_tcq *)vx) + (int64_t)row1 * n_groups;
    const block_q8_1_mmq * xrot_q8_1 = (const block_q8_1_mmq *)vxrot_q8_1;

    __shared__ alignas(16) int8_t cb_smem[1024];
    const int4 *       cb_src = reinterpret_cast<const int4 *>(s_codebook_i8_native);
    int4       * const cb_dst = reinterpret_cast<int4 *>(cb_smem);
    for (int e = tid; e < 64; e += 32 * (BLOCK_ROWS / 2)) {
        cb_dst[e] = cb_src[e];
    }
    __syncthreads();

    const int bit_base  = lane * 12;
    const int w_idx_lo  = bit_base >> 4;
    const int w_idx_hi  = w_idx_lo + 1;
    const int shift     = bit_base & 15;
    const float cb_scale = s_codebook_scale_native;

    const int pair_src_lo = (w_idx_lo + 1) >> 1;
    const int pair_bit_lo = ((w_idx_lo + 1) & 1) << 4;
    const int pair_src_hi = (w_idx_hi + 1) >> 1;
    const int pair_bit_hi = ((w_idx_hi + 1) & 1) << 4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    #pragma unroll 4
    for (int g = 0; g < n_groups; ++g) {
        const block_turboq3_tcq * blk0 = src_row0 + g;
        const block_turboq3_tcq * blk1 = src_row1 + g;
        const block_q8_1_mmq * xblk = xrot_q8_1 + g;

        const uint32_t * blk0_u32 = reinterpret_cast<const uint32_t *>(blk0);
        const uint32_t * blk1_u32 = reinterpret_cast<const uint32_t *>(blk1);
        const uint32_t own_pair0 = (lane < 13) ? blk0_u32[lane] : 0u;
        const uint32_t own_pair1 = (lane < 13) ? blk1_u32[lane] : 0u;

        const uint16_t norm_bits0 = (uint16_t)(__shfl_sync(0xFFFFFFFFu, own_pair0, 0, WARP_SIZE) & 0xFFFFu);
        const uint16_t norm_bits1 = (uint16_t)(__shfl_sync(0xFFFFFFFFu, own_pair1, 0, WARP_SIZE) & 0xFFFFu);
        const float    norm0      = __half2float(__ushort_as_half(norm_bits0)) * cb_scale;
        const float    norm1      = __half2float(__ushort_as_half(norm_bits1)) * cb_scale;

        const uint32_t p0_lo = __shfl_sync(0xFFFFFFFFu, own_pair0, pair_src_lo, WARP_SIZE);
        const uint32_t p0_hi = __shfl_sync(0xFFFFFFFFu, own_pair0, pair_src_hi, WARP_SIZE);
        const uint32_t p1_lo = __shfl_sync(0xFFFFFFFFu, own_pair1, pair_src_lo, WARP_SIZE);
        const uint32_t p1_hi = __shfl_sync(0xFFFFFFFFu, own_pair1, pair_src_hi, WARP_SIZE);
        const uint32_t w0_lo = (p0_lo >> pair_bit_lo) & 0xFFFFu;
        const uint32_t w0_hi = (p0_hi >> pair_bit_hi) & 0xFFFFu;
        const uint32_t w1_lo = (p1_lo >> pair_bit_lo) & 0xFFFFu;
        const uint32_t w1_hi = (p1_hi >> pair_bit_hi) & 0xFFFFu;
        const uint32_t merged0 = (w0_hi << 16) | w0_lo;
        const uint32_t merged1 = (w1_hi << 16) | w1_lo;

        const int state00 = (merged0 >> (shift + 0)) & 0x3FF;
        const int state01 = (merged0 >> (shift + 3)) & 0x3FF;
        const int state02 = (merged0 >> (shift + 6)) & 0x3FF;
        const int state03 = (merged0 >> (shift + 9)) & 0x3FF;
        const int state10 = (merged1 >> (shift + 0)) & 0x3FF;
        const int state11 = (merged1 >> (shift + 3)) & 0x3FF;
        const int state12 = (merged1 >> (shift + 6)) & 0x3FF;
        const int state13 = (merged1 >> (shift + 9)) & 0x3FF;

        const uint32_t cpack0 =
            ((uint32_t)(uint8_t)cb_smem[state00] << 0)  |
            ((uint32_t)(uint8_t)cb_smem[state01] << 8)  |
            ((uint32_t)(uint8_t)cb_smem[state02] << 16) |
            ((uint32_t)(uint8_t)cb_smem[state03] << 24);
        const uint32_t cpack1 =
            ((uint32_t)(uint8_t)cb_smem[state10] << 0)  |
            ((uint32_t)(uint8_t)cb_smem[state11] << 8)  |
            ((uint32_t)(uint8_t)cb_smem[state12] << 16) |
            ((uint32_t)(uint8_t)cb_smem[state13] << 24);

        const int qx = *reinterpret_cast<const int *>(xblk->qs + lane * 4);
        const float xd = xblk->d4[lane >> 3];

        const float acc0 = (float)ggml_cuda_dp4a((int)cpack0, qx, 0);
        const float acc1 = (float)ggml_cuda_dp4a((int)cpack1, qx, 0);

        sum0 = fmaf(acc0 * xd, norm0, sum0);
        sum1 = fmaf(acc1 * xd, norm1, sum1);
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    if (lane == 0) {
        dst[row0] = sum0;
        dst[row1] = sum1;
    }
}

template <int BLOCK_ROWS>
static __global__ void k_wq3_tcq_mmvq_q8_1_dual_rowpair_glu(
        const void  * __restrict__ vx_up,
        const void  * __restrict__ vx_gate,
        const void  * __restrict__ vxrot_up_q8_1,
        const void  * __restrict__ vxrot_gate_q8_1,
        float       * __restrict__ dst,
        const int ncols,
        const int nrows,
        ggml_glu_op glu_op) {
    static_assert((BLOCK_ROWS % 2) == 0, "dual rowpair kernel requires even BLOCK_ROWS");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int tid  = warp * 32 + lane;
    const int row0 = blockIdx.x * BLOCK_ROWS + warp * 2 + 0;
    const int row1 = row0 + 1;

    const int n_groups = ncols / 128;
    const block_turboq3_tcq * up_row0 =
        ((const block_turboq3_tcq *)vx_up) + (int64_t)row0 * n_groups;
    const block_turboq3_tcq * up_row1 =
        ((const block_turboq3_tcq *)vx_up) + (int64_t)row1 * n_groups;
    const block_turboq3_tcq * gate_row0 =
        ((const block_turboq3_tcq *)vx_gate) + (int64_t)row0 * n_groups;
    const block_turboq3_tcq * gate_row1 =
        ((const block_turboq3_tcq *)vx_gate) + (int64_t)row1 * n_groups;
    const block_q8_1_mmq * xrot_up_q8_1 = (const block_q8_1_mmq *)vxrot_up_q8_1;
    const block_q8_1_mmq * xrot_gate_q8_1 = (const block_q8_1_mmq *)vxrot_gate_q8_1;

    __shared__ alignas(16) int8_t cb_smem[1024];
    const int4 *       cb_src = reinterpret_cast<const int4 *>(s_codebook_i8_native);
    int4       * const cb_dst = reinterpret_cast<int4 *>(cb_smem);
    for (int e = tid; e < 64; e += 32 * (BLOCK_ROWS / 2)) {
        cb_dst[e] = cb_src[e];
    }
    __syncthreads();

    const int bit_base  = lane * 12;
    const int w_idx_lo  = bit_base >> 4;
    const int w_idx_hi  = w_idx_lo + 1;
    const int shift     = bit_base & 15;
    const float cb_scale = s_codebook_scale_native;

    const int pair_src_lo = (w_idx_lo + 1) >> 1;
    const int pair_bit_lo = ((w_idx_lo + 1) & 1) << 4;
    const int pair_src_hi = (w_idx_hi + 1) >> 1;
    const int pair_bit_hi = ((w_idx_hi + 1) & 1) << 4;

    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;

#pragma unroll 4
    for (int g = 0; g < n_groups; ++g) {
        const block_turboq3_tcq * up_blk0 = up_row0 + g;
        const block_turboq3_tcq * up_blk1 = up_row1 + g;
        const block_turboq3_tcq * gate_blk0 = gate_row0 + g;
        const block_turboq3_tcq * gate_blk1 = gate_row1 + g;
        const block_q8_1_mmq * xblk_up = xrot_up_q8_1 + g;
        const block_q8_1_mmq * xblk_gate = xrot_gate_q8_1 + g;

        const uint32_t * up0_u32 = reinterpret_cast<const uint32_t *>(up_blk0);
        const uint32_t * up1_u32 = reinterpret_cast<const uint32_t *>(up_blk1);
        const uint32_t * gate0_u32 = reinterpret_cast<const uint32_t *>(gate_blk0);
        const uint32_t * gate1_u32 = reinterpret_cast<const uint32_t *>(gate_blk1);
        const uint32_t up_pair0 = (lane < 13) ? up0_u32[lane] : 0u;
        const uint32_t up_pair1 = (lane < 13) ? up1_u32[lane] : 0u;
        const uint32_t gate_pair0 = (lane < 13) ? gate0_u32[lane] : 0u;
        const uint32_t gate_pair1 = (lane < 13) ? gate1_u32[lane] : 0u;

        const float up_norm0 = __half2float(__ushort_as_half((uint16_t)(__shfl_sync(0xFFFFFFFFu, up_pair0, 0, WARP_SIZE) & 0xFFFFu))) * cb_scale;
        const float up_norm1 = __half2float(__ushort_as_half((uint16_t)(__shfl_sync(0xFFFFFFFFu, up_pair1, 0, WARP_SIZE) & 0xFFFFu))) * cb_scale;
        const float gate_norm0 = __half2float(__ushort_as_half((uint16_t)(__shfl_sync(0xFFFFFFFFu, gate_pair0, 0, WARP_SIZE) & 0xFFFFu))) * cb_scale;
        const float gate_norm1 = __half2float(__ushort_as_half((uint16_t)(__shfl_sync(0xFFFFFFFFu, gate_pair1, 0, WARP_SIZE) & 0xFFFFu))) * cb_scale;

        const uint32_t up0_lo = __shfl_sync(0xFFFFFFFFu, up_pair0, pair_src_lo, WARP_SIZE);
        const uint32_t up0_hi = __shfl_sync(0xFFFFFFFFu, up_pair0, pair_src_hi, WARP_SIZE);
        const uint32_t up1_lo = __shfl_sync(0xFFFFFFFFu, up_pair1, pair_src_lo, WARP_SIZE);
        const uint32_t up1_hi = __shfl_sync(0xFFFFFFFFu, up_pair1, pair_src_hi, WARP_SIZE);
        const uint32_t gate0_lo = __shfl_sync(0xFFFFFFFFu, gate_pair0, pair_src_lo, WARP_SIZE);
        const uint32_t gate0_hi = __shfl_sync(0xFFFFFFFFu, gate_pair0, pair_src_hi, WARP_SIZE);
        const uint32_t gate1_lo = __shfl_sync(0xFFFFFFFFu, gate_pair1, pair_src_lo, WARP_SIZE);
        const uint32_t gate1_hi = __shfl_sync(0xFFFFFFFFu, gate_pair1, pair_src_hi, WARP_SIZE);

        const uint32_t up_merged0 = (((up0_hi >> pair_bit_hi) & 0xFFFFu) << 16) | ((up0_lo >> pair_bit_lo) & 0xFFFFu);
        const uint32_t up_merged1 = (((up1_hi >> pair_bit_hi) & 0xFFFFu) << 16) | ((up1_lo >> pair_bit_lo) & 0xFFFFu);
        const uint32_t gate_merged0 = (((gate0_hi >> pair_bit_hi) & 0xFFFFu) << 16) | ((gate0_lo >> pair_bit_lo) & 0xFFFFu);
        const uint32_t gate_merged1 = (((gate1_hi >> pair_bit_hi) & 0xFFFFu) << 16) | ((gate1_lo >> pair_bit_lo) & 0xFFFFu);

        const uint32_t up_cpack0 =
            ((uint32_t)(uint8_t)cb_smem[(up_merged0 >> (shift + 0)) & 0x3FF] << 0)  |
            ((uint32_t)(uint8_t)cb_smem[(up_merged0 >> (shift + 3)) & 0x3FF] << 8)  |
            ((uint32_t)(uint8_t)cb_smem[(up_merged0 >> (shift + 6)) & 0x3FF] << 16) |
            ((uint32_t)(uint8_t)cb_smem[(up_merged0 >> (shift + 9)) & 0x3FF] << 24);
        const uint32_t up_cpack1 =
            ((uint32_t)(uint8_t)cb_smem[(up_merged1 >> (shift + 0)) & 0x3FF] << 0)  |
            ((uint32_t)(uint8_t)cb_smem[(up_merged1 >> (shift + 3)) & 0x3FF] << 8)  |
            ((uint32_t)(uint8_t)cb_smem[(up_merged1 >> (shift + 6)) & 0x3FF] << 16) |
            ((uint32_t)(uint8_t)cb_smem[(up_merged1 >> (shift + 9)) & 0x3FF] << 24);
        const uint32_t gate_cpack0 =
            ((uint32_t)(uint8_t)cb_smem[(gate_merged0 >> (shift + 0)) & 0x3FF] << 0)  |
            ((uint32_t)(uint8_t)cb_smem[(gate_merged0 >> (shift + 3)) & 0x3FF] << 8)  |
            ((uint32_t)(uint8_t)cb_smem[(gate_merged0 >> (shift + 6)) & 0x3FF] << 16) |
            ((uint32_t)(uint8_t)cb_smem[(gate_merged0 >> (shift + 9)) & 0x3FF] << 24);
        const uint32_t gate_cpack1 =
            ((uint32_t)(uint8_t)cb_smem[(gate_merged1 >> (shift + 0)) & 0x3FF] << 0)  |
            ((uint32_t)(uint8_t)cb_smem[(gate_merged1 >> (shift + 3)) & 0x3FF] << 8)  |
            ((uint32_t)(uint8_t)cb_smem[(gate_merged1 >> (shift + 6)) & 0x3FF] << 16) |
            ((uint32_t)(uint8_t)cb_smem[(gate_merged1 >> (shift + 9)) & 0x3FF] << 24);

        const int qx_up = *reinterpret_cast<const int *>(xblk_up->qs + lane * 4);
        const float xd_up = xblk_up->d4[lane >> 3];
        const int qx_gate = *reinterpret_cast<const int *>(xblk_gate->qs + lane * 4);
        const float xd_gate = xblk_gate->d4[lane >> 3];

        up_sum0 = fmaf((float)ggml_cuda_dp4a((int)up_cpack0, qx_up, 0) * xd_up, up_norm0, up_sum0);
        up_sum1 = fmaf((float)ggml_cuda_dp4a((int)up_cpack1, qx_up, 0) * xd_up, up_norm1, up_sum1);
        gate_sum0 = fmaf((float)ggml_cuda_dp4a((int)gate_cpack0, qx_gate, 0) * xd_gate, gate_norm0, gate_sum0);
        gate_sum1 = fmaf((float)ggml_cuda_dp4a((int)gate_cpack1, qx_gate, 0) * xd_gate, gate_norm1, gate_sum1);
    }

    up_sum0 = warp_reduce_sum(up_sum0);
    up_sum1 = warp_reduce_sum(up_sum1);
    gate_sum0 = warp_reduce_sum(gate_sum0);
    gate_sum1 = warp_reduce_sum(gate_sum1);

    if (lane == 0) {
        if (glu_op == GGML_GLU_OP_SWIGLU) {
            dst[row0] = up_sum0 * (gate_sum0 / (1.0f + expf(-gate_sum0)));
            if (row1 < nrows) {
                dst[row1] = up_sum1 * (gate_sum1 / (1.0f + expf(-gate_sum1)));
            }
        } else if (glu_op == GGML_GLU_OP_GEGLU) {
            const float a = 0.044715f;
            const float b = 0.79788456080286535587989211986876f;
            const float gelu0 = 0.5f * gate_sum0 * (1.0f + tanhf(b * gate_sum0 * (1.0f + a * gate_sum0 * gate_sum0)));
            dst[row0] = up_sum0 * gelu0;
            if (row1 < nrows) {
                const float gelu1 = 0.5f * gate_sum1 * (1.0f + tanhf(b * gate_sum1 * (1.0f + a * gate_sum1 * gate_sum1)));
                dst[row1] = up_sum1 * gelu1;
            }
        } else if (glu_op == GGML_GLU_OP_SWIGLU_OAI) {
            const float alpha = 1.702f;
            const float limit = 7.0f;
            const float x0 = fminf(gate_sum0, limit);
            const float g0 = fmaxf(fminf(up_sum0, limit), -limit);
            dst[row0] = (x0 / (1.0f + expf(-x0 * alpha))) * (1.0f + g0);
            if (row1 < nrows) {
                const float x1 = fminf(gate_sum1, limit);
                const float g1 = fmaxf(fminf(up_sum1, limit), -limit);
                dst[row1] = (x1 / (1.0f + expf(-x1 * alpha))) * (1.0f + g1);
            }
        }
    }
}

template<int BLOCK_ROWS>
static __global__ void k_wq3_tcq_mmvq_q8_1_rowpair_cached(
        const block_wq3_tcq_i8_cache * __restrict__ vx_cache,
        const void  * __restrict__ vxrot_q8_1,
        float       * __restrict__ dst,
        const int ncols,
        const int nrows) {
    static_assert((BLOCK_ROWS % 2) == 0, "rowpair kernel requires even BLOCK_ROWS");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = blockIdx.x * BLOCK_ROWS + warp * 2 + 0;
    const int row1 = row0 + 1;

    const int n_groups = ncols / 128;
    const block_wq3_tcq_i8_cache * src_row0 =
        vx_cache + (int64_t)row0 * n_groups;
    const block_wq3_tcq_i8_cache * src_row1 =
        vx_cache + (int64_t)row1 * n_groups;
    const block_q8_1_mmq * xrot_q8_1 = (const block_q8_1_mmq *)vxrot_q8_1;

    const float cb_scale = s_codebook_scale_native;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    #pragma unroll 4
    for (int g = 0; g < n_groups; ++g) {
        const block_wq3_tcq_i8_cache * blk0 = src_row0 + g;
        const block_wq3_tcq_i8_cache * blk1 = src_row1 + g;
        const block_q8_1_mmq * xblk = xrot_q8_1 + g;

        const float norm0 = __half2float(__ushort_as_half(blk0->norm)) * cb_scale;
        const float norm1 = __half2float(__ushort_as_half(blk1->norm)) * cb_scale;

        const uint32_t cpack0 = *reinterpret_cast<const uint32_t *>(blk0->qs + lane * 4);
        const uint32_t cpack1 = *reinterpret_cast<const uint32_t *>(blk1->qs + lane * 4);

        const int qx = *reinterpret_cast<const int *>(xblk->qs + lane * 4);
        const float xd = xblk->d4[lane >> 3];

        const float acc0 = (float)ggml_cuda_dp4a((int)cpack0, qx, 0);
        const float acc1 = (float)ggml_cuda_dp4a((int)cpack1, qx, 0);

        sum0 = fmaf(acc0 * xd, norm0, sum0);
        sum1 = fmaf(acc1 * xd, norm1, sum1);
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    if (lane == 0) {
        dst[row0] = sum0;
        if (row1 < nrows) {
            dst[row1] = sum1;
        }
    }
}

static __device__ __forceinline__ uint32_t wq3_tcq_mul_u32(uint32_t x, uint32_t c) {
#if defined(__CUDA_ARCH__) && !defined(GGML_USE_HIP)
    // NVPTX-only fast path: mad.lo.u32 == x*c (low 32 bits). Under HIP, __CUDA_ARCH__
    // is defined too (vendors/hip.h), but this PTX is not valid amdgcn — fall through
    // to the portable multiply, which lowers to the same instruction on AMD.
    uint32_t r;
    asm ("mad.lo.u32 %0, %1, %2, 0;" : "=r"(r) : "r"(x), "r"(c));
    return r;
#else
    return x * c;
#endif
}

static __device__ __forceinline__ float wq3_tcq_norm_ppf_device(float p) {
    const float plow = 0.02425f;
    const float phigh = 1.0f - plow;

    if (p < plow) {
        const float q = sqrtf(-2.0f * logf(p));
        return (((((-7.784894e-03f * q - 3.2239646e-01f) * q - 2.4007583f) * q - 2.5497324f) * q + 4.3746643f) * q + 2.9381640f) /
               (((( 7.7846961e-03f * q + 3.2246712e-01f) * q + 2.4451342f) * q + 3.7544086f) * q + 1.0f);
    }
    if (p > phigh) {
        const float q = sqrtf(-2.0f * logf(1.0f - p));
        return -(((((-7.784894e-03f * q - 3.2239646e-01f) * q - 2.4007583f) * q - 2.5497324f) * q + 4.3746643f) * q + 2.9381640f) /
                 (((( 7.7846961e-03f * q + 3.2246712e-01f) * q + 2.4451342f) * q + 3.7544086f) * q + 1.0f);
    }

    const float q = p - 0.5f;
    const float r = q * q;
    return (((((-3.9696830e+01f * r + 2.2094609e+02f) * r - 2.7592851e+02f) * r + 1.3835775e+02f) * r - 3.0664799e+01f) * r + 2.5066283e+00f) * q /
           (((((-5.4476099e+01f * r + 1.6158583e+02f) * r - 1.5569898e+02f) * r + 6.6801315e+01f) * r - 1.3280682e+01f) * r + 1.0f);
}

static __device__ __forceinline__ int8_t wq3_tcq_proc_murmur_i8(uint32_t state) {
    uint32_t x = wq3_tcq_mul_u32(state, WQ3_TCQ_PROC_MURMUR_MUL) + WQ3_TCQ_PROC_MURMUR_ADD;
    x ^= x >> 16;
    x = wq3_tcq_mul_u32(x, WQ3_TCQ_PROC_MURMUR_MIX1);
    x ^= x >> 13;
    x = wq3_tcq_mul_u32(x, WQ3_TCQ_PROC_MURMUR_MIX2);
    x ^= x >> 16;

    const float u = ((float)x + 0.5f) * 2.3283064365386963e-10f;
    const float z = wq3_tcq_norm_ppf_device(fminf(fmaxf(u, 1e-9f), 1.0f - 1e-9f));
    int q = __float2int_rn(z * s_proc_qscale_native);
    if (q > 127)  q = 127;
    if (q < -127) q = -127;
    return (int8_t)q;
}

template <bool USE_U16>
static __device__ __forceinline__ uint32_t wq3_tcq_load_merged_words(
        const block_turboq3_tcq * blk,
        int lane,
        int pair_src_lo,
        int pair_src_hi,
        int pair_bit_lo,
        int pair_bit_hi,
        int w_idx_lo,
        int w_idx_hi) {
    if constexpr (USE_U16) {
        const uint16_t * words = reinterpret_cast<const uint16_t *>(blk);
        const uint32_t w_lo = words[w_idx_lo + 1];
        const uint32_t w_hi = words[w_idx_hi + 1];
        return (w_hi << 16) | w_lo;
    } else {
        const uint32_t * blk_u32 = reinterpret_cast<const uint32_t *>(blk);
        const uint32_t own_pair = (lane < 13) ? blk_u32[lane] : 0u;
        const uint32_t p_lo = __shfl_sync(0xFFFFFFFFu, own_pair, pair_src_lo, WARP_SIZE);
        const uint32_t p_hi = __shfl_sync(0xFFFFFFFFu, own_pair, pair_src_hi, WARP_SIZE);
        const uint32_t w_lo = (p_lo >> pair_bit_lo) & 0xFFFFu;
        const uint32_t w_hi = (p_hi >> pair_bit_hi) & 0xFFFFu;
        return (w_hi << 16) | w_lo;
    }
}

template <int BLOCK_ROWS, bool USE_U16>
static __global__ void k_wq3_tcq_mmvq_q8_1_rowpair_proc(
        const void  * __restrict__ vx,
        const void  * __restrict__ vxrot_q8_1,
        float       * __restrict__ dst,
        const int ncols,
        const int nrows) {
    static_assert((BLOCK_ROWS % 2) == 0, "rowpair kernel requires even BLOCK_ROWS");

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int row0 = blockIdx.x * BLOCK_ROWS + warp * 2 + 0;
    const int row1 = row0 + 1;

    const int n_groups = ncols / 128;
    const block_turboq3_tcq * src_row0 =
        ((const block_turboq3_tcq *)vx) + (int64_t)row0 * n_groups;
    const block_turboq3_tcq * src_row1 =
        ((const block_turboq3_tcq *)vx) + (int64_t)row1 * n_groups;
    const block_q8_1_mmq * xrot_q8_1 = (const block_q8_1_mmq *)vxrot_q8_1;

    const int bit_base  = lane * 12;
    const int w_idx_lo  = bit_base >> 4;
    const int w_idx_hi  = w_idx_lo + 1;
    const int shift     = bit_base & 15;
    const int pair_src_lo = (w_idx_lo + 1) >> 1;
    const int pair_bit_lo = ((w_idx_lo + 1) & 1) << 4;
    const int pair_src_hi = (w_idx_hi + 1) >> 1;
    const int pair_bit_hi = ((w_idx_hi + 1) & 1) << 4;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    #pragma unroll 4
    for (int g = 0; g < n_groups; ++g) {
        const block_turboq3_tcq * blk0 = src_row0 + g;
        const block_turboq3_tcq * blk1 = src_row1 + g;
        const block_q8_1_mmq * xblk = xrot_q8_1 + g;

        const float norm0 = __half2float(blk0->norm) * s_codebook_scale_native;
        const float norm1 = __half2float(blk1->norm) * s_codebook_scale_native;

        const uint32_t merged0 = wq3_tcq_load_merged_words<USE_U16>(
            blk0, lane, pair_src_lo, pair_src_hi, pair_bit_lo, pair_bit_hi, w_idx_lo, w_idx_hi);
        const uint32_t merged1 = wq3_tcq_load_merged_words<USE_U16>(
            blk1, lane, pair_src_lo, pair_src_hi, pair_bit_lo, pair_bit_hi, w_idx_lo, w_idx_hi);

        const int state00 = (merged0 >> (shift + 0)) & 0x3FF;
        const int state01 = (merged0 >> (shift + 3)) & 0x3FF;
        const int state02 = (merged0 >> (shift + 6)) & 0x3FF;
        const int state03 = (merged0 >> (shift + 9)) & 0x3FF;
        const int state10 = (merged1 >> (shift + 0)) & 0x3FF;
        const int state11 = (merged1 >> (shift + 3)) & 0x3FF;
        const int state12 = (merged1 >> (shift + 6)) & 0x3FF;
        const int state13 = (merged1 >> (shift + 9)) & 0x3FF;

        const uint32_t cpack0 =
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state00) << 0)  |
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state01) << 8)  |
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state02) << 16) |
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state03) << 24);
        const uint32_t cpack1 =
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state10) << 0)  |
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state11) << 8)  |
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state12) << 16) |
            ((uint32_t)(uint8_t)wq3_tcq_proc_murmur_i8((uint32_t)state13) << 24);

        const int qx = *reinterpret_cast<const int *>(xblk->qs + lane * 4);
        const float xd = xblk->d4[lane >> 3];

        const float acc0 = (float)ggml_cuda_dp4a((int)cpack0, qx, 0);
        const float acc1 = (float)ggml_cuda_dp4a((int)cpack1, qx, 0);

        sum0 = fmaf(acc0 * xd, norm0, sum0);
        sum1 = fmaf(acc1 * xd, norm1, sum1);
    }

    sum0 = warp_reduce_sum(sum0);
    sum1 = warp_reduce_sum(sum1);
    if (lane == 0) {
        dst[row0] = sum0;
        dst[row1] = sum1;
    }
}

// ── Stage MMQ: fused rotate + q8_1_mmq quantize (prefill path) ───────────────
//
// Produces xrot_q8_1 in the exact layout that MMQ consumes:
//   block_q8_1_mmq[ib], ib = group_idx * n_tokens + tok_idx
// Each block_q8_1_mmq is 128 int8 values + 4 fp32 sub-block scales (D4 layout).
// Replaces what quantize_mmq_q8_1<D4> would produce, but with the FWHT rotation
// baked in so the subsequent MMA against rotated-domain weights recovers the
// original-domain matmul.
//
// Grid: (n_groups, n_tokens). Block: 128 threads (one per element).

static __global__ void k_wq3_tcq_rotate_quantize_q8_1_mmq(
        const float * __restrict__ x,
        void        * __restrict__ vy,
        const int     ncols,
        const int     n_tokens,
        const int64_t x_stride_tok) {
    const int group_idx = blockIdx.x;
    const int tok_idx   = blockIdx.y;
    const int tid       = threadIdx.x;
    const int col       = group_idx * 128 + tid;

    const float * x_tok = x + (int64_t)tok_idx * x_stride_tok;

    __shared__ float smem[128];

    // FWHT rotation: v = signs2[i] * FWHT(signs1 * x)[i] / sqrt(128)
    float v = (col < ncols) ? x_tok[col] : 0.0f;
    v *= d_wq3_tcq_signs1[tid];
    v = wq3_fwht128(v, smem);
    constexpr float inv_sqrt_128 = 0.08838834764831845f;
    v *= inv_sqrt_128 * d_wq3_tcq_signs2[tid];

    // Warp-shuffle amax within each 32-thread sub-block (4 sub-blocks / 128 elems).
    float amax = fabsf(v);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, 32));
    }

    // Quantize: q = round(v * 127 / amax), clamped to [-127, 127].
    const float d_inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = (int)lrintf(v * d_inv);
    qi = max(-127, min(127, qi));

    block_q8_1_mmq * y = (block_q8_1_mmq *)vy;
    const int64_t ib = (int64_t)group_idx * n_tokens + tok_idx;

    y[ib].qs[tid] = (int8_t)qi;
    if ((tid & 31) == 0) {
        y[ib].d4[tid >> 5] = (d_inv > 0.0f) ? (1.0f / d_inv) : 0.0f;
    }
}

void ggml_cuda_wq3_tcq_rotate_quantize_q8_1_mmq(
        const float * x,
        void        * xrot_q8_1,
        int ncols,
        int n_tokens,
        int64_t x_stride_tok,
        cudaStream_t stream) {
    GGML_ASSERT(ncols % 128 == 0);
    GGML_ASSERT(d_wq3_tcq_signs_loaded && "WQ3_TCQ FWHT signs not loaded");
    const int n_groups = ncols / 128;
    dim3 grid(n_groups, n_tokens);
    k_wq3_tcq_rotate_quantize_q8_1_mmq<<<grid, 128, 0, stream>>>(
        x, xrot_q8_1, ncols, n_tokens, x_stride_tok);
}

// ── Native decode launchers ──────────────────────────────────────────────────
//
// Rotate-activation: one-shot over an N-length fp32 activation vector.
void ggml_cuda_wq3_tcq_rotate_activation_fp32(
        const float * x,
        float * xrot,
        int N,
        cudaStream_t stream) {
    GGML_ASSERT(N % 128 == 0);
    GGML_ASSERT(d_wq3_tcq_signs_loaded && "WQ3_TCQ FWHT signs not loaded");
    const int n_groups = N / 128;
    k_wq3_tcq_rotate_activation_fp32<<<n_groups, 128, 0, stream>>>(x, xrot, N);
}

static void wq3_tcq_mmvq_native_q8_1(
        const void * vx,
        const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache,
        const void * xrot_q8_1,
        float * dst,
        int ncols, int nrows, cudaStream_t stream) {
    // One warp per output row. The cached path is memory-pressure sensitive
    // after cache-line-aligned int8 loads; A100 sweep prefers BR=2 for large
    // matrices. Uncached/procedural paths keep the older BR=32/16/8/4 order.
    const bool use_proc = g_wq3_tcq_codebook_mode == WQ3_TCQ_CODEBOOK_PROC_MURMUR;
    const bool use_u16 = use_proc && wq3_tcq_use_proc_u16_layout();
    int cache_block_rows = decoded_cache != nullptr && nrows >= 2048 ? wq3_tcq_cached_block_rows_override() : 0;
    if (cache_block_rows != 0 && (nrows % cache_block_rows) != 0) {
        cache_block_rows = 0;
    }

    if ((nrows & 1) == 0 && nrows >= 2048 && (cache_block_rows == 2 || (decoded_cache != nullptr && cache_block_rows == 0))) {
        const int n_blocks = nrows / 2;
        const dim3 block(32, 1);
        if (decoded_cache != nullptr) {
            k_wq3_tcq_mmvq_q8_1_rowpair_cached<2><<<n_blocks, block, 0, stream>>>(
                decoded_cache->data, xrot_q8_1, dst, ncols, nrows);
        }
    } else if ((nrows & 63) == 0 && nrows >= 2048 && cache_block_rows == 64) {
        const int n_blocks = nrows / 64;
        const dim3 block(32, 32);
        if (decoded_cache != nullptr) {
            k_wq3_tcq_mmvq_q8_1_rowpair_cached<64><<<n_blocks, block, 0, stream>>>(
                decoded_cache->data, xrot_q8_1, dst, ncols, nrows);
        }
    } else if ((nrows & 31) == 0 && nrows >= 2048 && (cache_block_rows == 0 || cache_block_rows == 32)) {
        const int n_blocks = nrows / 32;
        const dim3 block(32, 16);
        if (decoded_cache != nullptr) {
            k_wq3_tcq_mmvq_q8_1_rowpair_cached<32><<<n_blocks, block, 0, stream>>>(
                decoded_cache->data, xrot_q8_1, dst, ncols, nrows);
        } else if (use_proc) {
            if (use_u16) {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<32, true><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            } else {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<32, false><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            }
        } else {
            k_wq3_tcq_mmvq_q8_1_rowpair<32><<<n_blocks, block, 0, stream>>>(
                vx, xrot_q8_1, dst, ncols, nrows);
        }
    } else if ((nrows & 15) == 0 && nrows >= 2048 && (cache_block_rows == 0 || cache_block_rows == 16)) {
        const int n_blocks = nrows / 16;
        const dim3 block(32, 8);
        if (decoded_cache != nullptr) {
            k_wq3_tcq_mmvq_q8_1_rowpair_cached<16><<<n_blocks, block, 0, stream>>>(
                decoded_cache->data, xrot_q8_1, dst, ncols, nrows);
        } else if (use_proc) {
            if (use_u16) {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<16, true><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            } else {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<16, false><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            }
        } else {
            k_wq3_tcq_mmvq_q8_1_rowpair<16><<<n_blocks, block, 0, stream>>>(
                vx, xrot_q8_1, dst, ncols, nrows);
        }
    } else if ((nrows & 7) == 0 && nrows >= 2048 && (cache_block_rows == 0 || cache_block_rows == 8)) {
        const int n_blocks = nrows / 8;
        const dim3 block(32, 4);
        if (decoded_cache != nullptr) {
            k_wq3_tcq_mmvq_q8_1_rowpair_cached<8><<<n_blocks, block, 0, stream>>>(
                decoded_cache->data, xrot_q8_1, dst, ncols, nrows);
        } else if (use_proc) {
            if (use_u16) {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<8, true><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            } else {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<8, false><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            }
        } else {
            k_wq3_tcq_mmvq_q8_1_rowpair<8><<<n_blocks, block, 0, stream>>>(
                vx, xrot_q8_1, dst, ncols, nrows);
        }
    } else if ((nrows & 3) == 0 && (cache_block_rows == 0 || cache_block_rows == 4)) {
        const int n_blocks = nrows / 4;
        const dim3 block(32, 2);
        if (decoded_cache != nullptr) {
            k_wq3_tcq_mmvq_q8_1_rowpair_cached<4><<<n_blocks, block, 0, stream>>>(
                decoded_cache->data, xrot_q8_1, dst, ncols, nrows);
        } else if (use_proc) {
            if (use_u16) {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<4, true><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            } else {
                k_wq3_tcq_mmvq_q8_1_rowpair_proc<4, false><<<n_blocks, block, 0, stream>>>(
                    vx, xrot_q8_1, dst, ncols, nrows);
            }
        } else {
            k_wq3_tcq_mmvq_q8_1_rowpair<4><<<n_blocks, block, 0, stream>>>(
                vx, xrot_q8_1, dst, ncols, nrows);
        }
    }
}

static __global__ void k_wq3_tcq_apply_glu(
        const float * __restrict__ gate,
        float * __restrict__ up_dst,
        int n,
        ggml_glu_op glu_op) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    const float gate_value = gate[i];
    const float up_value = up_dst[i];
    switch (glu_op) {
        case GGML_GLU_OP_SWIGLU:
            up_dst[i] = up_value * (gate_value / (1.0f + expf(-gate_value)));
            break;
        case GGML_GLU_OP_GEGLU: {
            const float gelu_coef_a = 0.044715f;
            const float sqrt_2_over_pi = 0.79788456080286535587989211986876f;
            const float gelu = 0.5f * gate_value *
                (1.0f + tanhf(sqrt_2_over_pi * gate_value * (1.0f + gelu_coef_a * gate_value * gate_value)));
            up_dst[i] = up_value * gelu;
            break;
        }
        case GGML_GLU_OP_SWIGLU_OAI: {
            const float alpha = 1.702f;
            const float limit = 7.0f;
            const float x = fminf(gate_value, limit);
            const float g = fmaxf(fminf(up_value, limit), -limit);
            up_dst[i] = (x / (1.0f + expf(-x * alpha))) * (1.0f + g);
            break;
        }
        default:
            break;
    }
}

// Native decode GEMV launcher. Pool-allocates xrot, runs rotate kernel,
// runs row-parallel FWHT-free trellis-decode GEMV. Wired into the batch=1
// WQ3_TCQ path from ggml_cuda_mul_mat.
void ggml_cuda_wq3_tcq_mmvq_native(
        ggml_cuda_pool & pool,
        const void * vx,
        const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache,
        const float * y, float * dst,
        int ncols, int nrows, cudaStream_t stream) {
    GGML_ASSERT(ncols % 128 == 0);
    GGML_ASSERT(d_wq3_tcq_codebook_loaded && "WQ3_TCQ codebook not loaded");
    GGML_ASSERT(d_wq3_tcq_signs_loaded   && "WQ3_TCQ FWHT signs not loaded");

    const int n_groups = ncols / 128;
    ggml_cuda_pool_alloc<block_q8_1_mmq> xrot_q8_1_buf(pool, n_groups);
    k_wq3_tcq_rotate_quantize_q8_1_mmq<<<n_groups, 128, 0, stream>>>(
        y, xrot_q8_1_buf.get(), ncols, /*n_tokens=*/1, /*x_stride_tok=*/ncols);

    if ((nrows & 3) == 0) {
        wq3_tcq_mmvq_native_q8_1(vx, decoded_cache, xrot_q8_1_buf.get(), dst, ncols, nrows, stream);
    } else {
        ggml_cuda_pool_alloc<float> xrot_buf(pool, ncols);
        k_wq3_tcq_rotate_activation_fp32<<<n_groups, 128, 0, stream>>>(y, xrot_buf.get(), ncols);
        const dim3 block(32, 1);
        k_wq3_tcq_mmvq_v2<1><<<nrows, block, 0, stream>>>(
            vx, xrot_buf.get(), dst, ncols, nrows);
    }
}

void ggml_cuda_wq3_tcq_mmvq_fused_gate_up_glu(
        ggml_cuda_pool & pool,
        const void * vx_up,
        const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache_up,
        const void * vx_gate,
        const ggml_cuda_wq3_tcq_decoded_cache * decoded_cache_gate,
        const float * y_up, const float * y_gate, float * dst,
        int ncols, int nrows, ggml_glu_op glu_op, cudaStream_t stream) {
    GGML_ASSERT(ncols % 128 == 0);
    GGML_ASSERT((nrows & 3) == 0);
    GGML_ASSERT(d_wq3_tcq_codebook_loaded && "WQ3_TCQ codebook not loaded");
    GGML_ASSERT(d_wq3_tcq_signs_loaded   && "WQ3_TCQ FWHT signs not loaded");

    static bool logged_fused = false;
    if (!logged_fused && getenv("GGML_CUDA_WQ3_FUSION_LOG") != nullptr) {
        logged_fused = true;
        fprintf(stderr, "WQ3_TCQ fusion: fused gate/up/GLU launcher active ncols=%d nrows=%d glu=%d cache_up=%d cache_gate=%d mode=%d\n",
            ncols, nrows, (int) glu_op, decoded_cache_up != nullptr, decoded_cache_gate != nullptr, (int) g_wq3_tcq_codebook_mode);
    }

    const int n_groups = ncols / 128;
    ggml_cuda_pool_alloc<block_q8_1_mmq> xrot_up_q8_1_buf(pool, n_groups);
    ggml_cuda_pool_alloc<block_q8_1_mmq> xrot_gate_q8_1_buf(pool, n_groups);
    k_wq3_tcq_rotate_quantize_q8_1_mmq<<<n_groups, 128, 0, stream>>>(
        y_up, xrot_up_q8_1_buf.get(), ncols, /*n_tokens=*/1, /*x_stride_tok=*/ncols);
    k_wq3_tcq_rotate_quantize_q8_1_mmq<<<n_groups, 128, 0, stream>>>(
        y_gate, xrot_gate_q8_1_buf.get(), ncols, /*n_tokens=*/1, /*x_stride_tok=*/ncols);

    if (decoded_cache_up == nullptr && decoded_cache_gate == nullptr &&
            g_wq3_tcq_codebook_mode == WQ3_TCQ_CODEBOOK_LUT) {
        static bool logged_dual = false;
        if (!logged_dual && getenv("GGML_CUDA_WQ3_FUSION_LOG") != nullptr) {
            logged_dual = true;
            fprintf(stderr, "WQ3_TCQ fusion: dual rowpair GLU kernel active\n");
        }
        if ((nrows & 31) == 0 && nrows >= 2048) {
            const int n_blocks = nrows / 32;
            const dim3 block(32, 16);
            k_wq3_tcq_mmvq_q8_1_dual_rowpair_glu<32><<<n_blocks, block, 0, stream>>>(
                vx_up, vx_gate, xrot_up_q8_1_buf.get(), xrot_gate_q8_1_buf.get(), dst, ncols, nrows, glu_op);
            return;
        }
        if ((nrows & 15) == 0 && nrows >= 2048) {
            const int n_blocks = nrows / 16;
            const dim3 block(32, 8);
            k_wq3_tcq_mmvq_q8_1_dual_rowpair_glu<16><<<n_blocks, block, 0, stream>>>(
                vx_up, vx_gate, xrot_up_q8_1_buf.get(), xrot_gate_q8_1_buf.get(), dst, ncols, nrows, glu_op);
            return;
        }
        if ((nrows & 7) == 0 && nrows >= 2048) {
            const int n_blocks = nrows / 8;
            const dim3 block(32, 4);
            k_wq3_tcq_mmvq_q8_1_dual_rowpair_glu<8><<<n_blocks, block, 0, stream>>>(
                vx_up, vx_gate, xrot_up_q8_1_buf.get(), xrot_gate_q8_1_buf.get(), dst, ncols, nrows, glu_op);
            return;
        }
        if ((nrows & 3) == 0) {
            const int n_blocks = nrows / 4;
            const dim3 block(32, 2);
            k_wq3_tcq_mmvq_q8_1_dual_rowpair_glu<4><<<n_blocks, block, 0, stream>>>(
                vx_up, vx_gate, xrot_up_q8_1_buf.get(), xrot_gate_q8_1_buf.get(), dst, ncols, nrows, glu_op);
            return;
        }
    }

    ggml_cuda_pool_alloc<float> gate_tmp(pool, nrows);
    wq3_tcq_mmvq_native_q8_1(vx_up,   decoded_cache_up,   xrot_up_q8_1_buf.get(),   dst,            ncols, nrows, stream);
    wq3_tcq_mmvq_native_q8_1(vx_gate, decoded_cache_gate, xrot_gate_q8_1_buf.get(), gate_tmp.get(), ncols, nrows, stream);

    const int block_size = 256;
    const int n_blocks = (nrows + block_size - 1) / block_size;
    k_wq3_tcq_apply_glu<<<n_blocks, block_size, 0, stream>>>(gate_tmp.get(), dst, nrows, glu_op);
}
