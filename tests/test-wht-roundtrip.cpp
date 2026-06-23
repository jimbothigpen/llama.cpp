// Round-trip regression guard for the WHT weight quants (WHT5_0/WHT6_0/WHT8_0).
//
// Background: the canonical Qwen3.5-9B WHT6_0/WHT8_0 test GGUFs were once produced by a
// pre-commit WIP quantizer and dequantized to garbage (PPL ~1020 / ~512k). The committed
// quantizer + dequantizer were correct; the files were stale. This test pins that the
// committed production quantize path (ggml_quantize_chunk) followed by to_float reconstructs
// a Gaussian source to a healthy, monotonically-improving error per bit-width — so a future
// regression in the WHT6/8 packing, codebook, or RHT is caught directly here (the generic
// test-quantize-fns currently aborts on an unrelated KV codec type before it reaches WHT).

#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double roundtrip_nrmse(ggml_type t, const std::vector<float> & src, int64_t nrows, int64_t ncol) {
    const ggml_type_traits * tr = ggml_get_type_traits(t);
    const size_t row_size = (size_t)(ncol / tr->blck_size) * tr->type_size;
    std::vector<char>  q(row_size * nrows);
    std::vector<float> deq((size_t)nrows * ncol);
    ggml_quantize_chunk(t, src.data(), q.data(), 0, nrows, ncol, nullptr);
    for (int64_t r = 0; r < nrows; r++) {
        tr->to_float(q.data() + r * row_size, deq.data() + r * ncol, ncol);
    }
    double se = 0.0, sx = 0.0;
    for (size_t i = 0; i < deq.size(); i++) {
        const double e = (double)deq[i] - (double)src[i];
        se += e * e;
        sx += (double)src[i] * (double)src[i];
    }
    const double nrmse = std::sqrt(se / sx);
    printf("  %-7s blck=%2lld tsz=%2lld  NRMSE=%.4f\n", ggml_type_name(t),
           (long long)tr->blck_size, (long long)tr->type_size, nrmse);
    return nrmse;
}

int main(void) {
    const int64_t nrows = 64, ncol = 4096; // ncol % 32 == 0 (WHT block size)
    std::vector<float> src((size_t)nrows * ncol);
    // Deterministic Gaussian N(0,1) (Box-Muller over a fixed-seed LCG) — WHT codebooks are
    // Lloyd-Max optimal for N(0,1), so this is the in-distribution case.
    unsigned int s = 12345u;
    for (size_t i = 0; i < src.size(); i += 2) {
        s = s * 1103515245u + 12345u; double u1 = ((s >> 8) & 0xFFFFFF) / 16777216.0 + 1e-9;
        s = s * 1103515245u + 12345u; double u2 = ((s >> 8) & 0xFFFFFF) / 16777216.0;
        const double r = std::sqrt(-2.0 * std::log(u1));
        src[i] = (float)(r * std::cos(2 * M_PI * u2));
        if (i + 1 < src.size()) src[i + 1] = (float)(r * std::sin(2 * M_PI * u2));
    }

    printf("WHT round-trip (Gaussian N(0,1), %lldx%lld):\n", (long long)nrows, (long long)ncol);
    const double n5 = roundtrip_nrmse(GGML_TYPE_WHT5_0, src, nrows, ncol);
    const double n6 = roundtrip_nrmse(GGML_TYPE_WHT6_0, src, nrows, ncol);
    const double n8 = roundtrip_nrmse(GGML_TYPE_WHT8_0, src, nrows, ncol);

    int failed = 0;
    // Absolute ceilings (healthy reference ~0.037 / 0.019 / 0.005; generous margins guard against
    // catastrophic regressions like the stale-GGUF garbage without being flaky).
    if (!(n5 < 0.06)) { printf("FAIL WHT5_0 NRMSE %.4f >= 0.06\n", n5); failed++; }
    if (!(n6 < 0.04)) { printf("FAIL WHT6_0 NRMSE %.4f >= 0.04\n", n6); failed++; }
    if (!(n8 < 0.02)) { printf("FAIL WHT8_0 NRMSE %.4f >= 0.02\n", n8); failed++; }
    // More bits must reconstruct strictly better.
    if (!(n5 > n6 && n6 > n8)) { printf("FAIL non-monotonic: %.4f %.4f %.4f\n", n5, n6, n8); failed++; }

    printf("%s\n", failed ? "FAILED" : "PASSED");
    return failed ? 1 : 0;
}
