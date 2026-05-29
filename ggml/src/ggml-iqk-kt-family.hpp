// IQ_KT trellis family — shared template infrastructure.
//
// Parameterizes the trellis weight-quant family over:
//   GROUP_SIZE : elements per codebook lookup group (4 for IQ4_KT; 8 for IQ2/3/1_KT)
//   NUM_BITS   : codebook index width in bits; kNumVal = 1 << NUM_BITS entries
//   IS_ABS     : true for absolute-value codebook; false (signed) for all current types
//
// Upstream QuantizerIQKT<block_size, group_size, num_bits, is_abs, is_int=true> instantiations:
//   IQ4_KT : <32, 4, 15, false, true>  — kNumVal=32768,  codebook A+B
//   IQ2_KT : <32, 8, 16, false, true>  — kNumVal=65536,  single codebook
//   IQ3_KT : <32, 8, 16, false, true>  — kNumVal=65536,  single codebook (signed output)
//   IQ1_KT : <32, 8, 13, false, true>  — kNumVal=8192,   single codebook
//
// This header provides:
//   - IQKTParams<...>                  compile-time parameter bundle + family aliases
//   - iqkt_gen_group_int<G, ABS>       deterministic codebook entry generator (is_int branch)
//   - IQKTCookedBook<G, N>             values array + soft-bin cluster index
//   - iqkt_build_cluster_index<G,N,A>  build soft-bin index for a CookedBook
//   - iqkt_cooked_book_init<G,N,A>     fill values + build index in one call
//   - iqkt_find_best_index<G,N,A>      cluster-accelerated NN search
//   - iqkt_find_best_index_brute<G,N>  brute-force NN search (final refinement)

#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

// ---------------------------------------------------------------------------
// Compile-time parameter bundles
// ---------------------------------------------------------------------------

template<int GROUP_SIZE_, int NUM_BITS_, bool IS_ABS_ = false>
struct IQKTParams {
    static constexpr int  kGroupSize = GROUP_SIZE_;
    static constexpr int  kNumBits   = NUM_BITS_;
    static constexpr bool kIsAbs     = IS_ABS_;
    static constexpr int  kNumVal    = 1 << NUM_BITS_;
};

// Family aliases matching upstream QuantizerIQKT instantiations.
using IQ4KTParams = IQKTParams<4, 15, false>;   // shipped
using IQ2KTParams = IQKTParams<8, 16, false>;   // P3a
using IQ3KTParams = IQKTParams<8, 16, false>;   // P3b
using IQ1KTParams = IQKTParams<8, 13, false>;   // P3c

// ---------------------------------------------------------------------------
// Codebook entry generator  (the is_int branch of upstream set_values)
//
// Produces GROUP_SIZE floats from a 32-bit index + offset using the
// 0xCBAC1FED multiplicative hash.  Each output is the sum of four 6-bit
// signed integers derived from one 32-bit state word, giving values in
// approximately [-126, +126].  IS_ABS takes std::abs() of each output.
// ---------------------------------------------------------------------------

template<int GROUP_SIZE, bool IS_ABS = false>
static inline void iqkt_gen_group_int(uint32_t idx, int offset, float * result) {
    constexpr uint32_t ka = 0xCBAC1FED;
    uint32_t x = idx + (uint32_t)offset;
    for (int k = 0; k < GROUP_SIZE; ++k) {
        x = ka * x;
        const uint32_t s  = x & 0x3f3f3f3f;
        const int8_t   i0 = (int8_t)( s        & 0xff);
        const int8_t   i1 = (int8_t)((s >>  8) & 0xff);
        const int8_t   i2 = (int8_t)((s >> 16) & 0xff);
        const int8_t   i3 = (int8_t)((s >> 24) & 0xff);
        const float val = (float)((int)i0 + (int)i1 + (int)i2 + (int)i3 - 126);
        result[k] = IS_ABS ? std::abs(val) : val;
    }
}

// ---------------------------------------------------------------------------
// Cooked codebook: values array + soft-binning cluster index
//
// The values array holds kNumVal × GROUP_SIZE floats (the full implicit codebook).
// bin_to_entries maps each bin to a list of candidate codebook indices — this is
// the soft-binning acceleration structure that makes NN-search fast.
// ---------------------------------------------------------------------------

template<int GROUP_SIZE, int NUM_BITS>
struct IQKTCookedBook {
    static constexpr int kNumVal    = 1 << NUM_BITS;
    static constexpr int kGroupSize = GROUP_SIZE;

    std::vector<float>             values;          // kNumVal × GROUP_SIZE floats
    std::vector<std::vector<int>>  bin_to_entries;  // per-bin candidate lists
};

// ---------------------------------------------------------------------------
// Binning helpers
//
// 5-bin step functions for one dimension, matching upstream bin5().
// The default (non-abs) breaks at -48/-16/+16/+48 (IQ4_KT range ≈ [-126,126]).
// The abs variant breaks at the corresponding positive thresholds.
// ---------------------------------------------------------------------------

static inline int iqkt_bin5_default(float x) {
    return x < -48.f ? 0 : x < -16.f ? 1 : x < 16.f ? 2 : x < 48.f ? 3 : 4;
}

static inline int iqkt_bin5_abs(float x) {
    return x < 11.2f ? 0 : x < 24.f ? 1 : x < 39.f ? 2 : x < 58.f ? 3 : 4;
}

// 3-bin step functions for GROUP_SIZE=8 (IQ2/3/1_KT).
// Thresholds at -16/+16 split the codebook range ≈ [-126, +126] into 3 parts.
static inline int iqkt_bin3_default(float x) {
    if (x < -16.f) return 0;
    if (x <  16.f) return 1;
    return 2;
}
static inline int iqkt_bin3_abs(float x) {
    const float ax = std::abs(x);
    if (ax < 16.f) return 0;
    if (ax < 48.f) return 1;
    return 2;
}

// Number of soft-bins for a given GROUP_SIZE.
//   GROUP_SIZE=4 → 5^4 = 625   (IQ4_KT)
//   GROUP_SIZE=8 → 3^8 = 6561  (IQ2/3/1_KT — full-8D base-3 hash)
//   Otherwise   → 1            (brute-force fallback)
template<int GROUP_SIZE>
static constexpr int iqkt_num_bins() {
    if constexpr (GROUP_SIZE == 4) return 625;
    if constexpr (GROUP_SIZE == 8) return 6561;  // 3^8 — all 8 dims via base-3
    return 1;
}

// Hash a GROUP_SIZE vector into a bin index.
//   GROUP_SIZE=4, non-abs → 5^4 grid using bin5_default
//   GROUP_SIZE=4, abs     → 5^4 grid using bin5_abs
//   GROUP_SIZE=8, non-abs → 3^8 grid using bin3_default (all 8 dims)
//   GROUP_SIZE=8, abs     → 3^8 grid using bin3_abs (all 8 dims)
//   Other                 → 0  (single-bin passthrough)
template<int GROUP_SIZE, bool IS_ABS>
static inline int iqkt_hash_bin(const float * v) {
    if constexpr (GROUP_SIZE == 4) {
        if constexpr (!IS_ABS) {
            return   iqkt_bin5_default(v[0])
                 + 5   * iqkt_bin5_default(v[1])
                 + 25  * iqkt_bin5_default(v[2])
                 + 125 * iqkt_bin5_default(v[3]);
        } else {
            return   iqkt_bin5_abs(v[0])
                 + 5   * iqkt_bin5_abs(v[1])
                 + 25  * iqkt_bin5_abs(v[2])
                 + 125 * iqkt_bin5_abs(v[3]);
        }
    }
    if constexpr (GROUP_SIZE == 8) {
        if constexpr (!IS_ABS) {
            return   iqkt_bin3_default(v[0])
                 + 3    * iqkt_bin3_default(v[1])
                 + 9    * iqkt_bin3_default(v[2])
                 + 27   * iqkt_bin3_default(v[3])
                 + 81   * iqkt_bin3_default(v[4])
                 + 243  * iqkt_bin3_default(v[5])
                 + 729  * iqkt_bin3_default(v[6])
                 + 2187 * iqkt_bin3_default(v[7]);
        } else {
            return   iqkt_bin3_abs(v[0])
                 + 3    * iqkt_bin3_abs(v[1])
                 + 9    * iqkt_bin3_abs(v[2])
                 + 27   * iqkt_bin3_abs(v[3])
                 + 81   * iqkt_bin3_abs(v[4])
                 + 243  * iqkt_bin3_abs(v[5])
                 + 729  * iqkt_bin3_abs(v[6])
                 + 2187 * iqkt_bin3_abs(v[7]);
        }
    }
    return 0;  // single-bin fallback
}

// ---------------------------------------------------------------------------
// Build soft-binning cluster index
//
// For GROUP_SIZE=4 (625 bins): runs Phase 1 (primary binning + centroid) +
// Phase 2 (k-nearest-bin soft assignment) — ~20M ops, runs once at init.
// For GROUP_SIZE!=4 (1 bin): passthrough — all entries in bin 0.
// k_neighbours: number of nearest bins each entry is registered into (e.g. 6).
// ---------------------------------------------------------------------------

template<int GROUP_SIZE, int NUM_BITS, bool IS_ABS>
static void iqkt_build_cluster_index(IQKTCookedBook<GROUP_SIZE, NUM_BITS> & cb, int k_neighbours) {
    constexpr int kNumVal   = 1 << NUM_BITS;
    constexpr int num_bins  = iqkt_num_bins<GROUP_SIZE>();
    const float * V = cb.values.data();

    cb.bin_to_entries.assign(num_bins, {});

    if constexpr (num_bins == 1) {
        // Single-bin: all entries in bin 0 → NN-search degenerates to brute force.
        cb.bin_to_entries[0].resize(kNumVal);
        std::iota(cb.bin_to_entries[0].begin(), cb.bin_to_entries[0].end(), 0);
        return;
    }

    // Phase 1: primary binning, compute bin centroids.
    std::vector<int>   bin_count(num_bins, 0);
    std::vector<float> bin_sum((size_t)num_bins * GROUP_SIZE, 0.f);
    for (int i = 0; i < kNumVal; ++i) {
        const float * v = V + (size_t)i * GROUP_SIZE;
        const int b = iqkt_hash_bin<GROUP_SIZE, IS_ABS>(v);
        bin_count[b]++;
        for (int k = 0; k < GROUP_SIZE; ++k) bin_sum[b * GROUP_SIZE + k] += v[k];
    }
    std::vector<float> centroid((size_t)num_bins * GROUP_SIZE, 0.f);
    for (int b = 0; b < num_bins; ++b) {
        if (bin_count[b] > 0) {
            for (int k = 0; k < GROUP_SIZE; ++k) {
                centroid[b * GROUP_SIZE + k] = bin_sum[b * GROUP_SIZE + k] / bin_count[b];
            }
        }
    }

    // Phase 2: for each entry, find k_neighbours nearest non-empty bin centroids
    // and register the entry into each of those bins (soft/overlapping assignment).
    // This fixes boundary misses — a query hashing to bin X can find entries
    // whose primary bin is adjacent to X.
    std::vector<float> best_d(k_neighbours);
    std::vector<int>   best_b(k_neighbours);
    for (int i = 0; i < kNumVal; ++i) {
        const float * v = V + (size_t)i * GROUP_SIZE;
        std::fill(best_d.begin(), best_d.end(), INFINITY);
        std::fill(best_b.begin(), best_b.end(), -1);
        for (int b = 0; b < num_bins; ++b) {
            if (bin_count[b] == 0) continue;
            const float * c = centroid.data() + (size_t)b * GROUP_SIZE;
            float dist = 0;
            for (int k = 0; k < GROUP_SIZE; ++k) {
                const float d = v[k] - c[k]; dist += d * d;
            }
            for (int j = 0; j < k_neighbours; ++j) {
                if (dist < best_d[j]) {
                    for (int kk = k_neighbours - 1; kk > j; --kk) {
                        best_d[kk] = best_d[kk - 1]; best_b[kk] = best_b[kk - 1];
                    }
                    best_d[j] = dist; best_b[j] = b;
                    break;
                }
            }
        }
        for (int j = 0; j < k_neighbours; ++j) {
            if (best_b[j] >= 0) cb.bin_to_entries[best_b[j]].push_back(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Initialize a CookedBook: generate all codebook entries then build index.
// ---------------------------------------------------------------------------

template<int GROUP_SIZE, int NUM_BITS, bool IS_ABS>
static void iqkt_cooked_book_init(IQKTCookedBook<GROUP_SIZE, NUM_BITS> & cb,
                                  int offset, int k_neighbours) {
    constexpr int kNumVal = 1 << NUM_BITS;
    cb.values.resize((size_t)kNumVal * GROUP_SIZE);
    for (int i = 0; i < kNumVal; ++i) {
        iqkt_gen_group_int<GROUP_SIZE, IS_ABS>(
            (uint32_t)i, offset, cb.values.data() + (size_t)i * GROUP_SIZE);
    }
    iqkt_build_cluster_index<GROUP_SIZE, NUM_BITS, IS_ABS>(cb, k_neighbours);
}

// ---------------------------------------------------------------------------
// Cluster-accelerated nearest-neighbour search
//
// Hashes the id-scaled query into a bin, then scans only the candidates in
// that bin (~kNumVal × k_neighbours / num_bins entries average).
// Falls back to brute force if the bin is empty (rare).
// ---------------------------------------------------------------------------

template<int GROUP_SIZE, int NUM_BITS, bool IS_ABS>
static int iqkt_find_best_index(const float * xb, const float * weight, float d,
                                const IQKTCookedBook<GROUP_SIZE, NUM_BITS> & cb) {
    constexpr int kNumVal = 1 << NUM_BITS;
    const float id = d != 0.f ? 1.f / d : 0.f;
    float xs[GROUP_SIZE];
    for (int k = 0; k < GROUP_SIZE; ++k) xs[k] = id * xb[k];

    const int b = iqkt_hash_bin<GROUP_SIZE, IS_ABS>(xs);
    const auto & candidates = cb.bin_to_entries[b];
    const float * V = cb.values.data();

    if (!candidates.empty()) {
        float best_dist = INFINITY;
        int   best_i    = candidates[0];
        for (int e : candidates) {
            const float * v = V + (size_t)e * GROUP_SIZE;
            float dist = 0;
            for (int k = 0; k < GROUP_SIZE; ++k) {
                const float diff = v[k] - xs[k]; dist += weight[k] * diff * diff;
            }
            if (dist < best_dist) { best_dist = dist; best_i = e; }
        }
        return best_i;
    }

    // Fallback: brute force (empty bin — very rare with soft-binning).
    float best_dist = INFINITY;
    int   best_i    = 0;
    for (int i = 0; i < kNumVal; ++i) {
        const float * v = V + (size_t)i * GROUP_SIZE;
        float dist = 0;
        for (int k = 0; k < GROUP_SIZE; ++k) {
            const float diff = v[k] - xs[k]; dist += weight[k] * diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_i = i; }
    }
    return best_i;
}

// ---------------------------------------------------------------------------
// Brute-force NN search (no clustering)
// Used for final per-group refinement with the committed scale value.
// ---------------------------------------------------------------------------

template<int GROUP_SIZE, int NUM_BITS>
static int iqkt_find_best_index_brute(const float * xb, const float * weight, float d,
                                      const float * codebook) {
    constexpr int kNumVal = 1 << NUM_BITS;
    const float id = d != 0.f ? 1.f / d : 0.f;
    float xs[GROUP_SIZE];
    for (int k = 0; k < GROUP_SIZE; ++k) xs[k] = id * xb[k];
    float best_dist = INFINITY;
    int   best_i    = 0;
    for (int i = 0; i < kNumVal; ++i) {
        const float * v = codebook + (size_t)i * GROUP_SIZE;
        float dist = 0;
        for (int k = 0; k < GROUP_SIZE; ++k) {
            const float diff = v[k] - xs[k]; dist += weight[k] * diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_i = i; }
    }
    return best_i;
}
