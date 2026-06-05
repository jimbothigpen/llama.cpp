// ===== TURBOQ8_0 SKELETON / DRAFT — NOT COMPILED, NOT WIRED INTO BUILD =====
// Native-VEC 8-bit TurboQuant KV codec. Port of buun TURBO8_0 (4ab44ae1c) adapted to
// OUR architecture: structural clone of turboq4_0 (turbo-quant.cuh:343-397), Lloyd-Max
// 256-centroid codebook (from buun CPU CENTROIDS_8BIT), group-WHT + per-block norm.
// We DISCARD buun's CUDA uniform-grid+absmax codec, fused-MMA tile loader, dequant-to-f16
// kernels and 39 MMA instance files (our tree has no fattn-mma-turbo path; HIP forces VEC).
// Full plan + risks + validation matrix: worker-scratch/.../RECON.md
//
// This file is a SCAFFOLD: data table is final; all functions are stubs with TODO anchors.
// To activate: fold these into turbo-quant.cuh + fattn-vec.cuh and wire the ~25 list sites
// (grep "GGML_TYPE_TURBOQ4_0" and mirror, skipping nibble-packing code). See RECON.md §5.
#pragma once
#if 0  // SKELETON GUARD — flip to enable once implemented & instances generated
#include "ggml-common.h"

// --- Lloyd-Max codebook for N(0,1/128): 256 monotone centroids (buun CENTROIDS_8BIT, exact) ---
static __constant__ float TURBO_CENTROIDS_8BIT[256] = {
    -0.34189706f, -0.29884648f, -0.27157635f, -0.25121314f, -0.23484838f, -0.22114745f, -0.20937878f, -0.19909346f,
    -0.19000012f, -0.18188631f, -0.17459596f, -0.16801505f, -0.16205003f, -0.15662252f, -0.15166721f, -0.14712896f,
    -0.14295785f, -0.13910911f, -0.13554055f, -0.13222068f, -0.12912571f, -0.12622918f, -0.12350196f, -0.12092574f,
    -0.11848717f, -0.11616507f, -0.11394363f, -0.11181760f, -0.10977627f, -0.10780649f, -0.10590563f, -0.10406567f,
    -0.10227606f, -0.10053416f, -0.09883731f, -0.09718554f, -0.09557616f, -0.09400389f, -0.09246874f, -0.09096270f,
    -0.08947787f, -0.08801425f, -0.08657184f, -0.08515064f, -0.08375065f, -0.08237188f, -0.08101431f, -0.07967795f,
    -0.07836547f, -0.07707417f, -0.07580144f, -0.07454460f, -0.07330103f, -0.07206804f, -0.07084568f, -0.06963391f,
    -0.06843275f, -0.06724219f, -0.06606225f, -0.06489290f, -0.06373417f, -0.06258603f, -0.06144851f, -0.06032158f,
    -0.05920527f, -0.05810222f, -0.05700977f, -0.05592792f, -0.05485668f, -0.05379604f, -0.05274602f, -0.05170659f,
    -0.05067778f, -0.04965957f, -0.04864930f, -0.04764700f, -0.04664999f, -0.04565829f, -0.04467189f, -0.04369080f,
    -0.04271500f, -0.04174452f, -0.04077933f, -0.03981945f, -0.03886486f, -0.03791293f, -0.03696631f, -0.03602498f,
    -0.03508631f, -0.03415029f, -0.03321957f, -0.03229416f, -0.03137139f, -0.03045128f, -0.02953382f, -0.02861901f,
    -0.02770950f, -0.02680265f, -0.02589845f, -0.02499689f, -0.02409534f, -0.02319644f, -0.02230284f, -0.02141190f,
    -0.02052095f, -0.01963266f, -0.01874701f, -0.01786137f, -0.01697838f, -0.01609804f, -0.01522035f, -0.01434266f,
    -0.01346497f, -0.01258993f, -0.01171755f, -0.01084516f, -0.00997278f, -0.00910304f, -0.00823331f, -0.00736357f,
    -0.00649649f, -0.00562941f, -0.00476233f, -0.00389524f, -0.00302816f, -0.00216373f, -0.00129930f, -0.00043487f,
     0.00043222f,  0.00129930f,  0.00216373f,  0.00302816f,  0.00389524f,  0.00476233f,  0.00562941f,  0.00649649f,
     0.00736357f,  0.00823331f,  0.00910304f,  0.00997278f,  0.01084516f,  0.01171755f,  0.01258993f,  0.01346497f,
     0.01434266f,  0.01522035f,  0.01609804f,  0.01697838f,  0.01786137f,  0.01874701f,  0.01963266f,  0.02052095f,
     0.02141190f,  0.02230284f,  0.02319644f,  0.02409534f,  0.02499689f,  0.02589845f,  0.02680265f,  0.02770950f,
     0.02861901f,  0.02953382f,  0.03045128f,  0.03137139f,  0.03229416f,  0.03321957f,  0.03415029f,  0.03508631f,
     0.03602498f,  0.03696631f,  0.03791293f,  0.03886486f,  0.03981945f,  0.04077933f,  0.04174452f,  0.04271500f,
     0.04369080f,  0.04467189f,  0.04565829f,  0.04664999f,  0.04764700f,  0.04864930f,  0.04965957f,  0.05067778f,
     0.05170659f,  0.05274602f,  0.05379604f,  0.05485668f,  0.05592792f,  0.05700977f,  0.05810222f,  0.05920527f,
     0.06032158f,  0.06144851f,  0.06258603f,  0.06373417f,  0.06489290f,  0.06606225f,  0.06724219f,  0.06843275f,
     0.06963391f,  0.07084568f,  0.07206804f,  0.07330103f,  0.07454460f,  0.07580144f,  0.07707417f,  0.07836547f,
     0.07967795f,  0.08101431f,  0.08237188f,  0.08375065f,  0.08515064f,  0.08657184f,  0.08801425f,  0.08947787f,
     0.09096270f,  0.09246874f,  0.09400389f,  0.09557616f,  0.09718554f,  0.09883731f,  0.10053416f,  0.10227606f,
     0.10406567f,  0.10590563f,  0.10780649f,  0.10977627f,  0.11181760f,  0.11394363f,  0.11616507f,  0.11848717f,
     0.12092574f,  0.12350196f,  0.12622918f,  0.12912571f,  0.13222068f,  0.13554055f,  0.13910911f,  0.14295785f,
     0.14712896f,  0.15166721f,  0.15662252f,  0.16205003f,  0.16801505f,  0.17459596f,  0.18188631f,  0.19000012f,
     0.19909346f,  0.20937878f,  0.22114745f,  0.23484838f,  0.25121314f,  0.27157635f,  0.29884648f,  0.34189706f,
};

// TODO(turboq8): TURBO_MID_8BIT[255] = midpoints for nearest-centroid bsearch
//   (buun MIDPOINTS_8BIT, ggml-turbo-quant.c) — drop in verbatim.

// --- set-rows encode: nearest-of-256 on ALREADY-ROTATED input (group-WHT done upstream) ---
// Mirror quantize_f32_turboq4_0_block (turbo-quant.cuh:382). norm set by set_rows wrapper.
static __device__ void quantize_f32_turboq8_0_block(const float * __restrict__ src,
                                                    block_turboq8_0 * __restrict__ dst) {
    // TODO(turboq8): for j in [0,128): dst->qs[j] = turbo_nearest_centroid_8bit(src[j]);
    //   bsearch over TURBO_MID_8BIT (256 levels). 1 byte/value, no packing.
    (void) src; (void) dst;
}

// --- VEC decode element: centroid * norm, in ROTATED domain (graph applies inverse WHT) ---
// Mirror turboq4_dequant_element (turbo-quant.cuh:394). 256 entries can't be register-cached
// like turboq4's sc[16]; read straight from __constant__ per element (coalesced by index).
static __device__ __forceinline__ float turboq8_dequant_element(
        const block_turboq8_0 * __restrict__ x, int j, float norm) {
    // TODO(turboq8): return TURBO_CENTROIDS_8BIT[x->qs[j]] * norm;
    (void) x; (void) j; (void) norm; return 0.0f;
}

// TODO(turboq8) fattn-vec.cuh: add type_K==TURBOQ8_0 K-scoring branch (Q·K rotated dot,
//   mirror turboq4 K path) and type_V==TURBOQ8_0 V-accum branch (per-element decode above).
// TODO(turboq8) set-rows.cu: set_rows_cuda_quant<block_turboq8_0,QK_TURBOQ8,quantize_f32_turboq8_0_block>
//   + add TURBOQ8_0 to extract_state==0, innerq_state==0, innerq_state==1 gate lists (match turboq4).
// TODO(turboq8) getrows.cu: get_rows_cuda_q<QK_TURBOQ8, QR_TURBOQ8(=2), turboq8 get-rows dequant>.
// TODO(turboq8) HOST ggml-turbo-quant.c: quantize_row_turboq8_0_ref / dequantize_row_turboq8_0 /
//   quantize_turboq8_0 (buun host code, renamed; reuse our matvec + turbo_rotation; CENTROIDS/MIDPOINTS).
// TODO(turboq8) traits: ggml.c type_traits + ggml_quantize_chunk case; ggml-cpu.c type_traits_cpu
//   (from_float=ref, vec_dot=NULL, vec_dot_type=F32); quants.h decl; ops.cpp clamp case.
// TODO(turboq8) fattn.cu: is_kv_compat / K->type switch (D%64, D<=512) / is_turbo_type / d_limit.
// TODO(turboq8) ggml-cuda.cu supports_op; llama-context/graph/kv-cache lists; arg.cpp kv_cache_types;
//   llama-bench ggml_type_from_name("turboq8").
// TODO(turboq8) generate 22 fattn-vec-instance-turboq8_0-*.cu (same pair-set as turboq4_0).
#endif // SKELETON GUARD
