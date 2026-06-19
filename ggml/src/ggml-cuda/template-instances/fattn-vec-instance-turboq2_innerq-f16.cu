// TURBOQ2_INNERQ K + F16 V CUDA flash attention vec kernel instances.
// Phase X-5 (TODO 249): the K-only InnerQ config (innerq K cache, plain f16 V)
// is the clean A/B vehicle for proving/retiring the standalone InnerQ types.
// Wire format is identical to TURBOQ2_0; the InnerQ per-channel scale_inv is
// applied entirely at the graph level (encode-side K pre-scale in set-rows.cu +
// Q-rotation compensation via ggml_turbo_wht_innerq in llama-graph.cpp), so this
// FA-vec kernel is byte-for-byte the TURBOQ2_0×F16 kernel.
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBOQ2_INNERQ, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBOQ2_INNERQ, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBOQ2_INNERQ, GGML_TYPE_F16);
