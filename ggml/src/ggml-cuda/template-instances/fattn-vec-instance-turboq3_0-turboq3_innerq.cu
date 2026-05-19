// TURBOQ3_0 K + TURBOQ3_INNERQ V CUDA flash attention vec kernel instances.
// Wire format is identical to TURBOQ3_0; InnerQ scale_inv is applied at the
// graph level via ggml_turbo_wht (Phase X-4 will pass scale tensor).
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_INNERQ);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_INNERQ);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_INNERQ);
