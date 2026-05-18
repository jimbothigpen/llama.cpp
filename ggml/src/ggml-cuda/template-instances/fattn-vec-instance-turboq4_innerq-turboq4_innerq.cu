// TURBOQ4_INNERQ K+V symmetric FA-vec kernel instances.
// Wire format is identical to TURBOQ4_0. TURBOQ4_INNERQ aliases the TURBOQ4_0
// encoder at SET_ROWS time (InnerQ pre-scaling hurts quality at 4-bit per ft2
// empirical data: PPL 9.08 with InnerQ vs 7.47 without, ccfe39d675).
#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBOQ4_INNERQ, GGML_TYPE_TURBOQ4_INNERQ);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBOQ4_INNERQ, GGML_TYPE_TURBOQ4_INNERQ);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBOQ4_INNERQ, GGML_TYPE_TURBOQ4_INNERQ);
