// TURBOQ3_TCQ K + Q4_0 V CUDA flash attention vec kernel instantiation
// gfx1102 excluded: unspecified launch failure (register pressure / wave64 edge case)

#include "../fattn-vec.cuh"

#ifndef __gfx1102__
DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBOQ3_TCQ, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBOQ3_TCQ, GGML_TYPE_Q4_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBOQ3_TCQ, GGML_TYPE_Q4_0);
#endif
