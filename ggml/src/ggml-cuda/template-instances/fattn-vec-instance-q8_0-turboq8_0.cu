// TurboQuant8 CUDA flash attention vec kernel instantiation
// This file is hand-maintained (turbo types are excluded from generate_cu_files.py).

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_Q8_0, GGML_TYPE_TURBOQ8_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_TURBOQ8_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_TURBOQ8_0);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_Q8_0, GGML_TYPE_TURBOQ8_0);
