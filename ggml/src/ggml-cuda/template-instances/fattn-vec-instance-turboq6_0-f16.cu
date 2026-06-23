// TurboQuant5/6 CUDA flash attention vec kernel instantiation
// This file is hand-maintained (turbo types are excluded from generate_cu_files.py). Invented ygg TODO 250.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBOQ6_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBOQ6_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBOQ6_0, GGML_TYPE_F16);
DECL_FATTN_VEC_CASE(512, GGML_TYPE_TURBOQ6_0, GGML_TYPE_F16);
