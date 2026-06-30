// Hand-created turbo MMA decode instance. Do NOT run generate_cu_files.py over it.

#include "../fattn-mma-f16.cuh"
#include "../fattn-mma-turbo.cuh"

DECL_FATTN_MMA_TURBO_CASE(128, 128, 4, 4, GGML_TYPE_TURBOQ4_0, GGML_TYPE_TURBOQ4_0);
DECL_FATTN_MMA_TURBO_CASE(128, 128, 4, 4, GGML_TYPE_TURBOQ3_0, GGML_TYPE_TURBOQ3_0);
DECL_FATTN_MMA_TURBO_CASE(128, 128, 4, 4, GGML_TYPE_TURBOQ2_0, GGML_TYPE_TURBOQ2_0);
