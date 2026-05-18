// RotorQuant: IsoQuant 4-bit FA V-dequant template instances (K=f16, V=rq_iso4_0)
// Source: carlosfundora @85ba5b945 fattn-vec-instance-f16-iso4_0.cu, renamed for ygg type IDs.

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_F16, GGML_TYPE_RQ_ISO4_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_F16, GGML_TYPE_RQ_ISO4_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_F16, GGML_TYPE_RQ_ISO4_0);
