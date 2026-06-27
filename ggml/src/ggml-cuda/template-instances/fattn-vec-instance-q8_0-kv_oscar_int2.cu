// Mixed KV: q8_0 K + OScaR INT2 V — asymmetric matrix bpw(K) >= bpw(V) (D=128, D=256)

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_KV_OSCAR_INT2);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_Q8_0, GGML_TYPE_KV_OSCAR_INT2);
