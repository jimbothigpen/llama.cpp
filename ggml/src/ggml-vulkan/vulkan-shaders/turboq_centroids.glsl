// turboq_centroids.glsl — shared TURBOQ PolarQuant Lloyd-Max centroid arrays.
// Single source of truth used by flash_attn_dequant.glsl, dequant_funcs.glsl,
// copy_to_quant.comp, and dequant_turboq{2,3,4}_0.comp. Matches CENTROIDS_2BIT
// / CENTROIDS_3BIT / CENTROIDS_4BIT in ggml-turbo-quant.c.

#ifndef TURBOQ_CENTROIDS_GLSL
#define TURBOQ_CENTROIDS_GLSL

const float TURBOQ2_CENTROIDS[4] = float[4](
    -0.133462, -0.039994, 0.039994, 0.133462
);

const float TURBOQ3_CENTROIDS[8] = float[8](
    -0.190685, -0.117832, -0.065717, -0.021460,
     0.021460,  0.065717,  0.117832,  0.190685
);

const float TURBOQ4_CENTROIDS[16] = float[16](
    -0.173926, -0.117195, -0.089527, -0.068756,
    -0.051262, -0.035597, -0.020989, -0.006938,
     0.006938,  0.020989,  0.035597,  0.051262,
     0.068756,  0.089527,  0.117195,  0.173926
);

#endif
