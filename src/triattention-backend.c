/*
 * triattention-backend.c — resolve GPU acceleration at init time
 *
 * Exactly one GPU backend is compiled in (per the build): TRIA_HIP_BACKEND for
 * ROCm/HIP, or TRIA_VULKAN_BACKEND for Vulkan. Both implement the same
 * g_tria_backend function-pointer table; the runtime is backend-agnostic.
 */
#include "triattention-backend.h"
#include <string.h>

struct tria_backend g_tria_backend;

#if defined(TRIA_HIP_BACKEND)
/* Phase B GPU scoring — requires triattention-hip.hip + TRIA_HIP_BACKEND define.
   Phase A uses CPU-backed capture only; this path is intentionally disabled. */
#include "triattention-hip.h"

int tria_backend_init(void) {
    g_tria_backend.stats_upload    = tria_hip_stats_upload;
    g_tria_backend.stats_free      = tria_hip_stats_free;
    g_tria_backend.score_q8_0      = tria_hip_score_q8_0;
    g_tria_backend.scores_download = tria_hip_scores_download;
    g_tria_backend.compact_rows    = tria_hip_compact_rows;
    return 1;
}

#elif defined(TRIA_CUDA_BACKEND)
/* Phase C GPU scoring on CUDA (T4 / sm_75) — requires triattention-cuda.cu +
   TRIA_CUDA_BACKEND define. Same g_tria_backend table as HIP; the raw-score
   kernel is register-frugal so it does not spill on sm_75. Falls back to CPU
   only if device ops fail at runtime. */
#include "triattention-cuda.h"

int tria_backend_init(void) {
    g_tria_backend.stats_upload    = tria_cuda_stats_upload;
    g_tria_backend.stats_free      = tria_cuda_stats_free;
    g_tria_backend.score_q8_0      = tria_cuda_score_q8_0;
    g_tria_backend.scores_download = tria_cuda_scores_download;
    g_tria_backend.compact_rows    = tria_cuda_compact_rows;
    return 1;
}

#elif defined(TRIA_VULKAN_BACKEND)
/* Phase C GPU scoring on Vulkan — requires triattention-vulkan.cpp +
   TRIA_VULKAN_BACKEND define. Self-contained Vulkan compute context (its own
   device/pipelines); falls back to CPU at runtime if device init fails. */
#include "triattention-vulkan.h"

int tria_backend_init(void) {
    g_tria_backend.stats_upload    = tria_vk_stats_upload;
    g_tria_backend.stats_free      = tria_vk_stats_free;
    g_tria_backend.score_q8_0      = tria_vk_score_q8_0;
    g_tria_backend.scores_download = tria_vk_scores_download;
    g_tria_backend.compact_rows    = tria_vk_compact_rows;
    return 1;
}

#else
/* Dynamic loading or no HIP — CPU fallback (no GPU scoring) */

int tria_backend_init(void) {
    memset(&g_tria_backend, 0, sizeof(g_tria_backend));
    return 0;
}

#endif
