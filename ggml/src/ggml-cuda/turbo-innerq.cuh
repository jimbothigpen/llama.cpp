#pragma once

// TurboQuant InnerQ per-channel equalization — cross-TU shared state
// The host-side state lives in turbo-innerq.cu; device-side state is per-TU
// in turbo-quant.cuh (only set-rows.cu needs device access).

// 256 to support the calibrated OSCAR INT2 K path on head_dim==256 models (OSCAR's primary target,
// e.g. Qwen3.5-9B: key_length=256). Turbo InnerQ only ever indexes scale_inv[t % group_size] with
// group_size<=128 (turbo-wht.cu), so the extra capacity is inert for the Turbo path. All InnerQ
// finalize/publish/copy sites size by the dynamic group_size, not this constant.
#define INNERQ_MAX_CHANNELS 256

#ifdef GGML_BACKEND_SHARED
#  if defined(_WIN32) && !defined(__MINGW32__)
#    ifdef GGML_BACKEND_BUILD
#      define TURBO_IQ_API __declspec(dllexport)
#    else
#      define TURBO_IQ_API __declspec(dllimport)
#    endif
#  else
#    define TURBO_IQ_API __attribute__((visibility("default")))
#  endif
#else
#  define TURBO_IQ_API
#endif

// Host-side shared state (defined in turbo-innerq.cu)
TURBO_IQ_API extern bool  g_innerq_finalized;
TURBO_IQ_API extern float g_innerq_scale_inv_host[INNERQ_MAX_CHANNELS];

// Called from set-rows.cu after InnerQ finalization to publish scale_inv
void turbo_innerq_publish(const float * scale_inv, int group_size);

// Called from llama-kv-cache.cpp (or equivalent) to check if tensor needs update
TURBO_IQ_API bool turbo_innerq_needs_tensor_update(void);

// Called after tensor update to clear the flag
TURBO_IQ_API void turbo_innerq_mark_tensor_updated(void);
