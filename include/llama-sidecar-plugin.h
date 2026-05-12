#pragma once

#include "llama-sidecar.h"

//
// Sidecar plugin loader contract
//
// An out-of-tree sidecar handler plugin is a shared library (.so on Linux)
// that:
//
//   1. Includes <llama-sidecar.h> and subclasses `llama_sidecar_handler`,
//      overriding any of the optional graph-build hooks (apply_to,
//      apply_to_expert, post_compute_logits) it needs.
//   2. Exports a single `extern "C"` symbol `llama_sidecar_plugin_init()`
//      that registers each of its handlers with the engine via
//      `llama_sidecar_register("type-tag", factory)`.
//   3. Returns 0 on success, non-zero on failure (logged + plugin rejected).
//
// The host engine loads the plugin with --sidecar-load-plugin <path.so>
// (repeatable). dlopen() happens before any sidecar GGUF is opened, so the
// plugin's registered types are available for type-discrimination at sidecar
// load time.
//
// ABI stability promise: the five virtuals on llama_sidecar_handler
// (type + load + apply_to + apply_to_expert + post_compute_logits) are the
// FROZEN base-class ABI and will not change without a major version bump.
// In particular, NO new virtual will ever be added directly to
// llama_sidecar_handler — doing so silently breaks every old plugin .so
// because the plugin's compiled-in derived-class vtable still has the old
// slot count, so engine calls to the new slot read garbage / segfault.
//
// New hook points are instead added as separate OPTIONAL interfaces (see
// llama_sidecar_handler_weights in <llama-sidecar.h> for the canonical
// example). The engine probes for each via dynamic_cast at the dispatch
// site, so:
//   - Plugins built against an old <llama-sidecar.h> that only inherit
//     llama_sidecar_handler keep working — they're transparently skipped
//     for any optional interface they don't implement.
//   - Plugins that want a new hook multiply-inherit from the optional
//     interface; only those plugins need rebuilding when the interface
//     itself changes.
//
// Build flags for plugin authors:
//   - Compile with C++17 or later, position-independent code.
//   - Link against libllama.so (and ggml + ggml-base which it depends on).
//   - Add the engine's installed include directory to the compiler include
//     path (e.g. -I/opt/llama-yggdrasil-vulkan/include).
//
// Skeleton:
//
//     #include <llama-sidecar-plugin.h>
//
//     struct my_handler : public llama_sidecar_handler {
//         std::string type() const override { return "my_handler"; }
//         bool load(...) override { ... }
//         ggml_tensor * apply_to(...) const override { ... }
//     };
//
//     extern "C" int llama_sidecar_plugin_init(void) {
//         llama_sidecar_register(
//             "my_handler",
//             []() -> llama_sidecar_handler_ptr {
//                 return std::make_shared<my_handler>();
//             });
//         return 0;
//     }
//

#ifdef __cplusplus
extern "C" {
#endif

// Plugin entry point. Implemented in each plugin .so (NOT in the engine).
// Engine dlsym()'s this symbol immediately after dlopen() and calls it once
// per plugin path.
//
// The plugin must export the symbol with default visibility, since the
// engine compiles with -fvisibility=hidden by default. Plugin authors should
// either compile their .so with -fvisibility=default, or annotate the
// definition explicitly:
//
//     extern "C" __attribute__((visibility("default")))
//     int llama_sidecar_plugin_init(void) { ... }
//
// Return 0 on success. Any non-zero value is treated as initialization
// failure; the engine logs it and refuses to use the plugin's registrations
// (though they may have already entered the registry — see plugin authoring
// notes).
int llama_sidecar_plugin_init(void);

// Convenience macro for the common visibility-default case.
#define LLAMA_SIDECAR_PLUGIN_INIT_DECL \
    extern "C" __attribute__((visibility("default"))) \
    int llama_sidecar_plugin_init(void)

#ifdef __cplusplus
} // extern "C"
#endif
