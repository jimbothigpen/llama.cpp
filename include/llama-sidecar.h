#pragma once

#include "llama.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_context;
struct llama_adapter_lora;
struct gguf_context;

//
// Sidecar plugin architecture
//
// A sidecar is a small GGUF carrying a runtime adapter that hooks into the
// model's forward graph at well-defined sites (residual stream, MoE expert
// outputs, etc.). Concrete sidecar types implement `llama_sidecar_handler`.
//
// On disk, a sidecar GGUF carries a `sidecar.type` string KV that identifies
// which handler to dispatch to. The KV is REQUIRED — producers must tag each
// sidecar with its handler type. The engine rejects sidecars without it.
//
// To add a new sidecar type:
//   1. Create src/sidecar/<type>.h + src/sidecar/<type>.cpp
//   2. Subclass llama_sidecar_handler, override the hook points you need
//   3. Use LLAMA_SIDECAR_REGISTER("<type>", YourClass) at file scope
//   4. Add the .cpp to src/CMakeLists.txt
//
// No changes elsewhere in the engine are required to add a handler.
//

struct llama_sidecar_handler {
    virtual ~llama_sidecar_handler() = default;

    // Type tag matching the sidecar GGUF's `sidecar.type` KV. Must be unique
    // per handler family.
    virtual std::string type() const = 0;

    // Load handler-specific state from an opened sidecar GGUF. Called once
    // per llama_context at attachment time. Handlers read their own KVs and
    // tensor data from `gguf` / `ctx_meta`. The optional `scale_override` and
    // `threshold_override` carry user-provided CLI args; handlers ignore
    // arguments that don't apply to their family.
    //
    // Return false on parse error; the engine will reject the sidecar.
    virtual bool load(
            const llama_model & model,
            gguf_context * gguf,
            ggml_context * ctx_meta,
            const std::string & path,
            float scale_override,
            float threshold_override) = 0;

    // ---- Optional graph-build hooks (default to no-op) ----

    // Hook at the residual-stream output of transformer layer `il`. Called by
    // llm_graph_context::build_sidecar() once per layer. Handlers that don't
    // need this hook can leave it as the default and only override the hooks
    // they care about.
    virtual ggml_tensor * apply_to(
            ggml_context * ctx,
            ggml_tensor  * cur,
            int            il) const {
        (void) ctx;
        (void) il;
        return cur;
    }

    // Hook inside MoE forward, on each layer's [n_embd, n_expert_used,
    // n_tokens] expert-output tensor BEFORE gating-weight mixing. Called by
    // llm_graph_context::build_sidecar_expert(). The `selected_experts`
    // [n_expert_used, n_tokens] tensor lets handlers gather per-active-expert
    // state via ggml_get_rows.
    virtual ggml_tensor * apply_to_expert(
            ggml_context * ctx,
            ggml_tensor  * experts,
            ggml_tensor  * selected_experts,
            int            il) const {
        (void) ctx;
        (void) selected_experts;
        (void) il;
        return experts;
    }

    // Hook called once-per-batch on the host-side logits buffer, after the
    // backend has finished writing it (post-synchronize) but before the user
    // reads it via llama_get_logits / llama_get_logits_ith. The buffer is
    // n_tokens * n_vocab floats, laid out row-major-by-token (i.e.
    // logits[token * n_vocab + vocab_id]). Modify in place. Order in the
    // chain is preserved.
    virtual void post_compute_logits(
            float * logits,
            int     n_vocab,
            int     n_tokens) const {
        (void) logits; (void) n_vocab; (void) n_tokens;
    }

    // ---- ABI stability ----
    //
    // The four virtuals above (type, load, apply_to, apply_to_expert,
    // post_compute_logits) are the FROZEN base-class ABI. Adding a virtual
    // here would extend this class's vtable, which silently breaks every
    // plugin .so that was built against the old header — the plugin's
    // derived-class vtable still has the old slot count, and engine calls
    // to the new slot read garbage / segfault.
    //
    // To add a new hook, define a SEPARATE optional interface (see
    // llama_sidecar_handler_weights below for the canonical pattern) and
    // dispatch via dynamic_cast at the engine call site. Plugins that opt
    // in inherit from both this class and the optional interface; plugins
    // that don't opt in are transparently skipped (dynamic_cast returns
    // null) and stay binary-compatible.
};

using llama_sidecar_handler_ptr = std::shared_ptr<llama_sidecar_handler>;

// ---- Optional handler interfaces ----
//
// Hook points added AFTER the initial base-class ABI live on separate
// optional interfaces. A handler opts in by multiply-inheriting from
// llama_sidecar_handler AND the optional interface(s) it wants to
// implement; the engine probes for each interface via dynamic_cast at the
// dispatch site and skips handlers that don't implement it.
//
// This pattern keeps the base-class vtable frozen, so old plugin .so
// binaries (which only inherit from llama_sidecar_handler) keep loading
// and running unchanged when a new hook is added — they're simply skipped
// for the new dispatch.
//
// To add a new optional hook:
//   1. Define a new struct here (e.g. llama_sidecar_handler_my_hook) with
//      a virtual destructor and the new pure-virtual method(s).
//   2. At the engine dispatch site, do:
//        auto * h = dynamic_cast<llama_sidecar_handler_my_hook *>(handler.get());
//        if (h) { h->my_hook(...); }
//   3. Plugins that want the hook inherit from the new interface in
//      addition to llama_sidecar_handler.

// Optional interface: weight-modification adapters.
//
// Handlers implementing this interface are called once at sidecar-attach
// time, after load() succeeds, to register weight-modification adapters
// (LoRA-shaped rank-r deltas, Heretic-style rank-1 surgery, Abliterix,
// etc.) that the engine wires into the forward graph via the existing
// build_lora_mm() machinery.
//
// Returns a vector of (adapter, scale) pairs. The handler typically
// constructs each adapter via llama_adapter_lora_init. `scale` is the
// per-adapter runtime multiplier the engine places in ctx->loras
// (composed with adapter.lora.alpha / rank inside build_lora_mm — see
// llama_adapter_lora_weight::get_scale). Ownership transfers to the
// engine: it frees adapters via llama_adapter_lora_free on chain
// replacement or context destruction. Handlers must NOT call free()
// themselves on returned adapters.
//
// Distinct from apply_to(): apply_to() runs every forward pass and
// transforms activations; apply_to_weights() runs once at attach time
// and hands the engine a set of weight deltas to fold into matmul outputs.
struct llama_sidecar_handler_weights {
    virtual ~llama_sidecar_handler_weights() = default;

    virtual std::vector<std::pair<llama_adapter_lora *, float>> apply_to_weights(
            llama_model & model,
            llama_context * lctx,
            const std::string & path) = 0;
};

// ---- Registry ----

using llama_sidecar_factory = std::function<llama_sidecar_handler_ptr()>;

// Register a concrete handler under its type tag. Typically invoked at
// static-init time via the LLAMA_SIDECAR_REGISTER macro. LLAMA_API decoration
// makes this symbol visible across the libllama.so boundary so out-of-tree
// plugin .so files can call it from their llama_sidecar_plugin_init().
LLAMA_API void llama_sidecar_register(const std::string & type, llama_sidecar_factory factory);

// Look up a handler factory by type and instantiate. Returns nullptr if no
// handler is registered for that type.
LLAMA_API llama_sidecar_handler_ptr llama_sidecar_create(const std::string & type);

// Enumerate registered types (for diagnostics or --list-sidecar-types).
LLAMA_API std::vector<std::string> llama_sidecar_list_types();

// Macro for in-tree handler registration. Place at file scope in the
// handler's .cpp:
//
//   LLAMA_SIDECAR_REGISTER("abliteration", llama_sidecar_abliteration);
//
#define LLAMA_SIDECAR_REGISTER(type_name, ClassName)                                  \
    namespace {                                                                       \
    struct ClassName##__sidecar_registrar {                                           \
        ClassName##__sidecar_registrar() {                                            \
            llama_sidecar_register(type_name, []() -> llama_sidecar_handler_ptr {     \
                return std::make_shared<ClassName>();                                 \
            });                                                                       \
        }                                                                             \
    };                                                                                \
    static ClassName##__sidecar_registrar ClassName##__sidecar_registrar_instance;    \
    }
