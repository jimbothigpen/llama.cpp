#include "llama-sidecar.h"

#include "llama-impl.h"

#include <map>
#include <mutex>

namespace {

// Returned by reference so static-init registrations from translation units
// in any link order all hit the same map. (Construct-on-first-use idiom.)
std::map<std::string, llama_sidecar_factory> & registry() {
    static std::map<std::string, llama_sidecar_factory> r;
    return r;
}

std::mutex & registry_mutex() {
    static std::mutex m;
    return m;
}

} // namespace

void llama_sidecar_register(const std::string & type, llama_sidecar_factory factory) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto & r = registry();
    if (r.find(type) != r.end()) {
        LLAMA_LOG_WARN("%s: sidecar type '%s' is already registered; replacing\n",
                       __func__, type.c_str());
    }
    r[type] = std::move(factory);
}

llama_sidecar_handler_ptr llama_sidecar_create(const std::string & type) {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto & r = registry();
    auto it = r.find(type);
    if (it == r.end()) {
        return nullptr;
    }
    return it->second();
}

std::vector<std::string> llama_sidecar_list_types() {
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto & r = registry();
    std::vector<std::string> out;
    out.reserve(r.size());
    for (const auto & kv : r) {
        out.push_back(kv.first);
    }
    return out;
}
