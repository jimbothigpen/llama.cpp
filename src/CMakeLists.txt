llama_add_compile_flags()

#
# libraries
#

# llama

file(GLOB LLAMA_MODELS_SOURCES "models/*.cpp")

add_library(llama
            ../include/llama.h
            llama.cpp
            llama-adapter.cpp
            llama-arch.cpp
            llama-batch.cpp
            llama-chat.cpp
            llama-context.cpp
            llama-cparams.cpp
            llama-grammar.cpp
            llama-graph.cpp
            llama-hparams.cpp
            llama-impl.cpp
            llama-io.cpp
            llama-kv-cache.cpp
            llama-kv-cache-iswa.cpp
            llama-memory.cpp
            llama-memory-hybrid.cpp
            llama-memory-hybrid-iswa.cpp
            llama-memory-recurrent.cpp
            llama-mmap.cpp
            llama-model-loader.cpp
            llama-model-saver.cpp
            llama-model.cpp
            llama-quant.cpp
            llama-sampler.cpp
            llama-sidecar.cpp
            llama-vocab.cpp
            unicode-data.cpp
            unicode.cpp
            unicode.h
            ${LLAMA_MODELS_SOURCES}
            llama-triattention.cpp
            llama-triattention.h
            triattention.c
            triattention.h
            triattention-backend.c
            triattention-backend.h
            triattention-bridge.cpp
            triattention-runtime.c
            triattention-runtime.h
            )

set_target_properties(llama PROPERTIES
    VERSION ${LLAMA_INSTALL_VERSION}
    SOVERSION 0
    MACHO_CURRENT_VERSION 0 # keep macOS linker from seeing oversized version number
)

target_include_directories(llama PRIVATE .)
target_include_directories(llama PUBLIC ../include)
target_compile_features   (llama PRIVATE cxx_std_17) # don't bump

target_link_libraries(llama PUBLIC ggml)

# TriAttention Phase C: GPU GQA scoring kernel (HIP).
# Compiled only for ROCm/HIP builds; enables the TRIA_HIP_BACKEND path in
# triattention-backend.c so g_tria_backend resolves to the real GPU kernels.
if (GGML_HIP)
    # Mirror ggml-hip's GPU target -> HIP architecture mapping (subdirectory-local).
    if (AMDGPU_TARGETS AND NOT GPU_TARGETS)
        set(GPU_TARGETS ${AMDGPU_TARGETS})
    endif()
    if (GPU_TARGETS AND NOT CMAKE_HIP_ARCHITECTURES)
        set(CMAKE_HIP_ARCHITECTURES ${GPU_TARGETS})
    endif()
    enable_language(HIP)
    find_package(hip REQUIRED)
    # Compile the scoring kernel as a HIP-ONLY static library so the HIP
    # architecture flags (--offload-arch) stay isolated from the host C++
    # compilation of the rest of the llama target.
    add_library(tria-hip STATIC triattention-hip.hip)
    set_source_files_properties(triattention-hip.hip PROPERTIES LANGUAGE HIP)
    set_target_properties(tria-hip PROPERTIES
        HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(tria-hip PRIVATE hip::device)
    target_link_libraries(llama PRIVATE tria-hip)
    target_compile_definitions(llama PRIVATE TRIA_HIP_BACKEND)
    # llama device-links tria-hip's static device code, so it needs a HIP arch.
    # llama has no HIP *sources* and links hip::device only transitively-PRIVATE
    # through tria-hip, so --offload-arch does not reach its host C++ compiles.
    set_target_properties(llama PROPERTIES HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
