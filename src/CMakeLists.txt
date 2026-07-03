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
            llama-kv-cache-dsa.cpp
            llama-kv-cache-dsv4.cpp
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

# KV cache compaction (Attention Matching) — ported from jandhyala-dev/modelai-llama.cpp.
# First-landing slice: SELECT method only. The deferred SOLVER/OMP/NONUNIFORM/CHUNKED
# implementation files (solver/query/budget/select/self-study/on-policy/prefill-q/exec)
# and the Darwin-only Metal/LAPACK linkage are intentionally NOT compiled here yet.
# When OFF (default), only the stubs TU is built and the default build is byte-unaffected.
if (LLAMA_KV_COMPACTION)
    target_sources(llama PRIVATE
        llama-kv-compacted-prefix.cpp
        llama-kv-compact-api.cpp
        llama-kv-compact-pipeline.cpp
    )
    target_compile_definitions(llama PUBLIC LLAMA_KV_COMPACTION)
else()
    target_sources(llama PRIVATE llama-kv-compact-stubs.cpp)
endif()

# EpiCache P1: block-wise prefill peak-mem bounding. No new translation units — the logic lives
# inside #ifdef LLAMA_EPICACHE in the already-compiled TriAttention runtime + llama-context. When
# OFF (default) the macro is undefined and the build is byte-unaffected.
if (LLAMA_EPICACHE)
    target_compile_definitions(llama PUBLIC LLAMA_EPICACHE)
endif()

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

# TriAttention Phase C: GPU GQA scoring kernel (CUDA / NVIDIA, incl. T4 sm_75).
# Compiled only for CUDA builds; enables the TRIA_CUDA_BACKEND path in
# triattention-backend.c so g_tria_backend resolves to the real GPU kernels.
# The raw-score kernel is register-frugal (loop-interchanged freq/offset, K
# dequantized on the fly) so it does not spill on sm_75 — bit-identical to the
# HIP path. See worker-scratch/triattn-cuda-scorer-2026-06-11/design.md.
if (GGML_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    # Inherit ggml-cuda's architecture selection; default to sm_75 (T4) if unset.
    if (NOT CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    # Compile the scoring kernel as a CUDA-ONLY static library so the CUDA
    # architecture flags stay isolated from the host C++ compilation of the
    # rest of the llama target (mirrors the tria-hip isolation above).
    add_library(tria-cuda STATIC triattention-cuda.cu)
    set_source_files_properties(triattention-cuda.cu PROPERTIES LANGUAGE CUDA)
    set_target_properties(tria-cuda PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        POSITION_INDEPENDENT_CODE ON)
    # Whole-program (non-separable) compilation: the fatbin is self-contained in
    # tria-cuda, so llama needs no CUDA device-link step — only the host runtime.
    # PRIVATE on a STATIC lib still propagates CUDA::cudart to llama's link line.
    target_link_libraries(tria-cuda PRIVATE CUDA::cudart)
    target_link_libraries(llama PRIVATE tria-cuda)
    target_compile_definitions(llama PRIVATE TRIA_CUDA_BACKEND)
endif()

# TriAttention Phase C: GPU GQA scoring kernel (Vulkan).
# Compiled only for Vulkan builds; enables the TRIA_VULKAN_BACKEND path in
# triattention-backend.c. Built as an isolated static library (its own Vulkan
# device/pipelines) so it stays independent of ggml-vulkan and works whether or
# not GGML_BACKEND_DL is set. The GLSL compute shaders are compiled to SPIR-V at
# build time with glslc and embedded as C arrays (no runtime shader files).
if (GGML_VULKAN)
    find_package(Vulkan COMPONENTS glslc REQUIRED)
    set(TRIA_VK_SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/triattention-vulkan-shaders)
    set(TRIA_VK_SHADER_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR}/triattention-vulkan-shaders)
    file(MAKE_DIRECTORY ${TRIA_VK_SHADER_BIN_DIR})
    set(TRIA_VK_SPV_INCS "")
    foreach(shader tria_raw_score tria_znorm_agg tria_znorm_global)
        set(_src ${TRIA_VK_SHADER_SRC_DIR}/${shader}.comp)
        set(_out ${TRIA_VK_SHADER_BIN_DIR}/${shader}.comp.spv.inc)
        add_custom_command(
            OUTPUT ${_out}
            COMMAND ${Vulkan_GLSLC_EXECUTABLE} -O -mfmt=num --target-env=vulkan1.2
                    ${_src} -o ${_out}
            DEPENDS ${_src}
            COMMENT "glslc (embed) ${shader}.comp -> ${shader}.comp.spv.inc"
            VERBATIM)
        list(APPEND TRIA_VK_SPV_INCS ${_out})
    endforeach()
    add_custom_target(tria-vulkan-shaders DEPENDS ${TRIA_VK_SPV_INCS})

    add_library(tria-vulkan STATIC triattention-vulkan.cpp)
    add_dependencies(tria-vulkan tria-vulkan-shaders)
    target_include_directories(tria-vulkan PRIVATE . ${TRIA_VK_SHADER_BIN_DIR})
    target_compile_features(tria-vulkan PRIVATE cxx_std_17)
    set_target_properties(tria-vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(tria-vulkan PRIVATE Vulkan::Vulkan)
    target_link_libraries(llama PRIVATE tria-vulkan)
    target_compile_definitions(llama PRIVATE TRIA_VULKAN_BACKEND)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_BUILD)
    target_compile_definitions(llama PUBLIC  LLAMA_SHARED)
endif()
