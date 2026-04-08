# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Build the CUDA Execution Provider as a plugin shared library.
# This file is included from the main CMakeLists.txt when onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON.

message(STATUS "Building CUDA EP as plugin shared library")



set(CUDA_PLUGIN_EP_DIR "${ONNXRUNTIME_ROOT}/core/providers/cuda/plugin")

# --- Collect standard CUDA EP sources ---
file(GLOB_RECURSE CUDA_EP_CC_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cc"
)

file(GLOB_RECURSE CUDA_EP_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/core/providers/cuda/*.cu"
)

# --- Collect contrib ops sources ---
file(GLOB_RECURSE CUDA_CONTRIB_OPS_CC_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cc"
)

file(GLOB_RECURSE CUDA_CONTRIB_OPS_CU_SRCS CONFIGURE_DEPENDS
    "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/*.cu"
)

list(APPEND CUDA_PLUGIN_EP_CC_SRCS
     ${CUDA_EP_CC_SRCS}
     ${CUDA_CONTRIB_OPS_CC_SRCS}
)

list(APPEND CUDA_PLUGIN_EP_CU_SRCS
     ${CUDA_EP_CU_SRCS}
     ${CUDA_CONTRIB_OPS_CU_SRCS}
)

list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/aten_ops/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/collective/.*")

list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/aten_ops/.*")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX "onnxruntime/contrib_ops/cuda/collective/.*")

# Exclude files that include cuda_execution_provider.h (directly or transitively),
# which conflicts with the adapter shim CUDAExecutionProvider class.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_execution_provider\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_provider_factory\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_provider_interface\\.cc$")

# Exclude the framework controlflow/ subdirectory — these inherit from CPU base
# classes (If, Loop, Scan). The plugin has its own control flow wrappers in
# plugin/cuda_controlflow_plugin.cc that delegate to OrtEpApi.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/core/providers/cuda/controlflow/.*")

# Exclude the entire tunable/ subdirectory — it depends on the real CudaTuningContext
# and CUDAExecutionProvider which are not available in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tunable/.*")

# Exclude real EP infrastructure files (replaced by plugin/ equivalents).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_stream_handle\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_execution_provider_info\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_graph\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_mempool_arena\\.cc$")

# Exclude cuda_common.cc — its HalfGemmOptions definitions conflict with the
# adapter's inline shim. Utility functions are replaced or not needed.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_common\\.cc$")

# Exclude cuda_nhwc_kernels.cc and cuda_contrib_kernels.cc — these files contain
# explicit BuildKernelCreateInfo<> registration tables that reference ALL kernel
# classes (including those in excluded source files like space_depth_ops.cc,
# controlflow/, transformers/, etc.), causing undefined symbols at link time.
# With PluginKernelCollector, individual kernel files self-register via macro
# overrides, so these centralized tables are not needed in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_nhwc_kernels\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/cuda_contrib_kernels\\.cc$")

# Exclude sequence_op.cc — uses TensorSeq (incomplete type in plugin build).
# identity_op.cc is now included: TensorSeq code path is guarded by
# BUILD_CUDA_EP_AS_PLUGIN and opset 14+ registrations use Tensor-only types.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/sequence_op\\.cc$")

# Permanently excluded — pure CPU ops, handled by GetCpuPreferredNodes.
# size.cc registers onnxruntime::Size (CPU op) whose Compute() body lives
# in the CPU provider and is not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/size\\.cc$")

# Permanently excluded — pure CPU ops, handled by GetCpuPreferredNodes.
# shape_op.cc inherits from onnxruntime::OpKernel (framework)
# which cannot convert to ep::adapter::OpKernel in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/shape_op\\.cc$")

# Exclude contrib llm/ for now. The core CUDA llm kernels are adapter-safe, but
# contrib llm kernels still need their own plugin pass.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/llm/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/llm/.*")

# Exclude contrib training ops (shrunken_gather depends on provider_api.h in header).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/tensor/shrunken_gather\\.cc$")


# Exclude contrib transformers/ (beam search, greedy search, sampling). Those need subgraph inference.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/transformers/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/transformers/.*")

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)
# Keep the plugin CUDA target aligned with the repo-wide C++20 baseline.
# Forcing CUDA C++17 here breaks newer protobuf/absl headers used by the plugin
# build, as absl::compare expects standard ordering support in this configuration.
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
)

# Suppress -Werror=maybe-uninitialized for local variables written by
# adapter OpKernelInfo::GetAttr<> (GCC falsely warns about variables that are
# initialized inside GetAttr’s output parameter path).
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:GNU>>:-Wno-maybe-uninitialized>
)
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    # Flash-attention, XQA, MoE, and other pure CUDA kernel .cu files must NOT
    # receive the ORT-framework force-include (it conflicts with cute::Tensor etc.).
    # cuda_plugin_kernels.cu already #include "cuda_kernel_adapter.h" directly.
    # Op-registration .cc files do not include it directly, so they need it here.
    #
    # IMPORTANT: The CXX force-include order matters — adapters.h MUST precede
    # cuda_kernel_adapter.h because the adapter establishes type aliases that the
    # kernel adapter header depends on.
    #
    # Force NVCC onto C++20 explicitly. With the VS generator the CUDA standard
    # property alone still leaves `-std=c++17` in AdditionalOptions.
    # Suppress NVCC cudafe warnings:
    #   550  - variable set but never used (in adapter headers)
    #   2810 - [[nodiscard]] false positive on Status assignments in op_kernel.h / kernel_registry.h
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--std c++20>"
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--diag_suppress=550>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=2810>"
)

# Force-include adapter headers for CXX files.
# MSVC uses /FI; GCC/Clang use -include.
if (MSVC)
    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:/FI \"${REPO_ROOT}/include/onnxruntime/ep/adapters.h\">"
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:/FI \"${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h\">"
    )
else()
    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${REPO_ROOT}/include/onnxruntime/ep/adapters.h>"
        "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h>"
    )
endif()

if (MSVC)
    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /permissive>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4834>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4127>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4211>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /Zc:__cplusplus>"
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /bigobj>"
    )

    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CXX>:/wd4127>"
    )
endif()

# Mirror the core CUDA provider's CUDA 12.8+ NVCC workarounds so the plugin
# target handles stricter cudafe diagnostics consistently.
if (DEFINED onnxruntime_NVCC_THREADS)
    set(onnxruntime_plugin_nvcc_threads "${onnxruntime_NVCC_THREADS}")
else()
    set(onnxruntime_plugin_nvcc_threads "1")
endif()
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--threads \"${onnxruntime_plugin_nvcc_threads}\">"
        "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=177>"
)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
            "$<$<COMPILE_LANGUAGE:CUDA>:--static-global-template-stub=false>"
            "$<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=221>"
    )

    if (MSVC)
        target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
                "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcompiler /wd4505>"
        )
    endif()
endif()

include(cudnn_frontend)
include(cutlass)

# --- Find cuDNN (may be at a custom path via onnxruntime_CUDNN_HOME) ---
set(_CUDNN_SEARCH_PATHS "")
if(onnxruntime_CUDNN_HOME)
  list(APPEND _CUDNN_SEARCH_PATHS "${onnxruntime_CUDNN_HOME}")
endif()
if(DEFINED ENV{CUDNN_HOME})
  list(APPEND _CUDNN_SEARCH_PATHS "$ENV{CUDNN_HOME}")
endif()

set(CUDA_PLUGIN_CUDNN_INCLUDE_DIR ${CUDNN_INCLUDE_DIR})
set(CUDA_PLUGIN_CUDNN_LIBRARY ${cudnn_LIBRARY})

if(NOT CUDA_PLUGIN_CUDNN_INCLUDE_DIR OR NOT CUDA_PLUGIN_CUDNN_LIBRARY)
  message(FATAL_ERROR "cuDNN not found (from main ORT search) for CUDA Plugin EP.")
endif()

message(STATUS "CUDA Plugin EP: cuDNN include: ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}")
message(STATUS "CUDA Plugin EP: cuDNN library: ${CUDA_PLUGIN_CUDNN_LIBRARY}")

# Include directories — only public ORT headers + CUDA toolkit + cuDNN + internal headers for adapter
target_include_directories(onnxruntime_providers_cuda_plugin PRIVATE
    ${REPO_ROOT}/include
    ${REPO_ROOT}/include/onnxruntime/core/session
    ${ONNXRUNTIME_ROOT}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDA_PLUGIN_CUDNN_INCLUDE_DIR}
    ${Eigen3_SOURCE_DIR}
    ${cutlass_SOURCE_DIR}/include
    ${cutlass_SOURCE_DIR}/examples
    ${cutlass_SOURCE_DIR}/tools/util/include
)

onnxruntime_add_include_to_target(
    onnxruntime_providers_cuda_plugin
    onnxruntime_common
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
    flatbuffers::flatbuffers
)

# Link libraries
target_link_libraries(onnxruntime_providers_cuda_plugin PRIVATE
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cufft
    CUDNN::cudnn_all
    cudnn_frontend
    Boost::mp11
    safeint_interface
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_mlas
    onnxruntime_flatbuffers
    onnxruntime_common
    cpuinfo::cpuinfo
    onnx
    onnx_proto
    ${PROTOBUF_LIB}
)

# Symbol visibility — only export CreateEpFactories and ReleaseEpFactory
target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_API_MANUAL_INIT BUILD_CUDA_EP_AS_PLUGIN ORT_USE_EP_API_ADAPTERS=1 ONNX_ML=1 ONNX_NAMESPACE=onnx ONNX_USE_LITE_PROTO=1)

if (onnxruntime_USE_CUDA_NHWC_OPS)
    target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ENABLE_CUDA_NHWC_OPS)
endif()

if(WIN32)
  # Windows: use .def file for symbol exports
  set(CUDA_PLUGIN_DEF_FILE ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_ep_symbols.def)
  if(EXISTS ${CUDA_PLUGIN_DEF_FILE})
    target_sources(onnxruntime_providers_cuda_plugin PRIVATE ${CUDA_PLUGIN_DEF_FILE})
  endif()
else()
  # Linux/macOS: hide all symbols by default, explicitly export via __attribute__((visibility("default")))
  set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
      C_VISIBILITY_PRESET hidden
      CXX_VISIBILITY_PRESET hidden
  )
endif()



# Set output name
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    OUTPUT_NAME "onnxruntime_providers_cuda_plugin"
)

# Install
install(TARGETS onnxruntime_providers_cuda_plugin
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
