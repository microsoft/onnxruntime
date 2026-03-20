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
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/einsum\\.cc$")

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

# integer_gemm.cc: dynamic_cast<CudaStream*> replaced with GetCublasHandle(cudaStream_t).
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/integer_gemm\\.cc$")  # REMOVED in Stage 5

# RNN ops: dual-build-compatible signatures are in place (void* alloc_stream,
# cudaStream_t, cudnnHandle_t), but the ORT C API lacks KernelInfoGetAttributeArray_string
# which rnn.h uses via GetAttrs<std::string>("activations", ...).
# Re-excluded until the C API is extended to support string array attributes.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/rnn/.*")

list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/rnn/.*")

# Exclude files that use TensorSeq (incomplete type in plugin build).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/identity_op\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/sequence_op\\.cc$")

# Permanently excluded — pure CPU ops, handled by GetCpuPreferredNodes.
# size.cc registers onnxruntime::Size (CPU op) whose Compute() body lives
# in the CPU provider and is not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/size\\.cc$")

# scatter_nd.cc: ValidateShapes inlined for plugin, GetComputeStream fixed.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/scatter_nd\\.cc$")  # REMOVED in Stage 5

# Exclude llm/ — attention.cc calls QkvToContext which dereferences
# onnxruntime::Stream* (not available in plugin build's adapter OpKernelContext).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/llm/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/llm/.*")

# Exclude constant_of_shape — inherits from ConstantOfShapeBase (CPU provider)
# which is not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/generator/constant_of_shape\\.cc$")

# matmul_integer.cc: GetComputeStream fixed, GemmInt8 signature updated.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/matmul_integer\.cc$")  # REMOVED in Stage 5

# matmul.cc: GetComputeStream fixed, GetTuningContext guarded.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/matmul\.cc$")  # REMOVED in Stage 5

# variadic_elementwise_ops.cc: adapter InputCount/RequiredInput/RequiredOutput supported.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/variadic_elementwise_ops\\.cc$")  # REMOVED in Stage 5C

# slice.cc: plugin-local wrappers added for SliceBase::PrepareForCompute/FlattenOutputDims.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/slice\\.cc$")  # REMOVED in Stage 5C.3

# Exclude space_depth_ops — inherits from SpaceDepthBase (CPU provider).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/space_depth_ops\\.cc$")

# concat.cc: InputArgCount/GetComputeStream usage fixed for adapter.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/concat\\.cc$")  # REMOVED in Stage 5A

# gather.cc: switched to GatherBase::PrepareForComputeImpl for adapter context.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/gather\\.cc$")  # REMOVED in Stage 5B

# gather_nd.cc: PrepareCompute signature changed to void*/cudaStream_t, GetComputeStream fixed.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/gather_nd\\.cc$")  # REMOVED in Stage 5

# pad.cc: plugin-local wrappers added for PadBase static helpers.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/pad\\.cc$")  # REMOVED in Stage 5C.2

# reshape.cc: GetComputeStream/CopyTensor framework dependency fixed for adapter.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/reshape\\.cc$")  # REMOVED in Stage 5A

# split.cc: GetComputeStream usage fixed for adapter via CudaKernel::GetComputeStream.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/split\\.cc$")  # REMOVED in Stage 5A

# Exclude object_detection/ — NonMaxSuppression and RoiAlign inherit from CPU
# base classes (NonMaxSuppressionBase, RoiAlignBase) not linked into the plugin.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/object_detection/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/object_detection/.*")

# Exclude upsample.cc — UpsampleBase uses InputDefs() and
# OpKernelInfo::GetAllocator() not available in adapter.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/upsample\\.cc$")

# Exclude resize.cc — Resize inherits from Upsample (excluded above).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/resize\\.cc$")

# Exclude einsum — einsum_auxiliary_ops.cc calls ReductionOps::ReduceCompute
# which is framework-only (guarded by #ifndef BUILD_CUDA_EP_AS_PLUGIN).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/einsum_utils/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/math/einsum_utils/.*")

# unsqueeze.cc: plugin-local PrepareCompute path added for adapter context.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/unsqueeze\\.cc$")  # REMOVED in Stage 5B

# Permanently excluded — pure CPU ops, handled by GetCpuPreferredNodes.
# shape_op.cc inherits from onnxruntime::OpKernel (framework)
# which cannot convert to ep::adapter::OpKernel in the plugin build.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/shape_op\\.cc$")

# cumsum.cc: axis parsing helper inlined for plugin build.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/math/cumsum\\.cc$")  # REMOVED in Stage 5B

# tile.cc: plugin-local IsTileMemcpy helper added.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/tensor/tile\\.cc$")  # REMOVED in Stage 5B

# --- Contrib op exclusions ---
# Exclude contrib ops that have dependencies not available in the plugin build.
# Note: aten_ops/ and collective/ exclusions are applied above (near the glob).

# Exclude contrib llm/ — uses onnxruntime::Stream* in QkvToContext.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/llm/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/llm/.*")

# Exclude contrib training ops (shrunken_gather depends on provider_api.h in header).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/tensor/shrunken_gather\\.cc$")

# Exclude contrib bert ops that use GetComputeStream() or framework OpKernelContext.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/decoder_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/decoder_masked_self_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/embed_layer_norm\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/fast_gelu\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/group_query_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/longformer_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/multihead_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/packed_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/packed_multihead_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/paged_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/relative_attn_bias\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/bert/remove_padding\\.cc$")

# Exclude contrib ops using GetComputeStream() or framework type deps.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/diffusion/group_norm\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/fused_conv\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/inverse\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/math/bias_dropout\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/math/fft_ops\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/moe/moe\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/sparse/sparse_attention\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/tensor/crop\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/tensor/dynamic_time_warping\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/tensor/dynamicslice\\.cc$")

# Exclude contrib quantization ops with GetComputeStream() deps.
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/quantization/attention_quantization\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/quantization/matmul_bnb4\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/quantization/matmul_nbits\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/quantization/moe_quantization\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/quantization/qordered_ops/.*")

# Exclude contrib transformers/ (beam search, greedy search, sampling).
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/transformers/.*")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/transformers/.*")

# Exclude gemm_float8.cc/.cu — ComputeInternal is in .cu which uses GetComputeStream().
list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/math/gemm_float8\\.cc$")
list(FILTER CUDA_PLUGIN_EP_CU_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/math/gemm_float8\\.cu$")

# fused_matmul.cc: matmul.cc is now included, so fused_matmul can be too.
# list(FILTER CUDA_PLUGIN_EP_CC_SRCS EXCLUDE REGEX ".*/contrib_ops/cuda/math/fused_matmul\\.cc$")  # REMOVED in Stage 5

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)
# Set CUDA standard and flags
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Suppress -Werror=maybe-uninitialized for local variables written by
# adapter OpKernelInfo::GetAttr<> (GCC falsely warns about variables that are
# initialised inside GetAttr’s output parameter path).
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-maybe-uninitialized>
)
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE
    # Flash-attention, XQA, MoE, and other pure CUDA kernel .cu files must NOT
    # receive the ORT-framework force-include (it conflicts with cute::Tensor etc.).
    # cuda_plugin_kernels.cu already #include "cuda_kernel_adapter.h" directly.
    # Op-registration .cc files do not include it directly, so they need it here.
    # Suppress NVCC cudafe warnings:
    #   550  - variable set but never used (in adapter headers)
    #   2810 - [[nodiscard]] false positive on Status assignments in op_kernel.h / kernel_registry.h
    "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--diag_suppress=550>"
    "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:-Xcudafe --diag_suppress=2810>"
    "$<$<COMPILE_LANGUAGE:CXX>:-include;${REPO_ROOT}/include/onnxruntime/ep/adapters.h>"
    "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h>"
)

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
    CUDNN::cudnn_all
    cudnn_frontend
    ${CUDA_PLUGIN_CUDNN_LIBRARY}
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
target_compile_definitions(onnxruntime_providers_cuda_plugin PRIVATE ORT_API_MANUAL_INIT BUILD_CUDA_EP_AS_PLUGIN ONNX_ML=1 ONNX_NAMESPACE=onnx ONNX_USE_LITE_PROTO=1)

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
