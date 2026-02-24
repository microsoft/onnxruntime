# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Build the CUDA Execution Provider as a plugin shared library.
# This file is included from the main CMakeLists.txt when onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON.

message(STATUS "Building CUDA EP as plugin shared library")

set(CUDA_PLUGIN_EP_DIR "${ONNXRUNTIME_ROOT}/core/providers/cuda/plugin")
set(CUDA_PLUGIN_REGISTRATION_SCRIPT "${REPO_ROOT}/tools/python/migrate_cuda_registrations.py")
set(CUDA_PLUGIN_REGISTRATION_INPUT "${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_execution_provider.cc")
set(CUDA_PLUGIN_CONTRIB_REGISTRATION_INPUT "${REPO_ROOT}/onnxruntime/contrib_ops/cuda/cuda_contrib_kernels.cc")
set(CUDA_PLUGIN_REGISTRATION_OUTPUT "${CUDA_PLUGIN_EP_DIR}/cuda_plugin_generated_registrations.inc")
set(CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT "${CUDA_PLUGIN_EP_DIR}/cuda_plugin_generated_contrib_registrations.inc")

# Source files (C++ and CUDA)
set(CUDA_PLUGIN_EP_CC_SRCS
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_ep.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_ep_factory.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_ep.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_allocator_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_data_transfer_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/cuda_stream_plugin.cc
    ${CUDA_PLUGIN_EP_DIR}/provider_host_bridge.cc
    ${CUDA_PLUGIN_EP_DIR}/provider_api_shims.cc
    ${ONNXRUNTIME_ROOT}/core/providers/shared/common.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_call.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/cudnn_fe_call.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/activation/activations.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/binary_elementwise_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/clip.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/softmax.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/softmax_common.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/cudnn_common.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/reduction/reduction_functions.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/reduction/reduction_ops.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/cast_op.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/concat.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/where.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/split.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gather.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/transpose.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/split.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/concat.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/transpose.cc
    ${ONNXRUNTIME_ROOT}/core/providers/cpu/tensor/gather.cc
)

set(CUDA_PLUGIN_EP_CU_SRCS
    ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_kernels.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/activation/activations_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/binary_elementwise_ops_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/clip_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/softmax_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/math/unary_elementwise_ops_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/fpgeneric.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/reduction/reduction_functions.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/cast_op.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/concat_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/where_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/split_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gather_impl.cu
    ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/transpose_impl.cu
)

if(NOT onnxruntime_DISABLE_CONTRIB_OPS)
    set(CUDA_PLUGIN_CONTRIB_TYPED_INSTANTIATION_SRCS
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/group_query_attention.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/rotary_embedding.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/gemma_rotary_emb.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/skip_layer_norm.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/embed_layer_norm.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/fast_gelu.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/decoder_masked_multihead_attention.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/multihead_attention.cc
    )

    set_source_files_properties(${CUDA_PLUGIN_CONTRIB_TYPED_INSTANTIATION_SRCS}
        PROPERTIES
            COMPILE_DEFINITIONS ORT_CUDA_PLUGIN_INSTANTIATE_TYPED_KERNEL_FROM_REGISTRATION=1
    )

    # XQA and FlashAttention CUDA kernels required for GQA ops.
    file(GLOB CUDA_PLUGIN_GQA_XQA_CU_SRCS CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/xqa/xqa_loader*.cu"
    )
    file(GLOB CUDA_PLUGIN_FLASH_ATTENTION_CU_SRCS CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/flash_attention/flash_fwd*.cu"
    )

    list(APPEND CUDA_PLUGIN_EP_CC_SRCS
        ${CUDA_PLUGIN_CONTRIB_TYPED_INSTANTIATION_SRCS}
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/transformer_common.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cpu/bert/attention_base.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cpu/bert/bias_gelu_helper.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cpu/bert/embed_layer_norm_helper.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_kernel_options.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/flash_attention/flash_api.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/lean_attention/lean_api.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/moe.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/moe_quantization.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/matmul_nbits.cc
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/gather_block_quantized.cc
    )

    list(APPEND CUDA_PLUGIN_EP_CU_SRCS
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/add_bias_transpose.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_qk.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_prepare_qkv.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_kv_cache.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_softmax.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_transpose.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/bert_padding.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/packed_attention_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/packed_multihead_attention_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/group_query_attention_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/rotary_embedding_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/gemma_rotary_emb_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/skip_layer_norm_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/embed_layer_norm_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/attention_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/fastertransformer_decoder_attention/decoder_masked_multihead_attention_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.cu
        ${CUDA_PLUGIN_FLASH_ATTENTION_CU_SRCS}
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/cutlass_fmha/fmha_sm50.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/cutlass_fmha/fmha_sm70.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/cutlass_fmha/fmha_sm75.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/cutlass_fmha/fmha_sm80.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/paged_attention_impl.cu
        ${ONNXRUNTIME_ROOT}/core/providers/cuda/nn/layer_norm_impl.cu
        ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gelu_impl.cu
        ${ONNXRUNTIME_ROOT}/core/providers/cuda/tensor/gelu_approximate_impl.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/moe/ft_moe/moe_kernel.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/gather_block_quantized.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/dequantize_blockwise_4bits.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/dequantize_blockwise_8bits.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/matmul_4bits.cu
        ${ONNXRUNTIME_ROOT}/contrib_ops/cuda/quantization/matmul_8bits.cu
    )

    file(GLOB CUDA_PLUGIN_XQA_CU_SRCS CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/xqa/*.cu"
    )
    list(APPEND CUDA_PLUGIN_EP_CU_SRCS ${CUDA_PLUGIN_XQA_CU_SRCS})

    file(GLOB CUDA_PLUGIN_TRT_FUSED_MHA_CC_SRCS CONFIGURE_DEPENDS
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/*.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/causal/*.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/*.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/flash_attention/*.cc"
        "${ONNXRUNTIME_ROOT}/contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/flash_attention_causal/*.cc"
    )
    list(APPEND CUDA_PLUGIN_EP_CC_SRCS ${CUDA_PLUGIN_TRT_FUSED_MHA_CC_SRCS})
endif()

# Create shared library target using the ORT helper function for plugins
onnxruntime_add_shared_library_module(onnxruntime_providers_cuda_plugin
    ${CUDA_PLUGIN_EP_CC_SRCS}
    ${CUDA_PLUGIN_EP_CU_SRCS}
)

add_custom_command(
    OUTPUT ${CUDA_PLUGIN_REGISTRATION_OUTPUT} ${CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT}
    COMMAND ${Python_EXECUTABLE} ${CUDA_PLUGIN_REGISTRATION_SCRIPT}
            --input ${CUDA_PLUGIN_REGISTRATION_INPUT}
            --output ${CUDA_PLUGIN_REGISTRATION_OUTPUT}
    COMMAND ${Python_EXECUTABLE} ${CUDA_PLUGIN_REGISTRATION_SCRIPT}
            --contrib
            --check-critical-contrib
            --input ${CUDA_PLUGIN_CONTRIB_REGISTRATION_INPUT}
            --output ${CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT}
    DEPENDS
        ${CUDA_PLUGIN_REGISTRATION_SCRIPT}
        ${CUDA_PLUGIN_REGISTRATION_INPUT}
        ${CUDA_PLUGIN_CONTRIB_REGISTRATION_INPUT}
    COMMENT "Generating CUDA plugin kernel registrations"
    VERBATIM
)

add_custom_target(onnxruntime_cuda_plugin_generate_registrations
    DEPENDS ${CUDA_PLUGIN_REGISTRATION_OUTPUT} ${CUDA_PLUGIN_CONTRIB_REGISTRATION_OUTPUT}
)
add_dependencies(onnxruntime_providers_cuda_plugin onnxruntime_cuda_plugin_generate_registrations)

# Set CUDA standard and flags
set_target_properties(onnxruntime_providers_cuda_plugin PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)
target_compile_options(onnxruntime_providers_cuda_plugin PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-Xcudafe;--diag_suppress=550>")
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
