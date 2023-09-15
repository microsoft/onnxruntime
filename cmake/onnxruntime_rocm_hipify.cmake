# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# GLOB pattern of file to be excluded
set(contrib_ops_excluded_files
  "bert/attention.cc"
  "bert/attention.h"
  "bert/attention_impl.cu"
  "bert/attention_softmax.h"
  "bert/attention_softmax.cu"
  "bert/attention_prepare_qkv.cu"
  "bert/decoder_masked_multihead_attention.h"
  "bert/decoder_masked_multihead_attention.cc"
  "bert/decoder_masked_self_attention.h"
  "bert/decoder_masked_self_attention.cc"
  "bert/fastertransformer_decoder_attention/*"
  "bert/multihead_attention.cc"
  "bert/multihead_attention.h"
  "bert/fast_gelu_impl.cu"
  "bert/fast_gelu_impl.h"
  "bert/fast_gelu.cc"
  "bert/fast_gelu.h"
  "bert/relative_attn_bias.cc"
  "bert/relative_attn_bias.h"
  "bert/relative_attn_bias_impl.cu"
  "bert/relative_attn_bias_impl.h"
  "bert/skip_layer_norm.cc"
  "bert/skip_layer_norm.h"
  "bert/skip_layer_norm_impl.cu"
  "bert/skip_layer_norm_impl.h"
  "bert/cutlass_fmha/*"
  "bert/tensorrt_fused_multihead_attention/*"
  "bert/transformer_common.h"
  "bert/transformer_common.cc"
  "bert/packed_attention.h"
  "bert/packed_attention.cc"
  "bert/packed_attention_impl.h"
  "bert/packed_attention_impl.cu"
  "bert/packed_multihead_attention.h"
  "bert/packed_multihead_attention.cc"
  "bert/packed_multihead_attention_impl.h"
  "bert/packed_multihead_attention_impl.cu"
  "diffusion/group_norm.cc"
  "diffusion/group_norm_impl.cu"
  "diffusion/group_norm_impl.h"
  "diffusion/nhwc_conv.cc"
  "math/complex_mul.cc"
  "math/complex_mul.h"
  "math/complex_mul_impl.cu"
  "math/complex_mul_impl.h"
  "math/cufft_plan_cache.h"
  "math/fft_ops.cc"
  "math/fft_ops.h"
  "math/fft_ops_impl.cu"
  "math/fft_ops_impl.h"
  "quantization/attention_quantization.cc"
  "quantization/attention_quantization.h"
  "quantization/attention_quantization_impl.cu"
  "quantization/attention_quantization_impl.cuh"
  "quantization/quantize_dequantize_linear.cc"
  "quantization/qordered_ops/qordered_attention_impl.cu"
  "quantization/qordered_ops/qordered_attention_impl.h"
  "quantization/qordered_ops/qordered_attention_input_enum.h"
  "quantization/qordered_ops/qordered_attention.cc"
  "quantization/qordered_ops/qordered_attention.h"
  "quantization/qordered_ops/qordered_common.cuh"
  "quantization/qordered_ops/qordered_layer_norm.h"
  "quantization/qordered_ops/qordered_layer_norm.cc"
  "quantization/qordered_ops/qordered_layer_norm_impl.h"
  "quantization/qordered_ops/qordered_layer_norm_impl.cu"
  "quantization/qordered_ops/qordered_longformer_attention.cc"
  "quantization/qordered_ops/qordered_longformer_attention.h"
  "quantization/qordered_ops/qordered_matmul.h"
  "quantization/qordered_ops/qordered_matmul.cc"
  "quantization/qordered_ops/qordered_matmul_utils.h"
  "quantization/qordered_ops/qordered_matmul_utils.cc"
  "quantization/qordered_ops/qordered_qdq_impl.cu"
  "quantization/qordered_ops/qordered_qdq_impl.h"
  "quantization/qordered_ops/qordered_qdq.cc"
  "quantization/qordered_ops/qordered_qdq.h"
  "quantization/qordered_ops/qordered_unary_ops.h"
  "quantization/qordered_ops/qordered_unary_ops.cc"
  "quantization/qordered_ops/qordered_unary_ops_impl.h"
  "quantization/qordered_ops/qordered_unary_ops_impl.cu"
  "tensor/crop.cc"
  "tensor/crop.h"
  "tensor/crop_impl.cu"
  "tensor/crop_impl.h"
  "tensor/dynamicslice.cc"
  "tensor/image_scaler.cc"
  "tensor/image_scaler.h"
  "tensor/image_scaler_impl.cu"
  "tensor/image_scaler_impl.h"
  "transformers/greedy_search.cc"
  "transformers/greedy_search.h"
  "conv_transpose_with_dynamic_pads.cc"
  "conv_transpose_with_dynamic_pads.h"
  "cuda_contrib_kernels.cc"
  "cuda_contrib_kernels.h"
  "inverse.cc"
  "fused_conv.cc"
)

if (NOT onnxruntime_ENABLE_ATEN)
  list(APPEND contrib_ops_excluded_files "aten_ops/aten_op.cc")
endif()
if (NOT onnxruntime_USE_NCCL)
  list(APPEND contrib_ops_excluded_files "collective/nccl_kernels.cc")
endif()

set(provider_excluded_files
  "atomic/common.cuh"
  "controlflow/if.cc"
  "controlflow/if.h"
  "controlflow/loop.cc"
  "controlflow/loop.h"
  "controlflow/scan.cc"
  "controlflow/scan.h"
  "cu_inc/common.cuh"
  "math/einsum_utils/einsum_auxiliary_ops.cc"
  "math/einsum_utils/einsum_auxiliary_ops.h"
  "math/einsum_utils/einsum_auxiliary_ops_diagonal.cu"
  "math/einsum_utils/einsum_auxiliary_ops_diagonal.h"
  "math/einsum.cc"
  "math/einsum.h"
  "math/gemm.cc"
  "math/matmul.cc"
  "math/softmax_impl.cu"
  "math/softmax_warpwise_impl.cuh"
  "math/softmax_common.cc"
  "math/softmax_common.h"
  "math/softmax.cc"
  "math/softmax.h"
  "nn/conv.cc"
  "nn/conv.h"
  "nn/conv_transpose.cc"
  "nn/conv_transpose.h"
  "nn/pool.cc"
  "nn/pool.h"
  "reduction/reduction_ops.cc"
  "rnn/cudnn_rnn_base.cc"
  "rnn/cudnn_rnn_base.h"
  "rnn/gru.cc"
  "rnn/gru.h"
  "rnn/lstm.cc"
  "rnn/lstm.h"
  "rnn/rnn.cc"
  "rnn/rnn.h"
  "rnn/rnn_impl.cu"
  "rnn/rnn_impl.h"
  "shared_inc/cuda_call.h"
  "shared_inc/fpgeneric.h"
  "cuda_allocator.cc"
  "cuda_allocator.h"
  "cuda_call.cc"
  "cuda_common.cc"
  "cuda_common.h"
  "cuda_execution_provider_info.cc"
  "cuda_execution_provider_info.h"
  "cuda_execution_provider.cc"
  "cuda_execution_provider.h"
  "cuda_memory_check.cc"
  "cuda_memory_check.h"
  "cuda_fence.cc"
  "cuda_fence.h"
  "cuda_fwd.h"
  "cuda_kernel.h"
  "cuda_pch.cc"
  "cuda_pch.h"
  "cuda_profiler.cc"
  "cuda_profiler.h"
  "cuda_provider_factory.cc"
  "cuda_provider_factory.h"
  "cuda_stream_handle.cc",
  "cuda_stream_handle.h",
  "cuda_utils.cu"
  "cudnn_common.cc"
  "cudnn_common.h"
  "cupti_manager.cc"
  "cupti_manager.h"
  "fpgeneric.cu"
  "gpu_data_transfer.cc"
  "gpu_data_transfer.h"
  "integer_gemm.cc"
  "tunable/*"
)

set(training_ops_excluded_files
  "activation/gelu_grad_impl_common.cuh"  # uses custom tanh
  "collective/adasum_kernels.cc"
  "collective/adasum_kernels.h"
  "math/div_grad.cc"  # miopen API differs from cudnn, no double type support
  "nn/batch_norm_grad.cc"  # no double type support
  "nn/batch_norm_grad.h"  # miopen API differs from cudnn
  "nn/batch_norm_internal.cc"  # miopen API differs from cudnn, no double type support
  "nn/batch_norm_internal.h"  # miopen API differs from cudnn, no double type support
  "nn/conv_grad.cc"
  "nn/conv_grad.h"
  "reduction/reduction_all.cc"  # deterministic = true, ignore ctx setting
  "reduction/reduction_ops.cc"  # no double type support
  "cuda_training_kernels.cc"
  "cuda_training_kernels.h"
  "nn/conv_shared.cc"
  "nn/conv_shared.h"
  "nn/conv_transpose_grad.cc"
  "nn/conv_transpose_grad.h"
)

function(auto_set_source_files_hip_language)
  foreach(f ${ARGN})
    if(f MATCHES ".*\\.cu$")
      set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
    endif()
  endforeach()
endfunction()

# cuda_dir must be relative to REPO_ROOT
function(hipify cuda_dir in_excluded_file_patterns out_generated_cc_files out_generated_cu_files)
  set(hipify_tool ${REPO_ROOT}/tools/ci_build/amd_hipify.py)

  file(GLOB_RECURSE srcs CONFIGURE_DEPENDS
    "${REPO_ROOT}/${cuda_dir}/cuda/*.h"
    "${REPO_ROOT}/${cuda_dir}/cuda/*.cc"
    "${REPO_ROOT}/${cuda_dir}/cuda/*.cuh"
    "${REPO_ROOT}/${cuda_dir}/cuda/*.cu"
  )

  # do exclusion
  set(excluded_file_patterns ${${in_excluded_file_patterns}})
  list(TRANSFORM excluded_file_patterns PREPEND "${REPO_ROOT}/${cuda_dir}/cuda/")
  file(GLOB_RECURSE excluded_srcs CONFIGURE_DEPENDS ${excluded_file_patterns})
  foreach(f ${excluded_srcs})
    message(STATUS "Excluded from hipify: ${f}")
  endforeach()
  list(REMOVE_ITEM srcs ${excluded_srcs})

  foreach(f ${srcs})
    file(RELATIVE_PATH cuda_f_rel "${REPO_ROOT}" ${f})
    string(REPLACE "cuda" "rocm" rocm_f_rel ${cuda_f_rel})
    set(f_out "${CMAKE_CURRENT_BINARY_DIR}/amdgpu/${rocm_f_rel}")
    add_custom_command(
      OUTPUT ${f_out}
      COMMAND Python3::Interpreter ${hipify_tool}
              --hipify_perl ${onnxruntime_HIPIFY_PERL}
              ${f} -o ${f_out}
      DEPENDS ${hipify_tool} ${f}
      COMMENT "Hipify: ${cuda_f_rel} -> amdgpu/${rocm_f_rel}"
    )
    if(f MATCHES ".*\\.cuh?$")
      list(APPEND generated_cu_files ${f_out})
    else()
      list(APPEND generated_cc_files ${f_out})
    endif()
  endforeach()

  set_source_files_properties(${generated_cc_files} PROPERTIES GENERATED TRUE)
  set_source_files_properties(${generated_cu_files} PROPERTIES GENERATED TRUE)
  auto_set_source_files_hip_language(${generated_cu_files})
  set(${out_generated_cc_files} ${generated_cc_files} PARENT_SCOPE)
  set(${out_generated_cu_files} ${generated_cu_files} PARENT_SCOPE)
endfunction()
