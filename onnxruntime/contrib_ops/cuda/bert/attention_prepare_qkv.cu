// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#if DEBUG_TENSOR_LEVEL > 1
// Dump the workspace for Q, K, V after processing QKV data.
template <typename T>
void DumpQkv(AttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  DUMP_TENSOR_INIT();
  if (data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH) {
    DUMP_TENSOR_D("q(BNSH)", data.q, batch_size, num_heads, sequence_length, qk_head_size);
    DUMP_TENSOR_D("k(BNSH)", data.k, batch_size, num_heads, kv_sequence_length, qk_head_size);
    DUMP_TENSOR_D("v(BNSH)", data.v, batch_size, num_heads, kv_sequence_length, v_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    DUMP_TENSOR_D("q(BSNH)", data.q, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", data.k, batch_size, kv_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", data.v, batch_size, kv_sequence_length, num_heads, v_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH) {
    DUMP_TENSOR_D("q(BNSH)", data.q, batch_size, num_heads, sequence_length, qk_head_size);
    DUMP_TENSOR_D("k(BNSH)", data.k, batch_size, num_heads, kv_sequence_length, qk_head_size);
    DUMP_TENSOR_D("v(BNSH)", data.v, batch_size, num_heads, kv_sequence_length, v_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::QKV_BSN3H) {
    DUMP_TENSOR_D("q(BSN3H)", data.q, batch_size, sequence_length, num_heads * 3, qk_head_size);
  }
}

// Dump the inputs before processing QKV data.
template <typename T>
void DumpInputs(contrib::AttentionParameters& parameters, AttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  DUMP_TENSOR_INIT();
  if (parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    DUMP_TENSOR_D("Query(BNSH)", data.query, batch_size, num_heads, sequence_length, qk_head_size);
    DUMP_TENSOR_D("Key(BNSH)", data.key, batch_size, num_heads, kv_sequence_length, qk_head_size);
    DUMP_TENSOR_D("Value(BNSH)", data.value, batch_size, num_heads, kv_sequence_length, v_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH) {
    DUMP_TENSOR_D("Query(BSNH)", data.query, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("Key(BSNH)", data.key, batch_size, kv_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("Value(BSNH)", data.value, batch_size, kv_sequence_length, num_heads, v_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH) {
    DUMP_TENSOR_D("Query(BNSH)", data.query, batch_size, num_heads, sequence_length, qk_head_size);
    DUMP_TENSOR_D("Key(BNSH)", data.key, batch_size, num_heads, kv_sequence_length, qk_head_size);
    DUMP_TENSOR_D("Value(BNSH)", data.value, batch_size, num_heads, kv_sequence_length, v_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::QKV_BSN3H) {
    DUMP_TENSOR_D("Query(BSN3H)", data.query, batch_size, sequence_length, num_heads * 3, qk_head_size);
  } else if (data.qkv_format == AttentionQkvFormat::Q_KV_BSNH_BSN2H) {
    DUMP_TENSOR_D("Query(BNSH)", data.query, batch_size, num_heads, sequence_length, qk_head_size);
    DUMP_TENSOR_D("Value(BSN2H)", data.value, batch_size, sequence_length, num_heads * 2, qk_head_size);
  }

  if (data.bias != nullptr) {
    DUMP_TENSOR_D("Q_bias", data.bias, num_heads, qk_head_size);
    DUMP_TENSOR_D("K_bias", data.bias + num_heads * qk_head_size, num_heads, qk_head_size);
    DUMP_TENSOR_D("V_bias", data.bias + 2 * num_heads * qk_head_size, num_heads, v_head_size);
  }

  if (data.attention_bias != nullptr) {
    DUMP_TENSOR_D("attention_bias", data.attention_bias,
                  parameters.broadcast_attn_bias_dim_0 ? 1 : batch_size,
                  parameters.broadcast_attn_bias_dim_1 ? 1 : num_heads,
                  sequence_length,
                  parameters.total_sequence_length);
  }

  if (data.mask_index != nullptr) {
    if (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
      DUMP_TENSOR_D("mask (2D)", data.mask_index, batch_size, parameters.total_sequence_length);
    }
    if (parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) {
      DUMP_TENSOR_D("mask (seqlen_k)", data.mask_index, 1, batch_size);
      DUMP_TENSOR_D("mask (cu_seqlen_q)", data.mask_index + batch_size, 1, batch_size + 1);
      DUMP_TENSOR_D("mask (cu_seqlen_k)", data.mask_index + 2 * batch_size + 1, 1, batch_size + 1);
    }
  }
}

// Dump the kernel outputs
template <typename T>
void DumpOutputs(AttentionData<T>& data) {
  DUMP_TENSOR_INIT();
  DUMP_TENSOR("output", data.output,
              parameters.batch_size, parameters.sequence_length, parameters.num_heads, parameters.v_head_size);
}
#endif

template <typename T>
Status PrepareQkv_Attention(contrib::AttentionParameters& parameters,
                            AttentionData<T>& data,
                            cudaStream_t stream,
                            int max_threads_per_block) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  const bool past_present_share_buffer = parameters.past_present_share_buffer;
  void* fused_runner = data.fused_runner;
  bool use_flash_or_efficient_attention = data.use_flash_attention || data.use_memory_efficient_attention;

  T* qkv = data.workspace;

  bool use_fused_kernel = (nullptr != fused_runner && !parameters.is_unidirectional);
  bool use_fused_causal = (nullptr != fused_runner && parameters.is_unidirectional);

  // For fused TRT attention, transpose qkv to BxSxNx3xH (format 2)
  // For flash or memory efficient attention, transpose to 3xBxSxNxH (format 3)
  // For unfused kernel, transpose to 3xBxNxSxH (format 1)
  // For fused causal kernel, use format 1 since we need have K and V to update present state,
  //   at the same time, we update gemm_buffer BxSx3xNxH with bias which is used as input for fused causal kernel.
  const int format = (use_fused_kernel ? 2 : (use_flash_or_efficient_attention ? 3 : 1));
  data.qkv_format = use_fused_kernel
                        ? AttentionQkvFormat::QKV_BSN3H
                        : (use_flash_or_efficient_attention
                               ? AttentionQkvFormat::Q_K_V_BSNH
                               : (use_fused_causal
                                      ? AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH
                                      : AttentionQkvFormat::Q_K_V_BNSH));

  // For fused causal, we will update gemm_buffer with bias directly.
  T* qkv_add_bias = use_fused_causal ? data.gemm_buffer : nullptr;

  int matrix_to_transpose = ((format == AttentionQkvFormat::Q_K_V_BNSH && past_present_share_buffer) ? 1 : 3);
  // format 1: BxSx(NH + NH + NH_v) => BxNxSxH + BxNxSxH + BxNxSxH_v
  // format 2: BxSx(NH + NH + NH) => BxSxNx(H + H + H)
  LaunchAddBiasTranspose(stream, matrix_to_transpose, format, max_threads_per_block,
                         batch_size, sequence_length, num_heads, qk_head_size,
                         data.gemm_buffer, data.bias, qkv, true, v_head_size, qkv_add_bias,
                         3, parameters.do_rotary, parameters.rotary_embedding,
                         parameters.past_sequence_length);
  return Status::OK();
}

// Return true if the workspace is not needed for Q, K, V inputs, false otherwise.
// This shall be in sync with the following function PrepareQkv_MHA_Cross.
template <typename T>
bool NoQkvWorkspace_MHA_Cross(AttentionData<T>& data) {
  // query, key and value are passed as Q, K and V for the following conditions.
  return (data.use_memory_efficient_attention ||
          data.use_flash_attention ||
          data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) &&
         data.bias == nullptr;
}

// For MultiHeadAttention with cross attention (Q_K_V_BSNH_BNSH_BNSH format)
template <typename T>
Status PrepareQkv_MHA_Cross(contrib::AttentionParameters& parameters,
                            AttentionData<T>& data,
                            cudaStream_t stream,
                            int max_threads_per_block) {
  assert(parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  // past_key or past_value is not supported for cross attention
  // present_key and present_value can be supported in theory, although we do not allow the senario for now.
  assert(data.past_key == nullptr);
  assert(data.past_value == nullptr);
  assert(data.has_qkv_workspace == !NoQkvWorkspace_MHA_Cross(data));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;

  if (data.use_memory_efficient_attention ||
      data.use_flash_attention ||
      data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    // Add bias for Q
    if (data.bias != nullptr) {
      LaunchAddBias(stream, max_threads_per_block, batch_size, sequence_length, num_heads, qk_head_size,
                    data.bias, data.query, data.q);
    } else {
      data.q = const_cast<T*>(data.query);
    }

    // Here we have assumption that there is no bias for key and value when they are in BNSH format.
    data.k = const_cast<T*>(data.key);
    data.v = const_cast<T*>(data.value);
    data.qkv_format = AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH;
  } else {  // unfused kernel
    assert(data.IsUnfused());
    if (data.bias == nullptr) {
      // Transpose query from BSNH to BNSH
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.query, data.q));
    } else {
      // Add bias to query, and transpose it: Query (BxSxNxH) => Q (BxNxSxH)
      constexpr int format = 0;
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, sequence_length, num_heads, qk_head_size,
                                data.query, data.bias, data.q,
                                true, -1);
    }

    // Here we have assumption that there is no bias for key and value when they are in BNSH format.
    // So we do not need to add bias for key and value. Just use the key and value directly.
    data.k = const_cast<T*>(data.key);
    data.v = const_cast<T*>(data.value);
    data.qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
  return Status::OK();
}

template <typename T>
bool NoQkvWorkspace_MHA_NoPast(AttentionData<T>& data) {
  // query, key and value are passed as Q, K and V for the following conditions.
  return (data.use_memory_efficient_attention ||
          data.use_flash_attention ||
          data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) &&
         data.bias == nullptr;
}

// For MultiHeadAttention without past state, with Q, K and V inputs
template <typename T>
Status PrepareQkv_MHA_NoPast(contrib::AttentionParameters& parameters,
                             AttentionData<T>& data,
                             cudaStream_t stream,
                             int max_threads_per_block) {
  assert(parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
  assert(data.query != nullptr);
  assert(data.key != nullptr);
  assert(data.value != nullptr);
  assert(data.past_key == nullptr);
  assert(data.past_value == nullptr);
  assert(data.present_key == nullptr);
  assert(data.present_value == nullptr);
  assert(!parameters.is_unidirectional);
  assert(data.has_qkv_workspace == !NoQkvWorkspace_MHA_NoPast(data));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  if (data.fused_cross_attention_kernel != nullptr) {
    assert(qk_head_size == v_head_size);
    assert(data.attention_bias == nullptr);
    assert(data.mask_index == nullptr);
    assert(parameters.hidden_size == parameters.v_hidden_size);

    // For fused cross attention, besides adding bias, K and V needed to be packed:
    //   Key (BxSxNxH), Value (BxSxNxH) => Q (BxSxNxH), K (BxSxNx2xH)
    LaunchAddBiasTransposeTrt(
        stream, max_threads_per_block,
        batch_size, sequence_length,
        num_heads, qk_head_size,
        data.bias, data.query, data.key, data.value, data.q, true, kv_sequence_length);
    data.v = nullptr;
    data.qkv_format = AttentionQkvFormat::Q_KV_BSNH_BSN2H;
  } else if (data.use_memory_efficient_attention ||
             data.use_flash_attention ||
             data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    if (data.bias != nullptr) {
      LaunchAddBias(stream, max_threads_per_block,
                    batch_size, sequence_length, kv_sequence_length,
                    num_heads, qk_head_size, v_head_size,
                    data.bias, data.query, data.key, data.value, data.q, data.k, data.v);
    } else {
      data.q = const_cast<T*>(data.query);
      data.k = const_cast<T*>(data.key);
      data.v = const_cast<T*>(data.value);
    }

    data.qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  } else if (data.fused_runner != nullptr) {
    assert(qk_head_size == v_head_size);
    assert(data.attention_bias == nullptr);

    // Query (BxSxNxH), Key (BxSxNxH), Value (BxSxNxH) => Q: BxSxNx(H + H + H)
    LaunchAddBiasTransposeTrt(
        stream, max_threads_per_block,
        batch_size, sequence_length,
        num_heads, qk_head_size,
        data.bias, data.query, data.key, data.value, data.q, false, kv_sequence_length);
    data.k = nullptr;
    data.v = nullptr;

    data.qkv_format = AttentionQkvFormat::QKV_BSN3H;
  } else {  // unfused kernel
    assert(data.IsUnfused());
    // Query (BxSxNxH) => Q (BxNxSxH)
    constexpr int format = 0;
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size,
        data.query, data.bias, data.q,
        true, -1);

    // Key (BxLxNxH) => K (BxNxLxH)
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, kv_sequence_length, num_heads, qk_head_size,
        data.key, nullptr == data.bias ? nullptr : data.bias + num_heads * qk_head_size, data.k,
        true, -1);

    // Value (BxLxNxH_v) => K (BxNxLxH_v)
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, kv_sequence_length, num_heads, v_head_size,
        data.value, nullptr == data.bias ? nullptr : data.bias + 2 * num_heads * qk_head_size, data.v,
        true, -1);

    data.qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }

  return Status::OK();
}

template <typename T>
bool NoQkvWorkspace_MHA_WithPast_NoBias(AttentionData<T>& data) {
  if (data.use_memory_efficient_attention ||
      data.use_flash_attention ||
      data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    // Q, K and V redirects to query, present_k and present_v, so we do not need extra workspace for QKV.
    return data.past_key == nullptr && data.present_key != nullptr;
  }
  return false;
}

// For MultiHeadAttention with kv cache (past or present), but no bias
template <typename T>
Status PrepareQkv_MHA_WithPast_NoBias(contrib::AttentionParameters& parameters,
                                      AttentionData<T>& data,
                                      cudaStream_t stream,
                                      int max_threads_per_block) {
  assert(parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
  assert(data.query != nullptr);
  assert(data.key != nullptr);
  assert(data.value != nullptr);
  assert(data.bias == nullptr);
  assert(data.fused_runner == nullptr);
  assert(data.fused_cross_attention_kernel == nullptr);
  assert(data.present_key != nullptr);
  assert(data.present_value != nullptr);
  assert(data.past_key == nullptr && data.past_value == nullptr ||
         data.past_key != nullptr && data.past_value != nullptr);
  assert(data.has_qkv_workspace == !NoQkvWorkspace_MHA_WithPast_NoBias(data));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  // When there is no past state and there is present state, we output K and V directly to present state.
  if (data.past_key == nullptr && data.present_key != nullptr) {
    data.k = data.present_key;
    data.v = data.present_value;
  }

  if (data.use_memory_efficient_attention ||
      data.use_flash_attention ||
      data.use_lean_attention ||
      data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    // Use oiginal Query (BSNH) since there is no bias.
    data.q = const_cast<T*>(data.query);

    // Key (BxLxNxH) => K (BxNxLxH)
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.key, data.k));
    // Value (BxLxNxH) => V (BxNxLxH)
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, v_head_size, num_heads,
                                       max_threads_per_block, false, data.value, data.v));
    data.qkv_format = AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH;
  } else {  // unfused kernel
    assert(data.IsUnfused());
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.query, data.q));
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.key, data.k));
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, v_head_size, num_heads,
                                       max_threads_per_block, false, data.value, data.v));
    data.qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }

  return Status::OK();
}

template <typename T>
constexpr bool NoQkvWorkspace_MHA_WithPast_Bias(AttentionData<T>& /*data*/) {
  return false;
}

// For MultiHeadAttention with both kv cache (past or present) and bias
template <typename T>
Status PrepareQkv_MHA_WithPast_Bias(contrib::AttentionParameters& parameters,
                                    AttentionData<T>& data,
                                    cudaStream_t stream,
                                    int max_threads_per_block) {
  assert(parameters.qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
  assert(data.bias != nullptr);
  assert(!(data.past_key != nullptr && data.present_key == nullptr));
  assert(data.fused_runner == nullptr);
  assert(data.fused_cross_attention_kernel == nullptr);
  assert(data.present_key != nullptr);
  assert(data.present_value != nullptr);
  assert(data.past_key == nullptr && data.past_value == nullptr ||
         data.past_key != nullptr && data.past_value != nullptr);
  assert(data.has_qkv_workspace == !NoQkvWorkspace_MHA_WithPast_Bias(data));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  // When there is no past state and there is present state, we output K and V directly to present state.
  if (data.past_key == nullptr && data.present_key != nullptr) {
    data.k = data.present_key;
    data.v = data.present_value;
  }

  if (data.use_memory_efficient_attention ||
      data.use_flash_attention ||
      data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    // Query(BxSxNxH) + Bias_Q => Q (BxSxNxH)
    LaunchAddBias(stream, max_threads_per_block, batch_size, sequence_length, num_heads, qk_head_size,
                  data.bias, data.query, data.q);

    // Key (BxLxNxH) + Bias_K => K (BxNxLxH)
    constexpr int format = 0;
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, kv_sequence_length, num_heads, qk_head_size,
        data.key, data.bias + num_heads * qk_head_size, data.k, true, -1);

    // Key (BxLxNxH) + Bias_K => K (BxNxLxH)
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, kv_sequence_length, num_heads, v_head_size,
        data.value, data.bias + 2 * num_heads * qk_head_size, data.v, true, -1);

    data.qkv_format = AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH;
  } else {  // unfused kernel
    assert(data.IsUnfused());

    constexpr int format = 0;
    // Query (BxSxNxH) => Q (BxNxSxH)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, sequence_length, num_heads, qk_head_size,
                              data.query, data.bias, data.q,
                              true, -1);

    // Key (BxLxNxH) => K (BxNxLxH)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, kv_sequence_length, num_heads, qk_head_size,
                              data.key, data.bias + num_heads * qk_head_size, data.k,
                              true, -1);

    // Value (BxLxNxH_v) => V (BxNxLxH_v)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, kv_sequence_length, num_heads, v_head_size,
                              data.value, data.bias + 2 * num_heads * qk_head_size, data.v,
                              true, -1);

    data.qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }

  return Status::OK();
}

template <typename T>
bool NoQkvWorkspace_MHA_PackedQKV(AttentionData<T>& data) {
  // query, key and value are passed as Q, K and V for the following conditions.
  return nullptr != data.fused_runner && data.bias == nullptr;
}

// For MultiHeadAttention without past state, with packed QKV inputs
template <typename T>
Status PrepareQkv_MHA_PackedQKV(contrib::AttentionParameters& parameters,
                                AttentionData<T>& data,
                                cudaStream_t stream,
                                int max_threads_per_block) {
  assert(parameters.qkv_format == AttentionQkvFormat::QKV_BSN3H);
  assert(data.past_key == nullptr);
  assert(data.past_value == nullptr);
  assert(data.present_key == nullptr);
  assert(data.present_value == nullptr);
  assert(parameters.head_size == parameters.v_head_size);
  assert(data.fused_cross_attention_kernel == nullptr);
  assert(!parameters.is_unidirectional);
  assert(data.has_qkv_workspace == !NoQkvWorkspace_MHA_PackedQKV(data));

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  if (data.use_memory_efficient_attention || data.use_flash_attention ||
      data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    // unpack qkv to BSNH.
    constexpr int format = 4;
    T* qkv_add_bias = nullptr;
    LaunchAddBiasTranspose(stream, 3, format, max_threads_per_block,
                           batch_size, sequence_length, num_heads, qk_head_size,
                           data.query, data.bias, data.q,
                           true, v_head_size, qkv_add_bias, 3);
    data.qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  } else if (nullptr != data.fused_runner) {
    assert(nullptr == data.attention_bias);
    if (data.bias == nullptr) {
      // When there is no bias, we can directly use the original packed QKV input.
      // Need revisit this when we add support for causal.
      data.q = const_cast<T*>(data.query);
      data.k = nullptr;
      data.v = nullptr;
    } else {  // data.bias != nullptr
      AddBiasTransposePacked<T>(
          data.query, data.key, data.value, data.bias, data.q,
          batch_size, sequence_length,
          num_heads, qk_head_size, v_head_size,
          AttentionQkvFormat::QKV_TN3H, AttentionQkvFormat::QKV_TN3H,
          nullptr, batch_size * sequence_length,
          stream);
    }

    data.qkv_format = AttentionQkvFormat::QKV_BSN3H;
  } else {  // unfused kernel
    assert(data.IsUnfused());
    // unpack qkv to BNSH
    constexpr int format = 5;
    T* qkv_add_bias = nullptr;
    LaunchAddBiasTranspose(stream, 3, format, max_threads_per_block,
                           batch_size, sequence_length, num_heads, qk_head_size,
                           data.query, data.bias, data.q,
                           true, v_head_size, qkv_add_bias, 3);

    data.qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }

  return Status::OK();
}

// This shall be in sync with the following function PrepareQkv_MHA_PackedQKV.
template <typename T>
bool NoQkvWorkspace_MHA_PackedKV(AttentionData<T>& data) {
  return data.fused_cross_attention_kernel != nullptr;
}

// For MultiHeadAttention without past state, with packed KV inputs
template <typename T>
Status PrepareQkv_MHA_PackedKV(contrib::AttentionParameters& parameters,
                               AttentionData<T>& data,
                               cudaStream_t stream,
                               int max_threads_per_block) {
  assert(parameters.qkv_format == AttentionQkvFormat::Q_KV_BSNH_BSN2H);
  assert(data.bias == nullptr);
  assert(data.past_key == nullptr);
  assert(data.past_value == nullptr);
  assert(data.present_key == nullptr);
  assert(data.present_value == nullptr);
  assert(parameters.head_size == parameters.v_head_size);
  assert(data.fused_runner == nullptr);
  assert(data.has_qkv_workspace == !NoQkvWorkspace_MHA_PackedKV(data));

  const int batch_size = parameters.batch_size;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  if (data.use_memory_efficient_attention || data.use_flash_attention ||
      data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    // Note that there is no bias so we need not output query to q.
    data.q = const_cast<T*>(data.query);
    // Unpack kv to BSNH.
    constexpr int format = 4;
    T* qkv_add_bias = nullptr;
    const T* kv_bias = (data.bias == nullptr ? data.bias : data.bias + parameters.hidden_size);
    LaunchAddBiasTranspose(stream, 2, format, max_threads_per_block,
                           batch_size, kv_sequence_length, num_heads, qk_head_size,
                           data.key, kv_bias, data.k,
                           true, v_head_size, qkv_add_bias);
    data.qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  } else if (data.fused_cross_attention_kernel != nullptr) {
    data.qkv_format = AttentionQkvFormat::Q_KV_BSNH_BSN2H;
    data.q = const_cast<T*>(data.query);
    data.k = const_cast<T*>(data.key);
    data.v = nullptr;
  } else {  // unfused kernel
    assert(data.IsUnfused());
    // Transpose q from BSNH to BNSH. Note that there is no bias.
    ORT_RETURN_IF_ERROR(Transpose_BSNH_to_BNSH(batch_size, parameters.sequence_length, num_heads, qk_head_size,
                                               data.query, data.q, stream, max_threads_per_block));

    // Unpack kv to BNSH.
    constexpr int format = 5;
    T* qkv_add_bias = nullptr;
    const T* kv_bias = (data.bias == nullptr ? data.bias : data.bias + parameters.hidden_size);
    LaunchAddBiasTranspose(stream, 2, format, max_threads_per_block,
                           batch_size, kv_sequence_length, num_heads, qk_head_size,
                           data.key, kv_bias, data.k,
                           true, v_head_size, qkv_add_bias, 2);
    data.qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }

  return Status::OK();
}

// Prepare Q, K and V for MultiHeadAttention operator.
template <typename T>
Status PrepareQkv_MultiHeadAttention(contrib::AttentionParameters& parameters,
                                     AttentionData<T>& data,
                                     cudaStream_t stream,
                                     int max_threads_per_block) {
  switch (parameters.qkv_format) {
    case AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH:
      ORT_RETURN_IF_ERROR(PrepareQkv_MHA_Cross(parameters, data, stream, max_threads_per_block));
      break;
    case AttentionQkvFormat::Q_KV_BSNH_BSN2H:
      ORT_RETURN_IF_ERROR(PrepareQkv_MHA_PackedKV(parameters, data, stream, max_threads_per_block));
      break;
    case AttentionQkvFormat::QKV_BSN3H:
      ORT_RETURN_IF_ERROR(PrepareQkv_MHA_PackedQKV(parameters, data, stream, max_threads_per_block));
      break;
    case AttentionQkvFormat::Q_K_V_BSNH:
      if (data.past_key != nullptr || data.present_key != nullptr) {
        if (data.bias == nullptr) {
          ORT_RETURN_IF_ERROR(PrepareQkv_MHA_WithPast_NoBias(parameters, data, stream, max_threads_per_block));
        } else {
          ORT_RETURN_IF_ERROR(PrepareQkv_MHA_WithPast_Bias(parameters, data, stream, max_threads_per_block));
        }
      } else {  // no past state
        ORT_RETURN_IF_ERROR(PrepareQkv_MHA_NoPast(parameters, data, stream, max_threads_per_block));
      }
      break;
    default:
      ORT_THROW("Unsupported QKV format: ", parameters.qkv_format);
  }
  return Status::OK();
}

// Check whether there is no needed to have workspace for Q, K and V for MultiHeadAttention operator.
// Please make it in sync with PrepareQkv_MultiHeadAttention.
template <typename T>
bool NoQkvWorkspace(contrib::AttentionParameters& parameters, AttentionData<T>& data) {
  switch (parameters.qkv_format) {
    case AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH:
      return NoQkvWorkspace_MHA_Cross(data);
    case AttentionQkvFormat::Q_KV_BSNH_BSN2H:
      return NoQkvWorkspace_MHA_PackedKV(data);
    case AttentionQkvFormat::QKV_BSN3H:
      return NoQkvWorkspace_MHA_PackedQKV(data);
    case AttentionQkvFormat::Q_K_V_BSNH:
      if (data.past_key != nullptr || data.present_key != nullptr) {
        if (data.bias == nullptr) {
          return NoQkvWorkspace_MHA_WithPast_NoBias(data);
        } else {
          return NoQkvWorkspace_MHA_WithPast_Bias(data);
        }
      } else {  // no past state
        return NoQkvWorkspace_MHA_NoPast(data);
      }
    default:
      ORT_THROW("Unsupported QKV format: ", parameters.qkv_format);
  }
}

template <typename T>
Status PrepareQkv(contrib::AttentionParameters& parameters,
                  AttentionData<T>& data,
                  cudaStream_t stream,
                  int max_threads_per_block) {
  if (data.has_qkv_workspace) {
    const int size_per_batch_q = parameters.sequence_length * parameters.head_size;
    const int size_per_batch_k = parameters.kv_sequence_length * parameters.head_size;
    const int size_per_batch_v = parameters.kv_sequence_length * parameters.v_head_size;
    const int batches = parameters.batch_size * parameters.num_heads;
    const size_t elements_q = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_q);
    const size_t elements_k = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_k);
    const size_t elements_v = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_v);
    data.q = data.workspace;
    data.k = data.workspace + elements_q;
    data.v = data.k + elements_k;
    data.scratch = data.v + elements_v;
  } else {
    data.q = nullptr;
    data.k = nullptr;
    data.v = nullptr;
    data.scratch = data.workspace;
  }

#if DEBUG_TENSOR_LEVEL > 1
  DumpInputs(parameters, data);
#endif

  if (nullptr != data.gemm_buffer) {  // Attention operator
    ORT_RETURN_IF_ERROR(PrepareQkv_Attention<T>(parameters, data, stream, max_threads_per_block));
  } else {  // MultiHeadAttention operator
    ORT_RETURN_IF_ERROR(PrepareQkv_MultiHeadAttention<T>(parameters, data, stream, max_threads_per_block));
  }

  assert(data.qkv_format != AttentionQkvFormat::UNKNOWN);

#if DEBUG_TENSOR_LEVEL > 1
  DumpQkv(data);
#endif

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

// Template Instantiation
template bool NoQkvWorkspace<float>(contrib::AttentionParameters& parameters, AttentionData<float>& data);
template bool NoQkvWorkspace<half>(contrib::AttentionParameters& parameters, AttentionData<half>& data);

template Status PrepareQkv<float>(
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data,
    cudaStream_t stream,
    int max_threads_per_block);

template Status PrepareQkv<half>(
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data,
    cudaStream_t stream,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
