// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
Status PrepareQkv_Attention(contrib::AttentionParameters& parameters,
                            AttentionData<T>& data,
                            cudaStream_t stream,
                            int max_threads_per_block,
                            AttentionQkvFormat& qkv_format) {
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

  if (data.bias == nullptr) {
    assert(nullptr == fused_runner);
    // For quantized attention, bias has been added so only need transpose here.
    // gemm_buffer should be BxSx3xNxH => qkv: 3xBxNxSxH
    assert(qk_head_size == v_head_size);
    int matrix_to_trans = (past_present_share_buffer ? 1 : 3);
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, matrix_to_trans, sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.gemm_buffer, qkv, 3));
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  } else {
    // For fused TRT attention, transpose qkv to BxSxNx3xH (format 2)
    // For flash or memory efficient attention, transpose to 3xBxSxNxH (format 3)
    // For unfused kernel, transpose to 3xBxNxSxH (format 1)
    // For fused causal kernel, use format 1 since we need have K and V to update present state,
    //   at the same time, we update gemm_buffer BxSx3xNxH with bias which is used as input for fused causal kernel.
    const int format = (use_fused_kernel ? 2 : (use_flash_or_efficient_attention ? 3 : 1));
    qkv_format = use_fused_kernel
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
                           3, parameters.do_rotary, parameters.past_sequence_length);
  }
  return Status::OK();
}

// For MultiHeadAttention with past state
template <typename T>
Status PrepareQkv_MHA_WithPast(contrib::AttentionParameters& parameters,
                               AttentionData<T>& data,
                               cudaStream_t stream,
                               int max_threads_per_block,
                               T* q, T* k, T* v, AttentionQkvFormat& qkv_format) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  DUMP_TENSOR_INIT();

  if (data.bias == nullptr) {
    // Below logic does not support fused attention with past without bias
    // When there is past state, the format shall be BxNxSxH, so we disable fused attention when there is past.

    // cross attention with past state
    if (data.past_key != nullptr && data.present_key == nullptr) {
      assert(data.past_value != nullptr);
      assert(data.query != nullptr);
      assert(data.key == nullptr);
      assert(data.value == nullptr);
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.query, q));
    }
    // cross attention with present state or self attention with present state
    else if (data.past_key == nullptr && data.present_key != nullptr) {
      assert(data.past_value == nullptr);
      assert(data.present_value != nullptr);
      assert(data.query != nullptr);
      assert(data.key != nullptr);
      assert(data.value != nullptr);

      // TODO: supporting packed qkv for self attention may benefit performance
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.query, q));

      // TODO: supporting packed kv for cross attention may benefit performance
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.key, data.present_key));
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, v_head_size, num_heads,
                                         max_threads_per_block, false, data.value, data.present_value));
    }
    // self attention with past and present state
    else {
      assert(data.past_key != nullptr);
      assert(data.past_value != nullptr);
      assert(data.present_key != nullptr);
      assert(data.present_value != nullptr);
      assert(data.query != nullptr);
      assert(data.key != nullptr);
      assert(data.value != nullptr);
      // TODO: supporting packed qkv for self attention may benefit performance
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.query, q));
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.key, k));
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, v_head_size, num_heads,
                                         max_threads_per_block, false, data.value, v));
    }
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
#if USE_MEMORY_EFFICIENT_ATTENTION || USE_FLASH_ATTENTION
  // When past_key/past_value are inputted directly as key/value and there is no present_key/present_value
  else if ((data.use_memory_efficient_attention || data.use_flash_attention) &&
           data.past_key != nullptr &&
           data.past_value != nullptr &&
           parameters.pass_past_in_kv) {
    // Transpose past_key and past_value to use memory efficient attention

    // past_key (BxNxSxH) => temp_k_workspace (BxSxNxH)
    ORT_RETURN_IF_ERROR(LaunchTransCtx(stream, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.past_key, data.temp_k_workspace));
    // past_value (BxNxSxH_v) => temp_v_workspace (BxSxNxH_v)
    ORT_RETURN_IF_ERROR(LaunchTransCtx(stream, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.past_value, data.temp_v_workspace));

    // query => q, temp_k_workspace => k, temp_v_workspace => v
    LaunchAddBias(stream, max_threads_per_block,
                  batch_size, sequence_length, kv_sequence_length,
                  num_heads, qk_head_size, v_head_size,
                  data.bias, data.query, data.temp_k_workspace, data.temp_v_workspace, q, k, v);

    DUMP_TENSOR_D("q(BSNH)", q, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", k, batch_size, kv_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", v, batch_size, kv_sequence_length, num_heads, v_head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;

    data.past_key = nullptr;
    data.past_value = nullptr;
  }
  // When there is no past_key/past_value and there is present_key/present_value
  // (e.g. get initial kv to use as past_kv in the next iteration)
  else if ((data.use_memory_efficient_attention || data.use_flash_attention) &&
           data.present_key != nullptr &&
           data.present_value != nullptr) {
    // Use memory efficient attention kernel
    LaunchAddBias(stream, max_threads_per_block,
                  batch_size, sequence_length, kv_sequence_length,
                  num_heads, qk_head_size, v_head_size,
                  data.bias, data.query, data.key, data.value, q, data.temp_k_workspace, data.temp_v_workspace);

    // temp_k_workspace (BxSxNxH) => present_k (BxNxSxH)
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, qk_head_size, num_heads,
                                       max_threads_per_block, false, data.temp_k_workspace, data.present_key));

    // temp_v_workspace (BxSxNxH_v) => present_v (BxNxSxH_v)
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, v_head_size, num_heads,
                                       max_threads_per_block, false, data.temp_v_workspace, data.present_value));

    DUMP_TENSOR_D("q(BSNH)", q, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", data.temp_k_workspace, batch_size, kv_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", data.temp_v_workspace, batch_size, kv_sequence_length, num_heads, v_head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  }
#endif
  else {
    // Use unfused kernel for Q, use unfused kernel for K and V if needed
    constexpr int format = 0;
    // Query (BxSxNxH) => Q (BxNxSxH)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, sequence_length, num_heads, qk_head_size,
                              data.query, data.bias, q,
                              true, -1);

    if (!parameters.pass_past_in_kv) {
      T* k_dest = (data.past_key == nullptr && data.present_key != nullptr) ? data.present_key : k;
      T* v_dest = (data.past_value == nullptr && data.present_value != nullptr) ? data.present_value : v;

      // Key (BxLxNxH) => K (BxNxLxH)
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, kv_sequence_length, num_heads, qk_head_size,
                                data.key, data.bias + num_heads * qk_head_size, k_dest,
                                true, -1);

      // Value (BxLxNxH_v) => V (BxNxLxH_v)
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, kv_sequence_length, num_heads, v_head_size,
                                data.value, data.bias + 2 * num_heads * qk_head_size, v_dest,
                                true, -1);

      DUMP_TENSOR_D("q(BNSH)", q, batch_size, num_heads, sequence_length, qk_head_size);
      DUMP_TENSOR_D("k(BNSH)", k_dest, batch_size, num_heads, kv_sequence_length, qk_head_size);
      DUMP_TENSOR_D("v(BNSH)", v_dest, batch_size, num_heads, kv_sequence_length, v_head_size);
    }
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
  return Status::OK();
}

// For MultiHeadAttention without past state, with packed QKV inputs
template <typename T>
Status PrepareQkv_MHA_PackedQKV(contrib::AttentionParameters& parameters,
                                AttentionData<T>& data,
                                cudaStream_t stream,
                                int max_threads_per_block,
                                T* q, T* k, T* v, AttentionQkvFormat& qkv_format) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  void* fused_runner = data.fused_runner;

  T* qkv = data.workspace;

  bool use_fused_kernel = (nullptr != fused_runner && !parameters.is_unidirectional);

  assert(data.bias == nullptr);
  assert(qk_head_size == v_head_size);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("packed_qkv", data.query, batch_size * sequence_length, num_heads, 3, qk_head_size);

  if (data.use_memory_efficient_attention || data.use_flash_attention) {
    // unpack qkv to BSNH. Note that there is no bias so we need not output query to q.
    constexpr int format = 4;
    T* qkv_add_bias = nullptr;
    LaunchAddBiasTranspose(stream, 3, format, max_threads_per_block,
                           batch_size, sequence_length, num_heads, qk_head_size,
                           data.query, data.bias, qkv,
                           true, v_head_size, qkv_add_bias, 3);
    DUMP_TENSOR_D("q(BSNH)", q, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", k, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", v, batch_size, sequence_length, num_heads, v_head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  } else {
    if (!use_fused_kernel) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, NOT_IMPLEMENTED,
          "packed QKV format is not implemented for current GPU. Please disable it in fusion options.");
    }

    qkv_format = AttentionQkvFormat::QKV_BSN3H;
  }
  return Status::OK();
}

// For MultiHeadAttention without past state, with packed KV inputs
template <typename T>
Status PrepareQkv_MHA_PackedKV(contrib::AttentionParameters& parameters,
                               AttentionData<T>& data,
                               cudaStream_t stream,
                               int max_threads_per_block,
                               T* q, T* k, T* v, AttentionQkvFormat& qkv_format) {
  const int batch_size = parameters.batch_size;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  // TODO: unpack kv to BNSH for unfused kernel so that we can remove the following constraint.
  // CheckInputs verified this constraint.
  assert(data.bias == nullptr);
  assert(qk_head_size == v_head_size);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("packed_kv", data.key, batch_size * kv_sequence_length, num_heads, 2, qk_head_size);

  if (data.use_memory_efficient_attention || data.use_flash_attention) {
    // unpack kv to BSNH. Note that there is no bias so we need not output query to q.
    constexpr int format = 4;
    T* qkv_add_bias = nullptr;
    const T* kv_bias = (data.bias == nullptr ? data.bias : data.bias + parameters.hidden_size);
    LaunchAddBiasTranspose(stream, 2, format, max_threads_per_block,
                           batch_size, kv_sequence_length, num_heads, qk_head_size,
                           data.key, kv_bias, k,
                           true, v_head_size, qkv_add_bias, 2);
    DUMP_TENSOR_D("k(BSNH)", k, batch_size, kv_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", v, batch_size, kv_sequence_length, num_heads, v_head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  } else {
    if (data.fused_cross_attention_kernel == nullptr) {
      return ORT_MAKE_STATUS(
          ONNXRUNTIME, NOT_IMPLEMENTED,
          "packed KV format is not implemented for current GPU. Please disable packed kv in fusion options.");
    }

    qkv_format = AttentionQkvFormat::Q_KV_BSNH_BSN2H;
  }
  return Status::OK();
}

// For MultiHeadAttention without past state, with Q, K and V inputs
template <typename T>
Status PrepareQkv_MHA_NotPacked(contrib::AttentionParameters& parameters,
                                AttentionData<T>& data,
                                cudaStream_t stream,
                                int max_threads_per_block,
                                T* q, T* k, T* v, AttentionQkvFormat& qkv_format) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  void* fused_runner = data.fused_runner;

  T* qkv = data.workspace;

  bool use_fused_kernel = (nullptr != fused_runner && !parameters.is_unidirectional);
  bool use_fused_causal = (nullptr != fused_runner && parameters.is_unidirectional);

  // gemm_buffer == nullptr and not packed
  assert(data.query != nullptr && data.key != nullptr && data.value != nullptr);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("query", data.query, batch_size, sequence_length, num_heads, qk_head_size);
  DUMP_TENSOR_D("key", data.key, batch_size, kv_sequence_length, num_heads, qk_head_size);
  DUMP_TENSOR_D("value", data.value, batch_size, kv_sequence_length, num_heads, v_head_size);

#if DUMP_TENSOR_LEVEL > 1
  if (data.bias != nullptr) {
    DUMP_TENSOR_D("query_bias", data.bias, num_heads, qk_head_size);
    DUMP_TENSOR_D("key_bias", data.bias + num_heads * qk_head_size, num_heads, qk_head_size);
    DUMP_TENSOR_D("value_bias", data.bias + 2 * num_heads * qk_head_size, num_heads, v_head_size);
  }
#endif

  if (data.relative_position_bias != nullptr && parameters.broadcast_res_pos_bias) {
    DUMP_TENSOR_D("relative_position_bias", data.relative_position_bias,
                  num_heads, sequence_length, kv_sequence_length);
  }

  if (data.mask_index != nullptr && parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START) {
    DUMP_TENSOR_D("mask_index", data.mask_index, 3 * batch_size + 2, 1);
  }

  if (data.fused_cross_attention_kernel != nullptr) {
    assert(qk_head_size == v_head_size);

    // For fused cross attention, besides adding bias, K and V needed to be packed:
    //   K (BxSxNxH), V (BxSxNxH) => BxSxNx2xH
    LaunchAddBiasTransposeTrt(
        stream, max_threads_per_block,
        batch_size, sequence_length,
        num_heads, qk_head_size,
        data.bias, data.query, data.key, data.value, qkv, true, kv_sequence_length);

    qkv_format = AttentionQkvFormat::Q_KV_BSNH_BSN2H;
  }
#if USE_MEMORY_EFFICIENT_ATTENTION || USE_FLASH_ATTENTION
  else if (data.use_memory_efficient_attention || data.use_flash_attention) {
    LaunchAddBias(stream, max_threads_per_block,
                  batch_size, sequence_length, kv_sequence_length,
                  num_heads, qk_head_size, v_head_size,
                  data.bias, data.query, data.key, data.value, q, k, v);

    DUMP_TENSOR_D("q(BSNH)", q, batch_size, sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("k(BSNH)", k, batch_size, kv_sequence_length, num_heads, qk_head_size);
    DUMP_TENSOR_D("v(BSNH)", v, batch_size, kv_sequence_length, num_heads, v_head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  }
#endif
  else if (use_fused_kernel) {
    assert(qk_head_size == v_head_size);

    // Q (BxSxNxH), K (BxSxNxH), V (BxSxNxH) => BxSxNx(H + H + H)
    LaunchAddBiasTransposeTrt(
        stream, max_threads_per_block,
        batch_size, sequence_length,
        num_heads, qk_head_size,
        data.bias, data.query, data.key, data.value, qkv, false, kv_sequence_length);
    DUMP_TENSOR_D("qkv(BSN3H)", qkv, batch_size, sequence_length, num_heads, 2 * qk_head_size + v_head_size);

    qkv_format = AttentionQkvFormat::QKV_BSN3H;
  } else {  // unfused kernel
    ORT_ENFORCE(!use_fused_causal, "MultiHeadAttention has not enabled fused causal");

    // Query (BxSxNxH) => Q (BxNxSxH)
    constexpr int format = 0;
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size,
        data.query, data.bias, q,
        true, -1);

    // Key (BxLxNxH) => K (BxNxLxH)
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, kv_sequence_length, num_heads, qk_head_size,
        data.key, nullptr == data.bias ? nullptr : data.bias + num_heads * qk_head_size, k,
        true, -1);

    // Value (BxLxNxH_v) => K (BxNxLxH_v)
    LaunchAddBiasTranspose<T>(
        stream, 1, format, max_threads_per_block,
        batch_size, kv_sequence_length, num_heads, v_head_size,
        data.value, nullptr == data.bias ? nullptr : data.bias + 2 * num_heads * qk_head_size, v,
        true, -1);

    DUMP_TENSOR_D("q(BNSH)", q, batch_size, num_heads, sequence_length, qk_head_size);
    DUMP_TENSOR_D("k(BNSH)", k, batch_size, num_heads, kv_sequence_length, qk_head_size);
    DUMP_TENSOR_D("v(BNSH)", v, batch_size, num_heads, kv_sequence_length, v_head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
  return Status::OK();
}

template <typename T>
Status PrepareQkv(contrib::AttentionParameters& parameters,
                  AttentionData<T>& data,
                  cudaStream_t stream,
                  int max_threads_per_block) {
  data.scratch = data.workspace;
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
  }

  if (nullptr != data.gemm_buffer) {  // Attention operator
    ORT_RETURN_IF_ERROR(PrepareQkv_Attention<T>(parameters, data, stream, max_threads_per_block,
                                                data.qkv_format));
  } else if (data.past_key != nullptr || data.present_key != nullptr) {  // mha operator with past/present state
    ORT_RETURN_IF_ERROR(PrepareQkv_MHA_WithPast(parameters, data, stream, max_threads_per_block,
                                                data.q, data.k, data.v, data.qkv_format));
  } else if (data.key == nullptr) {  // multihead attention operator, no past, packed qkv
    ORT_RETURN_IF_ERROR(PrepareQkv_MHA_PackedQKV(parameters, data, stream, max_threads_per_block,
                                                 data.q, data.k, data.v, data.qkv_format));
  } else if (data.value == nullptr) {  // multihead attention operator, no past, packed kv
    ORT_RETURN_IF_ERROR(PrepareQkv_MHA_PackedKV(parameters, data, stream, max_threads_per_block,
                                                data.q, data.k, data.v, data.qkv_format));
  } else {  // multihead attention operator, no past, separated Q/K/V inputs
    ORT_RETURN_IF_ERROR(PrepareQkv_MHA_NotPacked(parameters, data, stream, max_threads_per_block,
                                                 data.q, data.k, data.v, data.qkv_format));
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

// Template Instantiation
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
