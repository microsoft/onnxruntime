/*
 The implementation of this file is based on qkvToContext plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Modifications:
// (1) support GPT-2 past state, unidirectional mask and 4D attention mask from Megatron
// (2) support 2D attention mask
// (3) allow persistent softmax from PyTorch for debugging purpose.
// (4) support different input hidden size and model hidden size for pruned model
// (5) support different hidden sizes of Q/K and V
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/platform/env_var_utils.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "contrib_ops/cuda/bert/attention_qk.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/lean_attention/lean_api.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/utils/dump_cuda_tensor.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::attention_softmax_cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t kMemoryAlignment = 256;

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t AlignSize(size_t bytes) {
  const size_t bytesAligned = AlignTo(bytes, kMemoryAlignment);
  return bytesAligned;
}

const int32_t* CumulatedSequenceLengthCache::TryGet(int batch_size, int32_t seq_len, cudaStream_t stream) {
  if (this->sequence_length == 0 && seq_len > 0) {
    // Initialize only once with sequence length in the first request.
    std::call_once(init_once_flag_, [&]() {
      ORT_ENFORCE(buffer.get() != nullptr && this->max_batch_size > 0);
      LaunchTrtSequenceOffset(reinterpret_cast<int32_t*>(buffer.get()), nullptr,
                              this->max_batch_size, seq_len, stream);
      // Syncronize to ensure thread-safe since other thread will not wait for the above kernel finish.
      // Otherwise, the data might be consumed by other threads before it is ready and causes data race issue.
      cudaStreamSynchronize(stream);
      this->sequence_length = seq_len;
    });
  }

  if (this->sequence_length == seq_len && batch_size <= this->max_batch_size) {
    return reinterpret_cast<const int32_t*>(buffer.get());
  }

  return nullptr;
}

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t total_sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * total_sequence_length;
  return AlignSize(bytes);
}

size_t GetSequenceOffsetSize(int batch_size, bool has_padding) {
  // There are batch_size + 1 offsets Without padding (or padding removed), and 2 * batch_size + 1 with padding.
  size_t bytes = sizeof(int) * ((has_padding ? 2 * batch_size : batch_size) + 1);
  return AlignSize(bytes);
  ;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    size_t kv_sequence_length,
    size_t total_sequence_length,
    void* fused_runner,
    bool use_flash_attention,
    bool use_lean_attention,
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention,
    bool use_cudnn_flash_attention,
    bool no_qkv_workspace) {
  // Note that q, k and v might need alignment for fused attention kernels.
  const size_t qkv_size = element_size * batch_size * num_heads *
                          ((sequence_length + kv_sequence_length) * qk_head_size + kv_sequence_length * v_head_size);
  const size_t qkv_bytes = no_qkv_workspace ? 0 : qkv_size;

#if USE_FLASH_ATTENTION
  if (use_flash_attention) {
    return qkv_bytes;
  }
#else
  ORT_UNUSED_PARAMETER(use_flash_attention);
#endif

#if USE_LEAN_ATTENTION
  if (use_lean_attention) {
    return qkv_bytes;
  }
#else
  ORT_UNUSED_PARAMETER(use_lean_attention);
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (use_memory_efficient_attention) {
    size_t fmha_buffer_bytes = 0;
    if (MemoryEfficientAttentionParams::need_workspace(v_head_size, element_size == sizeof(float))) {
      fmha_buffer_bytes = batch_size * sequence_length * num_heads * v_head_size * sizeof(float);
    }

    return qkv_bytes + fmha_buffer_bytes;
  }
#else
  ORT_UNUSED_PARAMETER(use_memory_efficient_attention);
#endif

  if (fused_runner != nullptr) {
    return qkv_bytes + GetSequenceOffsetSize(static_cast<int>(batch_size), true);
  }

  if (use_fused_cross_attention) {
    return qkv_bytes + 2 * GetSequenceOffsetSize(static_cast<int>(batch_size), true);
  }

  if (use_cudnn_flash_attention) {
    return qkv_bytes;
  }

  return qkv_bytes + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length,
                                                 total_sequence_length);
}

template <typename T>
Status FusedTrtCrossAttention(
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data) {
  assert(data.qkv_format == AttentionQkvFormat::Q_KV_BSNH_BSN2H);

  // We only enable fused cross attention when there is no key padding mask.
  // Otherwise, key have effective batch size 2 * batch_size, which is different from batch_size of query.
  assert(data.mask_index == nullptr);
  assert(data.scratch != nullptr);
  assert(data.q != nullptr);
  assert(data.k != nullptr);

#ifndef NDEBUG
  char* scratch_end = reinterpret_cast<char*>(data.scratch) + 2 * GetSequenceOffsetSize(parameters.batch_size, false);
  char* buffer_end = reinterpret_cast<char*>(data.workspace) + data.workspace_bytes;
  assert(scratch_end <= buffer_end);
#endif
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;

  int32_t* q_sequence_offset = const_cast<int32_t*>(data.cumulated_sequence_length_q_cache);
  if (q_sequence_offset == nullptr) {
    q_sequence_offset = reinterpret_cast<int*>(data.scratch);
    LaunchTrtSequenceOffset(q_sequence_offset, data.mask_index, batch_size, sequence_length, stream);
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q_sequence_offset", q_sequence_offset, 1, batch_size + 1);

  int32_t* kv_sequence_offset = const_cast<int32_t*>(data.cumulated_sequence_length_kv_cache);
  if (kv_sequence_offset == nullptr) {
    int* scratch = reinterpret_cast<int*>(data.scratch) + (GetSequenceOffsetSize(batch_size, false) / sizeof(int));
    kv_sequence_offset = reinterpret_cast<int*>(scratch);
    LaunchTrtSequenceOffset(kv_sequence_offset, data.mask_index, batch_size, parameters.kv_sequence_length, stream);
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  DUMP_TENSOR_D("kv_sequence_offset", kv_sequence_offset, 1, batch_size + 1);

  FusedMultiHeadCrossAttentionKernel const* cross_attention_kernel =
      reinterpret_cast<FusedMultiHeadCrossAttentionKernel const*>(data.fused_cross_attention_kernel);

  run_fused_cross_attention(
      data.q,                         // Q
      data.k,                         // packed KV
      q_sequence_offset,              // cumulated sequence length of Q
      kv_sequence_offset,             // cumulated sequence length of KV
      data.output,                    // output
      cross_attention_kernel,         // kernels
      batch_size,                     // batch size
      parameters.num_heads,           // number of heads
      parameters.head_size,           // head size of Q/K/V
      sequence_length,                // sequence length of Q
      parameters.kv_sequence_length,  // sequence length of KV
      stream);

  return Status::OK();
}

template <>
Status FusedTrtCrossAttention<float>(
    cudaStream_t /*stream*/,
    contrib::AttentionParameters& /*parameters*/,
    AttentionData<float>& /*data*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                         "Trt fused cross attention does not support float tensor");
}

template <typename T>
Status FusedTrtSelfAttention(
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data) {
  assert(data.scratch != nullptr);
#ifndef NDEBUG
  char* scratch_end = reinterpret_cast<char*>(data.scratch) + GetSequenceOffsetSize(parameters.batch_size, false);
  char* buffer_end = reinterpret_cast<char*>(data.workspace) + data.workspace_bytes;
  assert(scratch_end <= buffer_end);
#endif

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const bool causal = parameters.is_unidirectional;

  const int32_t* sequence_offset = data.cumulated_sequence_length_q_cache;
  if (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
    LaunchTrtSequenceOffset2d(reinterpret_cast<int*>(data.scratch), data.mask_index, batch_size, sequence_length, stream);
    sequence_offset = reinterpret_cast<const int*>(data.scratch);
  } else {
    if (sequence_offset == nullptr) {
      LaunchTrtSequenceOffset(reinterpret_cast<int*>(data.scratch), data.mask_index, batch_size, sequence_length, stream);
      sequence_offset = reinterpret_cast<const int*>(data.scratch);
    }
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("sequence_offset", sequence_offset, 1, (data.mask_index != nullptr ? 2 : 1) * batch_size + 1);

  FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(data.fused_runner);

  const int s = causal ? sequence_length : fused_fp16_runner->NormalizeSequenceLength(sequence_length);

  // B = 2 * batch_size when there is padding in input, and B = batch_size when padding is removed.
  const int b = (nullptr == data.mask_index ? batch_size : 2 * batch_size);

  if (!causal) {
    assert(data.qkv_format == AttentionQkvFormat::QKV_BSN3H);
    fused_fp16_runner->Run(b, s, data.q, sequence_offset, data.output, stream);
  } else {
    assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH);
    fused_fp16_runner->Run(b, s, data.gemm_buffer, sequence_offset, data.output, stream);
  }

  return Status::OK();
}

// Template Specialization for float type
template <>
Status FusedTrtSelfAttention<float>(
    cudaStream_t /*stream*/,
    contrib::AttentionParameters& /*parameters*/,
    AttentionData<float>& /*data*/) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                         "Trt fused attention does not support float tensor");
}

#if USE_FLASH_ATTENTION
template <typename T>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    float scale) {
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH ||
         data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  assert(nullptr == data.mask_index);
  assert(nullptr == data.attention_bias);
  assert(parameters.head_size == parameters.v_head_size);

  constexpr bool is_bf16 = false;
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
      device_prop, stream, data.q, data.k, data.v, data.output, reinterpret_cast<void*>(data.softmax_lse),
      parameters.batch_size, parameters.num_heads, parameters.num_heads, parameters.head_size,
      parameters.sequence_length, parameters.total_sequence_length, scale, 0.0, parameters.is_unidirectional, is_bf16,
      false, data.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum),
      reinterpret_cast<void*>(data.out_accum), data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH));

  return Status::OK();
}

template <>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data,
    float scale) {
  ORT_UNUSED_PARAMETER(device_prop);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(parameters);
  ORT_UNUSED_PARAMETER(data);
  ORT_UNUSED_PARAMETER(scale);
  return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED, "flash attention does not support float tensor");
}
#endif

#if USE_LEAN_ATTENTION
template <typename T>
Status LeanAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    float scale) {
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH ||
         data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  assert(nullptr == data.mask_index);
  assert(nullptr == data.attention_bias);
  assert(parameters.head_size == parameters.v_head_size);

  constexpr bool is_bf16 = false;

  ORT_RETURN_IF_ERROR(onnxruntime::lean::mha_fwd_kvcache(
    device_prop, stream,
    data.q,
    data.k, // k_cache
    data.v, // v_cache
    nullptr,  // new_k (we have appended new_k to k_cache)
    nullptr,  // new_v (we have appended new_v to k_cache)
    data.output,
    reinterpret_cast<void*>(data.softmax_lse),
    nullptr, // seqlens_k
    nullptr, // cos_cache
    nullptr, // sin_cache
    nullptr, // block_table
    parameters.batch_size,
    parameters.num_heads,
    parameters.num_heads, // num_heads_k
    parameters.head_size,
    parameters.sequence_length, // seqlen_q
    parameters.total_sequence_length, // seqlen_k
    0, // seqlen_k_new
    0, // rotary_dim
    scale, // softmax_scale
    parameters.is_unidirectional,
    is_bf16,
    false, // past_bsnh
    data.num_splits,
    data.grid_dim_z,
    data.max_tiles_per_tb,
    data.high_load_tbs,
    data.tiles_per_head,
    reinterpret_cast<void*>(data.softmax_lse_accum),
    reinterpret_cast<void*>(data.out_accum),
    data.lean_sync_flag,
    -1, // local_window_size
    false, // is_rotary_interleaved
    false // is_packed_qkv
    ));

  return Status::OK();
}

template <>
Status LeanAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data,
    float scale) {
  ORT_UNUSED_PARAMETER(device_prop);
  ORT_UNUSED_PARAMETER(stream);
  ORT_UNUSED_PARAMETER(parameters);
  ORT_UNUSED_PARAMETER(data);
  ORT_UNUSED_PARAMETER(scale);
  return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED, "lean attention does not support float tensor");
}
#endif



template <typename T>
Status CudnnFlashAttention(
    cudnnHandle_t cudnn_handle,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    float scale) {
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH ||
         data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH ||
         data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH);
  assert(parameters.mask_type == AttentionMaskType::MASK_NONE ||
         parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN);
  constexpr bool is_bf16 = false;

  T* attention_bias = const_cast<T*>(data.attention_bias);
  int* mask_sequence_lengths_kv = const_cast<int*>(data.mask_index);

  cudnn_sdpa::run(
      data.output,
      data.q,
      data.k,
      data.v,
      attention_bias,
      nullptr,                                 // (optional) mask_sequence_lengths_q
      mask_sequence_lengths_kv,                // (optional) mask_sequence_lengths_kv
      parameters.batch_size,
      parameters.num_heads,                    // num_heads_q,
      parameters.num_heads,                    // num_heads_kv,
      parameters.head_size,                    // head_size_qk
      parameters.v_head_size,                  // head_size_v
      parameters.sequence_length,              // sequence_length_q
      parameters.total_sequence_length,        // sequence_length_kv
      scale,                                   // scaling factor applied prior softmax
      parameters.is_unidirectional,            // causal
      is_bf16,                                 // True if bfloat16, otherwise float16
      parameters.broadcast_attn_bias_dim_0,    // broadcast attention bias dimension 0 or not
      parameters.broadcast_attn_bias_dim_1,    // broadcast attention bias dimension 1 or not
      0,                                       // sliding window length. 0 means no sliding window.
      data.qkv_format,
      cudnn_handle,
      ort_stream,
      data.allocator);

  return Status::OK();
}

template <>
Status CudnnFlashAttention(
    cudnnHandle_t cudnn_handle,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data,
    float scale) {
  ORT_UNUSED_PARAMETER(cudnn_handle);
  ORT_UNUSED_PARAMETER(ort_stream);
  ORT_UNUSED_PARAMETER(parameters);
  ORT_UNUSED_PARAMETER(data);
  ORT_UNUSED_PARAMETER(scale);
  return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED,
                         "cudnn flash attention does not support float tensor");
}

#if USE_MEMORY_EFFICIENT_ATTENTION
template <typename T>
Status EfficientAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    float scale) {
  // We only enable fused cross attention when there is no key padding mask.
  // Otherwise, key have effective batch size 2 * batch_size, which is different from batch_size of query.
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH ||
         data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  assert(parameters.mask_type == AttentionMaskType::MASK_NONE ||
         parameters.mask_type == AttentionMaskType::MASK_1D_KEY_SEQ_LEN_START);

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_half = sizeof(T) == 2;
  p.batch_size = parameters.batch_size;
  p.num_heads = parameters.num_heads;
  p.sequence_length = parameters.sequence_length;
  p.kv_sequence_length = parameters.total_sequence_length;
  p.max_sequence_length = parameters.total_sequence_length;
  p.qk_head_size = parameters.head_size;
  p.v_head_size = parameters.v_head_size;
  p.causal = parameters.is_unidirectional;
  p.scale = scale;
  p.use_smooth_softmax = false;

  if (nullptr == data.mask_index) {
    p.seqlen_k_ptr = nullptr;
    p.seqstart_q_ptr = nullptr;
    p.seqstart_k_ptr = nullptr;
  } else {
    p.seqlen_k_ptr = const_cast<int32_t*>(reinterpret_cast<const int32_t*>(data.mask_index));
    p.seqstart_q_ptr = p.seqlen_k_ptr + parameters.batch_size;
    p.seqstart_k_ptr = p.seqlen_k_ptr + 2 * parameters.batch_size + 1;
  }

  p.query = data.q;
  p.key = data.k;
  p.value = data.v;

  p.attn_bias = data.attention_bias;
  p.broadcast_attn_bias_dim_0 = parameters.broadcast_attn_bias_dim_0;
  p.broadcast_attn_bias_dim_1 = parameters.broadcast_attn_bias_dim_1;

  p.output = data.output;
  p.is_kv_bsnh = data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(parameters.v_head_size, sizeof(T) == sizeof(float))
                    ? data.scratch
                    : nullptr;
  p.stream = stream;
  p.has_custom_right_padding = false;
  run_memory_efficient_attention(p);

  return Status::OK();
}
#endif

template <typename T, typename QK>
Status LaunchDecoderMaskedMultiHeadAttention(
  const DecoderMaskedMultiHeadAttentionParameters& parameters,
  cudaStream_t stream,
  const int head_size) {

  DUMP_STRING_INIT();
  DUMP_STRING("DMMHA parameters...");
  DUMP_STRING("is_mha = ", (parameters.is_mha == true));
  DUMP_STRING("is_cross_attention = ", (parameters.is_cross_attention == true));
  DUMP_STRING("is_packed_qkv = ", (parameters.is_packed_qkv == true));
  DUMP_STRING("kv_data_in_flight = ", (parameters.kv_data_in_flight == true));

  DUMP_STRING("Batch size = ", parameters.batch_size);
  DUMP_STRING("Sequence length = ", parameters.sequence_length);
  DUMP_STRING("Num heads = ", parameters.num_heads);
  DUMP_STRING("Head size = ", parameters.head_size);
  DUMP_STRING("Hidden size = ", parameters.hidden_size);

  DUMP_STRING("Past sequence length = ", parameters.past_sequence_length);
  DUMP_STRING("KV sequence length = ", parameters.kv_sequence_length);
  DUMP_STRING("Total sequence length = ", parameters.total_sequence_length);
  DUMP_STRING("Max sequence length = ", parameters.max_sequence_length);

  DUMP_STRING("parameters.k is null = ", (parameters.k == nullptr));
  DUMP_STRING("parameters.v is null = ", (parameters.v == nullptr));
  DUMP_STRING("parameters.k_cache is null = ", (parameters.k_cache == nullptr));
  DUMP_STRING("parameters.v_cache is null = ", (parameters.v_cache == nullptr));

  DUMP_STRING("parameters.q_bias is null = ", (parameters.q_bias == nullptr));
  DUMP_STRING("parameters.k_bias is null = ", (parameters.k_bias == nullptr));
  DUMP_STRING("parameters.v_bias is null = ", (parameters.v_bias == nullptr));

  DUMP_STRING("parameters.attention_bias is null = ", (parameters.attention_bias == nullptr));
  DUMP_STRING("Scale = ", parameters.scale);
  DUMP_STRING("Mask is null = ", (parameters.mask == nullptr));
  DUMP_STRING("Mask filter value = ", parameters.mask_filter_value);

  DUMP_STRING("Beam width = ", parameters.beam_width);
  DUMP_STRING("parameters.cache_indir is null = ", (parameters.cache_indir == nullptr));
  DUMP_STRING("parameters.out_qk is null = ", (parameters.out_qk == nullptr));

  switch (head_size) {
    case 32:
      mmha_launch_kernel<T, QK, 32>(parameters, stream);
      break;

    case 64:
      mmha_launch_kernel<T, QK, 64>(parameters, stream);
      break;

    case 128:
      mmha_launch_kernel<T, QK, 128>(parameters, stream);
      break;

    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                             "Unsupported head size in DecoderMaskedMultiHeadAttention. Got head size: ",
                             head_size);
  }

  return Status::OK();
}

template <typename T, typename QK>
Status DecoderMaskedMultiHeadAttention(
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    float scale) {
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH ||
         data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  assert(parameters.mask_type == AttentionMaskType::MASK_NONE ||
         parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING);
  assert(parameters.head_size == parameters.v_head_size);

  DecoderMaskedMultiHeadAttentionParameters p;
  p.is_mha = true;
  p.is_cross_attention = (data.past_key == nullptr && data.present_key == nullptr);
  p.is_packed_qkv = false;
  p.kv_data_in_flight = ParseEnvironmentVariableWithDefault<bool>(attention::kDecoderMaskedAttentionLoadKVDataInFlight, false);

  p.batch_size = parameters.batch_size;
  p.sequence_length = parameters.sequence_length;
  p.num_heads = parameters.num_heads;
  p.head_size = parameters.head_size;
  p.hidden_size = parameters.hidden_size;

  p.past_sequence_length = parameters.past_sequence_length;
  p.kv_sequence_length = parameters.kv_sequence_length;
  p.total_sequence_length = p.is_cross_attention ? parameters.kv_sequence_length : parameters.total_sequence_length;
  p.max_sequence_length = p.is_cross_attention ? parameters.kv_sequence_length : parameters.max_sequence_length;

  p.q = data.q;
  p.k = p.is_cross_attention ? nullptr : data.k;
  p.v = p.is_cross_attention ? nullptr : data.v;
  p.k_cache = p.is_cross_attention ? data.k : data.present_key;
  p.v_cache = p.is_cross_attention ? data.v : data.present_value;

  p.q_bias = data.q_bias;
  p.k_bias = data.k_bias;
  p.v_bias = data.v_bias;

  p.attention_bias = const_cast<T*>(data.attention_bias);
  p.broadcast_attn_bias_dim_0 = parameters.broadcast_attn_bias_dim_0;
  p.broadcast_attn_bias_dim_1 = parameters.broadcast_attn_bias_dim_1;

  p.scale = scale;
  p.mask = data.mask_index;
  p.mask_filter_value = parameters.mask_filter_value;

  p.beam_width = parameters.beam_width;
  p.cache_indir = data.cache_indirection;

  p.out = data.output;
  p.out_qk = data.output_qk;

  // DecoderMaskedMultiHeadAttention(T, QK) is defined for:
  // T = float, QK = float
  // T = float, QK = half
  // T = uint16_t, QK = float
  // T = uint16_t, QK = half
  if (std::is_same<T, float>::value) {
    return LaunchDecoderMaskedMultiHeadAttention<float, QK>(p, stream, parameters.head_size);
  }
  if (std::is_same<T, half>::value) {
    return LaunchDecoderMaskedMultiHeadAttention<uint16_t, QK>(p, stream, parameters.head_size);
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "DecoderMaskedMultiHeadAttention is only implemented for float32 and float16.");
}

template <typename T, typename QK>
Status UnfusedAttention(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data,
    float scale) {
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH);

  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int total_sequence_length = parameters.total_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  const int batches = batch_size * num_heads;

  const int* mask_index = data.mask_index;
  gsl::span<const int64_t>& mask_index_dims = data.mask_index_dims;

  // Raw attention mask could be 2D (BxT) or 3D (BxSxT) or 4D(Bx1xMxM), where M is the max sequence length.
  bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch: BxNxSxT
  // Q: BxNxSxH, K (present_k): BxNxTxH, Q*K': BxNxSxT
  float one = 1.0f;
  float zero = 0.f;

  float alpha = use_raw_attention_mask ? one : scale;

  cublasSetStream(cublas, stream);

  const int present_sequence_length = parameters.past_present_share_buffer
                                          ? parameters.max_sequence_length
                                          : total_sequence_length;
  const int present_size_per_batch_k = present_sequence_length * qk_head_size;
  const int present_size_per_batch_v = present_sequence_length * v_head_size;

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q", data.q, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("k", data.k, batch_size, num_heads, total_sequence_length, qk_head_size);
  DUMP_TENSOR_D("v", data.v, batch_size, num_heads, total_sequence_length, v_head_size);
  DUMP_TENSOR_D("mask_index", mask_index, mask_index_dims);

  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N,
      total_sequence_length, sequence_length, qk_head_size,
      &alpha, data.k, qk_head_size, present_size_per_batch_k,
      data.q, qk_head_size, sequence_length * qk_head_size,
      &zero, data.scratch, total_sequence_length, sequence_length * total_sequence_length, batches,
      device_prop, parameters.use_tf32));

  DUMP_TENSOR_D("QK", data.scratch, batch_size, num_heads, sequence_length, total_sequence_length);

  constexpr size_t element_size = sizeof(T);
  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, total_sequence_length);
  T* scratch2 = data.scratch + (bytes / element_size);

  const bool broadcast_attn_bias_dim_0 = parameters.broadcast_attn_bias_dim_0;
  const bool broadcast_attn_bias_dim_1 = parameters.broadcast_attn_bias_dim_1;

  // Apply softmax and store result R to scratch2: BxNxSxT
  if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
    const int mask_dimension = static_cast<int>(mask_index_dims.size());

    // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax used in Torch.
    const TransformerOptions* options = TransformerOptions::GetInstance();
    bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

    // replace Q*K' in place with masked score for persistent softmax.
    T* persistent_softmax_workspace = data.scratch;
    ORT_RETURN_IF_ERROR(
        ComputeSoftmaxWithRawMask<T>(
            ort_stream, total_sequence_length, sequence_length, batch_size, num_heads,
            mask_index, nullptr, data.attention_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
            data.scratch, scratch2, parameters.is_unidirectional, scale, mask_dimension,
            parameters.max_sequence_length, use_persistent_softmax, persistent_softmax_workspace,
            parameters.mask_filter_value));
  } else if (nullptr != mask_index) {  // 1d mask index
    assert(mask_index_dims.size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
        stream, total_sequence_length, sequence_length, batch_size, num_heads,
        mask_index, mask_start, data.attention_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
        data.scratch, scratch2, parameters.is_unidirectional));
  } else {  // no mask
    if (nullptr != data.output_qk) {
      int64_t qk_size = (int64_t)batch_size * num_heads * sequence_length * total_sequence_length;
      if (std::is_same<T, QK>::value) {
        cudaMemcpyAsync(data.output_qk, data.scratch, qk_size * sizeof(QK), cudaMemcpyDeviceToDevice, stream);
      } else {
        ORT_RETURN_IF_ERROR(
          (CopyQK<T, QK>(stream, static_cast<int>(qk_size), data.scratch, reinterpret_cast<QK*>(data.output_qk))));
      }
    }
    ORT_RETURN_IF_ERROR(
        ComputeSoftmax<T>(
            stream, total_sequence_length, sequence_length, batch_size, num_heads,
            data.attention_bias, broadcast_attn_bias_dim_0, broadcast_attn_bias_dim_1,
            data.scratch, scratch2, parameters.is_unidirectional));
  }

  DUMP_TENSOR_D("Softmax", scratch2, batch_size, num_heads, sequence_length, total_sequence_length);

  // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  T* temp_output = data.q;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      v_head_size, sequence_length, total_sequence_length,
      &one, data.v, v_head_size, present_size_per_batch_v,
      scratch2, total_sequence_length, sequence_length * total_sequence_length,
      &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop, parameters.use_tf32));

  // Temp_output is BxNxSxH_v, transpose to output BxSxNxH_v
  Status result = LaunchTransCtx(stream, sequence_length, batch_size, v_head_size, num_heads,
                                 device_prop.maxThreadsPerBlock, false, temp_output, data.output);
  DUMP_TENSOR_D("Attention Output", data.output, batch_size, sequence_length, num_heads, v_head_size);
  return result;
}

#ifndef USE_ROCM  // exclude the following from hipify since they are not used in ROCM EP

template <typename T>
Status ConcatPastToPresent(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                           int sequence_length, int total_sequence_length,
                           cudaStream_t stream, int max_threads_per_block,
                           AttentionData<T>& data) {
  // Concat past key value to present (2xBxNxLxH), where L is kv_sequence_length and T is total_sequence_length.
  // past_k (BxNxPxH) + k (BxNxLxH) => present_k (BxNxTxH)
  // past_v (BxNxPxH) + v (BxNxLxH) => present_v (BxNxTxH)
  // When there is past state, the head size for Q/K/V shall be same: H == H_v.

  if (nullptr != data.present) { // Attention op
    assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH ||
           data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH);

    ORT_RETURN_IF_ERROR(
        LaunchConcatTensorToTensor(
            stream, total_sequence_length, sequence_length, batch_size, qk_head_size, num_heads,
            max_threads_per_block, 2, data.past, data.k, data.present));



    // Update pointers to present_k and present_v.
    data.k = data.present;
    data.v = data.present + batch_size * num_heads * total_sequence_length * qk_head_size;
  } else  { // MultiHeadAttention op
    if (nullptr != data.present_key) {
      ORT_ENFORCE(data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH ||
                  data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
      if (nullptr != data.past_key) {
        assert(data.past_key != data.k);
        assert(data.past_value != data.v);

        ORT_RETURN_IF_ERROR(
            LaunchConcatTensorToTensor(stream, total_sequence_length, sequence_length,
                                      batch_size, qk_head_size, num_heads,
                                      max_threads_per_block, 1, data.past_key, data.k, data.present_key));
        ORT_RETURN_IF_ERROR(
            LaunchConcatTensorToTensor(stream, total_sequence_length, sequence_length,
                                      batch_size, v_head_size, num_heads,
                                      max_threads_per_block, 1, data.past_value, data.v, data.present_value));
        // Update pointers to present_k and present_v.
        data.k = data.present_key;
        data.v = data.present_value;
      } else { // nullptr == data.past_key && nullptr != data.present_key
        if (data.k != data.present_key) {
          int64_t k_size = (int64_t)batch_size * num_heads * total_sequence_length * qk_head_size;
          cudaMemcpyAsync(data.present_key, data.k, k_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }

        if (data.v != data.present_value) {
          int64_t v_size = (int64_t)batch_size * num_heads * total_sequence_length * v_head_size;
          cudaMemcpyAsync(data.present_value, data.v, v_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
      }
    }
  }

  return CUDA_CALL(cudaGetLastError());
}

// Template Instantiation
template Status ConcatPastToPresent<float>(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                                           int sequence_length, int total_sequence_length,
                                           cudaStream_t stream,
                                           int max_threads_per_block,
                                           AttentionData<float>& data);

template Status ConcatPastToPresent<half>(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                                          int sequence_length, int total_sequence_length,
                                          cudaStream_t stream,
                                          int max_threads_per_block,
                                          AttentionData<half>& data);
#endif

template <typename T>
Status PastPresentBufferShare(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                              int sequence_length, void* fused_runner,
                              contrib::AttentionParameters& parameters,
                              AttentionData<T>& data,
                              cudaStream_t stream,
                              int max_threads_per_block) {
  ORT_ENFORCE(qk_head_size == v_head_size);
  assert(data.fused_cross_attention_kernel == nullptr);
  assert(nullptr == fused_runner || parameters.is_unidirectional);
  assert(!data.use_memory_efficient_attention);
  assert(!data.use_flash_attention);
  assert(data.has_qkv_workspace);

  bool combined_key_value = nullptr != data.present;
  bool separate_key_value = nullptr != data.past_key && nullptr != data.present_key &&
                            nullptr != data.past_value && nullptr != data.present_value;

  // Return early if buffer sharing is not possible
  if (!combined_key_value && !separate_key_value) {
    return Status::OK();
  }

  if (combined_key_value) { // Attention op
    assert(data.gemm_buffer != nullptr);

    if (data.present != data.past) {
      // For easy testing. Production should better avoid this path.
      int64_t kv_size = 2LL * (int64_t)batch_size * num_heads * parameters.max_sequence_length * qk_head_size;
      cudaMemcpyAsync(data.present, data.past, kv_size * sizeof(T), cudaMemcpyDeviceToDevice, stream);
    }

    // For fused causal, bias has been added to gemm_buffer.
    const T* bias = (nullptr != fused_runner && parameters.is_unidirectional) ? nullptr : data.bias;

    // append last k v to present
    ORT_RETURN_IF_ERROR(LaunchAddBiasTransAppendKvToPresent(
        stream, parameters.max_sequence_length, parameters.past_sequence_length, sequence_length,
        batch_size, qk_head_size, num_heads, max_threads_per_block,
        bias, data.gemm_buffer, data.present));

    data.k = data.present;
    data.v = data.present + batch_size * num_heads * parameters.max_sequence_length * qk_head_size;
  } else if (data.use_decoder_masked_multihead_attention) {  // DecoderMaskedMultiHeadAttention op
    assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH ||
           data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);

    // DecoderMaskedMultiHeadAttention kernel manages the KV caches
    // so this case is empty
  } else {  // MultiHeadAttention op
    assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH ||
           data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
    assert(data.seqlens_k_total);

    // Using BNSH since AddBiasTranspose has already been applied
    constexpr bool is_past_kv_bnsh_format = true;
    constexpr bool is_new_kv_bnsh_format = true;
    ORT_RETURN_IF_ERROR(LaunchConcatKVInPlace(
      batch_size, num_heads, qk_head_size, parameters.max_sequence_length,
      data.seqlens_k_total, nullptr, parameters.sequence_length, data.k, data.v, data.present_key, data.present_value,
      is_past_kv_bnsh_format, is_new_kv_bnsh_format, stream, max_threads_per_block));

    data.k = data.present_key;
    data.v = data.present_value;
  }

  return CUDA_CALL(cudaGetLastError());
}

template Status PastPresentBufferShare<float>(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                                              int sequence_length, void* fused_runner,
                                              contrib::AttentionParameters& parameters,
                                              AttentionData<float>& data,
                                              cudaStream_t stream,
                                              int max_threads_per_block);

template Status PastPresentBufferShare<half>(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                                             int sequence_length, void* fused_runner,
                                             contrib::AttentionParameters& parameters,
                                             AttentionData<half>& data,
                                             cudaStream_t stream,
                                             int max_threads_per_block);

template <typename T, typename QK>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudnnHandle_t& cudnn,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int total_sequence_length = parameters.total_sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  void* fused_runner = data.fused_runner;

  // At most one fused kernel is enabled.
  assert((static_cast<int>(data.use_flash_attention) +
          static_cast<int>(data.use_memory_efficient_attention) +
          static_cast<int>(data.use_decoder_masked_multihead_attention) +
          static_cast<int>(fused_runner != nullptr) +
          static_cast<int>(data.fused_cross_attention_kernel != nullptr) +
          static_cast<int>(data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention)) <= 1);

  DUMP_STRING_INIT();
  DUMP_STRING("Preparing Q, K, V");
  ORT_RETURN_IF_ERROR(PrepareQkv<T>(parameters, data, stream, max_threads_per_block));

  if (!parameters.past_present_share_buffer) {
    ORT_RETURN_IF_ERROR(ConcatPastToPresent<T>(batch_size, num_heads, qk_head_size, v_head_size,
                                            sequence_length, total_sequence_length,
                                            stream, max_threads_per_block, data));

  } else {  // past_present_share_buffer
    ORT_RETURN_IF_ERROR(PastPresentBufferShare<T>(batch_size, num_heads, qk_head_size, v_head_size,
                                               sequence_length, fused_runner,
                                               parameters, data, stream, max_threads_per_block));
  }

  // Q, K and V are ready now
  if (data.fused_cross_attention_kernel != nullptr) {
    DUMP_STRING("FusedTrtCrossAttention");
    return FusedTrtCrossAttention<T>(stream, parameters, data);
  }

  // Run TRT fused attention.
  if (nullptr != fused_runner) {
    DUMP_STRING("FusedTrtSelfAttention");
    return FusedTrtSelfAttention<T>(stream, parameters, data);
  }

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                               : parameters.scale;
#if USE_LEAN_ATTENTION
  if (data.use_lean_attention) {
    DUMP_STRING("LeanAttention");
    return LeanAttention<T>(device_prop, stream, parameters, data, scale);
  }
#endif

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    DUMP_STRING("FlashAttention");
    return FlashAttention<T>(device_prop, stream, parameters, data, scale);
  }
#endif

  if (data.kernel_type == AttentionKernelType::AttentionKernel_CudnnFlashAttention) {
    DUMP_STRING("CudnnFlashAttention");
    return CudnnFlashAttention<T>(cudnn, ort_stream, parameters, data, scale);
  }

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    DUMP_STRING("EfficientAttention");
    return EfficientAttention<T>(device_prop, stream, parameters, data, scale);
  }
#endif

  if (data.use_decoder_masked_multihead_attention) {
    DUMP_STRING("DecoderMaskedMHA");
    return DecoderMaskedMultiHeadAttention<T, QK>(stream, parameters, data, scale);
  }

  DUMP_STRING("UnfusedAttention");
  return UnfusedAttention<T, QK>(device_prop, cublas, ort_stream, parameters, data, scale);
}

// Template Instantiation
template struct AttentionData<float>;

template struct AttentionData<half>;

template Status QkvToContext<float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudnnHandle_t& cudnn,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudnnHandle_t& cudnn,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data);

template Status QkvToContext<float, half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudnnHandle_t& cudnn,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data);

template Status QkvToContext<half, float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudnnHandle_t& cudnn,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data);

template Status LaunchDecoderMaskedMultiHeadAttention<float, float>(
  const DecoderMaskedMultiHeadAttentionParameters& parameters,
  cudaStream_t stream,
  const int head_size);

template Status LaunchDecoderMaskedMultiHeadAttention<float, half>(
  const DecoderMaskedMultiHeadAttentionParameters& parameters,
  cudaStream_t stream,
  const int head_size);

template Status LaunchDecoderMaskedMultiHeadAttention<uint16_t, float>(
  const DecoderMaskedMultiHeadAttentionParameters& parameters,
  cudaStream_t stream,
  const int head_size);

template Status LaunchDecoderMaskedMultiHeadAttention<uint16_t, half>(
  const DecoderMaskedMultiHeadAttentionParameters& parameters,
  cudaStream_t stream,
  const int head_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
