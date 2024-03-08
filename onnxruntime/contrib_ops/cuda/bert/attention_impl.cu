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
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

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

void CumulatedSequenceLengthCache::Initialize(int32_t seq_length, cudaStream_t stream) {
  if (this->sequence_length != seq_length) {
    ORT_ENFORCE(buffer.get() != nullptr && this->max_batch_size > 0);
    LaunchTrtSequenceOffset(reinterpret_cast<int32_t*>(buffer.get()), nullptr,
                            this->max_batch_size, seq_length, stream);
    this->sequence_length = seq_length;
  }
}

int* GetCumulatedSequenceLength(CumulatedSequenceLengthCache* cache,
                                const int* mask_index,
                                int batch_size,
                                int sequence_length,
                                cudaStream_t stream,
                                void* scratch_buffer) {
  if (mask_index == nullptr && cache != nullptr) {
    if (batch_size <= cache->max_batch_size) {
      cache->Initialize(sequence_length, stream);
      return reinterpret_cast<int*>(cache->buffer.get());
    }
  }

  int* sequence_offset = reinterpret_cast<int*>(scratch_buffer);
  LaunchTrtSequenceOffset(sequence_offset, mask_index, batch_size, sequence_length, stream);
  return sequence_offset;
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
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention) {
  // Note that q, k and v might need alignment for fused attention kernels.
  const size_t qkv_bytes = element_size * batch_size * num_heads *
                           ((sequence_length + kv_sequence_length) * qk_head_size + kv_sequence_length * v_head_size);

#if USE_FLASH_ATTENTION
  if (use_flash_attention) {
    return qkv_bytes + onnxruntime::flash::get_softmax_lse_size(sequence_length, batch_size, num_heads);
  }
#else
  ORT_UNUSED_PARAMETER(use_flash_attention);
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

  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  int* q_sequence_offset = GetCumulatedSequenceLength(data.cumulated_sequence_length_q_cache,
                                                      data.mask_index, batch_size,
                                                      sequence_length, stream,
                                                      data.scratch);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q_sequence_offset", q_sequence_offset, 1, batch_size + 1);

  int* kv_sequence_offset = q_sequence_offset + (GetSequenceOffsetSize(batch_size, false) / sizeof(int));
  kv_sequence_offset = GetCumulatedSequenceLength(data.cumulated_sequence_length_kv_cache,
                                                  data.mask_index, batch_size, parameters.kv_sequence_length, stream,
                                                  kv_sequence_offset);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  DUMP_TENSOR_D("kv_sequence_offset", kv_sequence_offset, 1, batch_size + 1);

  FusedMultiHeadCrossAttentionKernel const* cross_attention_kernel =
      reinterpret_cast<FusedMultiHeadCrossAttentionKernel const*>(data.fused_cross_attention_kernel);

  // When there is no bias, we can directly use q and packed kv from inputs.
  void const* query = data.q;
  void const* packed_kv = data.k;
  if (data.value == nullptr && data.bias == nullptr) {
    query = data.query;
    packed_kv = data.key;
  }

  run_fused_cross_attention(
      query,                          // Q
      packed_kv,                      // packed KV
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

  DUMP_TENSOR("trt cross output", data.output,
              batch_size, sequence_length, parameters.num_heads, parameters.v_head_size);
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
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const bool causal = parameters.is_unidirectional;

  int* sequence_offset = reinterpret_cast<int*>(data.scratch);

  DUMP_TENSOR_INIT();
  if (parameters.mask_type == AttentionMaskType::MASK_2D_KEY_PADDING) {
    DUMP_TENSOR_D("mask", reinterpret_cast<const int*>(data.mask_index), batch_size, sequence_length);
    LaunchTrtSequenceOffset2d(sequence_offset, data.mask_index, batch_size, sequence_length, stream);
  } else {
    sequence_offset = GetCumulatedSequenceLength(data.cumulated_sequence_length_q_cache,
                                                 data.mask_index, batch_size, sequence_length, stream,
                                                 sequence_offset);
  }
  DUMP_TENSOR_D("sequence_offset", sequence_offset, 1, (data.mask_index != nullptr ? 2 : 1) * batch_size + 1);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(data.fused_runner);

  const int S = causal ? sequence_length : fused_fp16_runner->getSFromMaxSeqLen(sequence_length);

  // B = 2 * batch_size when there is padding in input, and B = batch_size when padding is removed.
  const int B = (nullptr == data.mask_index ? batch_size : 2 * batch_size);

  fused_fp16_runner->setup(S, B);

  if (!causal) {
    assert(data.qkv_format == AttentionQkvFormat::QKV_BSN3H);

    // When there is no bias, we can directly use packed qkv from inputs.
    void const* packed_qkv = data.q;
    if (data.query != nullptr && data.key == nullptr && data.bias == nullptr) {
      packed_qkv = data.query;
    }

    fused_fp16_runner->run(packed_qkv, sequence_offset, data.output, stream);
    DUMP_TENSOR("fused output", data.output,
                batch_size, sequence_length, parameters.num_heads, parameters.v_head_size);
  } else {
    assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH);
    fused_fp16_runner->run(data.gemm_buffer, sequence_offset, data.output, stream);
    DUMP_TENSOR("fused causal output", data.output,
                batch_size, sequence_length, parameters.num_heads, parameters.v_head_size);
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
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
  assert(nullptr == data.mask_index);
  assert(nullptr == data.relative_position_bias);
  assert(parameters.head_size == parameters.v_head_size);

  void* query = reinterpret_cast<void*>(data.q);
  void* key = reinterpret_cast<void*>(data.k);
  void* value = reinterpret_cast<void*>(data.v);
  // For packed KV, we can use query input directly.
  if (data.gemm_buffer == nullptr && data.key != nullptr && data.value == nullptr && data.bias == nullptr) {
    query = reinterpret_cast<void*>(const_cast<T*>(data.query));
  }

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(query),
                parameters.batch_size, parameters.sequence_length, parameters.num_heads, parameters.head_size);
  DUMP_TENSOR_D("k(BSNH)", data.k,
                parameters.batch_size, parameters.total_sequence_length, parameters.num_heads, parameters.head_size);
  DUMP_TENSOR_D("v(BSNH)", data.v,
                parameters.batch_size, parameters.total_sequence_length,
                parameters.num_heads, parameters.v_head_size);

  bool is_bf16 = false;
  ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
      device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(data.scratch),
      parameters.batch_size, parameters.num_heads, parameters.num_heads, parameters.head_size,
      parameters.sequence_length, parameters.total_sequence_length, scale, parameters.is_unidirectional, is_bf16,
      parameters.num_splits, reinterpret_cast<void*>(data.softmax_lse_accum), reinterpret_cast<void*>(data.out_accum),
      true));

  DUMP_TENSOR("flash attention output", data.output,
              parameters.batch_size, parameters.sequence_length, parameters.num_heads, parameters.v_head_size);

  return Status::OK();
}

template <>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data,
    float scale) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, StatusCode::NOT_IMPLEMENTED, "flash attention does not support float tensor");
}
#endif

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
  assert(data.qkv_format == AttentionQkvFormat::Q_K_V_BSNH);

  const void* query = data.q;
  const void* key = data.k;
  const void* value = data.v;
  // For packed KV, we can use query input directly.
  if (data.gemm_buffer == nullptr && data.key != nullptr && data.value == nullptr) {
    assert(data.bias == nullptr);
    query = data.query;
  }

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(query),
                parameters.batch_size, parameters.sequence_length, parameters.num_heads, parameters.head_size);
  DUMP_TENSOR_D("k(BSNH)", data.k,
                parameters.batch_size, parameters.total_sequence_length, parameters.num_heads, parameters.head_size);
  DUMP_TENSOR_D("v(BSNH)", data.v,
                parameters.batch_size, parameters.total_sequence_length,
                parameters.num_heads, parameters.v_head_size);

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
  p.seqlen_k_ptr = nullptr == data.mask_index
                       ? nullptr
                       : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(data.mask_index));
  p.seqstart_q_ptr = nullptr == data.mask_index
                         ? nullptr
                         : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(
                               data.mask_index + parameters.batch_size));
  p.seqstart_k_ptr = nullptr == data.mask_index
                         ? nullptr
                         : const_cast<int32_t*>(reinterpret_cast<const int32_t*>(
                               data.mask_index + 2 * parameters.batch_size + 1));
  p.query = query;
  p.key = key;
  p.value = value;
  p.attn_bias = nullptr == data.relative_position_bias ? nullptr : data.relative_position_bias;
  p.is_attn_bias_batched = !parameters.broadcast_res_pos_bias;
  p.output = data.output;
  p.is_kv_bsnh = true;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(parameters.v_head_size, sizeof(T) == sizeof(float))
                    ? data.scratch
                    : nullptr;
  p.stream = stream;
  p.has_custom_right_padding = false;
  run_memory_efficient_attention(p);
  DUMP_TENSOR("efficient attention output", data.output,
              parameters.batch_size, parameters.sequence_length, parameters.num_heads, parameters.v_head_size);

  return Status::OK();
}
#endif

template <typename T>
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

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q[BNSH]", data.q, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("k[BNSH]", data.k, batch_size, num_heads, total_sequence_length, qk_head_size);

  const int present_sequence_length = parameters.past_present_share_buffer
                                          ? parameters.max_sequence_length
                                          : total_sequence_length;
  const int present_size_per_batch_k = present_sequence_length * qk_head_size;
  const int present_size_per_batch_v = present_sequence_length * v_head_size;

  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N,
      total_sequence_length, sequence_length, qk_head_size,
      &alpha, data.k, qk_head_size, present_size_per_batch_k,
      data.q, qk_head_size, sequence_length * qk_head_size,
      &zero, data.scratch, total_sequence_length, sequence_length * total_sequence_length, batches,
      device_prop, parameters.use_tf32));

  DUMP_TENSOR_D("Q", data.q, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("K", data.k, batch_size, num_heads, qk_head_size, sequence_length);
  DUMP_TENSOR_D("QK", data.scratch, batch_size, num_heads, sequence_length, total_sequence_length);

  constexpr size_t element_size = sizeof(T);
  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, total_sequence_length);
  T* scratch2 = data.scratch + (bytes / element_size);

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
            mask_index, nullptr, data.relative_position_bias, parameters.broadcast_res_pos_bias,
            data.scratch, scratch2, parameters.is_unidirectional, scale, mask_dimension,
            parameters.max_sequence_length, use_persistent_softmax, persistent_softmax_workspace,
            parameters.mask_filter_value));
  } else if (nullptr != mask_index) {  // 1d mask index
    assert(mask_index_dims.size() == 1);
    // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
    const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
    ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
        stream, total_sequence_length, sequence_length, batch_size, num_heads,
        mask_index, mask_start, data.relative_position_bias, parameters.broadcast_res_pos_bias,
        data.scratch, scratch2, parameters.is_unidirectional));
  } else {  // no mask
    ORT_RETURN_IF_ERROR(
        ComputeSoftmax<T>(
            stream, total_sequence_length, sequence_length, batch_size, num_heads, data.relative_position_bias,
            parameters.broadcast_res_pos_bias, data.scratch, scratch2, parameters.is_unidirectional));
  }

  DUMP_TENSOR_D("Softmax", scratch2, batch_size, num_heads, sequence_length, total_sequence_length);
  DUMP_TENSOR_D("V", data.v, batch_size, num_heads, sequence_length, v_head_size);

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
  DUMP_TENSOR("unfused output", data.output, batch_size, sequence_length, num_heads, v_head_size);
  return result;
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
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
  assert((int(data.use_flash_attention) +
          int(data.use_memory_efficient_attention) +
          int(fused_runner != nullptr) +
          int(data.fused_cross_attention_kernel != nullptr)) <= 1);

  ORT_RETURN_IF_ERROR(PrepareQkv<T>(parameters, data, stream, max_threads_per_block));

  if (!parameters.past_present_share_buffer) {
    ORT_RETURN_IF_ERROR(ConcatPastToPresent(batch_size, num_heads, qk_head_size, v_head_size,
                                            sequence_length, total_sequence_length, parameters.pass_past_in_kv,
                                            stream, max_threads_per_block, data));

  } else {  // past_present_share_buffer
    assert(qk_head_size == v_head_size);
    assert(data.fused_cross_attention_kernel == nullptr);
    assert(nullptr == fused_runner || parameters.is_unidirectional);
    assert(data.gemm_buffer != nullptr);
    assert(!data.use_memory_efficient_attention);
    assert(!data.use_flash_attention);
    assert(data.has_qkv_workspace);

    if (nullptr != data.past_key || nullptr != data.present_key) {
      // TODO: support this case.
      ORT_THROW("buffer sharing for no bias case between past and present is not supported yet.");
    }

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
  }

  // Q, K and V are ready now
  if (data.fused_cross_attention_kernel != nullptr) {
    return FusedTrtCrossAttention(stream, parameters, data);
  }

  // Run TRT fused attention.
  if (nullptr != fused_runner) {
    return FusedTrtSelfAttention(stream, parameters, data);
  }

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                               : parameters.scale;

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    return FlashAttention(device_prop, stream, parameters, data, scale);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    return EfficientAttention(device_prop, stream, parameters, data, scale);
  }
#endif

  return UnfusedAttention(device_prop, cublas, ort_stream, parameters, data, scale);
}

// Template Instantiation
template struct AttentionData<float>;

template struct AttentionData<half>;

template Status QkvToContext<float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
