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

#include <cassert>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

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
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t kv_num_heads,
    size_t head_size,
    size_t sequence_length,
    size_t kv_sequence_length,
    size_t total_sequence_length,
    bool use_flash_attention) {
  const size_t qkv_bytes = (element_size * batch_size * num_heads * sequence_length * head_size) +
                           (2 * element_size * batch_size * kv_num_heads * kv_sequence_length * head_size);

#if USE_FLASH_ATTENTION
  if (use_flash_attention) {
    return qkv_bytes + onnxruntime::flash::get_softmax_lse_size(sequence_length, batch_size, num_heads);
  }
#else
  ORT_UNUSED_PARAMETER(use_flash_attention);
#endif

  // TODO(aciddelgado): confirm call w kv_num_heads rt than num_heads
  return qkv_bytes + 2 * GetAttentionScratchSize(element_size, batch_size, kv_num_heads, sequence_length,
                                                 total_sequence_length);
}

// TODO(aciddelgado): what to do with these bias functions... names unclear
template <typename T>
__global__ void AddBiasTransAppendKvToPresentSmall(
    const T* qkv, const T* biases, T* present,
    const int head_size, const int past_sequence_length, const int max_sequence_length) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: (2, B, N, [P..P+S) of MaxS, H),
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int B = gridDim.y;

  constexpr int M = 3;           // Matrix count in qkv
  const int m = blockIdx.z + 1;  // k = 1, v = 2

  const int NH = N * head_size;
  const int NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int MsH = max_sequence_length * head_size;
  const int NMsH = N * MsH;
  const int BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

// TODO(aciddelgado): what to do with these bias functions... names unclear
template <typename T>
__global__ void AddBiasTransAppendKvToPresent(
    const T* qkv, const T* biases, T* present,
    const int head_size, const int past_sequence_length, const int max_sequence_length) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: (2, B, N, [P..P+S) of MaxS, H),
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  const int n = blockIdx.x;
  const int s = blockIdx.y;
  const int b = (blockIdx.z >> 1);
  const int N = gridDim.x;
  const int S = gridDim.y;
  const int B = (gridDim.z >> 1);

  constexpr int M = 3;                   // Matrix count in qkv
  const int m = (blockIdx.z & 0x1) + 1;  // k = 1, v = 2

  const int NH = N * head_size;
  const int NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int MsH = max_sequence_length * head_size;
  const int NMsH = N * MsH;
  const int BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

// TODO(aciddelgado): what to do with these bias functions... names unclear
// qkv buffer is merged tensor of shape (B,S,3,N,H), k v is the second/third of the 3.
// bias is of shape (3, NxH) or nullptr
// append to present of (2, B, N, (P..T) of M, H),
template <typename T>
Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                           const int max_sequence_length,
                                           const int past_sequence_length,
                                           const int sequence_length,
                                           const int batch_size,
                                           const int head_size,
                                           const int num_heads,
                                           const int max_threads_per_block,
                                           const T* biases,
                                           const T* qkv_buffer,
                                           T* present) {
  assert(head_size <= (1 << 30));

  int64_t nh = (int64_t)head_size * num_heads;
  if (nh <= max_threads_per_block) {
    const dim3 grid(sequence_length, batch_size, 2);  // 2 for k and v
    const dim3 block(max_threads_per_block / num_heads, num_heads, 1);

    AddBiasTransAppendKvToPresentSmall<T><<<grid, block, 0, stream>>>(
        qkv_buffer, biases, present, head_size, past_sequence_length, max_sequence_length);
  } else {
    const dim3 grid(num_heads, sequence_length, batch_size * 2);  // 2 for k and v
    const dim3 block(std::min(head_size, max_threads_per_block), 1, 1);
    AddBiasTransAppendKvToPresent<T><<<grid, block, 0, stream>>>(
        qkv_buffer, biases, present, head_size, past_sequence_length, max_sequence_length);
  }

  return CUDA_CALL(cudaGetLastError());
}

// TODO(aciddelgado): what to do with these bias functions... names unclear
template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const float* bias,
                                                    const float* qkv_buffer,
                                                    float* present);

// TODO(aciddelgado): what to do with these bias functions... names unclear
template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const half* bias,
                                                    const half* qkv_buffer,
                                                    half* present);

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
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  DUMP_TENSOR_INIT();

  if (data.bias == nullptr) {
    // Below logic does not support fused attention with past without bias
    // When there is past state, the format shall be BxNxSxH, so we disable fused attention when there is past.
    // cross attention with present state or self attention with present state
    if (data.past_key == nullptr && data.present_key != nullptr) {
      assert(data.past_value == nullptr);
      assert(data.present_value != nullptr);
      assert(data.query != nullptr);
      assert(data.key != nullptr);
      assert(data.value != nullptr);

      // TODO: supporting packed qkv for self attention may benefit performance
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, qk_head_size, num_heads,
                                         max_threads_per_block, false, data.query, q));
      // TODO: supporting packed kv for cross attention may benefit performance
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, head_size, kv_num_heads,
                                         max_threads_per_block, false, data.key, data.present_key));
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, head_size, kv_num_heads,
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
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, sequence_length, batch_size, head_size, num_heads,
                                         max_threads_per_block, false, data.query, q));
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, head_size, kv_num_heads,
                                         max_threads_per_block, false, data.key, k));
      ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, head_size, kv_num_heads,
                                         max_threads_per_block, false, data.value, v));
    }
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
#if USE_MEMORY_EFFICIENT_ATTENTION || USE_FLASH_ATTENTION
  // When there is no past_key/past_value and there is present_key/present_value
  // (e.g. get initial kv to use as past_kv in the next iteration)
  else if ((data.use_memory_efficient_attention || data.use_flash_attention) &&
           data.present_key != nullptr &&
           data.present_value != nullptr) {
    // Use flash or memory efficient attention kernel
    // TODO(aciddelgado): launch bias that works for group attention, maybe one for q and one for k and v??
    LaunchAddBias(stream, max_threads_per_block,
                  batch_size, sequence_length, kv_sequence_length,
                  num_heads, head_size, head_size,
                  data.bias, data.query, data.key, data.value, q, data.temp_k_workspace, data.temp_v_workspace);

    // temp_k_workspace (BxSxNxH) => present_k (BxNxSxH)
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, head_size, kv_num_heads,
                                       max_threads_per_block, false, data.temp_k_workspace, data.present_key));

    // temp_v_workspace (BxSxNxH_v) => present_v (BxNxSxH_v)
    ORT_RETURN_IF_ERROR(LaunchTransQkv(stream, 1, kv_sequence_length, batch_size, head_size, kv_num_heads,
                                       max_threads_per_block, false, data.temp_v_workspace, data.present_value));

    DUMP_TENSOR_D("q(BSNH)", q, batch_size, sequence_length, num_heads, head_size);
    DUMP_TENSOR_D("k(BSNH)", data.temp_k_workspace, batch_size, kv_sequence_length, kv_num_heads, head_size);
    DUMP_TENSOR_D("v(BSNH)", data.temp_v_workspace, batch_size, kv_sequence_length, kv_num_heads, head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  }
#endif
  else {
    // Use unfused kernel for Q, use unfused kernel for K and V if needed
    constexpr int format = 0;
    // Query (BxSxNxH) => Q (BxNxSxH)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, sequence_length, num_heads, head_size,
                              data.query, data.bias, q,
                              true, -1);

    if (!parameters.pass_past_in_kv) {
      T* k_dest = (data.past_key == nullptr && data.present_key != nullptr) ? data.present_key : k;
      T* v_dest = (data.past_value == nullptr && data.present_value != nullptr) ? data.present_value : v;

      // Key (BxLxNxH) => K (BxNxLxH)
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, kv_sequence_length, kv_num_heads, head_size,
                                data.key, data.bias + kv_num_heads * head_size, k_dest,
                                true, -1);

      // Value (BxLxNxH_v) => V (BxNxLxH_v)
      LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                                batch_size, kv_sequence_length, kv_num_heads, head_size,
                                data.value, data.bias + 2 * kv_num_heads * head_size, v_dest,
                                true, -1);

      DUMP_TENSOR_D("q(BNSH)", q, batch_size, num_heads, sequence_length, qk_head_size);
      DUMP_TENSOR_D("k(BNSH)", k_dest, batch_size, kv_num_heads, kv_sequence_length, head_size);
      DUMP_TENSOR_D("v(BNSH)", v_dest, batch_size, kv_num_heads, kv_sequence_length, head_size);
    }
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
  return Status::OK();
}

// For MultiHeadAttention without past state, with Q, K and V inputs
// TODO(aciddelgado): needs mod? this is MHA meaning no causal originally
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
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  T* qkv = data.workspace;

  // bool use_fused_kernel = (nullptr != fused_runner && !parameters.is_unidirectional);
  // bool use_fused_causal = (nullptr != fused_runner && parameters.is_unidirectional);

  // gemm_buffer == nullptr and not packed
  assert(data.query != nullptr && data.key != nullptr && data.value != nullptr);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("query", data.query, batch_size, sequence_length, num_heads, head_size);
  DUMP_TENSOR_D("key", data.key, batch_size, kv_sequence_length, kv_num_heads, head_size);
  DUMP_TENSOR_D("value", data.value, batch_size, kv_sequence_length, kv_num_heads, head_size);

#if DUMP_TENSOR_LEVEL > 1
  // TODO(aciddelgado): what is this dump tensor doing with bias dimensions
  if (data.bias != nullptr) {
    DUMP_TENSOR_D("query_bias", data.bias, num_heads, head_size);
    DUMP_TENSOR_D("key_bias", data.bias + kv_num_heads * head_size, num_heads, head_size);
    DUMP_TENSOR_D("value_bias", data.bias + 2 * kv_num_heads * head_size, num_heads, head_size);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION || USE_FLASH_ATTENTION
  if (data.use_memory_efficient_attention || data.use_flash_attention) {
    // TODO(aciddelgado): add bias kv_num_heads...
    LaunchAddBias(stream, max_threads_per_block,
                  batch_size, sequence_length, kv_sequence_length,
                  num_heads, head_size, head_size,
                  data.bias, data.query, data.key, data.value, q, k, v);

    DUMP_TENSOR_D("q(BSNH)", q, batch_size, sequence_length, num_heads, head_size);
    DUMP_TENSOR_D("k(BSNH)", k, batch_size, kv_sequence_length, kv_num_heads, head_size);
    DUMP_TENSOR_D("v(BSNH)", v, batch_size, kv_sequence_length, kv_num_heads, head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  }
#endif
  else {  // unfused kernel

    // Query (BxSxNxH) => Q (BxNxSxH)
    constexpr int format = 0;
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, sequence_length, num_heads, head_size,
                              data.query, data.bias, q,
                              true, -1);

    // Key (BxLxNxH) => K (BxNxLxH)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, kv_sequence_length, kv_num_heads, head_size,
                              data.key, nullptr == data.bias ? nullptr : data.bias + kv_num_heads * head_size, k,
                              true, -1);

    // Value (BxLxNxH_v) => K (BxNxLxH_v)
    LaunchAddBiasTranspose<T>(stream, 1, format, max_threads_per_block,
                              batch_size, kv_sequence_length, kv_num_heads, head_size,
                              data.value, nullptr == data.bias ? nullptr : data.bias + 2 * kv_num_heads * head_size, v,
                              true, -1);

    DUMP_TENSOR_D("q(BNSH)", q, batch_size, num_heads, sequence_length, head_size);
    DUMP_TENSOR_D("k(BNSH)", k, batch_size, kv_num_heads, kv_sequence_length, head_size);
    DUMP_TENSOR_D("v(BNSH)", v, batch_size, kv_num_heads, kv_sequence_length, head_size);
    qkv_format = AttentionQkvFormat::Q_K_V_BNSH;
  }
  return Status::OK();
}

template <typename T>
Status PrepareQkv(contrib::AttentionParameters& parameters,
                  AttentionData<T>& data,
                  cudaStream_t stream,
                  int max_threads_per_block,
                  T* q, T* k, T* v, AttentionQkvFormat& qkv_format) {
  if (data.past_key != nullptr || data.present_key != nullptr) {  // mha operator with past/present state
    ORT_RETURN_IF_ERROR(PrepareQkv_MHA_WithPast(parameters, data, stream, max_threads_per_block, q, k, v, qkv_format));
  } else {  // multihead attention operator, no past, separated Q/K/V inputs
    ORT_RETURN_IF_ERROR(PrepareQkv_MHA_NotPacked(parameters, data, stream, max_threads_per_block, q, k, v, qkv_format));
  }

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data) {
  auto stream = static_cast<cudaStream_t>(ort_stream->GetHandle());
  constexpr size_t element_size = sizeof(T);
  const int max_threads_per_block = device_prop.maxThreadsPerBlock;
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int kv_sequence_length = parameters.kv_sequence_length;
  const int total_sequence_length = parameters.total_sequence_length;
  const int num_heads = parameters.num_heads;
  const int kv_num_heads = parameters.kv_num_heads;
  const int head_size = parameters.head_size;

  const int q_batches = batch_size * num_heads;
  const int kv_batches = batch_size * kv_num_heads;

  T* qkv = nullptr;
  T* q = nullptr;
  T* k = nullptr;
  T* v = nullptr;
  T* scratch1 = data.workspace;
  if (data.has_qkv_workspace) {
    const int size_per_batch_q = sequence_length * head_size;
    const int size_per_batch_k = kv_sequence_length * head_size;
    const int size_per_batch_v = kv_sequence_length * head_size;
    const size_t elements_q = static_cast<size_t>(q_batches) * static_cast<size_t>(size_per_batch_q);
    const size_t elements_k = static_cast<size_t>(kv_batches) * static_cast<size_t>(size_per_batch_k);
    const size_t elements_v = static_cast<size_t>(kv_batches) * static_cast<size_t>(size_per_batch_v);
    qkv = data.workspace;
    q = qkv;
    k = q + elements_q;
    v = k + elements_k;
    scratch1 = v + elements_v;
  }

  AttentionQkvFormat qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  ORT_RETURN_IF_ERROR(PrepareQkv<T>(parameters, data, stream, max_threads_per_block, q, k, v, qkv_format));

  int present_size_per_batch_k = 0;
  int present_size_per_batch_v = 0;

  // Concat past key value to present (2xBxNxLxH), where L is kv_sequence_length and T is total_sequence_length.
  // past_k (BxNxPxH) + k (BxNxLxH) => present_k (BxNxTxH)
  // past_v (BxNxPxH) + v (BxNxLxH) => present_v (BxNxTxH)
  // When there is past state, the head size for Q/K/V shall be same: H == H_v.
  present_size_per_batch_k = total_sequence_length * head_size;
  present_size_per_batch_v = total_sequence_length * head_size;

  // TODO(aciddelgado): qkv_format was set above to be BSNH, what's going on here?
  if (nullptr != data.present) {
    assert(qkv_format == AttentionQkvFormat::Q_K_V_BNSH || qkv_format == AttentionQkvFormat::Q_K_V_BNSH_QKV_BS3NH);
    ORT_RETURN_IF_ERROR(
        LaunchConcatPastToPresent(
            stream, total_sequence_length, sequence_length, batch_size, qk_head_size, num_heads,
            max_threads_per_block, data.past, k, data.present));

    // Update pointers to present_k and present_v.
    k = data.present;
    v = data.present + kv_batches * present_size_per_batch_k;
  }

  if (nullptr != data.past_key || nullptr != data.present_key) {
    if (nullptr != data.past_key && nullptr == data.present_key) {
      k = const_cast<T*>(data.past_key);
      v = const_cast<T*>(data.past_value);
    } else if (nullptr == data.past_key && nullptr != data.present_key) {
      if (qkv_format == AttentionQkvFormat::Q_K_V_BNSH) {
        k = data.present_key;
        v = data.present_value;
      } else {
        assert(qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
        k = data.temp_k_workspace;
        v = data.temp_v_workspace;
      }
    } else {
      ORT_RETURN_IF_ERROR(
          LaunchConcatTensorToTensor(stream, parameters.total_sequence_length, sequence_length,
                                      batch_size, head_size, kv_num_heads,
                                      max_threads_per_block, 1, data.past_key, k, data.present_key));
      ORT_RETURN_IF_ERROR(
          LaunchConcatTensorToTensor(stream, parameters.total_sequence_length, sequence_length,
                                      batch_size, head_size, kv_num_heads,
                                      max_threads_per_block, 1, data.past_value, v, data.present_value));
      // Update pointers to present_k and present_v.
      k = data.present_key;
      v = data.present_value;
    }
  }

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  const float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(head_size)) : parameters.scale;
  assert(data.use_flash_attention);
#if USE_FLASH_ATTENTION
  // if (data.use_flash_attention) {
    assert(qkv_format == AttentionQkvFormat::Q_K_V_BSNH);
    assert(nullptr == data.mask_index);
    assert(nullptr == data.relative_position_bias);
    assert(parameters.num_heads == parameters.kv_num_heads);

    void* query = reinterpret_cast<void*>(q);
    void* key = reinterpret_cast<void*>(k);
    void* value = reinterpret_cast<void*>(v);
    // TODO(aciddelgado): packed KV, we can use query input directly.
    // if (data.gemm_buffer == nullptr && data.key != nullptr && data.value == nullptr && data.bias == nullptr) {
    //   query = reinterpret_cast<void*>(const_cast<T*>(data.query));
    // }

    DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(query), batch_size, sequence_length, num_heads, head_size);
    DUMP_TENSOR_D("k(BSNH)", k, batch_size, parameters.total_sequence_length, kv_num_heads, head_size);
    DUMP_TENSOR_D("v(BSNH)", v, batch_size, parameters.total_sequence_length, kv_num_heads, head_size);

    constexpr bool is_causal = parameters.is_unidirectional;
    ORT_RETURN_IF_ERROR(onnxruntime::flash::mha_fwd(
        device_prop, stream, query, key, value, data.output, reinterpret_cast<void*>(scratch1),
        parameters.batch_size, parameters.num_heads, parameters.kv_num_heads, parameters.head_size,
        parameters.sequence_length, parameters.total_sequence_length, scale, is_causal));

    DUMP_TENSOR("flash attention output", data.output, batch_size, sequence_length, kv_num_heads, head_size);

    return Status::OK();
  // }
#endif

  // // The following are unfused attention.
  // assert(qkv_format == AttentionQkvFormat::Q_K_V_BNSH);

  // // Raw attention mask could be 2D (BxT) or 3D (BxSxT) or 4D(Bx1xMxM), where M is the max sequence length.
  // bool use_raw_attention_mask = (nullptr != mask_index && mask_index_dims.size() >= 2);

  // // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxT
  // // Q: BxNxSxH, K (present_k): BxNxTxH, Q*K': BxNxSxT
  // float one = 1.0f;
  // float zero = 0.f;

  // float alpha = use_raw_attention_mask ? one : scale;

  // cublasSetStream(cublas, stream);

  // DUMP_TENSOR_D("q[BNSH]", q, batch_size, num_heads, sequence_length, qk_head_size);
  // DUMP_TENSOR_D("k[BNSH]", k, batch_size, num_heads, total_sequence_length, qk_head_size);
  // CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
  //     cublas, CUBLAS_OP_T, CUBLAS_OP_N,
  //     total_sequence_length, sequence_length, qk_head_size,
  //     &alpha, k, qk_head_size, present_size_per_batch_k,
  //     q, qk_head_size, sequence_length * qk_head_size,
  //     &zero, scratch1, total_sequence_length, sequence_length * total_sequence_length, batches, device_prop));

  // DUMP_TENSOR_D("Q", q, batch_size, num_heads, sequence_length, qk_head_size);
  // DUMP_TENSOR_D("K", k, batch_size, num_heads, qk_head_size, sequence_length);
  // DUMP_TENSOR_D("QK", scratch1, batch_size, num_heads, sequence_length, total_sequence_length);

  // const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
  //                                              sequence_length, total_sequence_length);
  // T* scratch2 = scratch1 + (bytes / element_size);

  // // Apply softmax and store result R to scratch2: BxNxSxT
  // if (use_raw_attention_mask) {  // 2d, 3d or 4d attention mask
  //   const int mask_dimension = static_cast<int>(mask_index_dims.size());

  //   // For testing, environment variable ORT_TRANSFORMER_OPTIONS=1 could enable persistent softmax used in Torch.
  //   const TransformerOptions* options = TransformerOptions::GetInstance();
  //   bool use_persistent_softmax = options->IsPrecisionMode() && !options->DisablePersistentSoftmax();

  //   T* persistent_softmax_workspace = scratch1;  // replace Q*K' in place with masked score for persistent softmax.
  //   ORT_RETURN_IF_ERROR(
  //       ComputeSoftmaxWithRawMask<T>(
  //           ort_stream, total_sequence_length, sequence_length, batch_size, num_heads,
  //           mask_index, nullptr, data.relative_position_bias, parameters.broadcast_res_pos_bias,
  //           scratch1, scratch2, parameters.is_unidirectional, scale, mask_dimension,
  //           parameters.max_sequence_length, use_persistent_softmax, persistent_softmax_workspace,
  //           mask_filter_value));
  // } else if (nullptr != mask_index) {  // 1d mask index
  //   assert(mask_index_dims.size() == 1);
  //   // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
  //   const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
  //   ORT_RETURN_IF_ERROR(ComputeSoftmaxWithMask1D<T>(
  //       stream, total_sequence_length, sequence_length, batch_size, num_heads,
  //       mask_index, mask_start, data.relative_position_bias, parameters.broadcast_res_pos_bias,
  //       scratch1, scratch2, parameters.is_unidirectional));
  // } else {  // no mask
  //   ORT_RETURN_IF_ERROR(
  //       ComputeSoftmax<T>(
  //           stream, total_sequence_length, sequence_length, batch_size, num_heads, data.relative_position_bias,
  //           parameters.broadcast_res_pos_bias, scratch1, scratch2, parameters.is_unidirectional));
  // }

  // DUMP_TENSOR_D("Softmax", scratch2, batch_size, num_heads, sequence_length, total_sequence_length);
  // DUMP_TENSOR_D("V", v, batch_size, num_heads, sequence_length, v_head_size);

  // // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  // T* temp_output = qkv;
  // CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
  //     cublas, CUBLAS_OP_N, CUBLAS_OP_N,
  //     v_head_size, sequence_length, total_sequence_length,
  //     &one, v, v_head_size, present_size_per_batch_v,
  //     scratch2, total_sequence_length, sequence_length * total_sequence_length,
  //     &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop));

  // // Temp_output is BxNxSxH_v, transpose to output BxSxNxH_v
  // Status result = LaunchTransCtx(stream, sequence_length, batch_size, v_head_size, num_heads,
  //                                max_threads_per_block, false, temp_output, data.output);
  // DUMP_TENSOR("unfused output", data.output, batch_size, sequence_length, num_heads, v_head_size);
  // return result;
}

// Template Instantiation
// template struct AttentionData<float>;

template struct AttentionData<half>;

// template Status QkvToContext<float>(
//     const cudaDeviceProp& device_prop,
//     cublasHandle_t& cublas,
//     Stream* ort_stream,
//     contrib::AttentionParameters& parameters,
//     AttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* ort_stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
