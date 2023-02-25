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
#include "contrib_ops/cuda/bert/packed_attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cpu/bert/attention_base.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"

using namespace onnxruntime::cuda;
using namespace cub;

#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)
#define CUDA_MEMORY_ALIGNMENT 256

namespace onnxruntime {
namespace contrib {
namespace cuda {

static size_t AlignTo(size_t a, size_t b) {
  return CeilDiv(a, b) * b;
}

size_t AlignSize(size_t bytes) {
  const size_t bytesAligned = AlignTo(bytes, CUDA_MEMORY_ALIGNMENT);
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
  ;
}

// Grid: (B, S)
// Block: 256
// Format 1 for unfused attention
//     Input: TxMxNxH
//     Output: MxBxNxSxH
// C is token_count
// B is batch_size
// S is sequence_length
// M is number of matrices
// N is num_heads
// H is head_size
template <typename T>
__global__ void AddBiasTransposeQKV(const T* input,
                                    const T* biases,
                                    int32_t N,
                                    int32_t H_QK,
                                    int32_t H_V,
                                    T* q,
                                    T* k,
                                    T* v,
                                    const int32_t* token_offset,
                                    int32_t token_count) {
  int b = blockIdx.x;
  int s = blockIdx.y;

  int B = gridDim.x;
  int S = gridDim.y;

  const int packing_token_idx = b * S + s;
  const int padding_token_idx = token_offset[packing_token_idx];
  b = padding_token_idx / S;
  s = padding_token_idx - b % S;

  input += packing_token_idx * N * (H_QK + H_QK + H_V);
  q += (b * N * S + s) * H_QK;
  k += (b * N * S + s) * H_QK;
  v += (b * N * S + s) * H_V;

  if (packing_token_idx < token_count) {
    for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = *input + *biases;
      input++;
      biases++;
    }

    for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      k[n * S * H_QK + h] = *input + *biases;
      input++;
      biases++;
    }

    for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
      int h = i % H_V;
      int n = i / H_V;
      v[n * S * H_V + h] = *input + *biases;
      input++;
      biases++;
    }
  } else {
    for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = *biases;
      input++;
      biases++;
    }

    for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      k[n * S * H_QK + h] = *biases;
      input++;
      biases++;
    }

    for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
      int h = i % H_V;
      int n = i / H_V;
      v[n * S * H_V + h] = *biases;
      input++;
      biases++;
    }
  }
}

template <typename T>
void InvokeAddBiasTranspose(
    const T* input, const T* biases, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    const int format, const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream) {
  const dim3 grid(batch_size, sequence_length);
  ORT_ENFORCE(format == 1);

  AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input,
                                                     biases,
                                                     num_heads,
                                                     qk_head_size,
                                                     v_head_size,
                                                     output,
                                                     output + batch_size * sequence_length * num_heads * qk_head_size,
                                                     output + 2 * batch_size * sequence_length * num_heads * qk_head_size,
                                                     token_offset,
                                                     token_count)

  // if (format == 2) {
  //   AddBiasTransposeTrt<T><<<grid, 256, 0, stream>>>(input, biases, output);
  // } else if (format == 1) {
  //   if (v_head_size == -1 || qk_head_size == v_head_size) {
  //     AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output, qkv_add_bias);
  //   } else {
  //     ORT_ENFORCE(total_matrix_count == 3);
  //     AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
  //   }
  // } else if (format == 3) {
  //   if (v_head_size == -1 || qk_head_size == v_head_size) {
  //     AddBiasTransposeCutlass<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output);
  //   } else {
  //     ORT_ENFORCE(total_matrix_count == 3);
  //     AddBiasTransposeCutlass<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
  //   }
  // } else if (format == 4) {  // format == 4
  //   AddBiasUnpack<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output);
  // } else {  // format == 0
  //   AddBiasTranspose<T><<<grid, block, 0, stream>>>(input, biases, output);
  // }
}

// Fused kernel of Add (bias) and Transpose.
// Shape of inputs and outputs:
//     biases:  (num_matrices, num_heads * head_size)
// format 0: (requires sequence_length = kv_sequence_length and qk_head_size = v_head_size when num_matrices == 3)
//     input:   (num_matrices, batch_size, sequence_length, num_heads, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)
// format 1:
//     input :  (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (num_matrices, batch_size, num_heads, sequence_length, head_size)
//     qkv_add_bias: (batch_size, sequence_length, num_matrices, num_heads, head_size) optional
// format 2:
//     input :  (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (batch_size, sequence_length, num_heads, num_matrices, head_size)
// format 3: (requires sequence_length = kv_sequence_length and qk_head_size = v_head_size when num_matrices == 3)
//     input:   (batch_size, sequence_length, num_matrices, num_heads, head_size)
//     output:  (num_matrices, batch_size, sequence_length, num_heads, head_size)
// format 4: (requires qk_head_size = v_head_size)
//     input:   (batch_size, sequence_length, num_heads, num_matrices, head_size)
//     output:  (num_matrices, batch_size, sequence_length, num_heads, head_size)

template <typename T>
struct T4;

template <>
struct T4<float> {
  using Type = float4;
};

template <>
struct T4<float> {
  using Type = Half4;
};

template <typename T>
struct T2;

template <>
struct T2<float> {
  using Type = float2;
};

template <>
struct T2<float> {
  using Type = half2;
};

template <typename T>
void LaunchAddBiasTranspose(
    const T* input, const T* biases, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    const int format, const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream) {
  if (0 == (qk_head_size & 3) && 0 == (v_head_size & 3)) {
    using T4Type = T4<T>::Type;
    const int H = qk_head_size / 4;
    const T4Type* input2 = reinterpret_cast<const T4Type*>(input);
    const T4Type* biases2 = reinterpret_cast<const T4Type*>(biases);
    T4Type* output2 = reinterpret_cast<T4Type*>(output);
    InvokeAddBiasTranspose<T4Type>(
        input2, biases2, output2,
        batch_size, sequence_length,
        num_heads, qk_head_size, v_head_size,
        format, token_offset, token_count, stream);
  } else if (0 == (qk_head_size & 1) && 0 == (v_head_size & 1)) {
    using T2Type = T2<T>::Type;
    const int H = qk_head_size / 2;
    const T2Type* input2 = reinterpret_cast<const T2Type*>(input);
    const T2Type* biases2 = reinterpret_cast<const T2Type*>(biases);
    T2Type* output2 = reinterpret_cast<T2Type*>(output);
    InvokeAddBiasTranspose<T2Type>(
        input2, biases2, output2,
        batch_size, sequence_length,
        num_heads, qk_head_size, v_head_size,
        format, token_offset, token_count, stream);
  } else {
    InvokeAddBiasTranspose<T>(
        input, biases, output,
        batch_size, sequence_length,
        num_heads, qk_head_size, v_head_size,
        format, token_offset, token_count, stream);
  }
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
    bool use_memory_efficient_attention) {
  // Note that q, k and v might need alignment for fused attention kernels.
  const size_t qkv_bytes = element_size * batch_size * num_heads *
                           ((sequence_length + kv_sequence_length) * qk_head_size + kv_sequence_length * v_head_size);

#if USE_FLASH_ATTENTION
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
    size_t sequence_offset_bytes = GetSequenceOffsetSize(static_cast<int>(batch_size), true);
    return qkv_bytes + sequence_offset_bytes;
  }

  return qkv_bytes + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length,
                                                 total_sequence_length);
}

template <typename T>
Status PrepareQkv(contrib::PackedAttentionParameters& parameters,
                  contrib::PackedAttentionData<T>& data,
                  AttentionQkvFormat& qkv_format,
                  cudaStream_t stream) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  bool use_memory_efficient_attention = data.use_memory_efficient_attention;

  // For fused TRT attention, transpose qkv to BxSxNx3xH (format 2)
  // For memory efficient attention, transpose to 3xBxSxNxH (format 3)
  // For unfused kernel, transpose to 3xBxNxSxH (format 1)
  // For fused causal kernel, use format 1 since we need have K and V to update present state,
  //   at the same time, we update gemm_buffer BxSx3xNxH with bias which is used as input for fused causal kernel.
  bool use_fused_kernel = nullptr != data.fused_runner;
  const int format = (use_fused_kernel ? 2 : (use_memory_efficient_attention ? 3 : 1));
  qkv_format = use_fused_kernel
                   ? AttentionQkvFormat::QKV_BSN3H
                   : (use_memory_efficient_attention
                          ? AttentionQkvFormat::Q_K_V_BSNH
                          : AttentionQkvFormat::Q_K_V_BNSH);

  // format 1: BxSx(NH + NH + NH_v) => BxNxSxH + BxNxSxH + BxNxSxH_v
  // format 2: BxSx(NH + NH + NH) => BxSxNx(H + H + H)
  LaunchAddBiasTranspose(data.gemm_buffer, data.bias, data.workspace,
                         batch_size, sequence_length, num_heads, qk_head_size, v_head_size,
                         format, data.token_offset, parameters.token_count, stream);

  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}

template <typename T>
Status QkvToContext(
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::PackedAttentionParameters& parameters,
    contrib::PackedAttentionData<T>& data) {
  constexpr size_t element_size = sizeof(T);
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  const float mask_filter_value = parameters.mask_filter_value;
  void* fused_runner = data.fused_runner;

  // At most one fused kernel is enabled.
  assert(int(data.use_memory_efficient_attention) + int(fused_runner != nullptr) <= 1);

  const int batches = batch_size * num_heads;
  const int size_per_batch_q = sequence_length * qk_head_size;
  const int size_per_batch_k = sequence_length * qk_head_size;
  const int size_per_batch_v = sequence_length * v_head_size;
  const size_t elements_q = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_q);
  const size_t elements_k = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_k);
  const size_t elements_v = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_v);

  // Q, K and V pointers when fused attention is not used
  T* qkv = data.workspace;
  T* q = qkv;
  T* k = q + elements_q;
  T* v = k + elements_k;

  bool use_fused_kernel = nullptr != fused_runner;

  AttentionQkvFormat qkv_format = AttentionQkvFormat::Q_K_V_BSNH;
  ORT_RETURN_IF_ERROR(PrepareQkv<T>(parameters, data, qkv_format, stream));

  T* scratch1 = qkv + elements_q + elements_k + elements_v;

  // Q, K and V are ready now
  DUMP_TENSOR_INIT();

  // Run TRT fused attention.
  if (use_fused_kernel) {
    FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(fused_runner);

    const int S = fused_fp16_runner->getSFromMaxSeqLen(sequence_length);
    fused_fp16_runner->setup(S, batch_size);

    assert(qkv_format == AttentionQkvFormat::QKV_BSN3H);
    fused_fp16_runner->run(qkv, data.cumulative_sequence_length, data.output, stream);
    DUMP_TENSOR("fused output", data.output, batch_size * sequence_length, num_heads, v_head_size);
    return Status::OK();
  }

#if USE_FLASH_ATTENTION
  if (data.use_memory_efficient_attention) {
    // We only enable fused cross attention when there is no key padding mask.
    // Otherwise, key have effective batch size 2 * batch_size, which is different from batch_size of query.
    assert(data.mask_index == nullptr);
    assert(qkv_format == AttentionQkvFormat::Q_K_V_BSNH);

    const void* query = q;
    const void* key = k;
    const void* value = v;
    if (data.gemm_buffer == nullptr && data.value == nullptr) {  // packed KV
      query = data.query;
    }

    MemoryEfficientAttentionParams p;
    p.sm = device_prop.major * 10 + device_prop.minor;
    p.is_half = sizeof(T) == 2;
    p.batch_size = data.mask_index == nullptr ? parameters.batch_size : 2 * parameters.batch_size;
    p.num_heads = parameters.num_heads;
    p.sequence_length = parameters.sequence_length;
    p.kv_sequence_length = parameters.total_sequence_length;
    p.qk_head_size = parameters.head_size;
    p.v_head_size = parameters.v_head_size;
    p.causal = parameters.is_unidirectional;
    p.cu_seqlens_q = nullptr;
    p.cu_seqlens_k = nullptr;
    p.query = query;
    p.key = key;
    p.value = value;
    p.output = data.output;
    p.workspace = MemoryEfficientAttentionParams::need_workspace(v_head_size, sizeof(T) == sizeof(float)) ? scratch1 : nullptr;
    p.stream = stream;
    run_memory_efficient_attention(p);

    DUMP_TENSOR("cutlass output", data.output, batch_size * sequence_length, num_heads, v_head_size);
    return Status::OK();
  }
#endif

  // The following are unfused attention.
  assert(qkv_format == AttentionQkvFormat::Q_K_V_BNSH);

  // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scratch1: BxNxSxT
  // Q: BxNxSxH, K (present_k): BxNxTxH, Q*K': BxNxSxT
  float one = 1.0f;
  float zero = 0.f;

  // For raw attention mask, the scalar 1/sqrt(H) is moved to combine with softmax computation.
  float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                         : parameters.scale;

  cublasSetStream(cublas, stream);

  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_T, CUBLAS_OP_N,
      sequence_length, sequence_length, qk_head_size,
      &scale,
      k, qk_head_size, sequence_length * qk_head_size,
      q, qk_head_size, sequence_length * qk_head_size,
      &zero,
      scratch1, sequence_length, sequence_length * sequence_length,
      batches, device_prop));

  DUMP_TENSOR_D("QK", scratch1, batch_size * num_heads, sequence_length, total_sequence_length);

  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length, total_sequence_length);
  T* scratch2 = scratch1 + (bytes / element_size);

  // Apply softmax and store result R to scratch2: BxNxSxT
  assert(mask_index_dims.size() == 1);
  // mask_index has 1D shape: either (batch_size) or (2*batch_size). Only the later one has start postions.
  const int* mask_start = (mask_index_dims[0] > batch_size) ? mask_index + batch_size : nullptr;
  ORT_RETURN_IF_ERROR(ComputeSoftmaxWithCumSeqLength<T>(
      scratch1,
      data.relative_position_bias,
      data.cumulative_sequence_length,
      batch_size,
      sequence_length,
      num_heads,
      scratch2, stream));

  DUMP_TENSOR_D("Softmax", scratch2, batch_size * num_heads, sequence_length, sequence_length);

  // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  T* temp_output = qkv;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      v_head_size, sequence_length, sequence_length,
      &one, v, v_head_size, sequence_length * v_head_size,
      scratch2, sequence_length, sequence_length * sequence_length,
      &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop));

  // Temp_output is BxNxSxH_v, transpose to output BxSxNxH_v
  Status result = LaunchTransCtx(stream, sequence_length, batch_size, v_head_size, num_heads,
                                 max_threads_per_block, false, temp_output, data.output);
  DUMP_TENSOR("unfused output", data.output, batch_size * sequence_length, num_heads, v_head_size);
  return result;
}

// Template Instantiation
template struct AttentionData<float>;

template struct AttentionData<half>;

template Status QkvToContext<float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
