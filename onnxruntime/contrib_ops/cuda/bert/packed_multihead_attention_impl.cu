// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "contrib_ops/cuda/bert/packed_attention_impl.h"
#include "contrib_ops/cuda/bert/packed_multihead_attention_impl.h"
#include "contrib_ops/cuda/bert/attention_softmax.h"
#include "contrib_ops/cuda/bert/transformer_common.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/rotary_embedding_util.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_api.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::attention_softmax_cuda;

#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace cuda {

static constexpr int32_t kMAX_THREADS_PER_BLOCK = 256;

#define ADD_BIAS(value, bias_value) (biases == nullptr) ? value : (value + bias_value)
#define GET_BIAS(bias_value) (biases == nullptr) ? T{} : bias_value

// Grid: (S, B)
// Block: 256
// For unfused PackedMultiHeadAttention
//     Inputs (query, key, value): TxNxH
//     Output: 3xBxNxSxH
// Where:
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void TransposeQKV_TNH_3BNSH(
    const T* query,
    const T* key,
    const T* value,
    const T* biases,
    int32_t N,
    int32_t H_QK,
    int32_t H_V,
    T* q,
    T* k,
    T* v,
    const int32_t* token_offset,
    int32_t token_count) {
  int s = blockIdx.x;
  int b = blockIdx.y;

  int S = gridDim.x;

  const int packing_token_idx = b * S + s;
  const int padding_token_idx = token_offset[packing_token_idx];
  b = padding_token_idx / S;
  s = padding_token_idx % S;

  const int D_QK = N * H_QK;
  const int D_V = N * H_V;
  query += packing_token_idx * D_QK;
  key += packing_token_idx * D_QK;
  value += packing_token_idx * D_V;

  int k_offset = D_QK;
  int v_offset = D_QK + D_QK;
  q += (b * N * S + s) * H_QK;
  k += (b * N * S + s) * H_QK;
  v += (b * N * S + s) * H_V;

  if (packing_token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = ADD_BIAS(query[i], biases[i]);
      k[n * S * H_QK + h] = ADD_BIAS(key[i], biases[i + k_offset]);
    }

    for (int i = threadIdx.x; i < D_V; i += blockDim.x) {
      int h = i % H_V;
      int n = i / H_V;
      v[n * S * H_V + h] = ADD_BIAS(value[i], biases[i + v_offset]);
    }
  } else {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = GET_BIAS(biases[i]);
      k[n * S * H_QK + h] = GET_BIAS(biases[i + k_offset]);
    }

    for (int i = threadIdx.x; i < D_V; i += blockDim.x) {
      int h = i % H_V;
      int n = i / H_V;
      v[n * S * H_V + h] = GET_BIAS(biases[i + v_offset]);
    }
  }
}

// Grid: (T)
// Block: 256
// For memory efficient fMHA from CUTLASS.
//     query, key, value: TxNxH
//     q, k, v: TxNxH
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void TransposeQKV_TNH_3TNH(
    const T* query,
    const T* key,
    const T* value,
    const T* biases,
    int32_t N,
    int32_t H_QK,
    int32_t H_V,
    T* q,
    T* k,
    T* v,
    int32_t token_count) {
  int token_idx = blockIdx.x;

  const int D_QK = N * H_QK;
  const int D_V = N * H_V;

  query += token_idx * D_QK;
  key += token_idx * D_QK;
  value += token_idx * D_V;

  q += token_idx * D_QK;
  k += token_idx * D_QK;
  v += token_idx * D_V;

  if (token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      q[i] = ADD_BIAS(query[i], biases[i]);
      k[i] = ADD_BIAS(key[i], biases[D_QK + i]);
    }

    for (int i = threadIdx.x; i < D_V; i += blockDim.x) {
      v[i] = ADD_BIAS(value[i], biases[D_QK + D_QK + i]);
    }
  }
}

// Grid: (T)
// Block: 256
// For Trt fused attention.
//     Inputs (query, key, value): TxNxH
//     Output: TxNx3xH
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void TransposeQKV_TNH_TN3H(
    const T* query,
    const T* key,
    const T* value,
    const T* biases,
    int32_t N,
    int32_t H_QK,
    int32_t H_V,
    T* output,
    int32_t token_count) {
  int token_idx = blockIdx.x;

  const int D_QK = N * H_QK;
  const int D_V = N * H_V;
  query += token_idx * D_QK;
  key += token_idx * D_QK;
  value += token_idx * D_V;
  output += token_idx * (D_QK + D_QK + D_V);

  if (token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int n = i / H_QK;
      int h = i % H_QK;

      int index = n * (H_QK + H_QK + H_V) + h;
      output[index] = ADD_BIAS(query[i], biases[i]);

      index = n * (H_QK + H_QK + H_V) + H_QK + h;
      output[index] = ADD_BIAS(key[i], biases[D_QK + i]);
    }

    for (int i = threadIdx.x; i < D_V; i += blockDim.x) {
      int n = i / H_V;
      int h = i % H_V;
      int index = n * (H_QK + H_QK + H_V) + H_QK + H_QK + h;
      output[index] = ADD_BIAS(value[i], biases[D_QK + D_QK + i]);
    }
  }
}

// Grid: (S, B)
// Block: 256
// For unfused PackedMultiHeadAttention
//     Input: TxNx3xH
//     Output: 3xBxNxSxH
// Where:
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void TransposeQKV_TN3H_3BNSH(
    const T* input,  // packed qkv
    const T* biases,
    int32_t N,
    int32_t H_QK,
    int32_t H_V,
    T* q,
    T* k,
    T* v,
    const int32_t* token_offset,
    int32_t token_count) {
  int s = blockIdx.x;
  int b = blockIdx.y;

  int S = gridDim.x;

  const int packing_token_idx = b * S + s;
  const int padding_token_idx = token_offset[packing_token_idx];
  b = padding_token_idx / S;
  s = padding_token_idx % S;

  int Hx3 = (H_QK + H_QK + H_V);

  input += packing_token_idx * N * Hx3;
  int k_offset = H_QK;
  int v_offset = H_QK + H_QK;
  q += (b * N * S + s) * H_QK;
  k += (b * N * S + s) * H_QK;
  v += (b * N * S + s) * H_V;

  if (packing_token_idx < token_count) {
    for (int i = threadIdx.x; i < N * Hx3; i += blockDim.x) {
      int n = i / Hx3;
      int h = i % Hx3;

      if (h < k_offset) {
        q[n * S * H_QK + h] = ADD_BIAS(input[i], biases[n * H_QK + h]);
      } else if (h < v_offset) {
        k[n * S * H_QK + (h - k_offset)] = ADD_BIAS(input[i], biases[(N + n) * H_QK + (h - H_QK)]);
      } else {
        v[n * S * H_V + (h - v_offset)] = ADD_BIAS(input[i], biases[(N + N) * H_QK + n * H_V + (h - H_QK - H_QK)]);
      }
    }
  } else {
    for (int i = threadIdx.x; i < N * Hx3; i += blockDim.x) {
      int n = i / Hx3;
      int h = i % Hx3;

      if (h < k_offset) {
        q[n * S * H_QK + h] = GET_BIAS(biases[n * H_QK + h]);
      } else if (h < v_offset) {
        k[n * S * H_QK + (h - k_offset)] = GET_BIAS(biases[(N + n) * H_QK + (h - H_QK)]);
      } else {
        v[n * S * H_V + (h - v_offset)] = GET_BIAS(biases[(N + N) * H_QK + n * H_V + (h - H_QK - H_QK)]);
      }
    }
  }
}

// TODO: merge TransposeQKV_TN3H_3TNH with AddBiasTransposeQKVPackedCutlass

// Grid: (T)
// Block: 256
// For memory efficient fMHA from CUTLASS.
//     Input: TxNx3xH
//     Output: 3xTxNxH
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void TransposeQKV_TN3H_3TNH(
    const T* input,
    const T* biases,
    int32_t N,
    int32_t H_QK,
    int32_t H_V,
    T* q,
    T* k,
    T* v,
    int32_t token_count) {
  int token_idx = blockIdx.x;

  const int D_QK = N * H_QK;
  const int D_V = N * H_V;

  input += token_idx * (D_QK + D_QK + D_V);
  q += token_idx * D_QK;
  k += token_idx * D_QK;
  v += token_idx * D_V;

  if (token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int n = i / H_QK;
      int h = i % H_QK;
      q[i] = ADD_BIAS(input[n * (H_QK + H_QK + H_V) + h], biases[i]);
      k[i] = ADD_BIAS(input[n * (H_QK + H_QK + H_V) + H_QK + h], biases[D_QK + i]);
    }

    for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
      int n = i / H_V;
      int h = i % H_V;
      v[i] = ADD_BIAS(input[n * (H_QK + H_QK + H_V) + H_QK + H_QK + h], biases[D_QK + D_QK + i]);
    }
  }
}

// Grid: (T)
// Block: 256
// For TRT fused attention.
//     Input: TxNx3xH
//     Output: TxNx3xH
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void AddBias_TN3H_TN3H(
    const T* input,
    const T* biases,
    int32_t N,
    int32_t H_QK,
    int32_t H_V,
    T* output,
    int32_t token_count) {
  int token_idx = blockIdx.x;

  const int D_QK = N * H_QK;
  const int D_V = N * H_V;

  input += token_idx * (D_QK + D_QK + D_V);
  output += token_idx * (D_QK + D_QK + D_V);

  if (token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int n = i / H_QK;
      int h = i % H_QK;

      int index = n * (H_QK + H_QK + H_V) + h;
      output[index] = ADD_BIAS(input[index], biases[i]);

      index = n * (H_QK + H_QK + H_V) + H_QK + h;
      output[index] = ADD_BIAS(input[index], biases[D_QK + i]);
    }

    for (int i = threadIdx.x; i < D_V; i += blockDim.x) {
      int n = i / H_V;
      int h = i % H_V;
      int index = n * (H_QK + H_QK + H_V) + H_QK + H_QK + h;
      output[index] = ADD_BIAS(input[index], biases[D_QK + D_QK + i]);
    }
  }
}

template <typename T>
void InvokeTranspose(
    const T* query, const T* key, const T* value, const T* bias, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    AttentionQkvFormat source_format, AttentionQkvFormat target_format,
    const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream) {
  if (key != nullptr && value != nullptr) {
    assert(source_format == AttentionQkvFormat::Q_K_V_TNH);

    if (target_format == AttentionQkvFormat::Q_K_V_BNSH) {
      const dim3 grid(sequence_length, batch_size);
      TransposeQKV_TNH_3BNSH<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
          query,
          key,
          value,
          bias,
          num_heads,
          qk_head_size,
          v_head_size,
          output,
          output + batch_size * sequence_length * num_heads * qk_head_size,
          output + 2 * batch_size * sequence_length * num_heads * qk_head_size,
          token_offset,
          token_count);
    } else if (target_format == AttentionQkvFormat::Q_K_V_TNH) {
      const dim3 grid(token_count);
      TransposeQKV_TNH_3TNH<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
          query,
          key,
          value,
          bias,
          num_heads,
          qk_head_size,
          v_head_size,
          output,
          output + token_count * num_heads * qk_head_size,
          output + 2 * token_count * num_heads * qk_head_size,
          token_count);
    } else {
      assert(target_format == AttentionQkvFormat::QKV_TN3H);
      const dim3 grid(token_count);
      TransposeQKV_TNH_TN3H<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
          query,
          key,
          value,
          bias,
          num_heads,
          qk_head_size,
          v_head_size,
          output,
          token_count);
    }

  } else {
    assert(key == nullptr && value == nullptr);
    assert(source_format == AttentionQkvFormat::QKV_TN3H);
    if (target_format == AttentionQkvFormat::Q_K_V_BNSH) {
      const dim3 grid(sequence_length, batch_size);
      TransposeQKV_TN3H_3BNSH<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
          query,
          bias,
          num_heads,
          qk_head_size,
          v_head_size,
          output,
          output + batch_size * sequence_length * num_heads * qk_head_size,
          output + 2 * batch_size * sequence_length * num_heads * qk_head_size,
          token_offset,
          token_count);
    } else if (target_format == AttentionQkvFormat::Q_K_V_TNH) {
      const dim3 grid(token_count);
      TransposeQKV_TN3H_3TNH<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
          query,
          bias,
          num_heads,
          qk_head_size,
          v_head_size,
          output,
          output + token_count * num_heads * qk_head_size,
          output + 2 * token_count * num_heads * qk_head_size,
          token_count);
    } else {
      assert(target_format == AttentionQkvFormat::QKV_TN3H);
      assert(bias != nullptr);
      const dim3 grid(token_count);
      AddBias_TN3H_TN3H<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
          query,
          bias,
          num_heads,
          qk_head_size,
          v_head_size,
          output,
          token_count);
    }
  }
}

template <typename T>
struct T4;

template <>
struct T4<float> {
  using Type = float4;
};

template <>
struct T4<half> {
  using Type = Half4;
};

template <typename T>
struct T2;

template <>
struct T2<float> {
  using Type = float2;
};

template <>
struct T2<half> {
  using Type = half2;
};

template <typename T>
void LaunchTranspose(
    const T* query, const T* key, const T* value, const T* bias, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    AttentionQkvFormat source_format, AttentionQkvFormat target_format,
    const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream) {
  if (0 == (qk_head_size & 3) && 0 == (v_head_size & 3)) {
    using T4Type = typename T4<T>::Type;
    const int H = qk_head_size / 4;
    const int H_v = v_head_size / 4;
    const T4Type* query2 = reinterpret_cast<const T4Type*>(query);
    const T4Type* key2 = reinterpret_cast<const T4Type*>(key);
    const T4Type* value2 = reinterpret_cast<const T4Type*>(value);
    const T4Type* bias2 = reinterpret_cast<const T4Type*>(bias);
    T4Type* output2 = reinterpret_cast<T4Type*>(output);
    InvokeTranspose<T4Type>(
        query2, key2, value2, bias2, output2,
        batch_size, sequence_length,
        num_heads, H, H_v,
        source_format, target_format,
        token_offset, token_count, stream);
  } else if (0 == (qk_head_size & 1) && 0 == (v_head_size & 1)) {
    using T2Type = typename T2<T>::Type;
    const int H = qk_head_size / 2;
    const int H_v = v_head_size / 2;
    const T2Type* query2 = reinterpret_cast<const T2Type*>(query);
    const T2Type* key2 = reinterpret_cast<const T2Type*>(key);
    const T2Type* value2 = reinterpret_cast<const T2Type*>(value);
    const T2Type* bias2 = reinterpret_cast<const T2Type*>(bias);
    T2Type* output2 = reinterpret_cast<T2Type*>(output);
    InvokeTranspose<T2Type>(
        query2, key2, value2, bias2, output2,
        batch_size, sequence_length,
        num_heads, H, H_v,
        source_format, target_format,
        token_offset, token_count, stream);
  } else {
    InvokeTranspose<T>(
        query, key, value, bias, output,
        batch_size, sequence_length,
        num_heads, qk_head_size, v_head_size,
        source_format, target_format,
        token_offset, token_count, stream);
  }
}

template <typename T>
Status FusedAttentionTrt(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  void* fused_runner = data.fused_runner;
  ORT_RETURN_IF_NOT(nullptr != fused_runner, "fused_runner cannot be NULL");

  // When packed QKV is used, we can directly pass it to fused runner. Otherwise, we need transpose to BSN3H format.
  const T* qkv = data.query;
  if (!data.no_qkv_workspace) {
    LaunchTranspose(data.query, data.key, data.value, data.bias, data.workspace,
                    batch_size, sequence_length,
                    num_heads, qk_head_size, v_head_size,
                    data.source_qkv_format, AttentionQkvFormat::QKV_TN3H,
                    data.token_offset, parameters.token_count, stream);
    qkv = data.workspace;
  }

  FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(fused_runner);
  const int S = fused_fp16_runner->getSFromMaxSeqLen(sequence_length);
  fused_fp16_runner->setup(S, batch_size);

  fused_fp16_runner->run(qkv, data.cumulative_sequence_length, data.output, stream);
  return Status::OK();
}

#if USE_FLASH_ATTENTION
template <typename T>
Status FlashAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  // Q, K and V pointers
  const int model_dimension_qk = num_heads * qk_head_size;
  const int model_dimension_v = num_heads * v_head_size;
  const size_t elements_qk = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_qk);
  const size_t elements_v = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_v);

  // When separated Q, K, V is used, we can directly use them in Cutlass FMHA. Otherwise, transpose BSN3H to 3BSNH
  if (!data.no_qkv_workspace) {
    LaunchTranspose(data.query, data.key, data.value, data.bias, data.workspace,
                    batch_size, sequence_length,
                    num_heads, qk_head_size, v_head_size,
                    data.source_qkv_format, AttentionQkvFormat::Q_K_V_TNH,
                    data.token_offset, parameters.token_count, stream);
  }

  float scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                         : parameters.scale;
  int32_t* cu_seqlens_q = const_cast<int32_t*>(data.cumulative_sequence_length);
  int32_t* cu_seqlens_k = const_cast<int32_t*>(data.cumulative_sequence_length);
  const void* query = data.no_qkv_workspace ? data.query : data.workspace;
  const void* key = data.no_qkv_workspace ? data.key : (data.workspace + elements_qk);
  const void* value = data.no_qkv_workspace ? data.value : (data.workspace + elements_qk + elements_qk);
  void* softmax_lse_buffer = data.no_qkv_workspace
                                 ? data.workspace
                                 : (data.workspace + elements_qk + elements_qk + elements_v);

  ORT_RETURN_IF_ERROR(
      onnxruntime::flash::mha_varlen_fwd(
          device_prop,
          stream,
          const_cast<void*>(query),
          const_cast<void*>(key),
          const_cast<void*>(value),
          data.output,
          cu_seqlens_q,
          cu_seqlens_k,
          softmax_lse_buffer,
          batch_size,
          num_heads,
          num_heads,  // num_heads_k
          qk_head_size,
          sequence_length,
          sequence_length,
          scale,
          false  // is causal
          ));

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(query), parameters.token_count, num_heads, qk_head_size);
  DUMP_TENSOR_D("k(BSNH)", reinterpret_cast<const T*>(key), parameters.token_count, num_heads, qk_head_size);
  DUMP_TENSOR_D("v(BSNH)", reinterpret_cast<const T*>(value), parameters.token_count, num_heads, v_head_size);
  DUMP_TENSOR_D("cumulative_sequence_length", data.cumulative_sequence_length, 1, batch_size + 1);
  DUMP_TENSOR("PackedMHA flash output", data.output, parameters.token_count, num_heads, v_head_size);

  return Status::OK();
}
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
template <typename T>
Status FusedAttentionCutlass(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  // Q, K and V pointers
  const int model_dimension_qk = num_heads * qk_head_size;
  const int model_dimension_v = num_heads * v_head_size;
  const size_t elements_qk = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_qk);
  const size_t elements_v = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_v);

  // When separated Q, K, V is used, we can directly use them in Cutlass FMHA. Otherwise, transpose BSN3H to 3BSNH
  if (!data.no_qkv_workspace) {
    LaunchTranspose(data.query, data.key, data.value, data.bias, data.workspace,
                    batch_size, sequence_length,
                    num_heads, qk_head_size, v_head_size,
                    data.source_qkv_format, AttentionQkvFormat::Q_K_V_TNH,
                    data.token_offset, parameters.token_count, stream);
  }

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_half = sizeof(T) == 2;
  p.batch_size = parameters.batch_size;
  p.num_heads = parameters.num_heads;
  p.sequence_length = parameters.sequence_length;
  p.kv_sequence_length = parameters.sequence_length;
  p.max_sequence_length = parameters.sequence_length;
  p.qk_head_size = parameters.head_size;
  p.v_head_size = parameters.v_head_size;
  p.causal = false;
  p.scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                     : parameters.scale;
  p.seqlen_k_ptr = nullptr;
  p.seqstart_q_ptr = const_cast<int32_t*>(data.cumulative_sequence_length);
  p.seqstart_k_ptr = const_cast<int32_t*>(data.cumulative_sequence_length);
  p.query = data.no_qkv_workspace ? data.query : data.workspace;
  p.key = data.no_qkv_workspace ? data.key : (data.workspace + elements_qk);
  p.value = data.no_qkv_workspace ? data.value : (data.workspace + elements_qk + elements_qk);
  p.attn_bias = data.relative_position_bias;
  p.is_attn_bias_batched = !parameters.broadcast_res_pos_bias;
  p.output = data.output;
  p.is_kv_bsnh = true;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(v_head_size, sizeof(T) == sizeof(float))
                    ? (data.workspace + (data.no_qkv_workspace ? 0 : (elements_qk + elements_qk + elements_v)))
                    : nullptr;
  p.stream = stream;
  run_memory_efficient_attention(p);

  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q(BSNH)", reinterpret_cast<const T*>(p.query), parameters.token_count, num_heads, qk_head_size);
  DUMP_TENSOR_D("k(BSNH)", reinterpret_cast<const T*>(p.key), parameters.token_count, num_heads, qk_head_size);
  DUMP_TENSOR_D("v(BSNH)", reinterpret_cast<const T*>(p.value), parameters.token_count, num_heads, v_head_size);
  DUMP_TENSOR_D("cumulative_sequence_length", data.cumulative_sequence_length, 1, batch_size + 1);
  DUMP_TENSOR("PackedMHA cutlass output", data.output, parameters.token_count, num_heads, v_head_size);

  return Status::OK();
}
#endif

template <typename T>
Status UnfusedAttention(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  constexpr size_t element_size = sizeof(T);
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;

  const int batches = batch_size * num_heads;
  const int size_per_batch_q = sequence_length * qk_head_size;
  const int size_per_batch_k = sequence_length * qk_head_size;
  const int size_per_batch_v = sequence_length * v_head_size;
  const size_t elements_q = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_q);
  const size_t elements_k = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_k);
  const size_t elements_v = static_cast<size_t>(batches) * static_cast<size_t>(size_per_batch_v);

  // Q, K and V pointers when fused attention is not used
  LaunchTranspose(data.query, data.key, data.value, data.bias, data.workspace,
                  batch_size, sequence_length,
                  num_heads, qk_head_size, v_head_size,
                  data.source_qkv_format, AttentionQkvFormat::Q_K_V_BNSH,
                  data.token_offset, parameters.token_count, stream);

  T* qkv = data.workspace;
  T* q = qkv;
  T* k = q + elements_q;
  T* v = k + elements_k;
  T* scaled_qk = qkv + elements_q + elements_k + elements_v;

  // Compute Q*K' (as K'*Q), scaled by 1/sqrt(H) and store in scaled_qk: BxNxSxT
  // Q: BxNxSxH, K: BxNxSxH, Q*K': BxNxSxS
  float one = 1.0f;
  float zero = 0.f;
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
      scaled_qk, sequence_length, sequence_length * sequence_length,
      batches, device_prop));

  // Q, K and V are ready now
  DUMP_TENSOR_INIT();
  DUMP_TENSOR_D("q (BNSH)", q, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("k (BNSH)", k, batch_size, num_heads, sequence_length, qk_head_size);
  DUMP_TENSOR_D("v (BNSH)", v, batch_size, num_heads, sequence_length, v_head_size);
  DUMP_TENSOR_D("QK", scaled_qk, batch_size, num_heads, sequence_length, sequence_length);

  const size_t bytes = GetAttentionScratchSize(element_size, batch_size, num_heads,
                                               sequence_length);
  T* attention_score = scaled_qk + (bytes / element_size);

  // Apply softmax and store result R to attention_score: BxNxSxS
  ORT_RETURN_IF_ERROR(ComputeSoftmaxWithCumSeqLength<T>(
      scaled_qk,
      data.relative_position_bias,
      parameters.broadcast_res_pos_bias,
      data.cumulative_sequence_length,
      batch_size,
      sequence_length,
      num_heads,
      attention_score, stream));

  DUMP_TENSOR_D("Softmax", attention_score, batch_size, num_heads, sequence_length, sequence_length);

  // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  T* temp_output = qkv;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      v_head_size, sequence_length, sequence_length,
      &one, v, v_head_size, sequence_length * v_head_size,
      attention_score, sequence_length, sequence_length * sequence_length,
      &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop));

  // Temp_output is BxNxSxH_v, transpose and remove padding to output TxNxH_v
  Status result = LaunchTransposeRemovePadding(
      data.output, temp_output,
      data.token_offset, parameters.token_count,
      batch_size, sequence_length, num_heads, v_head_size,
      stream);

  DUMP_TENSOR("PackedMHA unfused output", data.output, parameters.token_count, num_heads, v_head_size);
  return result;
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data) {
  void* fused_runner = data.fused_runner;
  if (nullptr != fused_runner) {
    return FusedAttentionTrt<T>(device_prop, stream, parameters, data);
  }

#if USE_FLASH_ATTENTION
  if (data.use_flash_attention) {
    return FlashAttention(device_prop, stream, parameters, data);
  }
#endif

#if USE_MEMORY_EFFICIENT_ATTENTION
  if (data.use_memory_efficient_attention) {
    return FusedAttentionCutlass(device_prop, stream, parameters, data);
  }
#endif

  return UnfusedAttention<T>(device_prop, cublas, stream, parameters, data);
}

template Status QkvToContext<float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
