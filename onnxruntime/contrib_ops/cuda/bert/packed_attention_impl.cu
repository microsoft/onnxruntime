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
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/mha_runner.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cross_attention/fmha_cross_attention.h"
#include "contrib_ops/cuda/bert/bert_padding.h"
#include "contrib_ops/cuda/transformers/dump_cuda_tensor.h"
#include "contrib_ops/cuda/bert/cutlass_fmha/memory_efficient_attention.h"
#include "contrib_ops/cuda/bert/rotary_embedding_util.h"

using namespace onnxruntime::cuda;
using namespace onnxruntime::contrib::attention_softmax_cuda;

#define CHECK_CUDA(expr) CUDA_RETURN_IF_ERROR(expr)

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr size_t kCUDAMemoryAlignment = 256;

constexpr int32_t kMAX_THREADS_PER_BLOCK = 256;

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length) {
  const size_t bytes = element_size * batch_size * num_heads * sequence_length * sequence_length;
  return ((bytes + kCUDAMemoryAlignment - 1) / kCUDAMemoryAlignment) * kCUDAMemoryAlignment;
}

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    void* fused_runner,
    bool use_memory_efficient_attention) {
  // Note that q, k and v might need alignment for fused attention kernels.
  const size_t qkv_bytes = element_size * batch_size * num_heads * sequence_length * (qk_head_size + qk_head_size + v_head_size);

  if (fused_runner != nullptr) {
    return qkv_bytes;
  }

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

  return qkv_bytes + 2 * GetAttentionScratchSize(element_size, batch_size, num_heads, sequence_length);
}

template <typename T, AttentionQkvFormat format>
__global__ void AddBiasTransposeQKVPacked(const T* input,
                                          const T* biases,
                                          int32_t N,
                                          int32_t H_QK,
                                          int32_t H_V,
                                          T* q,
                                          T* k,
                                          T* v,
                                          const int32_t* token_offset,
                                          int32_t token_count);

// Grid: (S, B)
// Block: 256
// For unfused PackedAttention
//     Input: Tx3xNxH
//     Output: 3xBxNxSxH
// Where:
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void AddBiasTransposeQKVPacked(
    const T* input,
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
  s = padding_token_idx - b * S;

  input += packing_token_idx * N * (H_QK + H_QK + H_V);
  int k_offset = N * H_QK;
  int v_offset = N * H_QK + N * H_QK;
  q += (b * N * S + s) * H_QK;
  k += (b * N * S + s) * H_QK;
  v += (b * N * S + s) * H_V;

  if (packing_token_idx < token_count) {
    for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = input[i] + biases[i];
      k[n * S * H_QK + h] = input[i + k_offset] + biases[i + k_offset];
    }

    for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
      int h = i % H_V;
      int n = i / H_V;
      v[n * S * H_V + h] = input[i + v_offset] + biases[i + v_offset];
    }
  } else {
    for (int i = threadIdx.x; i < N * H_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = biases[i];
      k[n * S * H_QK + h] = biases[i + k_offset];
    }

    for (int i = threadIdx.x; i < N * H_V; i += blockDim.x) {
      int h = i % H_V;
      int n = i / H_V;
      v[n * S * H_V + h] = biases[i + v_offset];
    }
  }
}

// Grid: (S, B)
// Block: 256
// For memory efficient fMHA from CUTLASS. For future use, doesn't support fMHA from CUTLASS yet.
//     Input: Tx3xNxH
//     Output: 3xTxNxH
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void AddBiasTransposeQKVPackedCutlass(
    const T* input,
    const T* biases,
    int32_t D_QK,
    int32_t D_V,
    T* q,
    T* k,
    T* v,
    int32_t token_count) {
  int token_idx = blockIdx.x;

  input += token_idx * (D_QK + D_QK + D_V);
  q += token_idx * D_QK;
  k += token_idx * D_QK;
  v += token_idx * D_V;

  if (token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      q[i] = input[i] + biases[i];
      k[i] = input[D_QK + i] + biases[D_QK + i];
    }

    for (int i = threadIdx.x; i < D_V; i += blockDim.x) {
      v[i] = input[D_QK + D_QK + i] + biases[D_QK + D_QK + i];
    }
  }
}

// Grid: (S, B)
// Block: 256
// For fMHA from TRT
//     Input: Tx3xNxH
//     Output: TxNx3xH
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void AddBiasTransposeQKVPackedTRT(
    const T* input,
    const T* biases,
    int32_t N,
    int32_t H,
    T* output) {
  int token_idx = blockIdx.x;

  int Hx3 = H * 3;
  int NxH = N * H;
  int NxHx2 = N * H + N * H;

  int offset = token_idx * N * Hx3;
  input += offset;
  output += offset;

  for (int i = threadIdx.x; i < N * H; i += blockDim.x) {
    int n = i / H;
    int h = i % H;
    output[n * Hx3 + h] = input[i] + biases[i];
    output[n * Hx3 + H + h] = input[i + NxH] + biases[i + NxH];
    output[n * Hx3 + H + H + h] = input[i + NxHx2] + biases[i + NxHx2];
  }
}

template <typename T>
void InvokeAddBiasTranspose(
    const T* input, const T* biases, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    AttentionQkvFormat format, const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream) {
  if (format == AttentionQkvFormat::Q_K_V_BNSH) {
    const dim3 grid(sequence_length, batch_size);
    AddBiasTransposeQKVPacked<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        input,
        biases,
        num_heads,
        qk_head_size,
        v_head_size,
        output,
        output + batch_size * sequence_length * num_heads * qk_head_size,
        output + 2 * batch_size * sequence_length * num_heads * qk_head_size,
        token_offset,
        token_count);
  } else if (format == AttentionQkvFormat::Q_K_V_BSNH) {
    const dim3 grid(token_count);
    AddBiasTransposeQKVPackedCutlass<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        input,
        biases,
        num_heads * qk_head_size,
        num_heads * v_head_size,
        output,
        output + token_count * num_heads * qk_head_size,
        output + 2 * token_count * num_heads * qk_head_size,
        token_count);
  } else {
    ORT_ENFORCE(format == AttentionQkvFormat::QKV_BSN3H);
    const dim3 grid(token_count);
    AddBiasTransposeQKVPackedTRT<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        input,
        biases,
        num_heads,
        qk_head_size,
        output);
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
void LaunchAddBiasTranspose(
    const T* input, const T* biases, T* output,
    const int batch_size, const int sequence_length,
    const int num_heads, const int qk_head_size, const int v_head_size,
    AttentionQkvFormat format, const int32_t* token_offset, int32_t token_count,
    cudaStream_t stream) {
  if (0 == (qk_head_size & 3) && 0 == (v_head_size & 3)) {
    using T4Type = typename T4<T>::Type;
    const int H = qk_head_size / 4;
    const int H_v = v_head_size / 4;
    const T4Type* input2 = reinterpret_cast<const T4Type*>(input);
    const T4Type* biases2 = reinterpret_cast<const T4Type*>(biases);
    T4Type* output2 = reinterpret_cast<T4Type*>(output);
    InvokeAddBiasTranspose<T4Type>(
        input2, biases2, output2,
        batch_size, sequence_length,
        num_heads, H, H_v,
        format, token_offset, token_count, stream);
  } else if (0 == (qk_head_size & 1) && 0 == (v_head_size & 1)) {
    using T2Type = typename T2<T>::Type;
    const int H = qk_head_size / 2;
    const int H_v = v_head_size / 2;
    const T2Type* input2 = reinterpret_cast<const T2Type*>(input);
    const T2Type* biases2 = reinterpret_cast<const T2Type*>(biases);
    T2Type* output2 = reinterpret_cast<T2Type*>(output);
    InvokeAddBiasTranspose<T2Type>(
        input2, biases2, output2,
        batch_size, sequence_length,
        num_heads, H, H_v,
        format, token_offset, token_count, stream);
  } else {
    InvokeAddBiasTranspose<T>(
        input, biases, output,
        batch_size, sequence_length,
        num_heads, qk_head_size, v_head_size,
        format, token_offset, token_count, stream);
  }
}

// Input:  BxNxSxH
// Output: TxNxH
// where:
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size

// Grid: T
// Block: 256
template <typename T>
__global__ void __launch_bounds__(kMAX_THREADS_PER_BLOCK)
    TransposeRemovePadding(T* target, const T* source, const int* token_offset,
                           const int B, const int S, const int N, const int H) {
  int token_idx = blockIdx.x;
  int source_idx = token_offset[token_idx];
  int b = source_idx / S;
  int s = source_idx - b * S;

  target += token_idx * N * H;
  source += b * N * S * H + s * H;
  for (int i = threadIdx.x; i < N * H; i += blockDim.x) {
    int n = i / H;
    int h = i - n * H;
    target[i] = source[n * S * H + h];
  }
}

template <typename T>
Status LaunchTransposeRemovePadding(
    T* output, const T* input,
    const int* token_offset, const int token_count,
    const int batch_size, const int seq_len, const int number_heads, const int head_size,
    cudaStream_t stream);

// input: [batch_size, number_heads, seq_len, head_size]
// output: [token_count, number_heads * head_size]
template <>
Status LaunchTransposeRemovePadding(
    half* output, const half* input,
    const int* token_offset, const int token_count,
    const int batch_size, const int seq_len, const int number_heads, const int head_size,
    cudaStream_t stream) {
  // Make sure memory is aligned to 128 bit
  ORT_ENFORCE(!(reinterpret_cast<size_t>(input) & 0xF) && !(reinterpret_cast<size_t>(output) & 0xF), "alignment");

  if (head_size % 8 == 0) {
    const int4* input2 = reinterpret_cast<const int4*>(input);
    int4* output2 = reinterpret_cast<int4*>(output);
    TransposeRemovePadding<int4><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size / 8);
  } else if (head_size % 4 == 0) {
    const int64_t* input2 = reinterpret_cast<const int64_t*>(input);
    int64_t* output2 = reinterpret_cast<int64_t*>(output);
    TransposeRemovePadding<int64_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size / 4);
  } else if (head_size % 2 == 0) {
    const int32_t* input2 = reinterpret_cast<const int32_t*>(input);
    int32_t* output2 = reinterpret_cast<int32_t*>(output);
    TransposeRemovePadding<int32_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size / 2);
  } else {
    const int16_t* input2 = reinterpret_cast<const int16_t*>(input);
    int16_t* output2 = reinterpret_cast<int16_t*>(output);
    TransposeRemovePadding<int16_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size);
  }

  return CUDA_CALL(cudaGetLastError());
}

// input: [batch_size, number_heads, seq_len, head_size]
// output: [token_count, number_heads * head_size]
template <>
Status LaunchTransposeRemovePadding(
    float* output, const float* input,
    const int* token_offset, const int token_count,
    const int batch_size, const int seq_len, const int number_heads, const int head_size,
    cudaStream_t stream) {
  ORT_ENFORCE(!(reinterpret_cast<size_t>(input) & 0xF) && !(reinterpret_cast<size_t>(output) & 0xF), "alignment");

  if (head_size % 4 == 0) {
    const int4* input2 = reinterpret_cast<const int4*>(input);
    int4* output2 = reinterpret_cast<int4*>(output);
    TransposeRemovePadding<int4><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size / 4);
  } else if (head_size % 2 == 0) {
    const int64_t* input2 = reinterpret_cast<const int64_t*>(input);
    int64_t* output2 = reinterpret_cast<int64_t*>(output);
    TransposeRemovePadding<int64_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size / 2);
  } else {
    const int32_t* input2 = reinterpret_cast<const int32_t*>(input);
    int32_t* output2 = reinterpret_cast<int32_t*>(output);
    TransposeRemovePadding<int32_t><<<token_count, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
        output2, input2, token_offset, batch_size, seq_len, number_heads, head_size);
  }

  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status FusedScaledDotProductAttention(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedAttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  void* fused_runner = data.fused_runner;
  ORT_RETURN_IF_NOT(nullptr != fused_runner, "fused_runner cannot be NULL");

  LaunchAddBiasTranspose(data.gemm_buffer, data.bias, data.workspace,
                         batch_size, sequence_length,
                         num_heads, qk_head_size, v_head_size,
                         AttentionQkvFormat::QKV_BSN3H, data.token_offset,
                         parameters.token_count, stream);

  FusedMHARunnerFP16v2* fused_fp16_runner = reinterpret_cast<FusedMHARunnerFP16v2*>(fused_runner);
  const int S = fused_fp16_runner->getSFromMaxSeqLen(sequence_length);
  fused_fp16_runner->setup(S, batch_size);

  fused_fp16_runner->run(data.workspace, data.cumulative_sequence_length, data.output, stream);
  return Status::OK();
}

#if USE_FLASH_ATTENTION
template <typename T>
Status FusedScaledDotProductAttentionCutlass(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedAttentionData<T>& data) {
  const int batch_size = parameters.batch_size;
  const int sequence_length = parameters.sequence_length;
  const int num_heads = parameters.num_heads;
  const int qk_head_size = parameters.head_size;
  const int v_head_size = parameters.v_head_size;
  LaunchAddBiasTranspose(data.gemm_buffer, data.bias, data.workspace,
                         batch_size, sequence_length,
                         num_heads, qk_head_size, v_head_size,
                         AttentionQkvFormat::Q_K_V_BSNH, data.token_offset,
                         parameters.token_count, stream);
  DUMP_TENSOR_INIT();

  DUMP_TENSOR_D("PackedAttention cutlass data.gemm_buffer", data.gemm_buffer, parameters.token_count, 3, num_heads * qk_head_size);
  DUMP_TENSOR_D("PackedAttention cutlass data.bias", data.bias, 1, 3 * num_heads * qk_head_size);

  // Q, K and V pointers
  const int model_dimension_qk = num_heads * qk_head_size;
  const int model_dimension_v = num_heads * v_head_size;
  const size_t elements_qk = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_qk);
  const size_t elements_v = static_cast<size_t>(parameters.token_count) * static_cast<size_t>(model_dimension_v);
  T* qkv = data.workspace;
  T* query = qkv;
  T* key = query + elements_qk;
  T* value = key + elements_qk;
  T* accum_workspace = value + elements_v;

  DUMP_TENSOR_D("PackedAttention cutlass q(BSNH)", query, parameters.token_count, num_heads * qk_head_size);
  DUMP_TENSOR_D("PackedAttention cutlass k(BSNH)", key, parameters.token_count, num_heads * qk_head_size);
  DUMP_TENSOR_D("PackedAttention cutlass v(BSNH)", value, parameters.token_count, num_heads * v_head_size);
  DUMP_TENSOR_D("PackedAttention cutlass cumulative_sequence_length", data.cumulative_sequence_length, 1, batch_size + 1);

  MemoryEfficientAttentionParams p;
  p.sm = device_prop.major * 10 + device_prop.minor;
  p.is_half = sizeof(T) == 2;
  p.batch_size = parameters.batch_size;
  p.num_heads = parameters.num_heads;
  p.sequence_length = parameters.sequence_length;
  p.kv_sequence_length = parameters.sequence_length;
  p.qk_head_size = parameters.head_size;
  p.v_head_size = parameters.v_head_size;
  p.causal = false;
  p.scale = parameters.scale == 0.0f ? 1.f / sqrt(static_cast<float>(qk_head_size))
                                     : parameters.scale;
  p.seqlen_k_ptr = nullptr;
  p.seqstart_q_ptr = const_cast<int32_t*>(data.cumulative_sequence_length);
  p.seqstart_k_ptr = const_cast<int32_t*>(data.cumulative_sequence_length);
  p.query = query;
  p.key = key;
  p.value = value;
  p.attn_bias = data.relative_position_bias;
  p.is_attn_bias_batched = !parameters.broadcast_res_pos_bias;
  p.output = data.output;
  p.workspace = MemoryEfficientAttentionParams::need_workspace(v_head_size, sizeof(T) == sizeof(float)) ? accum_workspace : nullptr;
  p.stream = stream;
  run_memory_efficient_attention(p);

  DUMP_TENSOR("PackedAttention cutlass output", data.output, parameters.token_count, num_heads, v_head_size);
  return Status::OK();
}
#endif

template <typename T>
Status UnfusedScaledDotProductAttention(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedAttentionData<T>& data) {
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
  T* qkv = data.workspace;
  T* q = qkv;
  T* k = q + elements_q;
  T* v = k + elements_k;

  LaunchAddBiasTranspose(data.gemm_buffer, data.bias, data.workspace,
                         batch_size, sequence_length,
                         num_heads, qk_head_size, v_head_size,
                         AttentionQkvFormat::Q_K_V_BNSH, data.token_offset,
                         parameters.token_count, stream);

  T* scaled_qk = qkv + elements_q + elements_k + elements_v;

  // Q, K and V are ready now
  DUMP_TENSOR_INIT();

  DUMP_TENSOR_D("PackedAttention unfused gemm_buffer", data.gemm_buffer, parameters.token_count, (num_heads * (qk_head_size * 2 + v_head_size)));
  DUMP_TENSOR_D("PackedAttention unfused data.workspace", data.workspace, 3 * batch_size, num_heads, sequence_length, qk_head_size);

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

  DUMP_TENSOR_D("PackedAttention unfused QK", scaled_qk, batch_size * num_heads, sequence_length, sequence_length);

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

  DUMP_TENSOR_D("PackedAttention unfused Softmax", attention_score, batch_size * num_heads, sequence_length, sequence_length);

  // compute R*V (as V*R), and store in temp_output (space used by Q): BxNxSxH_v
  T* temp_output = qkv;
  CUBLAS_RETURN_IF_ERROR(cublasGemmStridedBatchedHelper(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N,
      v_head_size, sequence_length, sequence_length,
      &one, v, v_head_size, sequence_length * v_head_size,
      attention_score, sequence_length, sequence_length * sequence_length,
      &zero, temp_output, v_head_size, sequence_length * v_head_size, batches, device_prop));

  // Temp_output is BxNxSxH_v, transpose and remove padding to output token_countxNxH_v
  Status result = LaunchTransposeRemovePadding(
      data.output, temp_output,
      data.token_offset, parameters.token_count,
      batch_size, sequence_length, num_heads, v_head_size,
      stream);

  DUMP_TENSOR("PackedAttention unfused output", data.output, parameters.token_count, num_heads, v_head_size);
  return result;
}

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedAttentionData<T>& data) {
  void* fused_runner = data.fused_runner;
  if (nullptr != fused_runner) {
    return FusedScaledDotProductAttention<T>(device_prop, stream, parameters, data);
  }

#if USE_FLASH_ATTENTION
  if (data.use_memory_efficient_attention) {
    return FusedScaledDotProductAttentionCutlass(device_prop, stream, parameters, data);
  }
#endif

  return UnfusedScaledDotProductAttention<T>(device_prop, cublas, stream, parameters, data);
}

template Status QkvToContext<float>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedAttentionData<float>& data);

template Status QkvToContext<half>(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    PackedAttentionParameters& parameters,
    PackedAttentionData<half>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
