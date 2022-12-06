// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"

namespace onnxruntime {
namespace cuda {

struct __align__(8) Half4 {
  half2 x;
  half2 y;
};

__device__ __forceinline__ Half4 operator+(const Half4& a, const Half4& b) {
  Half4 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  return r;
}

__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

}  // namespace cuda
}  // namespace onnxruntime

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void AddBiasTransposeTrt(const T* input, const T* biases, T* output) {
  // Input:  BxSxMxNxH (Format 2)
  // Output: BxSxNxMxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int H = blockDim.x;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const int NH = N * H;
  const int offset = (b * S + s) * M * NH;
  const int in_offset = offset + m * NH + n * H;
  const int out_offset = offset + (n * M + m) * H;

  const int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtLarge(const int head_size, const T* input, const T* biases, T* output) {
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int stride = blockDim.x;
  const int H = head_size;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const int NH = N * H;
  const int offset = (b * S + s) * M * NH;
  const int in_offset = offset + m * NH + n * H;
  const int out_offset = offset + (n * M + m) * H;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeTrt(const T* query, const T* key, const T* value, const T* biases, T* output) {
  // Q:  BxSxNxH
  // K:  BxSxNxH
  // V:  BxSxNxH
  // Output: BxSxNxMxH
  // B is batch_size, S is sequence_length, M is number of matrices (3), N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int H = blockDim.x;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const T* input = (m == 0 ? query : (m == 1 ? key : value));
  const int NH = N * H;
  const int in_offset = (b * S + s) * NH + n * H;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  const int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtLarge(const int head_size,
                                         const T* query, const T* key, const T* value, const T* biases, T* output) {
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int H = head_size;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const T* input = (m == 0 ? query : (m == 1 ? key : value));
  const int NH = N * H;
  const int in_offset = (b * S + s) * NH + n * H;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeQKV(const T* input, const T* biases, T* output) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int M = gridDim.z;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * head_size + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * head_size + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeQKV(const T* input, const T* biases, T* output, int v_head_size) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size

  int n = threadIdx.y;        // head_num_id
  int s = blockIdx.x;         // sequence_id
  int b = blockIdx.y;         // batch_id
  int m = blockIdx.z;         // matrix id (Q=0, K=1, V=2)
  const int h = threadIdx.x;  // head_element_id

  const int qk_head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int head_size = (m == 2 ? v_head_size : qk_head_size);

  const int total_head_size = num_heads * (qk_head_size + qk_head_size + v_head_size);

  int in_offset;
  int out_offset;
  int bias_offset;
  in_offset = b * (total_head_size * sequence_length) +  // B
              s * (total_head_size) +                    // S
              m * (qk_head_size * num_heads) +           // M
              n * head_size +                            // N
              h;                                         // H

  out_offset = m * (num_heads * qk_head_size * sequence_length * batch_size) +  // M
               b * (num_heads * head_size * sequence_length) +                  // B
               n * (sequence_length * head_size) +                              // N
               s * (head_size) +                                                // S
               h;                                                               // H

  bias_offset = m * (num_heads * qk_head_size) +  // M
                n * (head_size) +                 // N
                h;                                // H

  if (h < head_size) {
    output[out_offset] = input[in_offset] + biases[bias_offset];
  }
}

template <typename T>
__global__ void AddBiasTransposeQKVLarge(const int head_size, const T* input, const T* biases, T* output) {
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int M = gridDim.z;
  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  int in_offset = n * H + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTranspose(const T* input, const T* biases, T* output) {
  // Input:  MxBxSxNxH (Format 0)
  // Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;
  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = s * H + n * sequence_length * H + (b + m * batch_size) * NHS;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeLarge(const int head_size, const T* input, const T* biases, T* output) {
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = (s + n * sequence_length) * H + (b + m * batch_size) * NHS;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
void InvokeAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const T* input, const T* biases, T* output, const int v_head_size) {
  const dim3 grid(sequence_length, batch_size, num_matrices);
  if (qk_head_size * num_heads <= max_threads_per_block) {
    const dim3 block(qk_head_size, num_heads, 1);
    if (format == 2) {
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(input, biases, output);
    } else if (format == 1) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input, biases, output);
      } else {
        AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
      }
    } else {
      AddBiasTranspose<T><<<grid, block, 0, stream>>>(input, biases, output);
    }
  } else {
    const dim3 block(CeilDiv(max_threads_per_block, num_heads), num_heads, 1);
    if (format == 2) {
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
    } else if (format == 1) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeQKVLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
      } else {
        ORT_THROW("AddBiasTranspose (format 1) not implemented for hidden_size > max_threads_per_block");
      }
    } else {
      AddBiasTransposeLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
    }
  }
}

template <>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const half* input, const half* biases, half* output,
    bool enable_half4, const int v_head_size) {
  if (enable_half4 && 0 == (qk_head_size % 4) && 0 == (v_head_size % 4)) {
    const int H = qk_head_size / 4;
    const int H_v = v_head_size / 4;
    const Half4* input2 = reinterpret_cast<const Half4*>(input);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* output2 = reinterpret_cast<Half4*>(output);
    InvokeAddBiasTranspose<Half4>(stream, num_matrices, format, max_threads_per_block,
                                  batch_size, sequence_length, num_heads, H, input2, biases2, output2, H_v);
  } else if (0 == (qk_head_size & 1) && 0 == (v_head_size % 1)) {
    const int H = qk_head_size / 2;
    const int H_v = v_head_size / 2;
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* output2 = reinterpret_cast<half2*>(output);
    InvokeAddBiasTranspose<half2>(stream, num_matrices, format, max_threads_per_block,
                                  batch_size, sequence_length, num_heads, H, input2, biases2, output2, H_v);
  } else {
    InvokeAddBiasTranspose<half>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size, input, biases, output, v_head_size);
  }
}

template <>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const float* input, const float* biases, float* output,
    bool /*enable_half4*/, const int v_head_size) {
  if (0 == (qk_head_size % 4)) {
    const int H = qk_head_size / 4;
    const float4* input2 = reinterpret_cast<const float4*>(input);
    const float4* biases2 = reinterpret_cast<const float4*>(biases);
    float4* output2 = reinterpret_cast<float4*>(output);
    InvokeAddBiasTranspose<float4>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, H, input2, biases2, output2, v_head_size / 4);
  } else if (0 == (qk_head_size & 1)) {
    const int H = qk_head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    const float2* biases2 = reinterpret_cast<const float2*>(biases);
    float2* output2 = reinterpret_cast<float2*>(output);

    InvokeAddBiasTranspose<float2>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, H, input2, biases2, output2, v_head_size / 2);
  } else {
    InvokeAddBiasTranspose<float>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size, input, biases, output, v_head_size);
  }
}

template <typename T>
void InvokeAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const T* biases, const T* query, const T* key, const T* value, T* output) {
  constexpr int num_matrices = 3;
  const dim3 grid(sequence_length, batch_size, num_matrices);
  if (head_size * num_heads <= max_threads_per_block) {
    const dim3 block(head_size, num_heads, 1);
    AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, key, value, biases, output);
  } else {
    const dim3 block(CeilDiv(max_threads_per_block, num_heads), num_heads, 1);
    AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, key, value, biases, output);
  }
}

template <>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const float* biases, const float* query, const float* key, const float* value, float* output) {
  ORT_ENFORCE(false, "Shall not call this since fused kernel does not support float input.");
}

template <>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const half* biases, const half* query, const half* key, const half* value, half* output) {
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const Half4* query2 = reinterpret_cast<const Half4*>(query);
    const Half4* key2 = reinterpret_cast<const Half4*>(key);
    const Half4* value2 = reinterpret_cast<const Half4*>(value);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* output2 = reinterpret_cast<Half4*>(output);
    InvokeAddBiasTransposeTrt<Half4>(stream, max_threads_per_block,
                                     batch_size, sequence_length, num_heads, H,
                                     biases2, query2, key2, value2, output2);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const half2* query2 = reinterpret_cast<const half2*>(query);
    const half2* key2 = reinterpret_cast<const half2*>(key);
    const half2* value2 = reinterpret_cast<const half2*>(value);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* output2 = reinterpret_cast<half2*>(output);
    InvokeAddBiasTransposeTrt<half2>(stream, max_threads_per_block,
                                     batch_size, sequence_length, num_heads, H,
                                     biases2, query2, key2, value2, output2);
  } else {
    InvokeAddBiasTransposeTrt<half>(stream, max_threads_per_block,
                                    batch_size, sequence_length, num_heads, head_size,
                                    biases, query, key, value, output);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
