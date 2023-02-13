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
  // Format 2 for TensorRT fused attention (N*H <= 1024)
  //     Input:  BxSxMxNxH
  //     Output: BxSxNxMxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  // This kernel could support hidden size up to 4 * 1024 when T is Half4 and input is half.

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
  // Format 2 for TensorRT fused attention (N*H > 1024)
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
  // Separated Q/K/V inputs for TensorRT fused attention (N*H <= 1024)
  //     Q:  BxSxNxH
  //     K:  BxSxNxH
  //     V:  BxSxNxH
  //     Output: BxSxNxMxH
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
  // Separated Q/K/V inputs for TensorRT fused attention (N*H > 1024)
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
__global__ void AddBiasTransposeTrtKV(const T* key, const T* value, const T* biases, T* output) {
  // Separated K/V inputs for TensorRT fused cross attention (N*H <= 1024)
  //     K:  BxSxNxH
  //     V:  BxSxNxH
  //     Output: BxSxNxMxH (packed KV, requires H = H_v)
  // B is batch_size, S is sequence_length, M is number of matrices (2), N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int H = blockDim.x;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int NH = N * H;

  const int in_offset = (b * S + s) * NH + n * H;
  const T* input = (m == 0 ? key : value);

  constexpr int M = 2;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  const int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[(m + 1) * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtKVLarge(const int head_size,
                                           const T* key, const T* value, const T* biases,
                                           T* output) {
  // Separated K/V inputs for TensorRT fused cross attention (N*H > 1024)
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int H = head_size;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int NH = N * H;

  const int in_offset = (b * S + s) * NH + n * H;
  const T* input = (m == 0 ? key : value);

  constexpr int M = 2;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[(m + 1) * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeQKV(int M, const T* input, const T* biases, T* output, T* qkv_add_bias) {
  // Format 1 for unfused attention, or fused causal attention
  //     Input:  BxSxMxNxH
  //     Output: MxBxNxSxH
  //     qkv_add_bias: BxSxMxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * head_size + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * head_size + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    if (nullptr != qkv_add_bias) {
      qkv_add_bias[in_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    }
  }
}

// this suppose 3 matrix in total
template <typename T>
__global__ void AddBiasTransposeQKV(const T* input, const T* biases, T* output, int v_head_size) {
  // Format 1 for unfused attention
  //     Input:  BxSx(NxH + NxH + NxH_v)  (Packed QKV where K and V has different hidden sizes)
  //     Output: BxNxSxH + BxNxSxH + BxNxSxH_v
  // B is batch_size, S is sequence_length, N is num_heads, H is qk_head_size, H_v is v_head_size
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
__global__ void AddBiasTransposeQKVLarge(const int head_size, const T* input, const T* biases, T* output,
                                         T* qkv_add_bias, const int M) {
  // Format 1 for unfused attention (N*H > 1024), or fused causal attention
  //     Input:  BxSxMxNxH (Packed QKV)
  //     Output: MxBxNxSxH
  //     qkv_add_bias: BxSxMxNxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  int in_offset = n * H + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    if (nullptr != qkv_add_bias) {
      qkv_add_bias[in_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    }
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeCutlass(const T* input, const T* biases, T* output, int v_head_size) {
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSx(NxH + NxH + NxH_v)  (Packed QKV where K and V has different hidden sizes)
  //     Output: BxNxSxH + BxNxSxH + BxNxSxH_v
  // B is batch_size, S is sequence_length, N is num_heads, H is qk_head_size, H_v is v_head_size
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
               s * (num_heads * head_size) +                                    // S
               n * (head_size) +                                                // N
               h;                                                               // H

  bias_offset = m * (num_heads * qk_head_size) +  // M
                n * (head_size) +                 // N
                h;                                // H

  if (h < head_size) {
    output[out_offset] = input[in_offset] + biases[bias_offset];
  }
}

template <typename T>
__global__ void AddBiasUnpack(int M, const T* input, const T* biases, T* output) {
  // Format 4 to unpack TRT packed input format for memory efficient attention.
  //     Input:  BxSxNxMxH
  //     Output: MxBxSxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = m * head_size + n * M * H + (s * NH + b * NHS) * M;
  const int out_offset = n * head_size + s * NH + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    if (biases != nullptr) {
      output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    } else {
      output[out_offset + h] = input[in_offset + h];
    }
  }
}

template <typename T>
__global__ void AddBiasTransposeCutlass(int M, const T* input, const T* biases, T* output) {
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSxMxNxH
  //     Output: MxBxSxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * head_size + (m + s * M) * NH + b * NHS * M;
  const int out_offset = n * head_size + s * NH + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeCutlassLarge(const int head_size, const T* input, const T* biases, T* output,
                                             const int M) {
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSxMxNxH (Packed QKV)
  //     Output: MxBxSxNxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  int in_offset = n * H + (m + s * M) * NH + b * NHS * M;
  const int out_offset = n * H + s * NH + b * NHS + m * NHS * batch_size;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTranspose(const T* input, const T* biases, T* output) {
  // Format 0 for Separated Q, K, V (N*H <= 1024)
  //    Input:  MxBxSxNxH
  //    Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;
  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = (s + n * sequence_length) * H + (b + m * batch_size) * NHS;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeLarge(const int head_size, const T* input, const T* biases, T* output) {
  // Format 0 for Separated Q, K, V (N*H > 1024)
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

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
    const T* input, const T* biases, T* output, T* qkv_add_bias, const int v_head_size, int total_matrix_count) {
  assert(num_heads <= max_threads_per_block);
  const dim3 grid(sequence_length, batch_size, num_matrices);
  if (qk_head_size * num_heads <= max_threads_per_block) {
    const dim3 block(qk_head_size, num_heads, 1);
    if (format == 2) {
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(input, biases, output);
    } else if (format == 1) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output, qkv_add_bias);
      } else {
        ORT_ENFORCE(total_matrix_count == 3);
        AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
      }
    } else if (format == 3) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeCutlass<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output);
      } else {
        ORT_ENFORCE(total_matrix_count == 3);
        AddBiasTransposeCutlass<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
      }
    } else if (format == 4) {  // format == 4
      AddBiasUnpack<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output);
    } else {  // format == 0
      AddBiasTranspose<T><<<grid, block, 0, stream>>>(input, biases, output);
    }
  } else {
    const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
    if (format == 2) {
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
    } else if (format == 1) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeQKVLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output,
                                                                qkv_add_bias, total_matrix_count);
      } else {
        // It is rare for hidden size > 4096 (for half precision) and qk_head_size != v_head_size.
        ORT_THROW("AddBiasTranspose (format 1) not implemented for hidden_size > max_threads_per_block when qk_head_size != v_head_size");
      }
    } else if (format == 3) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeCutlassLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output,
                                                                    total_matrix_count);
      } else {
        ORT_THROW("AddBiasTranspose (format 3) not implemented for hidden_size > max_threads_per_block when qk_head_size != v_head_size");
      }
    } else if (format == 4) {  // format == 4
      ORT_THROW("AddBiasTranspose (format 4) not implemented for hidden_size > max_threads_per_block");
    } else {  // format 0
      AddBiasTransposeLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
    }
  }
}

template <>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const half* input, const half* biases, half* output,
    bool enable_half4, const int v_head_size, half* qkv_add_bias, int total_matrix_count) {
  total_matrix_count = std::max(num_matrices, total_matrix_count);
  if (enable_half4 && 0 == (qk_head_size % 4) && (v_head_size == -1 || 0 == (v_head_size % 4))) {
    const int H = qk_head_size / 4;
    const int H_v = v_head_size / 4;
    const Half4* input2 = reinterpret_cast<const Half4*>(input);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* output2 = reinterpret_cast<Half4*>(output);
    Half4* qkv_add_bias2 = reinterpret_cast<Half4*>(qkv_add_bias);
    InvokeAddBiasTranspose<Half4>(stream, num_matrices, format, max_threads_per_block,
                                  batch_size, sequence_length, num_heads, H, input2, biases2, output2,
                                  qkv_add_bias2, H_v, total_matrix_count);
  } else if (0 == (qk_head_size & 1) && (v_head_size == -1 || 0 == (v_head_size & 1))) {
    const int H = qk_head_size / 2;
    const int H_v = v_head_size / 2;
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* output2 = reinterpret_cast<half2*>(output);
    half2* qkv_add_bias2 = reinterpret_cast<half2*>(qkv_add_bias);
    InvokeAddBiasTranspose<half2>(stream, num_matrices, format, max_threads_per_block,
                                  batch_size, sequence_length, num_heads, H, input2, biases2, output2,
                                  qkv_add_bias2, H_v, total_matrix_count);
  } else {
    InvokeAddBiasTranspose<half>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size, input, biases, output,
        qkv_add_bias, v_head_size, total_matrix_count);
  }
}

template <>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const float* input, const float* biases, float* output,
    bool /*enable_half4*/, const int v_head_size, float* qkv_add_bias, int total_matrix_count) {
  total_matrix_count = std::max(num_matrices, total_matrix_count);
  if (0 == (qk_head_size % 4) && (v_head_size == -1 || 0 == (v_head_size % 4))) {
    const int H = qk_head_size / 4;
    const float4* input2 = reinterpret_cast<const float4*>(input);
    const float4* biases2 = reinterpret_cast<const float4*>(biases);
    float4* output2 = reinterpret_cast<float4*>(output);
    float4* qkv_add_bias2 = reinterpret_cast<float4*>(qkv_add_bias);
    InvokeAddBiasTranspose<float4>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, H, input2, biases2, output2,
        qkv_add_bias2, v_head_size / 4, total_matrix_count);
  } else if (0 == (qk_head_size & 1) && (v_head_size == -1 || 0 == (v_head_size & 1))) {
    const int H = qk_head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    const float2* biases2 = reinterpret_cast<const float2*>(biases);
    float2* output2 = reinterpret_cast<float2*>(output);
    float2* qkv_add_bias2 = reinterpret_cast<float2*>(qkv_add_bias);
    InvokeAddBiasTranspose<float2>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, H, input2, biases2, output2,
        qkv_add_bias2, v_head_size / 2, total_matrix_count);
  } else {
    InvokeAddBiasTranspose<float>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size, input, biases, output,
        qkv_add_bias, v_head_size, total_matrix_count);
  }
}

template <typename T>
void InvokeAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const T* biases, const T* query, const T* key, const T* value, T* output,
    bool is_cross_attention, int kv_sequence_length) {
  if (!is_cross_attention) {
    ORT_ENFORCE(sequence_length == kv_sequence_length);
    constexpr int num_matrices = 3;
    const dim3 grid(sequence_length, batch_size, num_matrices);
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, key, value, biases, output);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, key, value, biases, output);
    }
  } else {  // cross attention
    // Q: add bias
    {
      constexpr int num_matrices = 1;
      const dim3 grid(sequence_length, batch_size, num_matrices);
      if (head_size * num_heads <= max_threads_per_block) {
        const dim3 block(head_size, num_heads, 1);
        AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, biases, output);
      } else {
        const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
        AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, biases, output);
      }
    }

    // KV: add bias and pack kv
    {
      constexpr int num_matrices = 2;
      const dim3 grid(kv_sequence_length, batch_size, num_matrices);
      T* packed_kv = output + batch_size * sequence_length * num_heads * head_size;
      if (head_size * num_heads <= max_threads_per_block) {
        const dim3 block(head_size, num_heads, 1);
        AddBiasTransposeTrtKV<T><<<grid, block, 0, stream>>>(key, value, biases, packed_kv);
      } else {
        const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
        AddBiasTransposeTrtKVLarge<T><<<grid, block, 0, stream>>>(head_size, key, value, biases, packed_kv);
      }
    }
  }
}

template <>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const float* biases, const float* query, const float* key, const float* value, float* output,
    bool is_cross_attention, int kv_sequence_length) {
  ORT_ENFORCE(false, "Shall not call this since fused kernel does not support float input.");
}

template <>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const half* biases, const half* query, const half* key, const half* value, half* output,
    bool is_cross_attention, int kv_sequence_length) {
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const Half4* query2 = reinterpret_cast<const Half4*>(query);
    const Half4* key2 = reinterpret_cast<const Half4*>(key);
    const Half4* value2 = reinterpret_cast<const Half4*>(value);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* output2 = reinterpret_cast<Half4*>(output);
    InvokeAddBiasTransposeTrt<Half4>(stream, max_threads_per_block,
                                     batch_size, sequence_length, num_heads, H,
                                     biases2, query2, key2, value2, output2, is_cross_attention, kv_sequence_length);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const half2* query2 = reinterpret_cast<const half2*>(query);
    const half2* key2 = reinterpret_cast<const half2*>(key);
    const half2* value2 = reinterpret_cast<const half2*>(value);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* output2 = reinterpret_cast<half2*>(output);
    InvokeAddBiasTransposeTrt<half2>(stream, max_threads_per_block,
                                     batch_size, sequence_length, num_heads, H,
                                     biases2, query2, key2, value2, output2, is_cross_attention, kv_sequence_length);
  } else {
    InvokeAddBiasTransposeTrt<half>(stream, max_threads_per_block,
                                    batch_size, sequence_length, num_heads, head_size,
                                    biases, query, key, value, output, is_cross_attention, kv_sequence_length);
  }
}

template <typename T>
void InvokeAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const T* biases, const T* query, const T* key, const T* value, T* q, T* k, T* v) {
  assert(num_heads <= max_threads_per_block);
  constexpr int num_matrices = 1;
  // Q
  {
    const dim3 grid(sequence_length, batch_size, num_matrices);
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, biases, q);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, biases, q);
    }
  }
  // K
  {
    const dim3 grid(kv_sequence_length, batch_size, num_matrices);
    const T* biases_k = biases + num_heads * head_size;

    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(key, biases_k, k);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, key, biases_k, k);
    }
  }

  // V
  {
    const dim3 grid(kv_sequence_length, batch_size, num_matrices);

    const T* biases_v = biases + 2 * num_heads * head_size;
    if (v_head_size * num_heads <= max_threads_per_block) {
      const dim3 block(v_head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(value, biases_v, v);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(v_head_size, value, biases_v, v);
    }
  }
}

template <>
void LaunchAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const float* biases, const float* query, const float* key, const float* value, float* q, float* k, float* v) {
  if (0 == (head_size % 4) && 0 == (v_head_size % 4)) {
    const int H = head_size / 4;
    const int H_v = v_head_size / 4;
    const float4* query2 = reinterpret_cast<const float4*>(query);
    const float4* key2 = reinterpret_cast<const float4*>(key);
    const float4* value2 = reinterpret_cast<const float4*>(value);
    const float4* biases2 = reinterpret_cast<const float4*>(biases);
    float4* q2 = reinterpret_cast<float4*>(q);
    float4* k2 = reinterpret_cast<float4*>(k);
    float4* v2 = reinterpret_cast<float4*>(v);
    InvokeAddBias<float4>(stream, max_threads_per_block,
                          batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                          biases2, query2, key2, value2, q2, k2, v2);
  } else if (0 == (head_size & 1) && 0 == (v_head_size & 1)) {
    const int H = head_size / 2;
    const int H_v = v_head_size / 2;
    const float2* query2 = reinterpret_cast<const float2*>(query);
    const float2* key2 = reinterpret_cast<const float2*>(key);
    const float2* value2 = reinterpret_cast<const float2*>(value);
    const float2* biases2 = reinterpret_cast<const float2*>(biases);
    float2* q2 = reinterpret_cast<float2*>(q);
    float2* k2 = reinterpret_cast<float2*>(k);
    float2* v2 = reinterpret_cast<float2*>(v);
    InvokeAddBias<float2>(stream, max_threads_per_block,
                          batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                          biases2, query2, key2, value2, q2, k2, v2);
  } else {
    InvokeAddBias<float>(stream, max_threads_per_block,
                         batch_size, sequence_length, kv_sequence_length, num_heads, head_size, v_head_size,
                         biases, query, key, value, q, k, v);
  }
}

template <>
void LaunchAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const half* biases, const half* query, const half* key, const half* value, half* q, half* k, half* v) {
  if (0 == (head_size % 4) && 0 == (v_head_size % 4)) {
    const int H = head_size / 4;
    const int H_v = v_head_size / 4;
    const Half4* query2 = reinterpret_cast<const Half4*>(query);
    const Half4* key2 = reinterpret_cast<const Half4*>(key);
    const Half4* value2 = reinterpret_cast<const Half4*>(value);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* q2 = reinterpret_cast<Half4*>(q);
    Half4* k2 = reinterpret_cast<Half4*>(k);
    Half4* v2 = reinterpret_cast<Half4*>(v);
    InvokeAddBias<Half4>(stream, max_threads_per_block,
                         batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                         biases2, query2, key2, value2, q2, k2, v2);
  } else if (0 == (head_size & 1) && 0 == (v_head_size & 1)) {
    const int H = head_size / 2;
    const int H_v = v_head_size / 2;
    const half2* query2 = reinterpret_cast<const half2*>(query);
    const half2* key2 = reinterpret_cast<const half2*>(key);
    const half2* value2 = reinterpret_cast<const half2*>(value);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* q2 = reinterpret_cast<half2*>(q);
    half2* k2 = reinterpret_cast<half2*>(k);
    half2* v2 = reinterpret_cast<half2*>(v);
    InvokeAddBias<half2>(stream, max_threads_per_block,
                         batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                         biases2, query2, key2, value2, q2, k2, v2);
  } else {
    InvokeAddBias<half>(stream, max_threads_per_block,
                        batch_size, sequence_length, kv_sequence_length, num_heads, head_size, v_head_size,
                        biases, query, key, value, q, k, v);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
