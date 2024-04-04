/*
Copyright (c) Microsoft Corporation.
Licensed under the MIT License.
*/
/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/relative_attn_bias_impl.h"
#include "contrib_ops/cuda/bert/utils.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

static constexpr int32_t kMAX_THREADS_PER_BLOCK = 256;

// Grid: (S, B)
// Block: 256
// For packed input
//     query: TxNxH
//     Output: BxNxSxH
// Where:
// T is token_count
// B is batch_size
// S is sequence_length
// N is num_heads
// H is head_size
template <typename T>
__global__ void TransposeQKV_TNH_3BNSH(
    const T* query,
    const T* biases,
    int32_t N,
    int32_t H_QK,
    T* q,
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
  query += packing_token_idx * D_QK;

  q += (b * N * S + s) * H_QK;

  if (packing_token_idx < token_count) {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = (biases == nullptr) ? query[i] : (query[i] + biases[i]);
    }
  } else {
    for (int i = threadIdx.x; i < D_QK; i += blockDim.x) {
      int h = i % H_QK;
      int n = i / H_QK;
      q[n * S * H_QK + h] = (biases == nullptr) ? T{} : biases[i];
    }
  }
}

template <typename T>
void InvokeTranspose(
    const T* query, const T* bias, T* output,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const int32_t* token_offset, int32_t token_count, cudaStream_t stream) {
  const dim3 grid(sequence_length, batch_size);
  TransposeQKV_TNH_3BNSH<T><<<grid, kMAX_THREADS_PER_BLOCK, 0, stream>>>(
      query,
      bias,
      num_heads,
      qk_head_size,
      output,
      token_offset,
      token_count);
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
void RestorePaddingAddBiasTranspose(
    const T* query, const T* bias, T* output,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const int32_t* token_offset, int32_t token_count, cudaStream_t stream) {
  if (0 == (qk_head_size & 3)) {
    using T4Type = typename T4<T>::Type;
    const int H = qk_head_size / 4;
    const T4Type* query2 = reinterpret_cast<const T4Type*>(query);
    const T4Type* bias2 = reinterpret_cast<const T4Type*>(bias);
    T4Type* output2 = reinterpret_cast<T4Type*>(output);
    InvokeTranspose<T4Type>(
        query2, bias2, output2,
        batch_size, sequence_length,
        num_heads, H,
        token_offset, token_count, stream);
  } else if (0 == (qk_head_size & 1)) {
    using T2Type = typename T2<T>::Type;
    const int H = qk_head_size / 2;
    const T2Type* query2 = reinterpret_cast<const T2Type*>(query);
    const T2Type* bias2 = reinterpret_cast<const T2Type*>(bias);
    T2Type* output2 = reinterpret_cast<T2Type*>(output);
    InvokeTranspose<T2Type>(
        query2, bias2, output2,
        batch_size, sequence_length,
        num_heads, H,
        token_offset, token_count, stream);
  } else {
    InvokeTranspose<T>(
        query, bias, output,
        batch_size, sequence_length,
        num_heads, qk_head_size,
        token_offset, token_count, stream);
  }
}

template void RestorePaddingAddBiasTranspose(
    const float* query, const float* bias, float* output,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const int32_t* token_offset, int32_t token_count, cudaStream_t stream);

template void RestorePaddingAddBiasTranspose(
    const half* query, const half* bias, half* output,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const int32_t* token_offset, int32_t token_count, cudaStream_t stream);

template <typename T>
__global__ void buildRelativeAttentionBias(T* relative_attention_bias,
                                           const T* relative_attention_bias_table,
                                           const int head_num,
                                           const int seq_len,
                                           const int num_bucket,
                                           const bool is_bidirectional,
                                           const int max_distance) {
  const int head_id = blockIdx.x;
  for (int seq_id = blockDim.x * blockIdx.y + threadIdx.x; seq_id < seq_len * seq_len; seq_id += blockDim.x * gridDim.y) {
    int row_id = seq_id / seq_len;
    int col_id = seq_id % seq_len;

    int relative_position = col_id - row_id;

    int relative_buckets = 0;
    int tmp_num_bucket = num_bucket;

    if (is_bidirectional) {
      tmp_num_bucket /= 2;
      if (relative_position > 0) {
        relative_buckets += tmp_num_bucket;
      } else {
        relative_position *= -1;
      }
    } else {
      if (relative_position > 0) {
        relative_position = 0;
      } else {
        relative_position *= -1;
      }
    }

    int max_exact = tmp_num_bucket / 2;
    bool is_small = relative_position < max_exact;

    int relative_position_if_large =
        max_exact + (int)(logf(relative_position * 1.0f / max_exact) / logf((float)max_distance / max_exact) * (tmp_num_bucket - max_exact));

    relative_position_if_large = min(relative_position_if_large, tmp_num_bucket - 1);

    relative_buckets += is_small ? relative_position : relative_position_if_large;

    relative_attention_bias[head_id * seq_len * seq_len + seq_id] =
        relative_attention_bias_table[head_id * num_bucket + relative_buckets];
  }
}

template <typename T>
Status LaunchRelPosAttnBiasKernel(
    cudaStream_t stream,
    T* output,
    const T* bias_table,
    const int num_heads,
    const int seq_len,
    const int num_bucket,
    const int max_distance,
    const bool is_bidirectional,
    const int max_threads_per_block) {
  const int squared_sq_len = seq_len * seq_len;
  if (squared_sq_len <= max_threads_per_block) {
    dim3 grid(num_heads);
    dim3 block(squared_sq_len);
    buildRelativeAttentionBias<<<grid, block, 0, stream>>>(output,
                                                           bias_table,
                                                           num_heads,
                                                           seq_len,
                                                           num_bucket,
                                                           is_bidirectional,
                                                           max_distance);
    return CUDA_CALL(cudaGetLastError());
  } else if (seq_len >= 128 && seq_len <= 384) {
    dim3 grid(num_heads, seq_len);
    dim3 block(seq_len);
    buildRelativeAttentionBias<<<grid, block, 0, stream>>>(output,
                                                           bias_table,
                                                           num_heads,
                                                           seq_len,
                                                           num_bucket,
                                                           is_bidirectional,
                                                           max_distance);
    return CUDA_CALL(cudaGetLastError());
  } else {
    int blockSize = max_threads_per_block;
    const int grid_y_Size = (squared_sq_len + blockSize - 1) / blockSize;
    dim3 grid(num_heads, grid_y_Size);
    dim3 block(blockSize);
    buildRelativeAttentionBias<<<grid, block, 0, stream>>>(output,
                                                           bias_table,
                                                           num_heads,
                                                           seq_len,
                                                           num_bucket,
                                                           is_bidirectional,
                                                           max_distance);

    return CUDA_CALL(cudaGetLastError());
  }
}

template Status LaunchRelPosAttnBiasKernel<float>(cudaStream_t stream,
                                                  float* output,
                                                  const float* bias_table,
                                                  const int num_heads,
                                                  const int seq_len,
                                                  const int num_bucket,
                                                  const int max_distance,
                                                  const bool is_bidirectional,
                                                  const int max_threads_per_block);

template Status LaunchRelPosAttnBiasKernel<half>(cudaStream_t stream,
                                                 half* output,
                                                 const half* bias_table,
                                                 const int num_heads,
                                                 const int seq_len,
                                                 const int num_bucket,
                                                 const int max_distance,
                                                 const bool is_bidirectional,
                                                 const int max_threads_per_block);

namespace {

template <typename T, size_t size>
struct TypeMapper : public V_vec_m_<T, size> {};

// The following operator overriding is not common so we put it in anonymous namespace
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 530
inline __device__ half2 operator*(const float a, const half2 b) {
  return __hmul2(__float2half2_rn(a), b);
}
#else
inline __device__ half2 operator*(const float a, const half2 b) {
  half2 r;
  r.x = (half)(a * (float)b.x);
  r.y = (half)(a * (float)b.y);
  return r;
}
#endif

inline __device__ Half4 operator*(const float a, const Half4 b) {
  Half4 r;
  r.x = a * b.x;
  r.y = a * b.y;
  return r;
}

inline __device__ float2 operator*(const float a, const float2 b) {
  return make_float2(a * b.x, a * b.y);
}

inline __device__ float4 operator*(const float a, const float4 b) {
  return make_float4(a * b.x, a * b.y, a * b.z, a * b.w);
}

inline __device__ half operator*(const float a, const half b) {
  return (half)(a * float(b));
}
}  // anonymous namespace

template <typename T, typename VEC_T>
__global__ void GatedRelativePositionBiasKernelSmallD(
    VEC_T* output,         // (batch_size, num_heads, seq_len, seq_len)
    const VEC_T* rel_pos,  // (1, num_heads, seq_len, seq_len)
    const T* qw,           // (batch_size, num_heads, seq_len, D)
    const T* bias,         // (D)
    const T* eco_a,        // (1, num_heads, 1, 1)
    const int D,
    const int ldqw,
    const int equiv_seq_len) {
  __shared__ float gate[1];

  const int seq_len = gridDim.x;
  const int num_heads = gridDim.y;
  const int s = blockIdx.x;
  const int n = blockIdx.y;
  const int b = blockIdx.z;

  rel_pos += ((int64_t)n * seq_len + s) * equiv_seq_len;
  output += ((int64_t)b * num_heads * seq_len + (int64_t)n * seq_len + s) * equiv_seq_len;
  qw += ((int64_t)b * num_heads * seq_len + (int64_t)n * seq_len + s) * ldqw;

  float val = 0.0f;
  if (threadIdx.x < D) {
    val = (float)qw[threadIdx.x] + (bias ? (float)bias[threadIdx.x] : 0.0f);
  }

  float u = (threadIdx.x < D / 2) ? val : 0.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    u += __shfl_down_sync(0xffffffff, u, offset);
  }

  float r = (threadIdx.x >= D / 2) ? val : 0.0f;
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    r += __shfl_down_sync(0xffffffff, r, offset);
  }

  if (threadIdx.x == 0) {
    u = 1.0f / (1.0f + expf(-u));
    r = 1.0f / (1.0f + expf(-r));
    gate[0] = u * (r * (float)eco_a[n] - 1.0f) + 2.0f;
  }
  __syncthreads();

  for (int idx = threadIdx.x; idx < equiv_seq_len; idx += blockDim.x) {
    output[idx] = gate[0] * rel_pos[idx];
  }
}

template <typename T>
Status LaunchGatedRelativePositionBiasKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    T* output,
    const T* rel_pos,
    const T* qw,  // query * weight
    const T* bias,
    const T* eco_a,
    int batch_size,
    int num_heads,
    int seq_len,
    int D,
    int ldqw) {
  ORT_ENFORCE(D <= 32 && D > 0 && (D % 2 == 0));
  ORT_ENFORCE(ldqw == seq_len || ldqw == D);

  int equiv_seq_len = (seq_len & 1) == 0 ? (seq_len >> 1) : seq_len;
  equiv_seq_len = (equiv_seq_len & 1) == 0 ? (equiv_seq_len >> 1) : equiv_seq_len;

  int tpb = std::max(32, std::max(D, equiv_seq_len));
  tpb = std::min(tpb, device_prop.maxThreadsPerBlock);

  // round up tpb to power of 2
  --tpb;
  tpb |= (tpb >> 1);
  tpb |= (tpb >> 2);
  tpb |= (tpb >> 4);
  tpb |= (tpb >> 8);
  tpb |= (tpb >> 16);
  tpb++;

  dim3 block(tpb);
  dim3 grid(seq_len, num_heads, batch_size);

  if (seq_len % 4 == 0) {
    using vec_type = typename TypeMapper<T, 4>::Type;
    GatedRelativePositionBiasKernelSmallD<<<grid, block, sizeof(float), stream>>>(
        reinterpret_cast<vec_type*>(output),
        reinterpret_cast<const vec_type*>(rel_pos),
        qw, bias, eco_a, D, ldqw, equiv_seq_len);
  } else if ((seq_len & 1) == 0) {
    using vec_type = typename TypeMapper<T, 2>::Type;
    GatedRelativePositionBiasKernelSmallD<<<grid, block, sizeof(float), stream>>>(
        reinterpret_cast<vec_type*>(output),
        reinterpret_cast<const vec_type*>(rel_pos),
        qw, bias, eco_a, D, ldqw, equiv_seq_len);
  } else {
    GatedRelativePositionBiasKernelSmallD<<<grid, block, sizeof(float), stream>>>(
        output, rel_pos, qw, bias, eco_a, D, ldqw, equiv_seq_len);
  }

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchGatedRelativePositionBiasKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    float* output,
    const float* rel_pos,
    const float* qw,
    const float* bias,
    const float* eco_a,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int D,
    const int ldqw);

template Status LaunchGatedRelativePositionBiasKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    half* output,
    const half* rel_pos,
    const half* qw,
    const half* bias,
    const half* eco_a,
    const int batch_size,
    const int num_heads,
    const int seq_len,
    const int D,
    const int ldqw);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
