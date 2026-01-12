// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "contrib_ops/cuda/bert/rotary_common.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// ============================================================================
// ConcatTensorToTensor Kernel
// ============================================================================
// PURPOSE:
//   Concatenates past KV cache with new KV tokens to create present KV cache.
//   Used for non-shared buffer mode (separate past and present tensors).
//
// INPUTS:
//   tensor_add_sequence_length - Number of new tokens to append (L)
//   tensor_in  - Past KV cache [K, B, N, P, H] where P is past sequence length
//   tensor_add - New KV tokens [K, B, N, L, H] where L is new sequence length
//
// OUTPUTS:
//   tensor_out - Present KV cache [K, B, N, T, H] where T = P + L
//
// THREAD MAPPING:
//   threadIdx.x = h (head dimension element)
//   threadIdx.y = n (head index)
//   blockIdx.x  = s (sequence position in output)
//   blockIdx.y  = b (batch index)
//   blockIdx.z  = chunk_id (K dimension, typically 2 for K and V)
//
// ASSUMPTIONS:
//   - H * num_heads <= max_threads_per_block (use ConcatTensorToTensorLarge otherwise)
//   - Output format is BNSH
// ============================================================================
template <typename T>
__global__ void ConcatTensorToTensor(const int tensor_add_sequence_length,
                                     const T* tensor_in,
                                     const T* tensor_add,
                                     T* tensor_out) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int chunk_id = blockIdx.z;

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  // K: number of identical tensors
  // tensor_in:    K x BxNxPxH
  // tensor_add:   K x BxNxLxH
  // tensor_out:   K x BxNxTxH, where T = P + L
  const int tensor_in_sequence_length = all_sequence_length - tensor_add_sequence_length;

  const int64_t present_SH = int64_t(all_sequence_length) * H;
  const int64_t present_NSH = num_heads * present_SH;
  int64_t out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
  if (s < tensor_in_sequence_length) {
    const int64_t past_SH = int64_t(tensor_in_sequence_length) * H;
    const int64_t past_NSH = num_heads * past_SH;
    const int64_t in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
    tensor_out[out_offset] = tensor_in[in_offset];
  } else if (s < all_sequence_length) {
    const int64_t SH = int64_t(tensor_add_sequence_length) * H;
    const int64_t NSH = num_heads * SH;
    const int64_t in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
    tensor_out[out_offset] = tensor_add[in_offset];
  }
}

template <typename T>
__global__ void ConcatTensorToTensorLarge(const int tensor_add_sequence_length,
                                          const int H,
                                          const T* tensor_in,
                                          const T* tensor_add,
                                          T* tensor_out) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int chunk_id = blockIdx.z;

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int stride = blockDim.x;

  // K: number of identical tensor
  // tensor_in:    K x BxNxPxH
  // tensor_add:   K x BxNxLxH
  // tensor_out:   K x BxNxTxH
  const int tensor_in_sequence_length = all_sequence_length - tensor_add_sequence_length;

  const int64_t present_SH = int64_t(all_sequence_length) * H;
  const int64_t present_NSH = num_heads * present_SH;
  while (h < H) {
    int64_t out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
    if (s < tensor_in_sequence_length) {
      const int64_t past_SH = int64_t(tensor_in_sequence_length) * H;
      const int64_t past_NSH = num_heads * past_SH;
      const int64_t in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
      tensor_out[out_offset] = tensor_in[in_offset];
    } else if (s < all_sequence_length) {
      const int64_t SH = int64_t(tensor_add_sequence_length) * H;
      const int64_t NSH = num_heads * SH;
      const int64_t in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
      tensor_out[out_offset] = tensor_add[in_offset];
    }

    h += stride;
  }
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const float* tensor_in,
                                  const float* tensor_add,
                                  float* tensor_out) {
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                               reinterpret_cast<const float2*>(tensor_in),
                                                               reinterpret_cast<const float2*>(tensor_add),
                                                               reinterpret_cast<float2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(tensor_in),
                                                                    reinterpret_cast<const float2*>(tensor_add),
                                                                    reinterpret_cast<float2*>(tensor_out));
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<float><<<grid, block, 0, stream>>>(sequence_length, tensor_in, tensor_add, tensor_out);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float><<<grid, block, 0, stream>>>(sequence_length,
                                                                   head_size,
                                                                   tensor_in,
                                                                   tensor_add,
                                                                   tensor_out);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const half* tensor_in,
                                  const half* tensor_add,
                                  half* tensor_out) {
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                               reinterpret_cast<const float2*>(tensor_in),
                                                               reinterpret_cast<const float2*>(tensor_add),
                                                               reinterpret_cast<float2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(tensor_in),
                                                                    reinterpret_cast<const float2*>(tensor_add),
                                                                    reinterpret_cast<float2*>(tensor_out));
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<half2><<<grid, block, 0, stream>>>(sequence_length,
                                                              reinterpret_cast<const half2*>(tensor_in),
                                                              reinterpret_cast<const half2*>(tensor_add),
                                                              reinterpret_cast<half2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<half2><<<grid, block, 0, stream>>>(sequence_length,
                                                                   H,
                                                                   reinterpret_cast<const half2*>(tensor_in),
                                                                   reinterpret_cast<const half2*>(tensor_add),
                                                                   reinterpret_cast<half2*>(tensor_out));
    }
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<half><<<grid, block, 0, stream>>>(sequence_length, tensor_in, tensor_add, tensor_out);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<half><<<grid, block, 0, stream>>>(sequence_length,
                                                                  head_size,
                                                                  tensor_in,
                                                                  tensor_add,
                                                                  tensor_out);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const BFloat16* tensor_in,
                                  const BFloat16* tensor_add,
                                  BFloat16* tensor_out) {
  assert(num_heads <= max_threads_per_block);
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size % 8)) {
    const int H = head_size / 8;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float4><<<grid, block, 0, stream>>>(
          sequence_length,
          reinterpret_cast<const float4*>(tensor_in),
          reinterpret_cast<const float4*>(tensor_add),
          reinterpret_cast<float4*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float4><<<grid, block, 0, stream>>>(
          sequence_length,
          H,
          reinterpret_cast<const float4*>(tensor_in),
          reinterpret_cast<const float4*>(tensor_add),
          reinterpret_cast<float4*>(tensor_out));
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<__nv_bfloat162><<<grid, block, 0, stream>>>(
          sequence_length,
          reinterpret_cast<const __nv_bfloat162*>(tensor_in),
          reinterpret_cast<const __nv_bfloat162*>(tensor_add),
          reinterpret_cast<__nv_bfloat162*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<__nv_bfloat162><<<grid, block, 0, stream>>>(
          sequence_length,
          H,
          reinterpret_cast<const __nv_bfloat162*>(tensor_in),
          reinterpret_cast<const __nv_bfloat162*>(tensor_add),
          reinterpret_cast<__nv_bfloat162*>(tensor_out));
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<__nv_bfloat16><<<grid, block, 0, stream>>>(
          sequence_length,
          reinterpret_cast<const __nv_bfloat16*>(tensor_in),
          reinterpret_cast<const __nv_bfloat16*>(tensor_add),
          reinterpret_cast<__nv_bfloat16*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<__nv_bfloat16><<<grid, block, 0, stream>>>(
          sequence_length,
          head_size,
          reinterpret_cast<const __nv_bfloat16*>(tensor_in),
          reinterpret_cast<const __nv_bfloat16*>(tensor_add),
          reinterpret_cast<__nv_bfloat16*>(tensor_out));
    }
  }

  return CUDA_CALL(cudaGetLastError());
}

// ----------------------------------------------------------------------------------
// Below kernels are for past and present sharing buffer
// ----------------------------------------------------------------------------------

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

  constexpr int M = static_cast<int>(QKV::COUNT);  // Matrix count in qkv
  const int m = blockIdx.z + 1;                    // k = 1, v = 2

  const int64_t NH = N * head_size;
  const int64_t NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int64_t MsH = int64_t(max_sequence_length) * head_size;
  const int64_t NMsH = N * MsH;
  const int64_t BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

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

  constexpr int M = static_cast<int>(QKV::COUNT);  // Matrix count in qkv
  const int m = (blockIdx.z & 0x1) + 1;            // k = 1, v = 2

  const int64_t NH = N * head_size;
  const int64_t NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int64_t MsH = int64_t(max_sequence_length) * head_size;
  const int64_t NMsH = N * MsH;
  const int64_t BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

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

template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const BFloat16* bias,
                                                    const BFloat16* qkv_buffer,
                                                    BFloat16* present);

// Fused kernel to append new K and V to past in either BSNH or BNSH format.
// Adapted from ConcatTensorToTensor kernel.
//
// Grid.z encodes K vs V: blockIdx.z == 0 -> K (with optional RoPE), blockIdx.z == 1 -> V (no RoPE)
//
// Input Format Requirements:
//   - new_key/new_value: Must be contiguous BSNH format [batch, seq, kv_heads, head_size]
//   - past_key/past_value: Either BSNH or BNSH based on is_bsnh flag
//   - present_key/present_value: Same format as past (BSNH or BNSH)
//
// RoPE Requirements (when cos_cache != nullptr && rotary_dim > 0):
//   - new_key must be contiguous BSNH so RotaryDispatcher can read pair values
//   - The pair element for non-interleaved RoPE is read from new_key at offset (in_offset - h + pair_idx)
//   - cos_cache/sin_cache: [max_position, rotary_dim/2] contiguous

template <typename T, typename ElementT>
__global__ void ConcatNewToPastKVFused(const int new_seqlen,
                                       const int past_buffer_seqlen,
                                       const T* past_key,
                                       const T* past_value,
                                       const T* new_key,
                                       const T* new_value,
                                       T* present_key,
                                       T* present_value,
                                       const int* past_seq_lens,
                                       const int* total_seq_lens,
                                       const bool past_only,
                                       const bool is_bsnh,
                                       const T* cos_cache,
                                       const T* sin_cache,
                                       const int rotary_dim,
                                       const int64_t* position_ids,
                                       const bool interleaved) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int kind = blockIdx.z;  // 0 for K, 1 for V

  const int present_buffer_seqlen = gridDim.x;  // gridDim.x is present_sequence_length
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int64_t present_batch_stride = int64_t(present_buffer_seqlen) * num_heads * H;
  const int64_t row_stride = is_bsnh ? num_heads * H : H;
  const int64_t present_head_stride = is_bsnh ? H : int64_t(present_buffer_seqlen) * H;

  // Determine pointers based on kind
  const T* past_ptr = (kind == 0) ? past_key : past_value;
  const T* new_ptr = (kind == 0) ? new_key : new_value;
  T* present_ptr = (kind == 0) ? present_key : present_value;

  const int past_seqlen = past_seq_lens[b];

  int64_t out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;

  if (s < past_seqlen) {
    const int64_t past_batch_stride = int64_t(past_buffer_seqlen) * num_heads * H;
    const int64_t past_head_stride = is_bsnh ? H : int64_t(past_buffer_seqlen) * H;
    const int64_t in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
    present_ptr[out_offset] = past_ptr[in_offset];
  } else if (!past_only && s < past_seqlen + new_seqlen) {
    const int64_t new_batch_stride = int64_t(new_seqlen) * num_heads * H;
    const int64_t new_row_stride = num_heads * H;
    const int64_t new_head_stride = H;
    const int64_t in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;

    T val = new_ptr[in_offset];

    // Apply RoPE only for K (kind == 0)
    if (kind == 0 && cos_cache != nullptr && rotary_dim > 0) {
      int pos_id = 0;
      if (position_ids) {
        int new_s_idx = s - past_seqlen;
        if (new_s_idx >= 0 && new_s_idx < new_seqlen) {
          pos_id = static_cast<int>(position_ids[b * new_seqlen + new_s_idx]);
        } else {
          pos_id = s;
        }
      } else {
        pos_id = s;
      }

      // Check bounds for pos_id to be safe?
      // RoPE cache size usually matches max_seq_len.

      RotaryDispatcher<T, ElementT>::apply(val, cos_cache, sin_cache, rotary_dim, h, pos_id, interleaved, new_key, in_offset - h);
    }
    present_ptr[out_offset] = val;
  }
}

template <typename T, typename ElementT>
__global__ void ConcatNewToPastKVFusedLarge(const int new_seqlen,
                                            const int past_buffer_seqlen,
                                            const int H,
                                            const int num_heads,
                                            const T* past_key,
                                            const T* past_value,
                                            const T* new_key,
                                            const T* new_value,
                                            T* present_key,
                                            T* present_value,
                                            const int* past_seq_lens,
                                            const int* total_seq_lens,
                                            const bool past_only,
                                            const bool is_bsnh,
                                            const T* cos_cache,
                                            const T* sin_cache,
                                            const int rotary_dim,
                                            const int64_t* position_ids,
                                            const bool interleaved) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);

  if (i < H * num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z / 2;     // Integer div
    const int kind = blockIdx.z % 2;  // 0 for K, 1 for V

    const int present_buffer_seqlen = gridDim.y;
    // gridDim.z is batch_size * 2

    const int64_t present_batch_stride = int64_t(present_buffer_seqlen) * num_heads * H;
    const int64_t row_stride = is_bsnh ? num_heads * H : H;
    const int64_t present_head_stride = is_bsnh ? H : int64_t(present_buffer_seqlen) * H;

    const T* past_ptr = (kind == 0) ? past_key : past_value;
    const T* new_ptr = (kind == 0) ? new_key : new_value;
    T* present_ptr = (kind == 0) ? present_key : present_value;

    const int past_seqlen = past_seq_lens[b];

    const int64_t out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;

    if (s < past_seqlen) {
      const int64_t past_batch_stride = int64_t(past_buffer_seqlen) * num_heads * H;
      const int64_t past_head_stride = is_bsnh ? H : int64_t(past_buffer_seqlen) * H;
      const int64_t in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
      present_ptr[out_offset] = past_ptr[in_offset];
    } else if (!past_only && s < past_seqlen + new_seqlen) {
      const int64_t new_batch_stride = int64_t(new_seqlen) * num_heads * H;
      const int64_t new_row_stride = num_heads * H;
      const int64_t new_head_stride = H;
      const int64_t in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;

      T val = new_ptr[in_offset];

      if (kind == 0 && cos_cache != nullptr && rotary_dim > 0) {
        int pos_id = s;
        int new_s_idx = s - past_seqlen;
        if (position_ids && new_s_idx >= 0 && new_s_idx < new_seqlen) {
          pos_id = static_cast<int>(position_ids[b * new_seqlen + new_s_idx]);
        }

        RotaryDispatcher<T, ElementT>::apply(val, cos_cache, sin_cache, rotary_dim, h, pos_id, interleaved, new_key, in_offset - h);
      }
      present_ptr[out_offset] = val;
    }
  }
}

template <typename T>
Status LaunchConcatNewToPastKV(const int batch_size,
                               const int kv_num_heads,
                               const int head_size,
                               const int kv_sequence_length,
                               const int past_sequence_length,
                               const int present_sequence_length,
                               const bool is_bsnh,
                               const int* past_seq_lens,
                               const int* total_seq_lens,
                               const T* past_key,
                               const T* past_value,
                               const T* new_key,
                               const T* new_value,
                               T* present_key,
                               T* present_value,
                               cudaStream_t stream,
                               const int max_threads_per_block,
                               const bool past_only,
                               const T* cos_cache,
                               const T* sin_cache,
                               const int rotary_dim,
                               const int64_t* position_ids,
                               const bool interleaved) {
  constexpr int num_elements_per_thread = std::max(1, 8 / int(sizeof(T)));
  const int H = head_size / num_elements_per_thread;

  if (H * kv_num_heads <= max_threads_per_block) {
    // Grid Z dim is 2: 0 for K, 1 for V
    const dim3 grid(present_sequence_length, batch_size, 2);
    const dim3 block(H, kv_num_heads, 1);

    ConcatNewToPastKVFused<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                                  past_sequence_length,
                                                                  reinterpret_cast<const float2*>(past_key),
                                                                  reinterpret_cast<const float2*>(past_value),
                                                                  reinterpret_cast<const float2*>(new_key),
                                                                  reinterpret_cast<const float2*>(new_value),
                                                                  reinterpret_cast<float2*>(present_key),
                                                                  reinterpret_cast<float2*>(present_value),
                                                                  past_seq_lens,
                                                                  total_seq_lens,
                                                                  past_only,
                                                                  is_bsnh,
                                                                  reinterpret_cast<const float2*>(cos_cache),
                                                                  reinterpret_cast<const float2*>(sin_cache),
                                                                  rotary_dim, position_ids, interleaved);
  } else {
    // Large kernel version
    int steps = (H * kv_num_heads + 255) / 256;
    // Grid Z dim is batch_size * 2
    // We encode b and kind in blockIdx.z in the kernel
    const dim3 grid(steps, present_sequence_length, batch_size * 2);
    const dim3 block(256, 1, 1);

    ConcatNewToPastKVFusedLarge<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                                       past_sequence_length,
                                                                       H,
                                                                       kv_num_heads,
                                                                       reinterpret_cast<const float2*>(past_key),
                                                                       reinterpret_cast<const float2*>(past_value),
                                                                       reinterpret_cast<const float2*>(new_key),
                                                                       reinterpret_cast<const float2*>(new_value),
                                                                       reinterpret_cast<float2*>(present_key),
                                                                       reinterpret_cast<float2*>(present_value),
                                                                       past_seq_lens,
                                                                       total_seq_lens,
                                                                       past_only,
                                                                       is_bsnh,
                                                                       reinterpret_cast<const float2*>(cos_cache),
                                                                       reinterpret_cast<const float2*>(sin_cache),
                                                                       rotary_dim, position_ids, interleaved);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatNewToPastKV<half>(const int batch_size,
                                              const int kv_num_heads,
                                              const int head_size,
                                              const int kv_sequence_length,
                                              const int past_sequence_length,
                                              const int present_sequence_length,
                                              const bool is_bsnh,
                                              const int* past_seq_lens,
                                              const int* total_seq_lens,
                                              const half* past_key,
                                              const half* past_value,
                                              const half* new_key,
                                              const half* new_value,
                                              half* present_key,
                                              half* present_value,
                                              cudaStream_t stream,
                                              const int max_threads_per_block,
                                              const bool past_only,
                                              const half* cos_cache,
                                              const half* sin_cache,
                                              const int rotary_dim,
                                              const int64_t* position_ids,
                                              const bool interleaved);

template Status LaunchConcatNewToPastKV<BFloat16>(const int batch_size,
                                                  const int kv_num_heads,
                                                  const int head_size,
                                                  const int kv_sequence_length,
                                                  const int past_sequence_length,
                                                  const int present_sequence_length,
                                                  const bool is_bsnh,
                                                  const int* past_seq_lens,
                                                  const int* total_seq_lens,
                                                  const BFloat16* past_key,
                                                  const BFloat16* past_value,
                                                  const BFloat16* new_key,
                                                  const BFloat16* new_value,
                                                  BFloat16* present_key,
                                                  BFloat16* present_value,
                                                  cudaStream_t stream,
                                                  const int max_threads_per_block,
                                                  const bool past_only,
                                                  const BFloat16* cos_cache,
                                                  const BFloat16* sin_cache,
                                                  const int rotary_dim,
                                                  const int64_t* position_ids,
                                                  const bool interleaved);

template Status LaunchConcatNewToPastKV<float>(const int batch_size,
                                               const int kv_num_heads,
                                               const int head_size,
                                               const int kv_sequence_length,
                                               const int past_sequence_length,
                                               const int present_sequence_length,
                                               const bool is_bsnh,
                                               const int* past_seq_lens,
                                               const int* total_seq_lens,
                                               const float* past_key,
                                               const float* past_value,
                                               const float* new_key,
                                               const float* new_value,
                                               float* present_key,
                                               float* present_value,
                                               cudaStream_t stream,
                                               const int max_threads_per_block,
                                               const bool past_only,
                                               const float* cos_cache,
                                               const float* sin_cache,
                                               const int rotary_dim,
                                               const int64_t* position_ids,
                                               const bool interleaved);

// ============================================================================
// ConcatKVInPlace Kernel
// ============================================================================
// PURPOSE:
//   Appends new KV tokens to existing KV cache buffer IN-PLACE.
//   Used when past and present KV share the same memory (kv_share_buffer=true).
//
// INPUTS:
//   max_seqlen           - Maximum sequence length (buffer size)
//   new_kv               - New KV tokens to append
//   past_seq_lens        - Per-batch offset where to write (can be null)
//   total_seq_lens       - Per-batch total valid tokens after appending
//   is_past_kv_bnsh_format - True if KV buffer is BNSH, false for BSNH
//   is_new_kv_bnsh_format  - True if new_kv is BNSH, false for BSNH
//
// OUTPUTS:
//   kv_buff - Updated KV cache with new tokens appended at past_seq_len offset
//
// THREAD MAPPING:
//   threadIdx.x = h (head dimension element)
//   threadIdx.y = n (head index)
//   blockIdx.x  = s (new token sequence position, 0 to new_seqlen-1)
//   blockIdx.y  = b (batch index)
//
// BOUNDS CHECK:
//   Only writes when (s + past_seq_len < total_seq_lens[b]) to prevent
//   out-of-bounds access with variable-length sequences.
//
// ASSUMPTIONS:
//   - H * kv_num_heads <= max_threads_per_block (use ConcatKVInPlaceLarge otherwise)
// ============================================================================
template <typename T>
__global__ void ConcatKVInPlace(const int max_seqlen,
                                T* kv_buff,
                                const T* new_kv,
                                const int* past_seq_lens,
                                const int* total_seq_lens,
                                const bool is_past_kv_bnsh_format,
                                const bool is_new_kv_bnsh_format) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int kv_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

  int64_t out_offset = is_past_kv_bnsh_format
                           ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                           : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

  int64_t in_offset = is_new_kv_bnsh_format
                          ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                          : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

  if (s + past_seq_len < total_seq_lens[b]) {
    kv_buff[out_offset] = new_kv[in_offset];
  }
}

template <typename T>
__global__ void ConcatKVInPlaceLarge(const int max_seqlen,
                                     const int H,
                                     const int kv_num_heads,
                                     T* kv_buff,
                                     const T* new_kv,
                                     const int* past_seq_lens,
                                     const int* total_seq_lens,
                                     const bool is_past_kv_bnsh_format,
                                     const bool is_new_kv_bnsh_format) {  // refers to kv buff; otherwise bnsh
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * kv_num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int new_seqlen = gridDim.y;
    const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

    int64_t out_offset = is_past_kv_bnsh_format
                             ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                             : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

    int64_t in_offset = is_new_kv_bnsh_format
                            ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                            : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

    if (s + past_seq_len < total_seq_lens[b]) {
      kv_buff[out_offset] = new_kv[in_offset];
    }
  }
}

template <typename T>
Status LaunchConcatKVInPlace(int batch_size,
                             int kv_num_heads,
                             int head_size,
                             int max_sequence_length,
                             const int* past_seq_lens,
                             const int* total_seq_lens,
                             int new_seq_len,
                             const T* new_key,
                             const T* new_value,
                             T* present_key,
                             T* present_value,
                             const bool is_past_kv_bnsh_format,
                             const bool is_new_kv_bnsh_format,
                             cudaStream_t stream,
                             const int max_threads_per_block) {
  // static_assert(sizeof(T) == 2);
  assert(head_size % 4 == 0);

  const int H = head_size / 4;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(new_seq_len, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(present_key),
                                                        reinterpret_cast<const float2*>(new_key),
                                                        past_seq_lens,
                                                        total_seq_lens,
                                                        is_past_kv_bnsh_format,
                                                        is_new_kv_bnsh_format);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(present_value),
                                                        reinterpret_cast<const float2*>(new_value),
                                                        past_seq_lens,
                                                        total_seq_lens,
                                                        is_past_kv_bnsh_format,
                                                        is_new_kv_bnsh_format);
  } else {
    int steps = int(ceil(float(H * kv_num_heads) / 256.0));
    const dim3 grid(steps, new_seq_len, batch_size);
    const dim3 block(256, 1, 1);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(present_key),
                                                             reinterpret_cast<const float2*>(new_key),
                                                             past_seq_lens,
                                                             total_seq_lens,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(present_value),
                                                             reinterpret_cast<const float2*>(new_value),
                                                             past_seq_lens,
                                                             total_seq_lens,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatKVInPlace<half>(int batch_size,
                                            int kv_num_heads,
                                            int head_size,
                                            int max_sequence_length,
                                            const int* past_seq_lens,
                                            const int* total_seq_lens,
                                            int new_seq_len,
                                            const half* new_key,
                                            const half* new_value,
                                            half* present_key,
                                            half* present_value,
                                            bool is_past_kv_bnsh_format,
                                            bool is_new_kv_bnsh_format,
                                            cudaStream_t stream,
                                            const int max_threads_per_block);

template Status LaunchConcatKVInPlace<BFloat16>(int batch_size,
                                                int kv_num_heads,
                                                int head_size,
                                                int max_sequence_length,
                                                const int* past_seq_lens,
                                                const int* total_seq_lens,
                                                int new_seq_len,
                                                const BFloat16* new_key,
                                                const BFloat16* new_value,
                                                BFloat16* present_key,
                                                BFloat16* present_value,
                                                bool is_past_kv_bnsh_format,
                                                bool is_new_kv_bnsh_format,
                                                cudaStream_t stream,
                                                const int max_threads_per_block);

template Status LaunchConcatKVInPlace<float>(int batch_size,
                                             int kv_num_heads,
                                             int head_size,
                                             int max_sequence_length,
                                             const int* past_seq_lens,
                                             const int* total_seq_lens,
                                             int new_seq_len,
                                             const float* new_key,
                                             const float* new_value,
                                             float* present_key,
                                             float* present_value,
                                             bool is_past_kv_bnsh_format,
                                             bool is_new_kv_bnsh_format,
                                             cudaStream_t stream,
                                             const int max_threads_per_block);

// ============================================================================
// ConcatKVInPlaceFused Kernel
// ============================================================================
// PURPOSE:
//   Fused kernel that appends BOTH K and V in a single kernel launch.
//   Eliminates one kernel launch compared to calling ConcatKVInPlace twice.
//
// INPUTS:
//   max_seqlen, new_seqlen - Buffer and new sequence dimensions
//   new_k, new_v           - New K and V tokens to append (must be pre-rotated if RoPE is needed)
//   past_seq_lens          - Per-batch write offset (can be null)
//   total_seq_lens         - Per-batch total valid tokens
//   is_past_kv_bnsh_format - True if KV buffer is BNSH, false for BSNH
//   is_new_kv_bnsh_format  - True if new K/V is BNSH, false for BSNH
//
// OUTPUTS:
//   k_buff - Updated K cache
//   v_buff - Updated V cache
//
// NOTE:
//   RoPE should be applied BEFORE calling this kernel.
//   For fused RoPE+append, use ConcatNewToPastKVFused instead.
// ============================================================================
template <typename T>
__global__ void ConcatKVInPlaceFused(const int max_seqlen,
                                     const int new_seqlen,
                                     T* k_buff,
                                     T* v_buff,
                                     const T* new_k,
                                     const T* new_v,
                                     const int* past_seq_lens,
                                     const int* total_seq_lens,
                                     const bool is_past_kv_bnsh_format,
                                     const bool is_new_kv_bnsh_format) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int kv_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

  // Early exit to prevent out-of-bounds access and redundant writes
  if (s + past_seq_len >= total_seq_lens[b]) {
    return;
  }

  // Use int64_t for offsets to prevent overflow
  int64_t out_offset = is_past_kv_bnsh_format
                           ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                           : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

  int64_t in_offset = is_new_kv_bnsh_format
                          ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                          : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

  // Simple copy for K and V
  k_buff[out_offset] = new_k[in_offset];
  v_buff[out_offset] = new_v[in_offset];
}

// Large version for when H * kv_num_heads > max_threads_per_block
template <typename T>
__global__ void ConcatKVInPlaceFusedLarge(const int max_seqlen,
                                          const int new_seqlen,
                                          const int H,
                                          const int kv_num_heads,
                                          T* k_buff,
                                          T* v_buff,
                                          const T* new_k,
                                          const T* new_v,
                                          const int* past_seq_lens,
                                          const int* total_seq_lens,
                                          const bool is_past_kv_bnsh_format,
                                          const bool is_new_kv_bnsh_format) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * kv_num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;

    const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

    if (s + past_seq_len >= total_seq_lens[b]) {
      return;
    }

    int64_t out_offset = is_past_kv_bnsh_format
                             ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                             : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

    int64_t in_offset = is_new_kv_bnsh_format
                            ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                            : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

    k_buff[out_offset] = new_k[in_offset];
    v_buff[out_offset] = new_v[in_offset];
  }
}

// Launcher for fused K+V append
template <typename T>
Status LaunchConcatKVInPlaceFused(int batch_size,
                                  int kv_num_heads,
                                  int head_size,
                                  int max_sequence_length,
                                  const int* past_seq_lens,
                                  const int* total_seq_lens,
                                  int new_seq_len,
                                  const T* new_key,
                                  const T* new_value,
                                  T* present_key,
                                  T* present_value,
                                  bool is_past_kv_bnsh_format,
                                  bool is_new_kv_bnsh_format,
                                  cudaStream_t stream,
                                  const int max_threads_per_block) {
  // Determine vectorization factor (float2 is 8 bytes)
  constexpr int vector_bytes = sizeof(float2);
  constexpr int element_bytes = sizeof(T);
  constexpr int elements_per_vector = vector_bytes / element_bytes;

  if (head_size % elements_per_vector != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by ", elements_per_vector, " for vectorized kernel.");
  }

  const int H = head_size / elements_per_vector;

  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(new_seq_len, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);

    // Single kernel for both K and V
    ConcatKVInPlaceFused<float2><<<grid, block, 0, stream>>>(
        max_sequence_length,
        new_seq_len,
        reinterpret_cast<float2*>(present_key),
        reinterpret_cast<float2*>(present_value),
        reinterpret_cast<const float2*>(new_key),
        reinterpret_cast<const float2*>(new_value),
        past_seq_lens,
        total_seq_lens,
        is_past_kv_bnsh_format,
        is_new_kv_bnsh_format);
  } else {
    int steps = int(ceil(float(H * kv_num_heads) / 256.0));
    const dim3 grid(steps, new_seq_len, batch_size);
    const dim3 block(256, 1, 1);

    ConcatKVInPlaceFusedLarge<float2><<<grid, block, 0, stream>>>(
        max_sequence_length,
        new_seq_len,
        H,
        kv_num_heads,
        reinterpret_cast<float2*>(present_key),
        reinterpret_cast<float2*>(present_value),
        reinterpret_cast<const float2*>(new_key),
        reinterpret_cast<const float2*>(new_value),
        past_seq_lens,
        total_seq_lens,
        is_past_kv_bnsh_format,
        is_new_kv_bnsh_format);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatKVInPlaceFused<half>(int batch_size,
                                                 int kv_num_heads,
                                                 int head_size,
                                                 int max_sequence_length,
                                                 const int* past_seq_lens,
                                                 const int* total_seq_lens,
                                                 int new_seq_len,
                                                 const half* new_key,
                                                 const half* new_value,
                                                 half* present_key,
                                                 half* present_value,
                                                 bool is_past_kv_bnsh_format,
                                                 bool is_new_kv_bnsh_format,
                                                 cudaStream_t stream,
                                                 const int max_threads_per_block);

template Status LaunchConcatKVInPlaceFused<BFloat16>(int batch_size,
                                                     int kv_num_heads,
                                                     int head_size,
                                                     int max_sequence_length,
                                                     const int* past_seq_lens,
                                                     const int* total_seq_lens,
                                                     int new_seq_len,
                                                     const BFloat16* new_key,
                                                     const BFloat16* new_value,
                                                     BFloat16* present_key,
                                                     BFloat16* present_value,
                                                     bool is_past_kv_bnsh_format,
                                                     bool is_new_kv_bnsh_format,
                                                     cudaStream_t stream,
                                                     const int max_threads_per_block);

template Status LaunchConcatKVInPlaceFused<float>(int batch_size,
                                                  int kv_num_heads,
                                                  int head_size,
                                                  int max_sequence_length,
                                                  const int* past_seq_lens,
                                                  const int* total_seq_lens,
                                                  int new_seq_len,
                                                  const float* new_key,
                                                  const float* new_value,
                                                  float* present_key,
                                                  float* present_value,
                                                  bool is_past_kv_bnsh_format,
                                                  bool is_new_kv_bnsh_format,
                                                  cudaStream_t stream,
                                                  const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
