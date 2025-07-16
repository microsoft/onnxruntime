// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

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

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  int out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
  if (s < tensor_in_sequence_length) {
    const int past_SH = tensor_in_sequence_length * H;
    const int past_NSH = num_heads * past_SH;
    const int in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
    tensor_out[out_offset] = tensor_in[in_offset];
  } else if (s < all_sequence_length) {
    const int SH = tensor_add_sequence_length * H;
    const int NSH = num_heads * SH;
    const int in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
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

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  while (h < H) {
    int out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
    if (s < tensor_in_sequence_length) {
      const int past_SH = tensor_in_sequence_length * H;
      const int past_NSH = num_heads * past_SH;
      const int in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
      tensor_out[out_offset] = tensor_in[in_offset];
    } else if (s < all_sequence_length) {
      const int SH = tensor_add_sequence_length * H;
      const int NSH = num_heads * SH;
      const int in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
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

#ifndef USE_ROCM  // exclude the following from hipify since they are not used in ROCM EP

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
#endif

// Kernel to append new and past kv in either BSNH or BNSH format
// Adapted from ConcatTensorToTensor kernel in attention_kv_cache.cu file
template <typename T>
__global__ void ConcatNewToPastKV(const int new_seqlen,
                                  const int past_buffer_seqlen,
                                  const T* past_kv,
                                  const T* new_kv,
                                  T* present_kv,
                                  const int* seqlens_k,
                                  const bool past_only,
                                  // const int* seqlens_q,
                                  const bool is_bsnh) {  // refers to past; otherwise bnsh
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int present_buffer_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int present_batch_stride = present_buffer_seqlen * num_heads * H;
  const int row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : present_buffer_seqlen * H;

  // past_kv:     BPNH or BNPH
  // new_kv:      BLNH
  // present_kv:  BTNH or BNTH, where T = P + L

  // prompt, token, and interactive decoding cases
  const int past_seqlen = seqlens_k == nullptr ? 0 : seqlens_k[b] + 1 - new_seqlen;

  int out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
  if (s < past_seqlen) {
    const int past_batch_stride = past_buffer_seqlen * num_heads * H;
    const int past_head_stride = is_bsnh ? H : past_buffer_seqlen * H;
    const int in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
    present_kv[out_offset] = past_kv[in_offset];
  } else if (!past_only && s < past_seqlen + new_seqlen) {
    // Note: new KV always BSNH
    const int new_batch_stride = new_seqlen * num_heads * H;
    const int new_row_stride = num_heads * H;
    const int new_head_stride = H;
    const int in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;
    present_kv[out_offset] = new_kv[in_offset];
  }
}

// Use when (H*)*num_heads > 1024
template <typename T>
__global__ void ConcatNewToPastKVLarge(const int new_seqlen,
                                       const int past_buffer_seqlen,
                                       const int H,
                                       const int num_heads,
                                       const T* past_kv,
                                       const T* new_kv,
                                       T* present_kv,
                                       const int* seqlens_k,
                                       const bool past_only,
                                       const bool is_bsnh) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int present_buffer_seqlen = gridDim.y;

    const int present_batch_stride = present_buffer_seqlen * num_heads * H;
    const int row_stride = is_bsnh ? num_heads * H : H;
    const int present_head_stride = is_bsnh ? H : present_buffer_seqlen * H;

    // past_kv:     BPNH or BNPH
    // new_kv:      BLNH
    // present_kv:  BTNH or BNTH, where T = P + L

    // prompt, token, and interactive decoding cases
    const int past_seqlen = seqlens_k == nullptr ? 0 : seqlens_k[b] + 1 - new_seqlen;

    int out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
    if (s < past_seqlen) {
      const int past_batch_stride = past_buffer_seqlen * num_heads * H;
      const int past_head_stride = is_bsnh ? H : past_buffer_seqlen * H;
      const int in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
      present_kv[out_offset] = past_kv[in_offset];
    } else if (!past_only && s < past_seqlen + new_seqlen) {
      const int new_batch_stride = new_seqlen * num_heads * H;
      const int new_row_stride = num_heads * H;
      const int new_head_stride = H;
      const int in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;
      present_kv[out_offset] = new_kv[in_offset];
    }
  }
}

// Concat new to kv buffer in place
template <typename T>
Status LaunchConcatNewToPastKV(const int batch_size,
                               const int kv_num_heads,
                               const int head_size,
                               const int kv_sequence_length,
                               const int past_sequence_length,
                               const int present_sequence_length,
                               const bool is_bsnh,
                               const int* seqlens_k,
                               const T* past_key,
                               const T* past_value,
                               const T* new_key,
                               const T* new_value,
                               T* present_key,
                               T* present_value,
                               cudaStream_t stream,
                               const int max_threads_per_block,
                               const bool past_only) {
  const int H = head_size / 4;  // divide by 4 so kernel can operate on 4 float16 elements at a time.
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(present_sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatNewToPastKV<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                          past_sequence_length,
                                                          reinterpret_cast<const float2*>(past_key),
                                                          reinterpret_cast<const float2*>(new_key),
                                                          reinterpret_cast<float2*>(present_key),
                                                          seqlens_k,
                                                          past_only,
                                                          is_bsnh);
    ConcatNewToPastKV<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                          past_sequence_length,
                                                          reinterpret_cast<const float2*>(past_value),
                                                          reinterpret_cast<const float2*>(new_value),
                                                          reinterpret_cast<float2*>(present_value),
                                                          seqlens_k,
                                                          past_only,
                                                          is_bsnh);
  } else {
    int steps = (H * kv_num_heads + 255) / 256;
    const dim3 grid(steps, present_sequence_length, batch_size);
    const dim3 block(256, 1, 1);
    ConcatNewToPastKVLarge<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                               past_sequence_length,
                                                               H,
                                                               kv_num_heads,
                                                               reinterpret_cast<const float2*>(past_key),
                                                               reinterpret_cast<const float2*>(new_key),
                                                               reinterpret_cast<float2*>(present_key),
                                                               seqlens_k,
                                                               past_only,
                                                               is_bsnh);
    ConcatNewToPastKVLarge<float2><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                               past_sequence_length,
                                                               H,
                                                               kv_num_heads,
                                                               reinterpret_cast<const float2*>(past_value),
                                                               reinterpret_cast<const float2*>(new_value),
                                                               reinterpret_cast<float2*>(present_value),
                                                               seqlens_k,
                                                               past_only,
                                                               is_bsnh);
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
                                              const int* seqlens_k,
                                              const half* past_key,
                                              const half* past_value,
                                              const half* new_key,
                                              const half* new_value,
                                              half* present_key,
                                              half* present_value,
                                              cudaStream_t stream,
                                              const int max_threads_per_block,
                                              const bool past_only);

template Status LaunchConcatNewToPastKV<BFloat16>(const int batch_size,
                                                  const int kv_num_heads,
                                                  const int head_size,
                                                  const int kv_sequence_length,
                                                  const int past_sequence_length,
                                                  const int present_sequence_length,
                                                  const bool is_bsnh,
                                                  const int* seqlens_k,
                                                  const BFloat16* past_key,
                                                  const BFloat16* past_value,
                                                  const BFloat16* new_key,
                                                  const BFloat16* new_value,
                                                  BFloat16* present_key,
                                                  BFloat16* present_value,
                                                  cudaStream_t stream,
                                                  const int max_threads_per_block,
                                                  const bool past_only);

// Kernel to append new kv to kv buffer in place
template <typename T>
__global__ void ConcatKVInPlace(const int max_seqlen,
                                T* kv_buff,
                                const T* new_kv,
                                const int* seqlens_k,
                                const int* total_seqlens_k,
                                const bool is_past_kv_bnsh_format,
                                const bool is_new_kv_bnsh_format) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int kv_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int past_seq_len = (total_seqlens_k != nullptr)
                               ? (total_seqlens_k[b] - new_seqlen)
                               : (seqlens_k == nullptr ? 0 : (seqlens_k[b] + 1 - new_seqlen));

  int out_offset = is_past_kv_bnsh_format
                       ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                       : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

  int in_offset = is_new_kv_bnsh_format
                      ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                      : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

  kv_buff[out_offset] = new_kv[in_offset];
}

template <typename T>
__global__ void ConcatKVInPlaceLarge(const int max_seqlen,
                                     const int H,
                                     const int kv_num_heads,
                                     T* kv_buff,
                                     const T* new_kv,
                                     const int* seqlens_k,
                                     const int* total_seqlens_k,
                                     const bool is_past_kv_bnsh_format,
                                     const bool is_new_kv_bnsh_format) {  // refers to kv buff; otherwise bnsh
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * kv_num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int new_seqlen = gridDim.y;
    const int past_seq_len = (total_seqlens_k != nullptr)
                                 ? (total_seqlens_k[b] - new_seqlen)
                                 : (seqlens_k == nullptr ? 0 : (seqlens_k[b] + 1 - new_seqlen));

    int out_offset = is_past_kv_bnsh_format
                         ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                         : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

    int in_offset = is_new_kv_bnsh_format
                        ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                        : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

    kv_buff[out_offset] = new_kv[in_offset];
  }
}

// Concat new to kv buffer in place
template <typename T>
Status LaunchConcatKVInPlace(int batch_size,
                             int kv_num_heads,
                             int head_size,
                             int max_sequence_length,
                             const int* seqlens_k,
                             const int* total_seqlens_k,
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
                                                        seqlens_k,
                                                        total_seqlens_k,
                                                        is_past_kv_bnsh_format,
                                                        is_new_kv_bnsh_format);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(present_value),
                                                        reinterpret_cast<const float2*>(new_value),
                                                        seqlens_k,
                                                        total_seqlens_k,
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
                                                             seqlens_k,
                                                             total_seqlens_k,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(present_value),
                                                             reinterpret_cast<const float2*>(new_value),
                                                             seqlens_k,
                                                             total_seqlens_k,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatKVInPlace<half>(int batch_size,
                                            int kv_num_heads,
                                            int head_size,
                                            int max_sequence_length,
                                            const int* seqlens_k,
                                            const int* total_seqlens_k,
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
                                                const int* seqlens_k,
                                                const int* total_seqlens_k,
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
                                             const int* seqlens_k,
                                             const int* total_seqlens_k,
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
