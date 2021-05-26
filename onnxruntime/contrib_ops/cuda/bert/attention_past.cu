// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void ConcatPastToPresent(const int sequence_length,
                                    const T* past,
                                    const T* k_v,
                                    T* present) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int is_v = blockIdx.z;  // 0 for k, 1 for v

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  // past:    2 x BxNxS'xH   (past_k and past_v)
  // k_v:     2 x BxNxSxH    (k and v)
  // present: 2 x BxNxS*xH   (present_k and present_v)
  const int past_sequence_length = all_sequence_length - sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  int out_offset = b * present_NSH + n * present_SH + s * H + h + is_v * (present_NSH * batch_size);
  if (s < past_sequence_length) {
    const int past_SH = past_sequence_length * H;
    const int past_NSH = num_heads * past_SH;
    const int in_offset = b * past_NSH + n * past_SH + s * H + h + is_v * (past_NSH * batch_size);
    present[out_offset] = past[in_offset];
  } else if (s < all_sequence_length) {
    const int SH = sequence_length * H;
    const int NSH = num_heads * SH;
    const int in_offset = b * NSH + n * SH + (s - past_sequence_length) * H + h + is_v * (NSH * batch_size);
    present[out_offset] = k_v[in_offset];
  }
}

template <typename T>
__global__ void ConcatPastToPresentLarge(const int sequence_length,
                                         const int H,
                                         const T* past,
                                         const T* k_v,
                                         T* present) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int is_v = blockIdx.z;  // 0 for k, 1 for v

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int stride = blockDim.x;

  // past:    2 x BxNxS'xH   (past_k and past_v)
  // k_v:     2 x BxNxSxH    (k and v)
  // present: 2 x BxNxS*xH   (present_k and present_v)
  const int past_sequence_length = all_sequence_length - sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  while (h < H) {
    int out_offset = b * present_NSH + n * present_SH + s * H + h + is_v * (present_NSH * batch_size);
    if (s < past_sequence_length) {
      const int past_SH = past_sequence_length * H;
      const int past_NSH = num_heads * past_SH;
      const int in_offset = b * past_NSH + n * past_SH + s * H + h + is_v * (past_NSH * batch_size);
      present[out_offset] = past[in_offset];
    } else if (s < all_sequence_length) {
      const int SH = sequence_length * H;
      const int NSH = num_heads * SH;
      const int in_offset = b * NSH + n * SH + (s - past_sequence_length) * H + h + is_v * (NSH * batch_size);
      present[out_offset] = k_v[in_offset];
    }

    h += stride;
  }
}

bool LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const int max_threads_per_block,
                               const float* past,
                               const float* k_v,
                               float* present) {
  const dim3 grid(all_sequence_length, batch_size, 2);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatPastToPresent<float2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatPastToPresentLarge<float2><<<grid, block, 0, stream>>>(sequence_length, H, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatPastToPresent<float><<<grid, block, 0, stream>>>(sequence_length, past, k_v, present);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatPastToPresentLarge<float><<<grid, block, 0, stream>>>(sequence_length, head_size, past, k_v, present);
    }

  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const int max_threads_per_block,
                               const half* past,
                               const half* k_v,
                               half* present) {
  const dim3 grid(all_sequence_length, batch_size, 2);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatPastToPresent<float2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatPastToPresentLarge<float2><<<grid, block, 0, stream>>>(sequence_length, H, reinterpret_cast<const float2*>(past), reinterpret_cast<const float2*>(k_v), reinterpret_cast<float2*>(present));
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatPastToPresent<half2><<<grid, block, 0, stream>>>(sequence_length, reinterpret_cast<const half2*>(past), reinterpret_cast<const half2*>(k_v), reinterpret_cast<half2*>(present));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatPastToPresentLarge<half2><<<grid, block, 0, stream>>>(sequence_length, H, reinterpret_cast<const half2*>(past), reinterpret_cast<const half2*>(k_v), reinterpret_cast<half2*>(present));
    }
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatPastToPresent<half><<<grid, block, 0, stream>>>(sequence_length, past, k_v, present);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatPastToPresentLarge<half><<<grid, block, 0, stream>>>(sequence_length, head_size, past, k_v, present);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
