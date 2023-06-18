// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/bert/attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void StridedCopy(const T* in, const int H, longlong4 in_strides,  // coord (b,n,s,h)
                            T* out, longlong4 out_strides                    // coord (b,n,s,h)
) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  if (h < H) {
    const int in_offset = b * in_strides.x + n * in_strides.y + s * in_strides.z + h * in_strides.w;
    const int out_offset = b * out_strides.x + n * out_strides.y + s * out_strides.z + h * out_strides.w;
    out[out_offset] = in[in_offset];
  }
}

template <typename T>
__global__ void StridedCopyLarge(const T* in, const int H, longlong4 in_strides,  // coord (b,n,s,h)
                                 T* out, longlong4 out_strides                    // coord (b,n,s,h)
) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int h_step = blockDim.x;

  while (h < H) {
    const int in_offset = b * in_strides.x + n * in_strides.y + s * in_strides.z + h * in_strides.w;
    const int out_offset = b * out_strides.x + n * out_strides.y + s * out_strides.z + h * out_strides.w;
    out[out_offset] = in[in_offset];
    h += h_step;
  }
}

template <int NumBytes>
struct ToByteType;

template <>
struct ToByteType<2> {
  using T = int16_t;
};

template <>
struct ToByteType<4> {
  using T = int32_t;
};

template <>
struct ToByteType<8> {
  using T = int64_t;
};

template <>
struct ToByteType<16> {
  using T = uint4;
};

template <>
struct ToByteType<32> {
  using T = ulonglong4;
};

template <int NumBytes>
using ToBytes = typename ToByteType<NumBytes>::T;

template <typename T>
Status LaunchStridedCopy(cudaStream_t stream,
                         const T* in, int4 in_shape, longlong4 in_strides,  // coord (b,n,s,h)
                         T* out, longlong4 out_strides,                     // coord (b,n,s,h)
                         int max_threads_per_block) {
  int batch_size = in_shape.x;
  int num_heads = in_shape.y;
  int sequence_length = in_shape.z;
  int head_size = in_shape.w;
  if (sequence_length == 0) {
    return Status::OK();
  }

  const dim3 grid(sequence_length, batch_size);
  if (0 == (head_size % 4)) {  // pack 4 element together
    using Bytes = ToBytes<sizeof(T) * 4>;
    const int H = head_size / 4;
    in_strides.x /= 4;
    in_strides.y /= 4;
    in_strides.z /= 4;
    out_strides.x /= 4;
    out_strides.y /= 4;
    out_strides.z /= 4;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      StridedCopy<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_strides,
                                                     reinterpret_cast<Bytes*>(out), out_strides);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      StridedCopyLarge<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_strides,
                                                          reinterpret_cast<Bytes*>(out), out_strides);
    }
  } else if (0 == (head_size % 2)) {  // pack 2 element together
    using Bytes = ToBytes<sizeof(T) * 2>;
    const int H = head_size / 2;
    in_strides.x /= 2;
    in_strides.y /= 2;
    in_strides.z /= 2;
    out_strides.x /= 2;
    out_strides.y /= 2;
    out_strides.z /= 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      StridedCopy<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_strides,
                                                     reinterpret_cast<Bytes*>(out), out_strides);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      StridedCopyLarge<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), H, in_strides,
                                                          reinterpret_cast<Bytes*>(out), out_strides);
    }
  } else {
    using Bytes = ToBytes<sizeof(T)>;
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      StridedCopy<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), head_size, in_strides,
                                                     reinterpret_cast<Bytes*>(out), out_strides);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      StridedCopyLarge<Bytes><<<grid, block, 0, stream>>>(reinterpret_cast<const Bytes*>(in), head_size, in_strides,
                                                          reinterpret_cast<Bytes*>(out), out_strides);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchStridedCopy<float>(
    cudaStream_t stream,
    const float* in, int4 in_shape, longlong4 in_strides,
    float* out, longlong4 out_strides,
    int max_threads_per_block);

template Status LaunchStridedCopy<half>(
    cudaStream_t stream,
    const half* in, int4 in_shape, longlong4 in_strides,
    half* out, longlong4 out_strides,
    int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
