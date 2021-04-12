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

#include "core/providers/cuda/cuda_common.h"
#include "attention_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void TransposeCtx(const int H, const T* input, T* output) {
  // Input:  BxNxSxH
  // Output: BxSxNxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;

  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = s * H + n * sequence_length * H + b * NHS;
  const int out_offset = n * H + s * NH + b * NHS;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

template <typename T>
__global__ void TransposeCtxLarge(const int H, const T* input, T* output) {
  // Use when (H*)*num_heads > 1024

  // Input:  BxNxSxH
  // Output: BxSxNxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;

  int stride = blockDim.x;
  int num_heads = blockDim.y;
  int sequence_length = gridDim.x;

  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = s * H + n * sequence_length * H + b * NHS;
  const int out_offset = n * H + s * NH + b * NHS;

  int i = threadIdx.x;
  while (i < H) {
    output[out_offset + i] = input[in_offset + i];
    i += stride;
  }
}

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const float* input, float* output) {
  const dim3 grid(sequence_length, batch_size, 1);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      TransposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeCtxLarge<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      TransposeCtx<float><<<grid, block, 0, stream>>>(head_size, input, output);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeCtxLarge<float><<<grid, block, 0, stream>>>(head_size, input, output);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const half* input, half* output) {
  const dim3 grid(sequence_length, batch_size, 1);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      TransposeCtx<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeCtxLarge<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      TransposeCtx<half2><<<grid, block, 0, stream>>>(H, input2, output2);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeCtxLarge<half2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      TransposeCtx<half><<<grid, block, 0, stream>>>(head_size, input, output);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeCtxLarge<half><<<grid, block, 0, stream>>>(head_size, input, output);
    }
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

template <typename T>
__global__ void TransposeQKV(const int H, const T* input, T* output) {
  // Input:  BxSx3xNxH
  // Output: 3xBxNxSxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int i = threadIdx.x;
  if (i < H) {
    output[out_offset + i] = input[in_offset + i];
  }
}

template <typename T>
__global__ void TransposeQKVLarge(const int H, const T* input, T* output) {
  // Use when (H*)*num_heads > 1024

  // Input:  BxSx3xNxH
  // Output: 3xBxNxSxH

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  const int in_offset = n * H + m * NH + s * 3 * NH + b * NHS * 3;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  int i = threadIdx.x;
  while (i < H) {
    output[out_offset + i] = input[in_offset + i];
    i += stride;
  }
}

bool LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const float* input, float* output) {
  const dim3 grid(sequence_length, batch_size, 3);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeQKVLarge<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      TransposeQKV<float><<<grid, block, 0, stream>>>(head_size, input, output);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeQKVLarge<float><<<grid, block, 0, stream>>>(head_size, input, output);
    }

  }
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const half* input, half* output) {
  const dim3 grid(sequence_length, batch_size, 3);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    float2* output2 = reinterpret_cast<float2*>(output);
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      TransposeQKV<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeQKVLarge<float2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const half2* input2 = reinterpret_cast<const half2*>(input);
    half2* output2 = reinterpret_cast<half2*>(output);
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      TransposeQKV<half2><<<grid, block, 0, stream>>>(H, input2, output2);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeQKVLarge<half2><<<grid, block, 0, stream>>>(H, input2, output2);
    }
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel..
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      TransposeQKV<half><<<grid, block, 0, stream>>>(head_size, input, output);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      TransposeQKVLarge<half><<<grid, block, 0, stream>>>(head_size, input, output);
    }
  }
  return CUDA_CALL(cudaPeekAtLastError());
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
