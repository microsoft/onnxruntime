#include "hip/hip_runtime.h"
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

#pragma once

#include <type_traits>
#include <hipcub/hipcub.hpp>
#include <hip/hip_fp16.h>
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/rocm_common.h"
#include "core/providers/rocm/math/softmax.h"

#define ROCMRT_INF_F __int_as_float(0x7f800000)

using namespace onnxruntime::rocm;
using namespace hipcub;

namespace onnxruntime {
namespace contrib {
namespace rocm {

template <typename T, unsigned TPB>
__device__ inline void Softmax(const int all_sequence_length,
                               const int valid_end,
                               const int valid_start,
                               const T* add_before_softmax,
                               const T* input,
                               T* output) {
  using BlockReduce = hipcub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  float thread_data_max(-ROCMRT_INF_F);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      float input_at_idx = add_before_softmax == nullptr
                               ? static_cast<float>(input[index])
                               : static_cast<float>(input[index] + add_before_softmax[index]);
      if (thread_data_max < input_at_idx) {
        thread_data_max = input_at_idx;
      }
    }
  }

  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, hipcub::Max());

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_sum(0.f);
  for (int i = threadIdx.x; i < valid_end; i += TPB) {
    if (i >= valid_start) {
      const int index = offset + i;
      float val = add_before_softmax == nullptr ? input[index] : input[index] + add_before_softmax[index];
      thread_data_sum += expf(val - max_block);
    }
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_sum, hipcub::Sum());
  if (threadIdx.x == 0) {
    sum_reverse_block = 1.f / sum;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < all_sequence_length; i += TPB) {
    const int index = offset + i;
    float input_at_idx = add_before_softmax == nullptr
                             ? static_cast<float>(input[index])
                             : static_cast<float>(input[index] + add_before_softmax[index]);
    const float val = (i >= valid_start && i < valid_end) ? expf(input_at_idx - max_block) * sum_reverse_block : 0.f;
    output[index] = T(val);
  }
}

template <typename T, unsigned TPB>
__device__ inline void SoftmaxSmall(const int all_sequence_length,
                                    const int sequence_length,
                                    const int valid_end,
                                    const int valid_start,
                                    const T* add_before_softmax,
                                    const T* input,
                                    T* output,
                                    bool causal) {
  using BlockReduce = hipcub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  const int offset = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length;
  const int index = offset + threadIdx.x;

  bool is_valid = false;  // whether it has attention mask == 1.

  // Update end position for causal.
  int end = valid_end;
  if (causal) {
    const int end_causal = all_sequence_length - sequence_length + (blockIdx.x % sequence_length) + 1;
    if (end_causal < end) {
      end = end_causal;
    }
  }

  is_valid = (threadIdx.x >= valid_start && threadIdx.x < end);

  // e^x is represented as infinity if x is large enough, like 100.f.
  // Infinity divided by Infinity is a NAN. Thus, softmax gets a NAN if one or more item are large enough.
  // a math transform as below is leveraged to get a stable softmax:
  // e^xi/(e^x1 + ...e^xn) = e^(xi - max) / (e^(x1 - max) + ... + e^(xn - max))
  float input_data = add_before_softmax == nullptr
                         ? static_cast<float>(input[index])
                         : static_cast<float>(input[index] + add_before_softmax[index]);
  float thread_data_max = is_valid ? input_data : float(-ROCMRT_INF_F);
  const auto max = BlockReduce(tmp_storage).Reduce(thread_data_max, hipcub::Max(), end);

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  float thread_data_exp(0.f);
  if (is_valid) {
    thread_data_exp = expf(input_data - max_block);
  }

  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, hipcub::Sum(), end);

  // Store value of 1.0/sum.
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  // threadIdx.x might be larger than all_sequence_length due to alignment to 32x.
  if (threadIdx.x < all_sequence_length) {
    output[index] = is_valid ? T(thread_data_exp * sum_reverse_block) : T(0.f);
  }
}

// Note about the attention_mask_strides and attention_mask/key_padding_mask
// attention_mask accepts 2D, 3D or 4D tensor, but it will be viewed as 3D tensor uniformally and it will be indexed
// as [batch_index, sequence_index, token_index].
template <typename T, unsigned TPB>
__global__ void SoftmaxWithRawMaskSmallKernel(
    const int all_sequence_length,
    const int sequence_length,
    const int3 attention_mask_strides,
    const int* attention_mask,  // 2D, 3D or 4D attention mask
    const bool* key_padding_mask,
    const T* add_before_softmax,
    const T* input,
    T* output,
    const bool causal,
    const float rsqrt_head_size,
    const bool skip_softmax,
    const float mask_filter_value) {
  using BlockReduce = hipcub::BlockReduce<float, TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;

  __shared__ float sum_reverse_block;
  __shared__ float max_block;

  // Input dimension is BxNxSxS*; blockIdx.y is batch index b; gridDim.x=N*S;  blockIdx.x is index within N*S;
  int index = (blockIdx.y * gridDim.x + blockIdx.x) * all_sequence_length + threadIdx.x;

  // Mask all thread_data values to negative infinity to allow BlockReduce Max operation over all thread_data
  // members with all invalid members set to a value that does not impact the final result. This is necessary
  // to avoid the performance impact from using the valid_items interface.
  float thread_data = -ROCMRT_INF_F;
  if (threadIdx.x < all_sequence_length) {
    thread_data = float(input[index]) * rsqrt_head_size;

    const int sequence_index = blockIdx.x % sequence_length;
    if (causal) {
      int from_index = all_sequence_length - sequence_length + sequence_index;  // offset in all sequence length.
      if (threadIdx.x > from_index) {
        thread_data = -ROCMRT_INF_F;
      }
    }

    const int batch_index = blockIdx.y;
    int mask_offset = attention_mask_strides.x * batch_index +
                      attention_mask_strides.y * sequence_index +
                      attention_mask_strides.z * threadIdx.x;

    if (nullptr == key_padding_mask) {
      const int& mask = attention_mask[mask_offset];
      if (mask == 0)
        thread_data += mask_filter_value;
    } else {
      const bool mask = key_padding_mask[mask_offset];
      if (mask) {
        thread_data = -ROCMRT_INF_F;
      }
    }

    if (add_before_softmax != nullptr) {
      thread_data += float(add_before_softmax[index]);
    }
  }

  if (skip_softmax) {
    if (threadIdx.x < all_sequence_length) {
      output[index] = T(thread_data);
    }
    return;
  }

  const float max = BlockReduce(tmp_storage).Reduce(thread_data, hipcub::Max());

  // Store max value
  if (threadIdx.x == 0) {
    max_block = max;
  }
  __syncthreads();

  // Mask all thread_data_exp values to zero to allow BlockReduce Sum operation over all thread_data_exp
  // members with all invalid members set to a value that does not impact the final result. This is necessary
  // to avoid the performance impact from using the valid_items interface.
  float thread_data_exp = threadIdx.x < all_sequence_length ? expf(thread_data - max_block) : 0.0f;
  const auto sum = BlockReduce(tmp_storage).Reduce(thread_data_exp, hipcub::Sum());

  // Store value of 1.0/sum
  if (threadIdx.x == 0) {
    sum_reverse_block = (1.f) / sum;
  }
  __syncthreads();

  if (threadIdx.x < all_sequence_length) {
    output[index] = T(thread_data_exp * sum_reverse_block);
  }
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernelSmall(const int all_sequence_length, const int sequence_length,
                                   const T* add_before_softmax, const T* input, T* output, bool causal) {
  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, all_sequence_length, 0,
                       add_before_softmax, input, output, causal);
}

template <typename T, unsigned TPB>
__global__ void SoftmaxKernel(const int all_sequence_length, const T* add_before_softmax, const T* input, T* output) {
  Softmax<T, TPB>(all_sequence_length, all_sequence_length, 0, add_before_softmax, input, output);
}

template <typename T>
Status ComputeSoftmax(
    hipStream_t stream,
    const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
    const T* add_before_softmax, const T* input, T* output, bool causal) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);
  if (all_sequence_length <= 32) {
    const int blockSize = 32;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, add_before_softmax, input, output, causal);
  } else if (all_sequence_length <= 64) {
    const int blockSize = 64;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, add_before_softmax, input, output, causal);
  } else if (all_sequence_length <= 128) {
    const int blockSize = 128;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, add_before_softmax, input, output, causal);
  } else if (all_sequence_length <= 256) {
    const int blockSize = 256;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, add_before_softmax, input, output, causal);
  } else if (all_sequence_length <= 512) {
    const int blockSize = 512;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, add_before_softmax, input, output, causal);
  } else if (all_sequence_length <= 1024) {
    const int blockSize = 1024;
    SoftmaxKernelSmall<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, sequence_length, add_before_softmax, input, output, causal);
  } else if (!causal) {
    const int blockSize = 1024;
    SoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, add_before_softmax, input, output);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attention ROCM operator does not support total sequence length > 1024.");
  }

  return HIP_CALL(hipPeekAtLastError());
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernelSmall(const int all_sequence_length, const int sequence_length,
                                         const int* mask_end, const int* mask_start,
                                         const T* add_before_softmax, const T* input, T* output,
                                         bool causal) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(all_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = all_sequence_length;
    }
  }
  __syncthreads();

  SoftmaxSmall<T, TPB>(all_sequence_length, sequence_length, end_position, start_position,
                       add_before_softmax, input, output, causal);
}

template <typename T, unsigned TPB>
__global__ void MaskedSoftmaxKernel(const int all_sequence_length, const int* mask_end, const int* mask_start,
                                    const T* add_before_softmax, const T* input, T* output) {
  __shared__ int start_position;
  __shared__ int end_position;

  if (threadIdx.x == 0) {
    const int batch = blockIdx.y;
    start_position = mask_start != nullptr ? max(0, mask_start[batch]) : 0;
    end_position = min(all_sequence_length, mask_end[batch]);

    // Attend to no word has same effect as attend to all words. This is added to get parity with CPU result.
    if (start_position >= end_position) {
      start_position = 0;
      end_position = all_sequence_length;
    }
  }
  __syncthreads();

  Softmax<T, TPB>(all_sequence_length, end_position, start_position, add_before_softmax, input, output);
}

template <typename T>
Status ComputeSoftmaxWithMask1D(
    hipStream_t stream,
    const int all_sequence_length, const int sequence_length, const int batch_size, const int num_heads,
    const int* mask_index, const int* mask_start,
    const T* add_before_softmax, const T* input, T* output, const bool causal) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

#define DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(block_size)                    \
  MaskedSoftmaxKernelSmall<T, block_size><<<grid, block_size, 0, stream>>>( \
      all_sequence_length, sequence_length, mask_index, mask_start,         \
      add_before_softmax, input, output, causal);

  if (all_sequence_length <= 32) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(32);
  } else if (all_sequence_length <= 64) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(64);
  } else if (all_sequence_length <= 128) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(128);
  } else if (all_sequence_length <= 256) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(256);
  } else if (all_sequence_length <= 512) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(512);
  } else if (all_sequence_length <= 1024) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(1024);
  } else if (!causal) {
    const int blockSize = 1024;
    MaskedSoftmaxKernel<T, blockSize><<<grid, blockSize, 0, stream>>>(
        all_sequence_length, mask_index, mask_start,
        add_before_softmax, input, output);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attention ROCM operator does not support total sequence length > 1024.");
  }

#undef DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE

  return HIP_CALL(hipPeekAtLastError());
}

template <typename T>
Status ComputeSoftmaxWithRawMask(Stream* ort_stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int num_heads,
                                 const int3 attention_mask_strides,
                                 const int* attention_mask,
                                 const bool* key_padding_mask,
                                 const T* add_before_softmax,
                                 const T* input,
                                 T* output,
                                 const bool causal,
                                 const float rsqrt_head_size,
                                 const bool use_persistent_softmax,
                                 T* persistent_softmax_workspace,
                                 const float mask_filter_value) {
  const dim3 grid(sequence_length * num_heads, batch_size, 1);

  T* out = use_persistent_softmax ? persistent_softmax_workspace : output;
  auto stream = static_cast<hipStream_t>(ort_stream->GetHandle());

#define DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(block_size)                         \
  SoftmaxWithRawMaskSmallKernel<T, block_size><<<grid, block_size, 0, stream>>>( \
      all_sequence_length, sequence_length, attention_mask_strides,              \
      attention_mask, key_padding_mask, add_before_softmax, input, out,          \
      causal, rsqrt_head_size,                                                   \
      use_persistent_softmax, mask_filter_value);

  if (all_sequence_length <= 32) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(32);
  } else if (all_sequence_length <= 64) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(64);
  } else if (all_sequence_length <= 128) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(128);
  } else if (all_sequence_length <= 256) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(256);
  } else if (all_sequence_length <= 512) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(512);
  } else if (all_sequence_length <= 1024) {
    DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE(1024);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Attention ROCM operator does not support total sequence length > 1024.");
  }

#undef DISPATCH_KERNEL_SMALL_WITH_BLOCKSIZE

  if (use_persistent_softmax) {
    return dispatch_warpwise_softmax_forward<T, T, float, false>(ort_stream,
                                                                 output,
                                                                 persistent_softmax_workspace,
                                                                 all_sequence_length,
                                                                 all_sequence_length,
                                                                 batch_size * num_heads * sequence_length);
  }

  return HIP_CALL(hipPeekAtLastError());
}

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
