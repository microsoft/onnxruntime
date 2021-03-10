// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/reduction/reduction_all.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/atomic/common.cuh"
#include "core/providers/cuda/reduction/reduction_utils.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

template <typename Tin, typename Tout>
__global__ void ScalarSqrtKernel(Tin* input, Tout* output) {
  *output = (Tout)_Sqrt(*input);
}

template <typename Tin, typename Tout>
void ScalarSqrt(cudaStream_t stream, Tin* input, Tout* output) {
  ScalarSqrtKernel<<<1, 1, 0, stream>>>(input, output);
};

template void ScalarSqrt(cudaStream_t stream, float* input, float* output);
template void ScalarSqrt(cudaStream_t stream, half* input, half* output);
template void ScalarSqrt(cudaStream_t stream, float* input, half* output);
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
template void ScalarSqrt(cudaStream_t stream, nv_bfloat16* input, nv_bfloat16* output);
template void ScalarSqrt(cudaStream_t stream, float* input, nv_bfloat16* output);
#endif

template <typename TIn, typename TOut, typename TBuf, typename TInOp, typename TOutOp>
__launch_bounds__(ChunkGroup<1>::thread_count_per_block)
__global__ void MultiTensorReduceKernel(ChunkGroup<1> chunk_group, TOut* output) {
  const int group_index = chunk_group.block_index_to_tensor_group_index[blockIdx.x];
  const int tensor_size = chunk_group.tensor_sizes[group_index];
  const int chunk_size = chunk_group.chunk_size;
  const int chunk_start = chunk_group.block_index_to_chunk_start_index[blockIdx.x];
  const TIn* w = reinterpret_cast<const TIn*>(chunk_group.tensor_ptrs[0][group_index]) + chunk_start;
  TOut* w_norm = output;

  TBuf w_sum = TBuf(0.f);
  constexpr int load_count_per_thread = 4;
  for (int i = threadIdx.x; i < chunk_size && i + chunk_start < tensor_size; i += blockDim.x * load_count_per_thread) {
#pragma unroll
    for (int j = 0; j < load_count_per_thread; ++j) {
      const int index_in_chunk = i + j * blockDim.x;
      const int index_in_tensor = chunk_start + index_in_chunk;
      if (index_in_chunk < chunk_size && index_in_tensor < tensor_size) {
        const TBuf w_element = TBuf(w[index_in_chunk]);
        w_sum += TInOp()(w_element);
      }
    }
  }

// Thread count in a block must be a multiple of GPU_WARP_SIZE.
#pragma unroll
  for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
    w_sum += WARP_SHFL_DOWN(w_sum, stride);
  }

  const int warp_count_in_block = blockDim.x / GPU_WARP_SIZE;
  const int lid = threadIdx.x % GPU_WARP_SIZE;
  const int wid = threadIdx.x / GPU_WARP_SIZE;

  // Shape is 2 x warp_count_in_block.
  extern __shared__ unsigned char shared_memory_[];
  TBuf* shared_memory = reinterpret_cast<TBuf*>(shared_memory_);

  if (lid == 0) {
    shared_memory[wid] = w_sum;
  }

  __syncthreads();

#pragma unroll
  for (int stride = warp_count_in_block / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      shared_memory[threadIdx.x] += shared_memory[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    atomic_add(w_norm, TOutOp()(TOut(shared_memory[0])));
  }
}

template <typename TIn, typename TOut, typename TBuf, typename TInOp, typename TOutOp>
void MultiTensorReduce(cudaStream_t stream, ChunkGroup<1> chunk_group, TOut* output) {
  // thread count per block.
  constexpr int thread_count = ChunkGroup<1>::thread_count_per_block;
  // shared memory's size per block.
  const int shared_memory_size = thread_count / GPU_WARP_SIZE * sizeof(TBuf);

  // Enforce assumptions used inside this reduction CUDA kernel.
  ORT_ENFORCE(thread_count % GPU_WARP_SIZE == 0);
  ORT_ENFORCE((thread_count & (thread_count - 1)) == 0);

  MultiTensorReduceKernel<TIn, TOut, TBuf, TInOp, TOutOp><<<chunk_group.chunk_count, thread_count, shared_memory_size, stream>>>(chunk_group, output);
}

template <typename TIn, typename TOut>
void MultiTensorReduceL2<TIn, TOut>::operator()(cudaStream_t stream, ChunkGroup<1> chunk_group, TOut* output) {
  using TBuf = AccumulationType_t<TIn>;
  MultiTensorReduce<TIn, TOut, TBuf, Square, Identity>(stream, chunk_group, output);
}

#define INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(TIn, TOut) \
  template void MultiTensorReduceL2<TIn, TOut>::operator()(cudaStream_t stream, ChunkGroup<1> chunk_group, TOut* output);

INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(double, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(float, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(half, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(float, half)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(half, half)
#if CUDA_VERSION >= 11000 && (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(nv_bfloat16, float)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(float, nv_bfloat16)
INSTANTIATE_MULTI_TENSOR_REDUCTION_L2_FUNCTOR(nv_bfloat16, nv_bfloat16)
#endif

}  // namespace cuda
}  // namespace onnxruntime