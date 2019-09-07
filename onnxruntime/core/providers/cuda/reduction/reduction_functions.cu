#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "reduction_functions.h"

#define MAX_BLOCK_COUNT 256
#define ONE_THREAD_LOAD_COUNT 4
#define WARP_THREAD_COUNT 32
#define MAX_BLOCK_WARP_COUNT 8
#define ALL_ONE_MASK 0xFFFFFFFF
#define ONE_MASK 0x00000001

namespace onnxruntime {
namespace cuda {

int compute_block_number(int size) {
  return std::min(
    MAX_BLOCK_COUNT,
    std::max(1, size / WARP_THREAD_COUNT / MAX_BLOCK_WARP_COUNT));
}

int compute_reduction_buffer_size(int element_size, int size) {
  const int block_count = compute_block_number(size);
  return static_cast<int>(block_count * element_size + sizeof(int));
}

template<typename TAccumulated, typename TValue>
struct Identity {
  __forceinline__ __device__ TAccumulated operator()(const TValue& value) {
    return TAccumulated(value);
  }
};

template<typename TAccumulated, typename TValue>
struct Square {
  __forceinline__ __device__ TAccumulated operator()(const TValue& value) {
    return TAccumulated(value) * TAccumulated(value);
  }
};

template<typename TAccumulated, typename TValue>
struct Abs {
  __forceinline__ __device__ TAccumulated operator()(const TValue& value) {
    TAccumulated value_ = TAccumulated(value);
    return value_ > TAccumulated(0) ? value_ : -value_;
  }
};

__forceinline__ __device__ int least_pow2_bound(int value) {
  unsigned int value_ = static_cast<unsigned int>(value);
  --value_;
  value_ |= value_ >> 1;
  value_ |= value_ >> 2;
  value_ |= value_ >> 4;
  value_ |= value_ >> 8;
  value_ |= value_ >> 16;
  return static_cast<unsigned int>(++value_);
}

template<typename TIn, typename TOut, typename TOp>
__global__ void reduce_all_kernel(const int size, const TIn * data, TOut* output, TOut* buffer) {
  extern __shared__ unsigned char shared_memory_[];
  TOut* shared_memory = reinterpret_cast<TOut*>(shared_memory_);
  // Thread-level indexes:
  // Linear index of thread in block.
  const int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
  // Total number of threads in a 2-D block.
  const int thread_count_in_block = blockDim.x * blockDim.y;

  // Warp-level indexes:
  // Warp index of thread.
  const int wid_in_block = tid_in_block / WARP_THREAD_COUNT;
  // Lane index of thread.
  const int lid_in_block = tid_in_block % WARP_THREAD_COUNT;
  // Warp count per block.
  const int warp_count_in_block = thread_count_in_block / WARP_THREAD_COUNT;

  // Grid-level indexes:
  // Linear index of block in grid.
  const int bid_in_grid = blockIdx.x + blockIdx.y * gridDim.x;
  // Linear index of thread in grid.
  const int tid_in_grid = bid_in_grid * (blockDim.x * blockDim.y) + tid_in_block;
  // Total number of blocks in a 2-D grid.
  const int block_count_in_grid = gridDim.x * gridDim.y;
  // Total number of threads in a 2-D grid with 2-D blocks.
  const int thread_count_in_grid = block_count_in_grid * thread_count_in_block;

  // Thread-level reduction (storage change: global memory -> register).
  // One thread reduces ONE_THREAD_LOAD_COUNT elements to a thread register
  // in one iteration.
  const int total_thread_launch_count = size / ONE_THREAD_LOAD_COUNT;
  TOut value = 0;
  for (int thread_launch_count = 0; thread_launch_count < total_thread_launch_count; thread_launch_count += thread_count_in_grid) {
    const int offset = (thread_launch_count + tid_in_grid) * ONE_THREAD_LOAD_COUNT;
    if (thread_launch_count + tid_in_grid < total_thread_launch_count) {
      const TIn* addr = data + offset;
#pragma unroll
      for (int i = 0; i < ONE_THREAD_LOAD_COUNT; ++i) {
        value += TOp()(addr[i]);
      }
    }
  }

  if (tid_in_grid == 0) {
    const int rest_thread_launch_count = size % ONE_THREAD_LOAD_COUNT;
#pragma unroll(4)
    for (int thread_launch_count = 0; thread_launch_count < rest_thread_launch_count; ++thread_launch_count) {
      value += TOp()(data[size - thread_launch_count - 1]);
    }
  }

  // If we have less than ONE_THREAD_LOAD_COUNT elements, the task is done.
  if (size <= ONE_THREAD_LOAD_COUNT) {
    if (tid_in_grid == 0) {
      output[0] = value;
    }
    return;
  }

  __syncthreads();

  // Warp-level reduction (storage change: register -> register).
  // The values in a warp will be summed up to a scalar. After warp-level
  // reduction, each block holds warp_count_in_block values in the shared
  // memory.
  TOut value_ = value;
#pragma unroll
  for (int stride = WARP_THREAD_COUNT / 2; stride > 0; stride /= 2) {
    value_ += __shfl_down_sync(ALL_ONE_MASK, value_, stride);
  }

  // If we have less than ONE_THREAD_LOAD_COUNT * WARP_THREAD_COUNT elements,
  // the task is done because one warp can at most reduce
  // ONE_THREAD_LOAD_COUNT * WARP_THREAD_COUNT values.
  if (size <= ONE_THREAD_LOAD_COUNT * WARP_THREAD_COUNT) {
    if (tid_in_grid == 0) {
      output[0] = value_;
    }
    return;
  }

  if (lid_in_block == 0) {
    shared_memory[wid_in_block] = value_;
  }

  __syncthreads();

  // Block-level reduction (storage change: shared memory -> global memory).
  // The values in a block will be summed up to a scalar.
  // Note that the values are stored in the shared memory.
  // Here we assume that the size of shared_memory is smaller
  // than warp_count_in_block, so we just keep halving the number
  // of threads in each iteartion. Our assumption is always true because
  // the size of shared_memory equals to the number of warps.
#pragma unroll
  for (int stride = MAX_BLOCK_WARP_COUNT / 2; stride > 0; stride /= 2) {
    if (tid_in_block + stride < warp_count_in_block) {
      shared_memory[tid_in_block] += shared_memory[tid_in_block + stride];
    }
    __syncthreads();
  }

  // If we have less than ONE_THREAD_LOAD_COUNT * thread_count_in_block elements,
  // the task is done because one block can at most reduce
  // ONE_THREAD_LOAD_COUNT * thread_count_in_block values.
  if (size <= ONE_THREAD_LOAD_COUNT * thread_count_in_block) {
    if (tid_in_grid == 0) {
      output[0] = shared_memory[0];
    }
    return;
  }

  if (tid_in_block == 0) {
    buffer[bid_in_grid] = shared_memory[0];
  }

  __threadfence();
  __syncthreads();

  // Grid-level reduciton. We use the last block to sum up values
  // stored in the global buffer.
  __shared__ bool is_last_block_done;

  if (tid_in_block == 0) {
    int* p_lock = reinterpret_cast<int*>(buffer + block_count_in_grid);
    int count = atomicAdd(p_lock, 1);
    is_last_block_done = (count == (block_count_in_grid - 1));
  }

  // All threads in each block see if they belong the last active block 
  // (i.e., the value of is_last_block_done).
  __syncthreads();

  // Only the block which saw that count equals to block_count_in_grid - 1 can
  // enter the following block.
  if (is_last_block_done) {
    const int pow2_bound = least_pow2_bound(block_count_in_grid);
    for (int stride = pow2_bound / 2; stride > 0; stride /= 2) {
      if (tid_in_block < stride && tid_in_block + stride < block_count_in_grid)
        buffer[tid_in_block] += buffer[tid_in_block + stride];
      __syncthreads();
    }

    // The first thread in the last block assigns the final output.
    if (tid_in_block == 0) {
      output[0] = buffer[0];
    }
  }
}

template<typename TIn, typename TOut, typename TOp>
void call_reduce_all_kernel(const TIn *data, TOut *output, int size, TOut *buffer)
{
  const int block_count = compute_block_number(size);
  cudaMemset(buffer + block_count, 0, sizeof(int));
  const dim3 grid(block_count, 1, 1);
  const dim3 block(WARP_THREAD_COUNT, MAX_BLOCK_WARP_COUNT, 1);
  reduce_all_kernel<TIn, TOut, TOp><<<grid, block, MAX_BLOCK_WARP_COUNT * sizeof(TOut)>>>(size, data, output, buffer);
}

template<typename TIn, typename TOut>
void reduce_sum(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Identity<TOut, TIn>>(
    data, output, size, buffer);
}

template<typename TIn, typename TOut>
void reduce_square_sum(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Square<TOut, TIn>>(
    data, output, size, buffer);
}

template void reduce_sum<half, float>(
  const half* data, float* output, int size, float* buffer);
template void reduce_sum<float, float>(
  const float* data, float* output, int size, float* buffer);
template void reduce_sum<double, double>(
  const double* data, double* output, int size, double* buffer);

template void reduce_square_sum<half, float>(
  const half* data, float* output, int size, float* buffer);
template void reduce_square_sum<float, float>(
  const float* data, float* output, int size, float* buffer);
template void reduce_square_sum<double, double>(
  const double* data, double* output, int size, double* buffer);

}  // namespace cuda
}  // namespace onnxruntime