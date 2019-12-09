#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"
#include "reduction_functions.h"
#include "reduction_utils.cuh"

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

template<typename TIn, typename TOut, typename TOp, typename TFinalOp, bool DivideResultBySize>
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
      // Compilation time if-else branch controlled by template argument can be
      // optimized out, so there will be no branch in real computation phase.
      if (DivideResultBySize) {
        output[0] = TFinalOp()(value / TOut(size));
      } else {
        output[0] = TFinalOp()(value);
      }
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
      // Compilation time if-else branch controlled by template argument can be
      // optimized out, so there will be no branch in real computation phase.
      if (DivideResultBySize) {
        output[0] = TFinalOp()(value_ / TOut(size));
      } else {
        output[0] = TFinalOp()(value_);
      }
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
      // Compilation time if-else branch controlled by template argument can be
      // optimized out, so there will be no branch in real computation phase.
      if (DivideResultBySize) {
        output[0] = TFinalOp()(shared_memory[0] / TOut(size));
      } else {
        output[0] = TFinalOp()(shared_memory[0]);
      }
    }
    return;
  }

  if (tid_in_block == 0) {
    // Compilation time if-else branch controlled by template argument can be
    // optimized out, so there will be no branch in real computation phase.
    if (DivideResultBySize) {
      buffer[bid_in_grid] = shared_memory[0];
    } else {
      buffer[bid_in_grid] = shared_memory[0];
    }
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
      // Compilation time if-else branch controlled by template argument can be
      // optimized out, so there will be no branch in real computation phase.
      if (DivideResultBySize) {
        output[0] = TFinalOp()(buffer[0] / TOut(size));
      } else {
        output[0] = TFinalOp()(buffer[0]);
      }
    }
  }
}

template<typename TIn, typename TOut, typename TOp, typename TFinalOp, bool DivideResultBySize>
void call_reduce_all_kernel(const TIn *data, TOut *output, int size, TOut *buffer)
{
  const int block_count = compute_block_number(size);
  cudaMemset(buffer + block_count, 0, sizeof(int));
  const dim3 grid(block_count, 1, 1);
  const dim3 block(WARP_THREAD_COUNT, MAX_BLOCK_WARP_COUNT, 1);
  reduce_all_kernel<TIn, TOut, TOp, TFinalOp, DivideResultBySize><<<grid, block, MAX_BLOCK_WARP_COUNT * sizeof(TOut)>>>(size, data, output, buffer);
}

template<typename TIn, typename TOut>
void reduce_sum(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Cast<TOut, TIn>, Identity<TOut>, false>(
    data, output, size, buffer);
}

template<typename TIn, typename TOut>
void reduce_square_sum(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Square<TOut, TIn>, Identity<TOut>, false>(
    data, output, size, buffer);
}

template<typename TIn, typename TOut>
void reduce_l2_norm(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Square<TOut, TIn>, Sqrt<TOut>, false>(
    data, output, size, buffer);
}

template<typename TIn, typename TOut>
void reduce_mean(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Cast<TOut, TIn>, Identity<TOut>, true>(
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

template void reduce_l2_norm<half, float>(
  const half* data, float* output, int size, float* buffer);
template void reduce_l2_norm<float, float>(
  const float* data, float* output, int size, float* buffer);
template void reduce_l2_norm<double, double>(
  const double* data, double* output, int size, double* buffer);

template void reduce_mean<half, float>(
  const half* data, float* output, int size, float* buffer);
template void reduce_mean<float, float>(
  const float* data, float* output, int size, float* buffer);
template void reduce_mean<double, double>(
  const double* data, double* output, int size, double* buffer);

bool is_matrix_row_reduction(
    const cudnnReduceTensorOp_t cudnnReduceOp,
    const int m,
    const int n,
    const size_t rank,
    std::vector<int64_t> axes) {
  if (m < 1)
    return false;

  if (n < 1)
    return false;

  if (rank < 2)
    return false;

  if (cudnnReduceOp != CUDNN_REDUCE_TENSOR_ADD)
    return false;

  // Check if all but the last axis are reduced. For example, reducing
  // [N, C, H, W]-tensor to [W]-tensor can pass these two checks but reducing
  // [N, C]-tensor to [N, 1]-tensor cannot.
  if (axes.size() != rank - 1)
    return false;

  // The last reduced axis should be the second last axis. For
  // [N, C, H, W]-input, the sorted axes should be [0, 1, 2].
  std::sort(axes.begin(), axes.end());
  if (axes.back() != rank - 2)
    return false;

  return true;
}

template<typename TIn, typename TOut, typename TBuf>
__global__ void reduce_matrix_rows_kernel(const TIn *input, TOut *output, int m, int n) {
  constexpr int x_load_count_per_thread = 1;
  constexpr int y_load_count_per_thread = 4;
  const int t_count_x_in_grid = blockDim.x * gridDim.x;
  const int t_count_y_in_grid = blockDim.y * gridDim.y;
  const int x_grid_stride = t_count_x_in_grid * x_load_count_per_thread;
  const int y_grid_stride = t_count_y_in_grid * y_load_count_per_thread;
  const int tid_x_in_grid = threadIdx.x + blockDim.x * blockIdx.x;
  const int tid_y_in_grid = threadIdx.y + blockDim.y * blockIdx.y;
  const int tid_in_block = threadIdx.x + blockDim.x * threadIdx.y;

  // Shape is blockDim.y-by-blockDim.x and element type is TBuf.
  extern __shared__ unsigned char shared_memory_[];
  TBuf *shared_memory = reinterpret_cast<TBuf*>(shared_memory_);

  for (int col = tid_x_in_grid; col < n; col += x_grid_stride) {
    shared_memory[tid_in_block] = TBuf(0.0f);

    // This loops load multiple blockDim.y-by-blockDim.x sub-tensors from the input.
    for (int row = tid_y_in_grid; row < m; row += y_grid_stride) {
      TBuf sum = 0.0f;
      // Thread-level reduction. Each thread loads y_load_count_per_thread values
      // and aggregrate them.
#pragma unroll(y_load_count_per_thread)
      for (int row_inner = 0; row_inner < y_load_count_per_thread; ++row_inner) {
        int row_final = row + row_inner * t_count_y_in_grid;
        int col_final = col;
        if (row_final < m && col_final < n) {
          sum += TBuf(input[row_final * n + col_final]);
        }
      }
      // Write thread-level reduction result into shared memory.
      shared_memory[tid_in_block] += sum;
    }

    // Wait all threads to finish their thread-level reductions.
    __syncthreads();

    // This loop conducts reduction on elements stored in shared memory.
    // Each block reduces blockDim.y-by-blockDim.x tensor to 1-by-blockDim.x tensor.
#pragma unroll(4)
    for (int stride = blockDim.y / 2; stride > 0; stride /= 2) {
      if (threadIdx.y < stride) {
        shared_memory[tid_in_block] += shared_memory[tid_in_block + stride * blockDim.x];
      }
      __syncthreads();
    }

    if (threadIdx.y == 0) {
      atomic_add(output + col, TOut(shared_memory[threadIdx.x]));
    }

    // Make sure all values in shared memory have been written into the output memory.
    __syncthreads();
  }
}

// This function reduces the given input tensor along all but the last axis.
// For example, [N, C, H, W]-tensor may lead to a output [W]-tensor.
// It's implementation is in reduction_ops.cu and called in reduction_ops.cc.
template<typename TIn, typename TOut, typename TBuf>
void call_reduce_matrix_rows(const TIn *input, TOut *output, int m, int n) {
  constexpr int max_thread_count_in_block = 512;
  constexpr int max_block_count_in_grid = 512;
  constexpr int warp_size = 32;
  constexpr int load_count_per_thread = 4;

  const int block_x_dim = least_pow2_bound(std::max(1, std::min(n, warp_size)));
  const int block_y_dim = least_pow2_bound(std::max(1, std::min(max_thread_count_in_block / block_x_dim, m / load_count_per_thread)));
  const int grid_x_dim = std::max(1, std::min(n / block_x_dim, max_block_count_in_grid));
  const int grid_y_dim = std::max(1, std::min(max_block_count_in_grid / grid_x_dim, m / block_y_dim / 4));

  const dim3 grid(grid_x_dim, grid_y_dim, 1);
  const dim3 block(block_x_dim, block_y_dim, 1);

  reduce_matrix_rows_kernel<TIn, TOut, TBuf><<<grid, block, block.y * block.x * sizeof(TBuf)>>>(
      input, output, m, n);
}

template<typename TIn, typename TOut>
void reduce_matrix_rows(const TIn* data, TOut* output, int m, int n)
{
  call_reduce_matrix_rows<TIn, TOut, TOut>(data, output, m, n);
}

template<> void reduce_matrix_rows<half, half>(const half* data, half* output, int m, int n)
{
  call_reduce_matrix_rows<half, half, float>(data, output, m, n);
}

template void reduce_matrix_rows<float, float>(
  const float* data, float* output, int m, int n);
template void reduce_matrix_rows<double, double>(
  const double* data, double* output, int m, int n);

}  // namespace cuda
}  // namespace onnxruntime