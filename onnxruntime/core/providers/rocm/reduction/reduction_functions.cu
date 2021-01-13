// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/rocm/reduction/reduction_functions.h"

#include <algorithm>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "core/common/common.h"
#include "core/providers/rocm/atomic/common.cuh"
#include "core/providers/rocm/cu_inc/common.cuh"
#include "core/providers/rocm/shared_inc/rocm_utils.h"
#include "core/providers/rocm/reduction/reduction_utils.cuh"

#define NUM_ELEMENTS_PER_THREAD 4
#define NUM_WARPS_PER_BLOCK 8
#define MAX_NUM_BLOCKS 256

#define ALL_ONE_MASK 0xFFFFFFFF
#define ONE_MASK 0x00000001

namespace onnxruntime {
namespace rocm {

namespace detail {
constexpr auto MAX_NUM_ELEMENTS_PER_THREAD = 4;
constexpr auto MAX_NUM_WARPS_PER_BLOCK = 8;
constexpr auto MAX_NUM_BLOCKS_IN_GRID_ROW = 256;
constexpr auto MAX_NUM_GRID_ROWS = 32768;

dim3 compute_block_dim(int num_cols) {
  const int x = GPU_WARP_SIZE;
  const int y = std::min(MAX_NUM_WARPS_PER_BLOCK, std::max(1, num_cols / (MAX_NUM_ELEMENTS_PER_THREAD * x)));
  return dim3(x, y);
}

std::pair<dim3, dim3> compute_grid_and_block_dims(int num_rows, int num_cols) {
  const auto block_dim = compute_block_dim(num_cols);
  const auto grid_x =
      std::min<int>(
          MAX_NUM_BLOCKS_IN_GRID_ROW,
          std::max<int>(1, num_cols / (MAX_NUM_ELEMENTS_PER_THREAD * block_dim.x * block_dim.y)));
  const auto grid_y = std::min(MAX_NUM_GRID_ROWS, num_rows);
  const dim3 grid_dim(grid_x, grid_y);
  return {grid_dim, block_dim};
}

uintptr_t round_up_to_aligned(uintptr_t original, size_t alignment) {
  assert((alignment & (alignment - 1)) == 0);
  const size_t alignment_mask = ~(alignment - 1);
  return (original + alignment - 1) & alignment_mask;
}

/**
 * call_reduce_matrix_columns() intermediate buffer layout
 *
 * Given buffer element type TBuf, the intermediate buffer layout looks like this:
 *
 * -----
 * m * num_blocks_per_row * sizeof(TBuf) bytes for block reductions per row
 * alignment padding bytes as needed
 * m * sizeof(int) bytes for block done counts per row
 * -----
 */

size_t compute_reduce_matrix_columns_intermediate_buffer_size(
    int element_size, int num_rows, int num_cols) {
  ORT_ENFORCE(element_size >= 0 && num_rows >= 0 && num_cols >= 0);

  const auto grid_dim = compute_grid_and_block_dims(num_rows, num_cols).first;

  size_t buffer_size{};

  // at the beginning, for sizing purposes, assume we are aligned
  buffer_size += static_cast<size_t>(num_rows) * grid_dim.x * element_size;

  buffer_size = round_up_to_aligned(buffer_size, alignof(int));
  buffer_size += static_cast<size_t>(num_rows) * sizeof(int);

  // add padding to give us room to align
  buffer_size += alignof(max_align_t) - 1;

  return buffer_size;
}

template <typename TBuf>
Status get_reduction_buffers(
    int num_rows, int num_cols, void* buffer, size_t buffer_size,
    TBuf*& block_reductions_buffer, int*& block_done_counts_buffer) {
  const auto grid_dim = compute_grid_and_block_dims(num_rows, num_cols).first;

  const uintptr_t begin_addr = reinterpret_cast<uintptr_t>(buffer);
  const uintptr_t block_reductions_addr =
      round_up_to_aligned(begin_addr, alignof(TBuf));
  const uintptr_t block_done_counts_buffer_addr =
      round_up_to_aligned(
          block_reductions_addr + static_cast<size_t>(num_rows) * grid_dim.x * sizeof(TBuf), alignof(int));
  const uintptr_t end_addr =
      block_done_counts_buffer_addr + static_cast<size_t>(num_rows) * sizeof(int);
  const size_t required_size = end_addr - begin_addr;

  ORT_RETURN_IF_NOT(
      required_size <= buffer_size,
      "Buffer size is too small (", buffer_size, " bytes). ",
      "At least ", required_size, " bytes are needed from the given base address (", buffer, ").");

  block_reductions_buffer = reinterpret_cast<TBuf*>(block_reductions_addr);
  block_done_counts_buffer = reinterpret_cast<int*>(block_done_counts_buffer_addr);

  return Status::OK();
}

template <typename TIn, typename TOut, typename TBuf, typename TOp, typename TFinalOp, bool DivideResultBySize>
__device__ void reduce_all(
    const int num_elements, const TIn* const input, TOut* const output,
    TBuf* const block_reductions_buffer, int* const block_done_count_buffer) {
  extern __shared__ unsigned char shared_memory_bytes[];
  TBuf* shared_memory = reinterpret_cast<TBuf*>(shared_memory_bytes);
  // Thread-level indices:
  // Linear index of thread in block.
  const int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
  // Total number of threads in a 2-D block.
  const int num_threads_in_block = blockDim.x * blockDim.y;

  // Warp-level indices:
  // Warp index of thread.
  const int wid_in_block = tid_in_block / GPU_WARP_SIZE;
  // Lane index of thread.
  const int lid_in_block = tid_in_block % GPU_WARP_SIZE;
  // Warp count per block.
  const int num_warps_in_block = num_threads_in_block / GPU_WARP_SIZE;

  // Grid-level indices:
  // Linear index of block in grid row.
  const int bid_in_grid_row = blockIdx.x;
  // Linear index of thread in grid row.
  const int tid_in_grid_row = bid_in_grid_row * (blockDim.x * blockDim.y) + tid_in_block;
  // Total number of blocks in a grid row.
  const int num_blocks_in_grid_row = gridDim.x;
  // Total number of threads in a grid row with 2-D blocks.
  const int num_threads_in_grid_row = num_blocks_in_grid_row * num_threads_in_block;

  const auto write_result = [&output, &num_elements](const TOut result) {
    // Compilation time if-else branch controlled by template argument can be
    // optimized out, so there will be no branch in real computation phase.
    if (DivideResultBySize) {
      output[0] = TFinalOp()(result / TOut(num_elements));
    } else {
      output[0] = TFinalOp()(result);
    }
  };

  // Thread-level reduction (storage change: global memory -> register).
  // One thread reduces MAX_NUM_ELEMENTS_PER_THREAD elements to a thread register
  // in one iteration.
  TBuf value = 0;
  for (int id = tid_in_grid_row; id < num_elements; id += MAX_NUM_ELEMENTS_PER_THREAD * num_threads_in_grid_row) {
    TIn v[MAX_NUM_ELEMENTS_PER_THREAD];

#pragma unroll
    for (int i = 0; i < MAX_NUM_ELEMENTS_PER_THREAD; i++) {
      const int offset = id + i * num_threads_in_grid_row;
      if (offset < num_elements) {
        v[i] = input[offset];
      }
    }

#pragma unroll
    for (int i = 0; i < MAX_NUM_ELEMENTS_PER_THREAD; i++) {
      const int offset = id + i * num_threads_in_grid_row;
      if (offset < num_elements) {
        value += TOp()(TBuf(v[i]));
      }
    }
  }

#if __ROCM_ARCH__ >= 700
  __syncwarp();
#else
  __syncthreads();
#endif

  // Warp-level reduction (storage change: register -> register).
  // The values in a warp will be summed up to a scalar. After warp-level
  // reduction, each block holds num_warps_in_block values in the shared memory.
#pragma unroll
  for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
    value += WARP_SHFL_DOWN(value, stride);
  }

  // Return early if only one warp is used for reduction.
  // Given a fixed amount of threads, we prefer threads over warps over blocks so that we never have cases such as
  // 1. two blocks and each of them has only 1 warp (32 threads).
  // 2. two warps and each of them has only 2 threads.
  if (num_warps_in_block == 1) {
    if (tid_in_grid_row == 0) {
      write_result(value);
    }
    return;
  }

  if (lid_in_block == 0) {
    shared_memory[wid_in_block] = value;
  }

  __syncthreads();

  // Block-level reduction (storage change: shared memory -> global memory).
  // The values in a block will be summed up to a scalar.
  // Note that the values are stored in the shared memory.
  // Here we assume that the size of shared_memory is smaller
  // than num_warps_in_block, so we just keep halving the number
  // of threads in each iteration. Our assumption is always true because
  // the size of shared_memory equals to the number of warps.
#pragma unroll
  for (int stride = MAX_NUM_WARPS_PER_BLOCK / 2; stride > 0; stride /= 2) {
    if (tid_in_block + stride < num_warps_in_block) {
      shared_memory[tid_in_block] += shared_memory[tid_in_block + stride];
    }
    __syncthreads();
  }

  // Return early if only one block is used for reduction.
  if (num_blocks_in_grid_row == 1) {
    if (tid_in_grid_row == 0) {
      write_result(shared_memory[0]);
    }
    return;
  }

  if (tid_in_block == 0) {
    block_reductions_buffer[bid_in_grid_row] = shared_memory[0];
  }

  __threadfence();
  __syncthreads();

  // Grid-level reduction. We use the last block to sum up values
  // stored in the global block_reductions_buffer.
  __shared__ bool is_last_block_done;

  if (tid_in_block == 0) {
    const int count = atomicAdd(block_done_count_buffer, 1);
    is_last_block_done = (count == (num_blocks_in_grid_row - 1));
  }

  // All threads in each block see if they belong the last active block
  // (i.e., the value of is_last_block_done).
  __syncthreads();

  // Only the block which saw that count equals to num_blocks_in_grid_row - 1 can
  // enter the following block.
  if (is_last_block_done) {
    const int pow2_bound = least_pow2_bound(num_blocks_in_grid_row);
    for (int stride = pow2_bound / 2; stride > 0; stride /= 2) {
      if (tid_in_block < stride && tid_in_block + stride < num_blocks_in_grid_row) {
        block_reductions_buffer[tid_in_block] += block_reductions_buffer[tid_in_block + stride];
      }
      __syncthreads();
    }

    // The first thread in the last block assigns the final output.
    if (tid_in_block == 0) {
      write_result(block_reductions_buffer[0]);
    }
  }
}

template <typename TIn, typename TOut, typename TBuf, typename TOp, typename TFinalOp, bool DivideResultBySize>
__global__ void reduce_matrix_columns_kernel(
    const int num_rows, const int num_cols, const TIn* const input, TOut* const output,
    TBuf* const block_reductions_buffer, int* const block_done_counts_buffer) {
  const int num_blocks_in_grid_row = gridDim.x;
  const int row_id_in_grid = blockIdx.y;
  const int num_grid_rows = gridDim.y;

  // one row per iteration
  // row_id is int64_t to avoid int overflow in offset calculations
  for (int64_t row_id = row_id_in_grid; row_id < num_rows; row_id += num_grid_rows) {
    const TIn* const row_data = input + row_id * num_cols;
    TOut* const row_output = output + row_id;
    TBuf* const row_block_reductions_buffer = block_reductions_buffer + row_id * num_blocks_in_grid_row;
    int* const row_block_done_counts_buffer = block_done_counts_buffer + row_id;

    reduce_all<TIn, TOut, TBuf, TOp, TFinalOp, DivideResultBySize>(
        num_cols, row_data, row_output,
        row_block_reductions_buffer, row_block_done_counts_buffer);
  }
}

template <typename TIn, typename TOut, typename TOp, typename TFinalOp, bool DivideResultBySize>
Status call_reduce_matrix_columns(
    const TIn* input, TOut* output, const int num_rows, const int num_cols, void* buffer, size_t buffer_size) {
  ORT_ENFORCE(num_rows >= 0 && num_cols >= 0);

  using TBuf = AccumulationType_t<TIn>;

  const auto grid_and_block_dims = compute_grid_and_block_dims(num_rows, num_cols);
  const dim3& grid_dim = grid_and_block_dims.first;
  const dim3& block_dim = grid_and_block_dims.second;

  TBuf* block_reductions_buffer;
  int* block_done_counts_buffer;
  ORT_RETURN_IF_ERROR(get_reduction_buffers(
      num_rows, num_cols, buffer, buffer_size,
      block_reductions_buffer, block_done_counts_buffer));

  // If more than one block is used per grid row, then inter-block reduction is needed.
  if (grid_dim.x > 1) {
    HIP_RETURN_IF_ERROR(hipMemsetAsync(block_done_counts_buffer, 0, num_rows * sizeof(int)));
  }

  const int shared_mem_size = sizeof(TBuf) * block_dim.x * block_dim.y / GPU_WARP_SIZE;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_matrix_columns_kernel<TIn, TOut, TBuf, TOp, TFinalOp, DivideResultBySize>),
      grid_dim, block_dim, shared_mem_size, 0,
      num_rows, num_cols, input, output, block_reductions_buffer, block_done_counts_buffer);

  return Status::OK();
}
}  // namespace detail

template <typename TIn, typename TOut>
Status reduce_sum(
    const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Identity2, Identity2, false>(
      input, output, 1, size, buffer, buffer_size);
}

template <typename TIn, typename TOut>
Status reduce_square_sum(
    const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Square2, Identity2, false>(
      input, output, 1, size, buffer, buffer_size);
}

template <typename TIn, typename TOut>
Status reduce_l2_norm(
    const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Square2, Sqrt2, false>(
      input, output, 1, size, buffer, buffer_size);
}

template <typename TIn, typename TOut>
Status reduce_mean(
    const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Identity2, Identity2, true>(
      input, output, 1, size, buffer, buffer_size);
}

#define INSTANTIATE_REDUCE_SUM(TIn, TOut) \
  template Status reduce_sum<TIn, TOut>(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_SUM(half, float);
INSTANTIATE_REDUCE_SUM(float, float);
INSTANTIATE_REDUCE_SUM(double, double);
#undef INSTANTIATE_REDUCE_SUM

#define INSTANTIATE_REDUCE_SQUARE_SUM(TIn, TOut) \
  template Status reduce_square_sum<TIn, TOut>(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_SQUARE_SUM(half, float);
INSTANTIATE_REDUCE_SQUARE_SUM(float, float);
INSTANTIATE_REDUCE_SQUARE_SUM(double, double);
#undef INSTANTIATE_REDUCE_SQUARE_SUM

#define INSTANTIATE_REDUCE_L2_NORM(TIn, TOut) \
  template Status reduce_l2_norm<TIn, TOut>(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_L2_NORM(half, float);
INSTANTIATE_REDUCE_L2_NORM(float, float);
INSTANTIATE_REDUCE_L2_NORM(double, double);
#undef INSTANTIATE_REDUCE_L2_NORM

#define INSTANTIATE_REDUCE_MEAN(TIn, TOut) \
  template Status reduce_mean<TIn, TOut>(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_MEAN(half, float);
INSTANTIATE_REDUCE_MEAN(float, float);
INSTANTIATE_REDUCE_MEAN(double, double);
#undef INSTANTIATE_REDUCE_MEAN

namespace detail {
template <typename TIn, typename TOut, typename TBuf>
__global__ void reduce_matrix_rows_kernel(const TIn* input, TOut* output, int m, int n) {
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
  extern __shared__ unsigned char shared_memory_bytes[];
  TBuf* shared_memory = reinterpret_cast<TBuf*>(shared_memory_bytes);

  // to prevent int overflow in index calculation for input size m*n
  const int64_t n_int64 = static_cast<int64_t>(n);

  for (int col = tid_x_in_grid; col < n; col += x_grid_stride) {
    shared_memory[tid_in_block] = TBuf(0.0f);
    TBuf sum = TBuf(0.0f);
    // This loops load multiple blockDim.y-by-blockDim.x sub-tensors from the input.
    for (int row = tid_y_in_grid; row < m; row += y_grid_stride) {
      // Thread-level reduction. Each thread loads y_load_count_per_thread values
      // and aggregrate them.
#pragma unroll(y_load_count_per_thread)
      for (int row_inner = 0; row_inner < y_load_count_per_thread; ++row_inner) {
        int row_final = row + row_inner * t_count_y_in_grid;
        int col_final = col;
        if (row_final < m && col_final < n) {
          sum += TBuf(input[row_final * n_int64 + col_final]);
        }
      }
    }
    // Write thread-level reduction result into shared memory.
    shared_memory[tid_in_block] = sum;

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
  }
}

template <typename TIn, typename TOut, typename TBuf>
Status call_reduce_matrix_rows(const TIn* input, TOut* output, int m, int n, bool reset_initial_output) {
  ORT_ENFORCE(m >= 0 && n >= 0);

  if (reset_initial_output) {
    HIP_RETURN_IF_ERROR(hipMemsetAsync(output, 0, n * sizeof(TOut)));
  }

  constexpr int max_num_threads_in_block = 512;
  constexpr int max_num_blocks_in_grid = 512;
  constexpr int load_count_per_thread = 4;

  const int block_x_dim = least_pow2_bound(std::max(1, std::min(n, GPU_WARP_SIZE)));
  const int block_y_dim = least_pow2_bound(std::max(1, std::min(max_num_threads_in_block / block_x_dim, m / load_count_per_thread)));
  const int grid_x_dim = std::max(1, std::min(n / block_x_dim, max_num_blocks_in_grid));
  const int grid_y_dim = std::max(1, std::min(max_num_blocks_in_grid / grid_x_dim, m / block_y_dim / 4));

  const dim3 grid(grid_x_dim, grid_y_dim, 1);
  const dim3 block(block_x_dim, block_y_dim, 1);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_matrix_rows_kernel<TIn, TOut, TBuf>),
      grid, block, block.y * block.x * sizeof(TBuf), 0,
      input, output, m, n);

  return Status::OK();
}
}  // namespace detail

template <typename TIn, typename TOut>
Status reduce_matrix_rows(const TIn* input, TOut* output, int m, int n, bool reset_initial_output) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::call_reduce_matrix_rows<TIn, TOut, TBuf>(input, output, m, n, reset_initial_output);
}

#define INSTANTIATE_REDUCE_MATRIX_ROWS(T) \
  template Status reduce_matrix_rows<T, T>(const T* input, T* output, int m, int n, bool reset_initial_output)
INSTANTIATE_REDUCE_MATRIX_ROWS(half);
INSTANTIATE_REDUCE_MATRIX_ROWS(float);
INSTANTIATE_REDUCE_MATRIX_ROWS(double);
#undef INSTANTIATE_REDUCE_MATRIX_ROWS

template <typename TIn, typename TOut>
Status reduce_matrix_columns(const TIn* input, TOut* output, int m, int n, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Identity2, Identity2, false>(
      input, output, m, n, buffer, buffer_size);
}

#define INSTANTIATE_REDUCE_MATRIX_COLUMNS(T) \
  template Status reduce_matrix_columns<T, T>(const T* input, T* output, int m, int n, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_MATRIX_COLUMNS(half);
INSTANTIATE_REDUCE_MATRIX_COLUMNS(float);
INSTANTIATE_REDUCE_MATRIX_COLUMNS(double);
#undef INSTANTIATE_REDUCE_MATRIX_COLUMNS







//
// TODO: DELETE EVERYTHING BELOW
//

std::pair<int, int> compute_block_size(int size) {
  int x = GPU_WARP_SIZE;
  int y = std::min(NUM_WARPS_PER_BLOCK, std::max(1, size / (NUM_ELEMENTS_PER_THREAD * GPU_WARP_SIZE)));
  return std::make_pair(x, y);
}

int compute_grid_size(int size) {
  const auto block = compute_block_size(size);
  return std::min(MAX_NUM_BLOCKS, std::max(1, size / (NUM_ELEMENTS_PER_THREAD * block.first * block.second)));
}

int compute_reduction_buffer_size(int element_size, int size) {
  const int num_blocks = compute_grid_size(size);
  return static_cast<int>(num_blocks * element_size + sizeof(int));
}

template<typename TIn, typename TOut, typename TOp, typename TFinalOp, bool DivideResultBySize>
__global__ void reduce_all_kernel(const int size, const TIn * data, TOut* output, TOut* buffer) {
  extern __shared__ unsigned char shared_memory_[];
  TOut* shared_memory = reinterpret_cast<TOut*>(shared_memory_);
  // Thread-level indexes:
  // Linear index of thread in block.
  const int tid_in_block = threadIdx.y * blockDim.x + threadIdx.x;
  // Total number of threads in a 2-D block.
  const int num_threads_in_block = blockDim.x * blockDim.y;

  // Warp-level indexes:
  // Warp index of thread.
  const int wid_in_block = tid_in_block / GPU_WARP_SIZE;
  // Lane index of thread.
  const int lid_in_block = tid_in_block % GPU_WARP_SIZE;
  // Warp count per block.
  const int num_warps_in_block = num_threads_in_block / GPU_WARP_SIZE;

  // Grid-level indexes:
  // Linear index of block in grid.
  const int bid_in_grid = blockIdx.x + blockIdx.y * gridDim.x;
  // Linear index of thread in grid.
  const int tid_in_grid = bid_in_grid * (blockDim.x * blockDim.y) + tid_in_block;
  // Total number of blocks in a 2-D grid.
  const int num_blocks_in_grid = gridDim.x * gridDim.y;
  // Total number of threads in a 2-D grid with 2-D blocks.
  const int num_threads_in_grid = num_blocks_in_grid * num_threads_in_block;

  // Thread-level reduction (storage change: global memory -> register).
  // One thread reduces NUM_ELEMENTS_PER_THREAD elements to a thread register
  // in one iteration.
  TOut value = 0;
  for (int id = tid_in_grid; id < size; id += NUM_ELEMENTS_PER_THREAD * num_threads_in_grid) {
    TOut v[NUM_ELEMENTS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
      int offset = id + i * num_threads_in_grid;
      if (offset < size) {
        v[i] = TOut(TOp()(data[offset]));
      } else {
        v[i] = TOut(0.0f);
      }
    }

    #pragma unroll
    for (int i = 0; i < NUM_ELEMENTS_PER_THREAD; i++) {
      value += v[i];
    }
  }

  __syncthreads();

  // Warp-level reduction (storage change: register -> register).
  // The values in a warp will be summed up to a scalar. After warp-level
  // reduction, each block holds num_warps_in_block values in the shared memory.
  TOut value_ = value;
#pragma unroll
  for (int stride = GPU_WARP_SIZE / 2; stride > 0; stride /= 2) {
    value_ += WARP_SHFL_DOWN(value_, stride);
  }

  // Return early if only one warp is used for reduction.
  // Given a fixed amount of threads, we perfer threads over warps over blocks so that we never have cases such as
  // 1. two blocks and each of them has only 1 warp (32 threads).
  // 2. two warps and each of them has only 2 threads.
  if (num_warps_in_block == 1) {
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
  // than num_warps_in_block, so we just keep halving the number
  // of threads in each iteartion. Our assumption is always true because
  // the size of shared_memory equals to the number of warps.
#pragma unroll
  for (int stride = NUM_WARPS_PER_BLOCK / 2; stride > 0; stride /= 2) {
    if (tid_in_block + stride < num_warps_in_block) {
      shared_memory[tid_in_block] += shared_memory[tid_in_block + stride];
    }
    __syncthreads();
  }

  // Return early if only one block is used for reduction.
  if (num_blocks_in_grid == 1) {
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
    buffer[bid_in_grid] = shared_memory[0];
  }

  __threadfence();
  __syncthreads();

  // Grid-level reduciton. We use the last block to sum up values
  // stored in the global buffer.
  __shared__ bool is_last_block_done;

  if (tid_in_block == 0) {
    int* p_lock = reinterpret_cast<int*>(buffer + num_blocks_in_grid);
    int count = atomicAdd(p_lock, 1);
    is_last_block_done = (count == (num_blocks_in_grid - 1));
  }

  // All threads in each block see if they belong the last active block
  // (i.e., the value of is_last_block_done).
  __syncthreads();

  // Only the block which saw that count equals to num_blocks_in_grid - 1 can
  // enter the following block.
  if (is_last_block_done) {
    const int pow2_bound = least_pow2_bound(num_blocks_in_grid);
    for (int stride = pow2_bound / 2; stride > 0; stride /= 2) {
      if (tid_in_block < stride && tid_in_block + stride < num_blocks_in_grid) {
        buffer[tid_in_block] += buffer[tid_in_block + stride];
      }
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
void call_reduce_all_kernel(const TIn *data, TOut *output, int size, TOut *buffer) {
  const auto block_size = compute_block_size(size);
  const int num_blocks = compute_grid_size(size);
  const dim3 block(block_size.first, block_size.second, 1);
  const dim3 grid(num_blocks, 1, 1);

  // If more than one blocks are used, then inter-blocks reduction is needed.
  if (num_blocks != 1) {
    HIP_CALL_THROW(hipMemsetAsync(buffer + num_blocks, 0, sizeof(int)));
  }

  const int shared_mem_size = sizeof(TOut) * block_size.first * block_size.second / GPU_WARP_SIZE;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_all_kernel<TIn, TOut, TOp, TFinalOp, DivideResultBySize>), dim3(grid), dim3(block), shared_mem_size, 0, size, data, output, buffer);
}

template <typename TIn, typename TOut>
void reduce_sum(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Cast<TOut, TIn>, Identity<TOut>, false>(
      data, output, size, buffer);
}

template <typename TIn, typename TOut>
void reduce_square_sum(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Square<TOut, TIn>, Identity<TOut>, false>(
      data, output, size, buffer);
}

template <typename TIn, typename TOut>
void reduce_l2_norm(const TIn* data, TOut* output, int size, TOut* buffer) {
  call_reduce_all_kernel<TIn, TOut, Square<TOut, TIn>, Sqrt<TOut>, false>(
      data, output, size, buffer);
}

template <typename TIn, typename TOut>
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

template <typename TIn, typename TOut, typename TBuf>
__global__ void reduce_matrix_rows_kernel(const TIn* input, TOut* output, int m, int n) {
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
  HIP_DYNAMIC_SHARED( unsigned char, shared_memory_)
  TBuf* shared_memory = reinterpret_cast<TBuf*>(shared_memory_);

  // to prevent int overflow in index calculation for input size m*n
  const int64_t n_int64 = static_cast<int64_t>(n);

  for (int col = tid_x_in_grid; col < n; col += x_grid_stride) {
    shared_memory[tid_in_block] = TBuf(0.0f);
    TBuf sum = TBuf(0.0f);

    // This loops load multiple blockDim.y-by-blockDim.x sub-tensors from the input.
    for (int row = tid_y_in_grid; row < m; row += y_grid_stride) {
      // Thread-level reduction. Each thread loads y_load_count_per_thread values
      // and aggregrate them.
#pragma unroll(y_load_count_per_thread)
      for (int row_inner = 0; row_inner < y_load_count_per_thread; ++row_inner) {
        int row_final = row + row_inner * t_count_y_in_grid;
        int col_final = col;
        if (row_final < m && col_final < n) {
          sum += TBuf(input[row_final * n_int64 + col_final]);
        }
      }
    }

    // Write thread-level reduction result into shared memory.
    shared_memory[tid_in_block] = sum;

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
  }
}

// This function reduces the given input tensor along all but the last axis.
// For example, [N, C, H, W]-tensor may lead to a output [W]-tensor.
// It's implementation is in reduction_ops.cu and called in reduction_ops.cc.
template <typename TIn, typename TOut, typename TBuf>
void call_reduce_matrix_rows(const TIn* input, TOut* output, int m, int n) {
  constexpr int max_num_threads_in_block = 512;
  constexpr int max_num_blocks_in_grid = 512;
  constexpr int load_count_per_thread = 4;

  const int block_x_dim = least_pow2_bound(std::max(1, std::min(n, GPU_WARP_SIZE)));
  const int block_y_dim = least_pow2_bound(std::max(1, std::min(max_num_threads_in_block / block_x_dim, m / load_count_per_thread)));
  const int grid_x_dim = std::max(1, std::min(n / block_x_dim, max_num_blocks_in_grid));
  const int grid_y_dim = std::max(1, std::min(max_num_blocks_in_grid / grid_x_dim, m / block_y_dim / 4));

  const dim3 grid(grid_x_dim, grid_y_dim, 1);
  const dim3 block(block_x_dim, block_y_dim, 1);

  hipLaunchKernelGGL(HIP_KERNEL_NAME(reduce_matrix_rows_kernel<TIn, TOut, TBuf>), dim3(grid), dim3(block), block.y * block.x * sizeof(TBuf), 0, 
      input, output, m, n);
}

template <typename TIn, typename TOut>
void reduce_matrix_rows(const TIn* data, TOut* output, int m, int n) {
  call_reduce_matrix_rows<TIn, TOut, TOut>(data, output, m, n);
}

template <>
void reduce_matrix_rows<half, half>(const half* data, half* output, int m, int n) {
  call_reduce_matrix_rows<half, half, float>(data, output, m, n);
}

template void reduce_matrix_rows<float, float>(
    const float* data, float* output, int m, int n);
template void reduce_matrix_rows<double, double>(
    const double* data, double* output, int m, int n);

}  // namespace rocm
}  // namespace onnxruntime
