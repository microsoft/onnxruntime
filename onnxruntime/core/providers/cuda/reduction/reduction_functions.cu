// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/reduction/reduction_functions.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp16.h>
#include "core/common/common.h"
#include "core/providers/cuda/atomic/common.cuh"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/reduction/reduction_utils.cuh"
#include "core/providers/cuda/cu_inc/unary_elementwise_impl.cuh"

namespace onnxruntime {
namespace cuda {

namespace detail {
constexpr auto MAX_NUM_ELEMENTS_PER_THREAD = 4;
constexpr auto MAX_NUM_WARPS_PER_BLOCK = 8;
constexpr auto MAX_NUM_BLOCKS_IN_GRID_ROW = 256;
constexpr auto MAX_NUM_GRID_ROWS = 32768;

dim3 compute_block_dim(int num_cols) {
  const int x = GPU_WARP_SIZE_HOST;
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

#if __CUDA_ARCH__ >= 700
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
    if (tid_in_block < stride && tid_in_block + stride < num_warps_in_block) {
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
    cudaStream_t stream, const TIn* input, TOut* output, const int num_rows, const int num_cols, void* buffer, size_t buffer_size) {
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
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(block_done_counts_buffer, 0, num_rows * sizeof(int), stream));
  }

  const int shared_mem_size = sizeof(TBuf) * block_dim.x * block_dim.y / GPU_WARP_SIZE_HOST;
  reduce_matrix_columns_kernel<TIn, TOut, TBuf, TOp, TFinalOp, DivideResultBySize>
      <<<grid_dim, block_dim, shared_mem_size, stream>>>(
          num_rows, num_cols, input, output, block_reductions_buffer, block_done_counts_buffer);

  return Status::OK();
}
}  // namespace detail

template <typename TIn, typename TOut>
Status reduce_sum(
    cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Identity, Identity, false>(
      stream, input, output, 1, size, buffer, buffer_size);
}

template <typename TIn, typename TOut>
Status reduce_square_sum(
    cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Square, Identity, false>(
      stream, input, output, 1, size, buffer, buffer_size);
}

template <typename TIn, typename TOut>
Status reduce_l2_norm(
    cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Square, Sqrt, false>(
      stream, input, output, 1, size, buffer, buffer_size);
}

template <typename TIn, typename TOut>
Status reduce_mean(
    cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Identity, Identity, true>(
      stream, input, output, 1, size, buffer, buffer_size);
}

#define INSTANTIATE_REDUCE_SUM(TIn, TOut) \
  template Status reduce_sum<TIn, TOut>(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_SUM(half, half);
INSTANTIATE_REDUCE_SUM(half, float);
INSTANTIATE_REDUCE_SUM(float, float);
INSTANTIATE_REDUCE_SUM(double, double);
INSTANTIATE_REDUCE_SUM(BFloat16, BFloat16);
INSTANTIATE_REDUCE_SUM(BFloat16, float);
#undef INSTANTIATE_REDUCE_SUM

#define INSTANTIATE_REDUCE_SQUARE_SUM(TIn, TOut) \
  template Status reduce_square_sum<TIn, TOut>(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_SQUARE_SUM(half, float);
INSTANTIATE_REDUCE_SQUARE_SUM(float, float);
INSTANTIATE_REDUCE_SQUARE_SUM(double, double);
INSTANTIATE_REDUCE_SQUARE_SUM(BFloat16, float);
#undef INSTANTIATE_REDUCE_SQUARE_SUM

#define INSTANTIATE_REDUCE_L2_NORM(TIn, TOut) \
  template Status reduce_l2_norm<TIn, TOut>(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_L2_NORM(half, float);
INSTANTIATE_REDUCE_L2_NORM(float, float);
INSTANTIATE_REDUCE_L2_NORM(double, double);
#undef INSTANTIATE_REDUCE_L2_NORM

#define INSTANTIATE_REDUCE_MEAN(TIn, TOut) \
  template Status reduce_mean<TIn, TOut>(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size)
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
#pragma unroll y_load_count_per_thread
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
#pragma unroll 4
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
Status call_reduce_matrix_rows(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, bool reset_initial_output) {
  ORT_ENFORCE(m >= 0 && n >= 0);

  if (reset_initial_output) {
    CUDA_RETURN_IF_ERROR(cudaMemsetAsync(output, 0, n * sizeof(TOut), stream));
  }

  constexpr int max_num_threads_in_block = 512;
  constexpr int max_num_blocks_in_grid = 512;
  constexpr int load_count_per_thread = 4;

  const int block_x_dim = least_pow2_bound(std::max(1, std::min(n, GPU_WARP_SIZE_HOST)));
  const int block_y_dim = least_pow2_bound(std::max(1, std::min(max_num_threads_in_block / block_x_dim, m / load_count_per_thread)));
  const int grid_x_dim = std::max(1, std::min(n / block_x_dim, max_num_blocks_in_grid));
  const int grid_y_dim = std::max(1, std::min(max_num_blocks_in_grid / grid_x_dim, m / block_y_dim / 4));

  const dim3 grid(grid_x_dim, grid_y_dim, 1);
  const dim3 block(block_x_dim, block_y_dim, 1);

  reduce_matrix_rows_kernel<TIn, TOut, TBuf><<<grid, block, block.y * block.x * sizeof(TBuf), stream>>>(
      input, output, m, n);

  return Status::OK();
}
}  // namespace detail

template <typename T>
struct OP_Div {
  __device__ __inline__ T operator()(const T& a) const {
    return a / v_;
  }

  OP_Div(T v) : v_(v) {}

  T v_;
};

template <typename T>
void UnaryDiv(cudaStream_t stream, const T* input, T* output, T denominator, size_t count) {
  UnaryElementWiseImpl(stream, input, output, OP_Div<T>(denominator), count);
}

#define INSTANTIATE_UNARY_DIV(T) \
  template void UnaryDiv<T>(cudaStream_t stream, const T* input, T* output, T denominator, size_t count)
INSTANTIATE_UNARY_DIV(half);
INSTANTIATE_UNARY_DIV(float);
INSTANTIATE_UNARY_DIV(double);
INSTANTIATE_UNARY_DIV(BFloat16);
#undef INSTANTIATE_UNARY_DIV

template <typename TIn, typename TOut>
Status reduce_matrix_rows(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, bool reset_initial_output) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::call_reduce_matrix_rows<TIn, TOut, TBuf>(stream, input, output, m, n, reset_initial_output);
}

#define INSTANTIATE_REDUCE_MATRIX_ROWS(T) \
  template Status reduce_matrix_rows<T, T>(cudaStream_t stream, const T* input, T* output, int m, int n, bool reset_initial_output)
INSTANTIATE_REDUCE_MATRIX_ROWS(half);
INSTANTIATE_REDUCE_MATRIX_ROWS(float);
INSTANTIATE_REDUCE_MATRIX_ROWS(double);
INSTANTIATE_REDUCE_MATRIX_ROWS(BFloat16);
#undef INSTANTIATE_REDUCE_MATRIX_ROWS

template <typename TIn, typename TOut>
Status reduce_matrix_columns(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, void* buffer, size_t buffer_size) {
  return detail::call_reduce_matrix_columns<TIn, TOut, Identity, Identity, false>(
      stream, input, output, m, n, buffer, buffer_size);
}

#define INSTANTIATE_REDUCE_MATRIX_COLUMNS(T) \
  template Status reduce_matrix_columns<T, T>(cudaStream_t stream, const T* input, T* output, int m, int n, void* buffer, size_t buffer_size)
INSTANTIATE_REDUCE_MATRIX_COLUMNS(half);
INSTANTIATE_REDUCE_MATRIX_COLUMNS(float);
INSTANTIATE_REDUCE_MATRIX_COLUMNS(double);
INSTANTIATE_REDUCE_MATRIX_COLUMNS(BFloat16);
#undef INSTANTIATE_REDUCE_MATRIX_COLUMNS

namespace detail {
constexpr int kMaxReduceRank = 16;

struct ReduceSumNdMetadata {
  int output_segment_count{};
  int reduction_segment_count{};
  int64_t output_segment_sizes[kMaxReduceRank]{};
  int64_t output_segment_strides[kMaxReduceRank]{};
  int64_t reduction_segment_sizes[kMaxReduceRank]{};
  int64_t reduction_segment_strides[kMaxReduceRank]{};
  int64_t output_count{};
  int64_t reduction_count{};
};

template <typename T>
struct SumState {
  T sum{};

  __device__ __forceinline__ void Add(T value) { sum += value; }
  __device__ __forceinline__ T Result() const { return sum; }
};

template <>
struct SumState<double> {
  double sum{};
  double correction{};

  // Neumaier compensation is used instead of Kahan so that independently accumulated
  // thread partials can be merged without losing a small value between large values.
  __device__ __forceinline__ void Add(double value) {
    const double next = __dadd_rn(sum, value);
    const double error = fabs(sum) >= fabs(value)
                             ? __dadd_rn(__dsub_rn(sum, next), value)
                             : __dadd_rn(__dsub_rn(value, next), sum);
    correction = __dadd_rn(correction, error);
    sum = next;
  }

  __device__ __forceinline__ double Result() const { return __dadd_rn(sum, correction); }
};

template <typename T>
struct MergeSumState {
  __device__ __forceinline__ SumState<T> operator()(SumState<T> lhs, const SumState<T>& rhs) const {
    lhs.Add(rhs.sum);
    if constexpr (std::is_same_v<T, double>) {
      lhs.Add(rhs.correction);
    }
    return lhs;
  }
};

template <typename T, typename TAccum>
__device__ __forceinline__ T CastReduceSumResult(TAccum value) {
  if constexpr (std::is_integral_v<T>) {
    const double value_as_double = static_cast<double>(value);
    const double max_value = static_cast<double>(std::numeric_limits<T>::max());
    const double min_value = static_cast<double>(std::numeric_limits<T>::min());
    if (value_as_double >= max_value) return std::numeric_limits<T>::max();
    if (value_as_double <= min_value) return std::numeric_limits<T>::min();
  }
  return static_cast<T>(value);
}

template <typename T, int BlockSize>
__global__ void reduce_sum_nd_kernel(const T* input, T* output, ReduceSumNdMetadata metadata) {
  using TAccum = std::conditional_t<std::is_integral_v<T>, double, AccumulationType_t<T>>;
  using BlockReduce = cub::BlockReduce<SumState<TAccum>, BlockSize>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ int64_t input_base;

  // One cooperative block reduces each output. Grid-striding keeps the launch bounded for
  // large outputs while retaining enough blocks to saturate the device.
  for (int64_t output_index = blockIdx.x;
       output_index < metadata.output_count;
       output_index += gridDim.x) {
    if (threadIdx.x == 0) {
      int64_t remaining = output_index;
      int64_t base = 0;
      for (int segment = metadata.output_segment_count - 1; segment >= 0; --segment) {
        const int64_t coordinate = segment == 0 ? remaining : remaining % metadata.output_segment_sizes[segment];
        if (segment != 0) remaining /= metadata.output_segment_sizes[segment];
        base += coordinate * metadata.output_segment_strides[segment];
      }
      input_base = base;
    }
    __syncthreads();

    SumState<TAccum> thread_sum{};
    for (int64_t reduction_index = threadIdx.x; reduction_index < metadata.reduction_count;
         reduction_index += BlockSize) {
      int64_t remaining = reduction_index;
      int64_t input_index = input_base;
      // Adjacent reduced dimensions are collapsed on the host, so this loop performs
      // divisions only at reduced/non-reduced boundaries rather than once per rank.
      for (int segment = metadata.reduction_segment_count - 1; segment >= 0; --segment) {
        const int64_t coordinate = segment == 0 ? remaining : remaining % metadata.reduction_segment_sizes[segment];
        if (segment != 0) remaining /= metadata.reduction_segment_sizes[segment];
        input_index += coordinate * metadata.reduction_segment_strides[segment];
      }
      thread_sum.Add(static_cast<TAccum>(input[input_index]));
    }

    const SumState<TAccum> block_sum = BlockReduce(reduce_storage).Reduce(thread_sum, MergeSumState<TAccum>{});
    if (threadIdx.x == 0) {
      output[output_index] = CastReduceSumResult<T>(block_sum.Result());
    }
    __syncthreads();  // reduce_storage is reused by the next grid-stride iteration.
  }
}
}  // namespace detail

template <typename T>
Status reduce_sum_nd(cudaStream_t stream, const T* input, T* output,
                     gsl::span<const int64_t> dims, gsl::span<const int64_t> axes) {
  ORT_RETURN_IF_NOT(dims.size() <= detail::kMaxReduceRank,
                    "The general CUDA ReduceSum kernel supports ranks up to ", detail::kMaxReduceRank, ".");

  detail::ReduceSumNdMetadata metadata;
  const int rank = gsl::narrow_cast<int>(dims.size());
  std::array<int64_t, detail::kMaxReduceRank> strides{};
  SafeInt<int64_t> stride = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    ORT_RETURN_IF_NOT(dims[axis] > 0, "ReduceSum dimensions must be positive.");
    strides[axis] = static_cast<int64_t>(stride);
    stride *= dims[axis];
  }

  std::array<bool, detail::kMaxReduceRank> reduced{};
  if (axes.empty()) {
    for (int axis = 0; axis < rank; ++axis) reduced[axis] = true;
  } else {
    for (int64_t axis : axes) {
      if (axis < 0) axis += rank;
      ORT_RETURN_IF_NOT(axis >= 0 && axis < rank, "ReduceSum axis is out of range.");
      ORT_RETURN_IF_NOT(!reduced[axis], "ReduceSum axes must not contain duplicates.");
      reduced[axis] = true;
    }
  }

  SafeInt<int64_t> output_count = 1;
  SafeInt<int64_t> reduction_count = 1;
  for (int axis = 0; axis < rank;) {
    const bool is_reduced = reduced[axis];
    SafeInt<int64_t> segment_size = 1;
    int last_axis = axis;
    do {
      segment_size *= dims[axis];
      last_axis = axis++;
    } while (axis < rank && reduced[axis] == is_reduced);

    if (is_reduced) {
      const int segment = metadata.reduction_segment_count++;
      metadata.reduction_segment_sizes[segment] = static_cast<int64_t>(segment_size);
      metadata.reduction_segment_strides[segment] = strides[last_axis];
      reduction_count *= static_cast<int64_t>(segment_size);
    } else {
      const int segment = metadata.output_segment_count++;
      metadata.output_segment_sizes[segment] = static_cast<int64_t>(segment_size);
      metadata.output_segment_strides[segment] = strides[last_axis];
      output_count *= static_cast<int64_t>(segment_size);
    }
  }
  metadata.output_count = static_cast<int64_t>(output_count);
  metadata.reduction_count = static_cast<int64_t>(reduction_count);

  constexpr int block_size = 256;
  constexpr int max_blocks = 65535;
  const int grid_size = static_cast<int>(std::min<int64_t>(max_blocks, metadata.output_count));
  detail::reduce_sum_nd_kernel<T, block_size><<<grid_size, block_size, 0, stream>>>(input, output, metadata);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_REDUCE_SUM_ND(T)                                               \
  template Status reduce_sum_nd<T>(cudaStream_t stream, const T* input, T* output, \
                                   gsl::span<const int64_t> dims, gsl::span<const int64_t> axes)
INSTANTIATE_REDUCE_SUM_ND(half);
INSTANTIATE_REDUCE_SUM_ND(float);
INSTANTIATE_REDUCE_SUM_ND(double);
INSTANTIATE_REDUCE_SUM_ND(BFloat16);
INSTANTIATE_REDUCE_SUM_ND(int32_t);
INSTANTIATE_REDUCE_SUM_ND(int64_t);
#undef INSTANTIATE_REDUCE_SUM_ND

namespace detail {
template <typename TIn, bool IsArgMax>
__global__ void arg_min_max_last_axis_kernel(const TIn* input, int64_t* output, int m, int n) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= m) return;

  const int64_t row_offset = static_cast<int64_t>(row) * n;
  TIn best_value = input[row_offset];
  int64_t best_index = 0;
  for (int i = 1; i < n; ++i) {
    const TIn value = input[row_offset + i];
    if constexpr (IsArgMax) {
      if (value > best_value) {
        best_value = value;
        best_index = i;
      }
    } else {
      if (value < best_value) {
        best_value = value;
        best_index = i;
      }
    }
  }

  output[row] = best_index;
}
}  // namespace detail

template <typename TIn, bool IsArgMax>
Status arg_min_max_last_axis(cudaStream_t stream, const TIn* input, int64_t* output, int m, int n) {
  // The kernel reads input[row_offset] unconditionally, so a non-empty reduction axis is required.
  if (m == 0 || n <= 0) return Status::OK();
  constexpr int block_size = 256;
  const int grid_size = (m + block_size - 1) / block_size;
  detail::arg_min_max_last_axis_kernel<TIn, IsArgMax><<<grid_size, block_size, 0, stream>>>(input, output, m, n);
  return CUDA_CALL(cudaGetLastError());
}

#define INSTANTIATE_ARG_MIN_MAX_LAST_AXIS(T)                                                                          \
  template Status arg_min_max_last_axis<T, true>(cudaStream_t stream, const T* input, int64_t* output, int m, int n); \
  template Status arg_min_max_last_axis<T, false>(cudaStream_t stream, const T* input, int64_t* output, int m, int n)
INSTANTIATE_ARG_MIN_MAX_LAST_AXIS(half);
INSTANTIATE_ARG_MIN_MAX_LAST_AXIS(float);
INSTANTIATE_ARG_MIN_MAX_LAST_AXIS(double);
#undef INSTANTIATE_ARG_MIN_MAX_LAST_AXIS

}  // namespace cuda
}  // namespace onnxruntime

// =============================================================================
// Saturating absolute value for norm no-op reductions.
//
// When reducing a singleton axis (input_count == output_count), the cuDNN
// workaround copies data directly. For NORM1/NORM2 the result must be
// non-negative. The standard _Abs(a) uses `a > 0 ? a : -a` which is undefined
// behavior for the minimum value of signed types (e.g., INT32_MIN) because
// -INT32_MIN overflows int32.
//
// This saturating variant clamps the result to numeric_limits<T>::max() when
// the true absolute value is unrepresentable. This is mathematically the
// closest value within the type's range.
// =============================================================================
namespace onnxruntime {
namespace cuda {

template <typename T>
struct OP_SaturatingAbs {
  __device__ __inline__ T operator()(const T& a) const {
    if constexpr (std::is_signed_v<T> && std::is_integral_v<T>) {
      // For the minimum value of a signed type, -a overflows.
      // Saturate to max representable value instead.
      if (a == std::numeric_limits<T>::min()) {
        return std::numeric_limits<T>::max();
      }
    }
    return a > T(0) ? a : -a;
  }
};

template <typename T>
void Impl_SaturatingAbs(cudaStream_t stream, const T* input_data, T* output_data, size_t count) {
  UnaryElementWiseImpl(stream, input_data, output_data, OP_SaturatingAbs<T>(), count);
}

/**
 * Saturating cast from double to integer type T.
 *
 * A plain C++ cast from double to an integer type is undefined behavior when the value
 * exceeds the integer's representable range (C++ standard [conv.fpint]/1). While some
 * CUDA compiler/PTX combinations may produce saturating behavior in practice, this is
 * NOT guaranteed by the CUDA specification. This functor explicitly clamps the double
 * to [numeric_limits<T>::min(), numeric_limits<T>::max()] before casting, making the
 * result well-defined and matching the CPU side's explicit clamping logic.
 */
template <typename T>
struct OP_SaturatingCastFromDouble {
  __device__ __inline__ T operator()(const double& a) const {
    constexpr double t_max = static_cast<double>(std::numeric_limits<T>::max());
    constexpr double t_min = static_cast<double>(std::numeric_limits<T>::min());
    if (a >= t_max) return std::numeric_limits<T>::max();
    if (a <= t_min) return std::numeric_limits<T>::min();
    // NaN: neither >= t_max nor <= t_min, falls through to cast.
    // static_cast<T>(NaN) is UB in C++, but cuDNN reductions on finite inputs
    // cannot produce NaN for norm/sum operations, so this path is unreachable.
    return static_cast<T>(a);
  }
};

template <typename T>
void Impl_SaturatingCastFromDouble(cudaStream_t stream, const double* input_data, T* output_data, size_t count) {
  UnaryElementWiseImpl(stream, input_data, output_data, OP_SaturatingCastFromDouble<T>(), count);
}

// Explicit instantiations for types used by ReduceL1/L2 on CUDA.
// The SPECIALIZED_REDUCEKERNEL_COMPUTEIMPL macro expands for int32_t, int64_t, int8_t, uint8_t.
// Even though ReduceL1/L2 aren't registered for int64/int8/uint8, the runtime check
// on cudnn_reduce_op causes the compiler to emit the call, so the linker needs symbols.
template void Impl_SaturatingAbs<float>(cudaStream_t, const float*, float*, size_t);
template void Impl_SaturatingAbs<double>(cudaStream_t, const double*, double*, size_t);
template void Impl_SaturatingAbs<half>(cudaStream_t, const half*, half*, size_t);
template void Impl_SaturatingAbs<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, size_t);
template void Impl_SaturatingAbs<int32_t>(cudaStream_t, const int32_t*, int32_t*, size_t);
template void Impl_SaturatingAbs<int64_t>(cudaStream_t, const int64_t*, int64_t*, size_t);
template void Impl_SaturatingAbs<int8_t>(cudaStream_t, const int8_t*, int8_t*, size_t);
template void Impl_SaturatingAbs<uint8_t>(cudaStream_t, const uint8_t*, uint8_t*, size_t);

template void Impl_SaturatingCastFromDouble<int32_t>(cudaStream_t, const double*, int32_t*, size_t);
template void Impl_SaturatingCastFromDouble<int64_t>(cudaStream_t, const double*, int64_t*, size_t);
template void Impl_SaturatingCastFromDouble<int8_t>(cudaStream_t, const double*, int8_t*, size_t);
template void Impl_SaturatingCastFromDouble<uint8_t>(cudaStream_t, const double*, uint8_t*, size_t);

// FixExpForReduceLogSumExp: after computing exp(X - max), fix NaN results
// that arise from inf - inf or -inf - (-inf).
// For each element:
//   - If original X[i] is -inf: exp should be 0 (exp(-inf) = 0)
//   - If original X[i] is +inf: exp should be 1 (exp(+inf - +inf) = exp(0) = 1)
//   - If original X[i] is NaN: preserve NaN (propagate)
//   - Otherwise: keep exp_result[i] as-is
template <typename T>
__global__ void _FixExpForReduceLogSumExp(const T* original_input, T* exp_result, size_t count) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) return;
  if (_IsNan<T>{}(exp_result[idx])) {
    T x = original_input[idx];
    if (_IsInf<T, true, true>{}(x)) {
      // +inf input: exp(+inf - (+inf)) should be 1
      // -inf input: exp(-inf - (-inf)) should be 0
      exp_result[idx] = _IsInf<T, true, false>{}(x) ? T(1) : T(0);
    }
    // else: x is NaN, keep NaN in exp_result (propagate)
  }
}

template <typename T>
void Impl_FixExpForReduceLogSumExp(cudaStream_t stream, const T* original_input,
                                   T* exp_result, size_t count) {
  if (count == 0) return;
  constexpr int block_size = 256;
  int grid_size = static_cast<int>((count + block_size - 1) / block_size);
  _FixExpForReduceLogSumExp<T><<<grid_size, block_size, 0, stream>>>(original_input, exp_result, count);
}

template void Impl_FixExpForReduceLogSumExp<float>(cudaStream_t, const float*, float*, size_t);
template void Impl_FixExpForReduceLogSumExp<double>(cudaStream_t, const double*, double*, size_t);
template void Impl_FixExpForReduceLogSumExp<half>(cudaStream_t, const half*, half*, size_t);
template void Impl_FixExpForReduceLogSumExp<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, size_t);

}  // namespace cuda
}  // namespace onnxruntime
