// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "tile_impl.h"

namespace onnxruntime {
namespace cuda {

#ifdef USE_ROCM
constexpr int num_elements_per_thread = 2;
constexpr int num_threads_per_block = 512;
#else
constexpr int num_elements_per_thread = GridDim::maxElementsPerThread;
constexpr int num_threads_per_block = GridDim::maxThreadsPerBlock;
#endif

template <typename T>
__global__ void _UnRolledTileKernel(const size_t shape_rank, const TArray<fast_divmod> fdm_input_shape,
                                    const TArray<int64_t> input_strides, const T* input_data,
                                    const TArray<fast_divmod> fdm_output_strides, T* output_data, const CUDA_LONG N) {
  CUDA_LONG start = num_elements_per_thread * num_threads_per_block * blockIdx.x + threadIdx.x;
  T value[num_elements_per_thread];
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < num_elements_per_thread; ++i) {
    if (id < N) {
      CUDA_LONG input_index = 0;
      CUDA_LONG offset = id;
#pragma unroll
      for (auto dim = 0; dim < fdm_output_strides.Capacity(); ++dim) {
        if (dim >= shape_rank) {
          break;
        }

        int out_coord, r;
        fdm_output_strides[dim].divmod(offset, out_coord, r);
        int in_coord = fdm_input_shape[dim].mod(out_coord);
        input_index += input_strides[dim] * in_coord;
        offset = r;
      }

      value[i] = input_data[input_index];
      id += num_threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < num_elements_per_thread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += num_threads_per_block;
    }
  }
}

template <typename T>
void TileImpl(cudaStream_t stream, const size_t shape_rank, const TArray<fast_divmod>& fdm_input_shape,
              const TArray<int64_t>& input_stride, const T* input_data, const TArray<fast_divmod>& fdm_output_strides,
              T* output_data, const size_t N) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N, num_threads_per_block * num_elements_per_thread));
  _UnRolledTileKernel<T><<<blocksPerGrid, num_threads_per_block, 0, stream>>>(shape_rank, fdm_input_shape, input_stride,
                                                                              input_data, fdm_output_strides,
                                                                              output_data, static_cast<CUDA_LONG>(N));
}

template <typename T>
__global__ void _TileMemcpyKernelFromOutput(const T* input_data, T* output_data,
                                            const fast_divmod divmod_num_input_elements, const CUDA_LONG N) {
  CUDA_LONG start = num_elements_per_thread * num_threads_per_block * blockIdx.x + threadIdx.x;
  T value[num_elements_per_thread];
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < num_elements_per_thread; ++i) {
    if (id < N) {
      value[i] = input_data[divmod_num_input_elements.mod(id)];
      id += num_threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < num_elements_per_thread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += num_threads_per_block;
    }
  }
}

template <typename T>
__global__ void _TileMemcpyKernelFromInput(const T* input_data, T* output_data, const CUDA_LONG N,
                                           const size_t repeats) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  T input_val = input_data[id];
  for (size_t i = 0; i < repeats; ++i) {
    output_data[id] = input_val;
    id += N;
  }
}

template <typename T>
size_t GetVectorizedSize(size_t num_input_elements, size_t num_elements_per_batch, uint64_t address_input,
                         uint64_t address_output, CUDA_LONG& N, int& blocksPerGrid) {
  constexpr int vec4_alignment = std::alignment_of<aligned_vector<T, 4>>::value;
  constexpr int vec2_alignment = std::alignment_of<aligned_vector<T, 2>>::value;
  N = static_cast<CUDA_LONG>(num_input_elements);
  size_t vectorized_size = 1;
  if (num_elements_per_batch % 4 == 0 && address_input % vec4_alignment == 0 && address_output % vec4_alignment == 0) {
    N /= 4;
    vectorized_size = 4;
  } else if (num_elements_per_batch % 2 == 0 && address_input % vec2_alignment == 0 &&
             address_output % vec2_alignment == 0) {
    N /= 2;
    vectorized_size = 2;
  }
  blocksPerGrid = CeilDiv(N, num_threads_per_block);
  return vectorized_size;
}

template <typename T>
void TileMemcpyImpl(cudaStream_t stream, const T* input_data, T* output_data, const size_t num_input_elements,
                    const size_t repeats) {
  // If the block number from input size is too small to fill all streaming multiprocessors,
  // it won't have perf gain to launch from inputs. In this case we will use the output based kernel.
  CUDA_LONG N;
  int blocksPerGrid;
  size_t vectorized_size =
      GetVectorizedSize<T>(num_input_elements, num_input_elements, reinterpret_cast<uint64_t>(input_data),
                           reinterpret_cast<uint64_t>(output_data), N, blocksPerGrid);
  if (blocksPerGrid < 128) {
    N = static_cast<CUDA_LONG>(num_input_elements * repeats);
    blocksPerGrid = CeilDiv(N, num_threads_per_block * num_elements_per_thread);
    _TileMemcpyKernelFromOutput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
        input_data, output_data, fast_divmod(static_cast<int>(num_input_elements)), N);
    return;
  }

  if (vectorized_size == 4) {
    using Vec4T = aligned_vector<T, 4>;
    _TileMemcpyKernelFromInput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
        reinterpret_cast<const Vec4T*>(input_data), reinterpret_cast<Vec4T*>(output_data), N, repeats);
    return;
  } else if (vectorized_size == 2) {
    using Vec2T = aligned_vector<T, 2>;
    _TileMemcpyKernelFromInput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
        reinterpret_cast<const Vec2T*>(input_data), reinterpret_cast<Vec2T*>(output_data), N, repeats);
    return;
  }

  _TileMemcpyKernelFromInput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(input_data, output_data, N, repeats);
}

template <typename T>
__global__ void _TileBatchedMemcpyKernelFromOutput(const T* input_data, T* output_data,
                                                   const fast_divmod divmod_size_output_row,
                                                   const size_t size_input_row, const fast_divmod divmod_batch,
                                                   const fast_divmod divmod_size_input_row, const CUDA_LONG N) {
  CUDA_LONG start = num_elements_per_thread * num_threads_per_block * blockIdx.x + threadIdx.x;
  T value[num_elements_per_thread];
  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < num_elements_per_thread; ++i) {
    if (id < N) {
      int batch_idx, element_idx;
      divmod_size_output_row.divmod(id, batch_idx, element_idx);
      value[i] = input_data[divmod_batch.mod(batch_idx) * size_input_row + divmod_size_input_row.mod(element_idx)];
      id += num_threads_per_block;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < num_elements_per_thread; ++i) {
    if (id < N) {
      output_data[id] = value[i];
      id += num_threads_per_block;
    }
  }
}

// Input size is [batch, data], output size is [batch * batch_repeats, data * repeats_per_batch].
// Here size_input_row = data, size_output_row = data * repeats_per_batch,
// size_output_batch = batch * data * repeats_per_batch
template <typename T>
__global__ void _TileBatchedMemcpyKernelFromInput(const T* input_data, T* output_data,
                                                  const fast_divmod divmod_size_input_row,
                                                  const CUDA_LONG size_input_row, const CUDA_LONG size_output_row,
                                                  const CUDA_LONG size_output_batch, const size_t batch_repeats,
                                                  const size_t repeats_per_batch, const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  T input_val = input_data[id];
  CUDA_LONG q, r;
  divmod_size_input_row.divmod(id, q, r);
  CUDA_LONG batch_offset = q * size_output_row + r;
  for (size_t i = 0; i < batch_repeats; ++i) {
    CUDA_LONG offset = batch_offset;
    for (size_t j = 0; j < repeats_per_batch; ++j) {
      output_data[offset] = input_val;
      offset += size_input_row;
    }
    batch_offset += size_output_batch;
  }
}

// Input size is [batch, data], output size is [batch * batch_repeats, data * repeats_per_batch].
// Here size_input_row = data, num_input_elements = batch * data
template <typename T>
void TileBatchedMemcpyImpl(cudaStream_t stream, const T* input_data, T* output_data, const size_t size_input_row,
                           const size_t num_input_elements, const size_t batch_repeats,
                           const size_t repeats_per_batch) {
  // If the block number from input size is too small to fill all streaming multiprocessors,
  // it won't have perf gain to launch from inputs. In this case we will use the output based kernel.
  CUDA_LONG N;
  int blocksPerGrid;
  size_t vectorized_size =
      GetVectorizedSize<T>(num_input_elements, size_input_row, reinterpret_cast<uint64_t>(input_data),
                           reinterpret_cast<uint64_t>(output_data), N, blocksPerGrid);
  if (blocksPerGrid < 128) {
    N = static_cast<CUDA_LONG>(num_input_elements * batch_repeats * repeats_per_batch);
    blocksPerGrid = CeilDiv(N, num_threads_per_block * num_elements_per_thread);
    _TileBatchedMemcpyKernelFromOutput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
        input_data, output_data, fast_divmod(static_cast<int>(size_input_row * repeats_per_batch)), size_input_row,
        fast_divmod(static_cast<int>(num_input_elements / size_input_row)),
        fast_divmod(static_cast<int>(size_input_row)), N);
    return;
  }

  CUDA_LONG size_input_row_vec = static_cast<CUDA_LONG>(size_input_row);
  if (vectorized_size == 4) {
    using Vec4T = aligned_vector<T, 4>;
    size_input_row_vec /= 4;
    _TileBatchedMemcpyKernelFromInput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
        reinterpret_cast<const Vec4T*>(input_data), reinterpret_cast<Vec4T*>(output_data),
        fast_divmod(size_input_row_vec), size_input_row_vec,
        size_input_row_vec * static_cast<CUDA_LONG>(repeats_per_batch), N * static_cast<CUDA_LONG>(repeats_per_batch),
        batch_repeats, repeats_per_batch, N);
    return;
  } else if (vectorized_size == 2) {
    using Vec2T = aligned_vector<T, 2>;
    size_input_row_vec /= 2;
    _TileBatchedMemcpyKernelFromInput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
        reinterpret_cast<const Vec2T*>(input_data), reinterpret_cast<Vec2T*>(output_data),
        fast_divmod(size_input_row_vec), size_input_row_vec,
        size_input_row_vec * static_cast<CUDA_LONG>(repeats_per_batch), N * static_cast<CUDA_LONG>(repeats_per_batch),
        batch_repeats, repeats_per_batch, N);
    return;
  }

  _TileBatchedMemcpyKernelFromInput<<<blocksPerGrid, num_threads_per_block, 0, stream>>>(
      input_data, output_data, fast_divmod(size_input_row_vec), size_input_row_vec,
      size_input_row_vec * static_cast<CUDA_LONG>(repeats_per_batch), N * static_cast<CUDA_LONG>(repeats_per_batch),
      batch_repeats, repeats_per_batch, N);
}

#define SPECIALIZED_IMPL(T)                                                                                           \
  template void TileImpl<T>(cudaStream_t stream, const size_t shape_rank, const TArray<fast_divmod>& fdm_input_shape, \
                            const TArray<int64_t>& input_stride, const T* input_data,                                 \
                            const TArray<fast_divmod>& fdm_output_strides, T* output_data, const size_t N);           \
  template void TileMemcpyImpl<T>(cudaStream_t stream, const T* input_data, T* output_data,                           \
                                  const size_t num_input_elements, const size_t repeats);                             \
  template void TileBatchedMemcpyImpl<T>(cudaStream_t stream, const T* input_data, T* output_data,                    \
                                         const size_t size_input_row, const size_t num_input_elements,                \
                                         const size_t batch_repeats, const size_t repeats_per_batch);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
