// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "transpose_impl.h"

namespace onnxruntime {
namespace cuda {

constexpr unsigned int kNumElementsPerThread = 4;
constexpr unsigned int kTileSize = 32;

// TileSize for current implementation is always 32, but still use template parameter to make it flexible for future.
// For each batch, transpose matrix [m, n] to [n, m].
template <typename T, unsigned int TileSize>
__global__ void Transpose3DKernel(const int64_t m, const int64_t n, const int64_t batch_stride, const T* input_data,
                                  T* output_data) {
  __shared__ T tile[TileSize][TileSize + 1];

  int x = blockIdx.x * TileSize + threadIdx.x;
  int y = blockIdx.y * TileSize + threadIdx.y;

  if (x < n) {
#pragma unroll
    for (unsigned int i = 0; i < TileSize; i += (TileSize / kNumElementsPerThread)) {
      int y_idx = y + i;
      if (y_idx < m) {
        tile[threadIdx.y + i][threadIdx.x] = input_data[blockIdx.z * batch_stride + y_idx * n + x];
      }
    }
  }
  __syncthreads();

  x = blockIdx.y * TileSize + threadIdx.x;
  y = blockIdx.x * TileSize + threadIdx.y;

  if (x < m) {
#pragma unroll
    for (unsigned int i = 0; i < TileSize; i += (TileSize / kNumElementsPerThread)) {
      int y_idx = y + i;
      if (y_idx < n) {
        output_data[blockIdx.z * batch_stride + y_idx * m + x] = tile[threadIdx.x][threadIdx.y + i];
      }
    }
  }
}

bool CanDoTranspose3D(const cudaDeviceProp& prop, size_t rank, const gsl::span<const int64_t>& input_dims,
                      const gsl::span<const size_t>& permutations, dim3& grid_size, dim3& block_size) {
  // Permutation is done in the last two dimensions.
  if (rank == 3 && permutations[rank - 2] == (rank - 1) && permutations[rank - 1] == (rank - 2)) {
    // Normally maxGridSize.x is a large number but maxGridSize.y and maxGridSize.z are limited. Ideally we can check
    // the input sizes to see if a dimension is too large so that we can use grid.x for it to avoid returning false.
    // But this requires different versions of kernel implementation with different index compute logics.
    // Below code is good enough for most of the cases for now, and if we see any case that input_dims[0] or
    // input_dims[1] is too large in the future, we will handle it accordingly.
    int grid_size_x = CeilDiv(static_cast<int>(input_dims[2]), kTileSize);
    int grid_size_y = CeilDiv(static_cast<int>(input_dims[1]), kTileSize);
    int grid_size_z = static_cast<int>(input_dims[0]);

    if (grid_size_x <= prop.maxGridSize[0] && grid_size_y <= prop.maxGridSize[1] &&
        grid_size_z <= prop.maxGridSize[2]) {
      block_size = dim3(kTileSize, kTileSize / kNumElementsPerThread);
      grid_size = dim3(static_cast<unsigned int>(grid_size_x), static_cast<unsigned int>(grid_size_y),
                       static_cast<unsigned int>(grid_size_z));
      return true;
    } else {
      return false;
    }
  }
  return false;
}

#define HANDLE_TRANSPOSE_3D_TILE_DIM(type)                                                                        \
  case sizeof(type): {                                                                                            \
    Transpose3DKernel<type, kTileSize>                                                                            \
        <<<grid_size, block_size, 0, stream>>>(input_shape[1], input_shape[2], input_strides[0],                  \
                                               reinterpret_cast<const ToCudaType<type>::MappedType*>(input_data), \
                                               reinterpret_cast<ToCudaType<type>::MappedType*>(output_data));     \
  } break

Status Transpose3DImpl(cudaStream_t stream, size_t element_size, const TArray<int64_t>& input_shape,
                       const TArray<int64_t>& input_strides, const void* input_data, void* output_data, int64_t /*N*/,
                       const dim3& grid_size, const dim3& block_size) {
  switch (element_size) {
    HANDLE_TRANSPOSE_3D_TILE_DIM(int8_t);
    HANDLE_TRANSPOSE_3D_TILE_DIM(int16_t);
    HANDLE_TRANSPOSE_3D_TILE_DIM(int32_t);
    HANDLE_TRANSPOSE_3D_TILE_DIM(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }

  return Status::OK();
}

template <int element_size>
__global__ void Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim(
    const TArray<int64_t> input_strides, const void* input_data,
    const TArray<int64_t> output_strides, void* output_data,
    int64_t input_shape_2, CUDA_LONG N) {
  // coordinates will be: [d0, d1, d2, d3]
  CUDA_LONG d0 = blockIdx.z;
  CUDA_LONG d1 = blockIdx.y;
  CUDA_LONG d2 = threadIdx.y + blockIdx.x * blockDim.y;
  CUDA_LONG d3 = threadIdx.x;

  CUDA_LONG input_index = (d0 * input_strides[0] +
                           d1 * input_strides[1] +
                           d2 * input_strides[2]) /
                              (4 * sizeof(int) / element_size) +
                          d3 * input_strides[3];

  CUDA_LONG output_index = (d0 * output_strides[0] +
                            d1 * output_strides[1] +
                            d2 * output_strides[2]) /
                               (4 * sizeof(int) / element_size) +
                           d3 * output_strides[3];

  const int4* v_input = reinterpret_cast<const int4*>(input_data);
  int4* v_output = reinterpret_cast<int4*>(output_data);

  if (input_index < N && output_index < N && d2 < input_shape_2) {
    v_output[output_index] = v_input[input_index];
  }
}

bool CanDoTranspose4DParallelizeMultipleElementsPerThreadInInnermostDim(const cudaDeviceProp& prop,
                                                                        size_t element_size,
                                                                        int32_t rank,
                                                                        const gsl::span<const int64_t>& input_dims,
                                                                        const gsl::span<const size_t>& permutations,
                                                                        dim3& grid_size, dim3& block_size) {
  if (rank == 4 &&
      // the permutations is not on the last dimension.
      permutations[3] == 3) {
    unsigned int num_elements_per_thread = 4 * sizeof(int) / static_cast<unsigned int>(element_size);  // int4 is used in the kernel to access data.

    // dims[3]: block.x
    // dims[2]: block.y + grid.x
    // dims[1]: grid.y
    // dims[0]: grid.z
    if (input_dims[3] / num_elements_per_thread <= prop.maxThreadsPerBlock &&
        (input_dims[3] % num_elements_per_thread) == 0 &&
        input_dims[1] <= prop.maxGridSize[1] &&
        input_dims[0] <= prop.maxGridSize[2]) {
      // There are 2 constrains when luanching the kernels
      // 1. block_size_x * block_size_y <= prop.maxThreadsPerBlock
      // 2. block_size_y * num_block_ext >= input_dims[2]
      int64_t block_size_x = input_dims[3] / num_elements_per_thread;
      int64_t max_block_size_y = prop.maxThreadsPerBlock / block_size_x;
      int64_t block_size_y = min(input_dims[2], max_block_size_y);
      int64_t num_block_ext = CeilDiv(input_dims[2], block_size_y);

      if (num_block_ext <= prop.maxGridSize[0]) {
        block_size = dim3(static_cast<unsigned int>(block_size_x), static_cast<unsigned int>(block_size_y));
        grid_size = dim3(static_cast<unsigned int>(num_block_ext),
                         static_cast<unsigned int>(input_dims[1]),
                         static_cast<unsigned int>(input_dims[0]));
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

Status Transpose4DParallelizeMultipleElementsPerThreadInInnermostDim(
    cudaStream_t stream, size_t element_size,
    const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides,
    const void* input_data, const TArray<int64_t>& output_strides,
    void* output_data, int N, const dim3& grid_size, const dim3& block_size) {
  unsigned int num_elements_per_thread = 4 * sizeof(int) / static_cast<unsigned int>(element_size);  // int4 is used in the kernel to access data.

  switch (element_size) {
    case sizeof(int8_t):
      Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim<sizeof(int8_t)>
          <<<grid_size, block_size, 0, stream>>>(
              input_strides, input_data,
              output_strides, output_data,
              input_shape[2],
              N / num_elements_per_thread);
      break;
    case sizeof(int16_t):
      Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim<sizeof(int16_t)>
          <<<grid_size, block_size, 0, stream>>>(
              input_strides, input_data,
              output_strides, output_data,
              input_shape[2],
              N / num_elements_per_thread);
      break;
    case sizeof(int32_t):
      Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim<sizeof(int32_t)>
          <<<grid_size, block_size, 0, stream>>>(
              input_strides, input_data,
              output_strides, output_data,
              input_shape[2],
              N / num_elements_per_thread);
      break;
    case sizeof(int64_t):
      Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim<sizeof(int64_t)>
          <<<grid_size, block_size, 0, stream>>>(
              input_strides, input_data,
              output_strides, output_data,
              input_shape[2],
              N / num_elements_per_thread);
      break;
    default:
      // User will not hit this as this kernel is for fixed element size tensors only
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }

  return Status::OK();
}

__global__ void Transpose4DKernelParallelizeOneElementPerThread(
    const TArray<int64_t> input_strides, const int8_t* input_data,
    const TArray<int64_t> output_strides, int8_t* output_data,
    size_t element_size, int64_t input_shape_2, CUDA_LONG N) {
  // coordinates will be: [d0, d1, d2, d3]
  CUDA_LONG d0 = blockIdx.z;
  CUDA_LONG d1 = blockIdx.y;
  CUDA_LONG d2 = threadIdx.y + blockIdx.x * blockDim.y;
  CUDA_LONG d3 = threadIdx.x;

  CUDA_LONG input_index = d0 * input_strides[0] +
                          d1 * input_strides[1] +
                          d2 * input_strides[2] +
                          d3 * input_strides[3];

  CUDA_LONG output_index = d0 * output_strides[0] +
                           d1 * output_strides[1] +
                           d2 * output_strides[2] +
                           d3 * output_strides[3];

  if (input_index < N && output_index < N && d2 < input_shape_2) {
    const int8_t* input_data_to_be_copied = input_data + (input_index * element_size);
    int8_t* output_data_to_be_copied = output_data + (output_index * element_size);

    // copy over the bytes
    for (size_t iter = 0; iter < element_size; ++iter) {
      *output_data_to_be_copied++ = *input_data_to_be_copied++;
    }
  }
}

bool CanDoTranspose4DParallelizeOneElementPerThread(const cudaDeviceProp& prop,
                                                    size_t /*element_size*/,
                                                    int32_t rank,
                                                    const gsl::span<const int64_t>& input_dims,
                                                    const gsl::span<const size_t>& /*permutations*/,
                                                    dim3& grid_size, dim3& block_size) {
  if (rank == 4) {
    // dims[3]: block.x
    // dims[2]: block.y + grid.x
    // dims[1]: grid.y
    // dims[0]: grid.z
    if (input_dims[3] <= prop.maxThreadsPerBlock &&
        input_dims[1] <= prop.maxGridSize[1] &&
        input_dims[0] <= prop.maxGridSize[2]) {
      // There are 2 constrains when luanching the kernels
      // 1. block_size_x * block_size_y <= prop.maxThreadsPerBlock
      // 2. block_size_y * num_block_ext >= input_dims[2]
      int64_t block_size_x = input_dims[3];
      int64_t max_block_size_y = prop.maxThreadsPerBlock / block_size_x;
      int64_t block_size_y = std::min(input_dims[2], max_block_size_y);
      int64_t num_block_ext = CeilDiv(input_dims[2], block_size_y);

      if (num_block_ext <= prop.maxGridSize[0]) {
        block_size = dim3(static_cast<unsigned int>(block_size_x), static_cast<unsigned int>(block_size_y));
        grid_size = dim3(static_cast<unsigned int>(num_block_ext),
                         static_cast<unsigned int>(input_dims[1]),
                         static_cast<unsigned int>(input_dims[0]));
        return true;
      } else {
        return false;
      }
    }
  }
  return false;
}

Status Transpose4DParallelizeOneElementPerThread(
    cudaStream_t stream, size_t element_size,
    const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides,
    const void* input_data, const TArray<int64_t>& output_strides,
    void* output_data, int N, const dim3& grid_size, const dim3& block_size) {
  if (element_size != sizeof(int8_t) &&
      element_size != sizeof(int16_t) &&
      element_size != sizeof(int32_t) &&
      element_size != sizeof(int64_t)) {
    // User will not hit this as this kernel is for fixed element size tensors only
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                           element_size);
  }

  Transpose4DKernelParallelizeOneElementPerThread<<<grid_size, block_size, 0, stream>>>(
      input_strides, reinterpret_cast<const int8_t*>(input_data),
      output_strides, reinterpret_cast<int8_t*>(output_data),
      element_size, input_shape[2], N);

  return Status::OK();
}

template <typename T>
__global__ void TransposeKernel(int32_t shape_rank, const TArray<int64_t> input_strides,
                                const T* input_data, const TArray<fast_divmod> output_strides, T* output_data, CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

#pragma unroll
  for (auto dim = 0; dim < input_strides.Capacity(); ++dim) {
    if (dim >= shape_rank) {
      break;
    }
    int out_coord, r;
    output_strides[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}

Status TransposeImpl(cudaStream_t stream, size_t element_size, int32_t shape_rank, const TArray<int64_t>& input_strides,
                     const void* input_data, const TArray<fast_divmod>& fdm_output_strides, void* output_data, int N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int8_t):
      TransposeKernel<int8_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int16_t):
      TransposeKernel<int16_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int32_t):
      TransposeKernel<int32_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int64_t):
      TransposeKernel<int64_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          shape_rank, input_strides,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
          N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
