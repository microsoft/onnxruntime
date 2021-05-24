// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "transpose_impl.h"

namespace onnxruntime {
namespace cuda {

constexpr unsigned int TILE_DIM = 16;

template <typename T>
__global__ void Transpose3DKernel(const TArray<int64_t> input_shape,
                                  const TArray<int64_t> input_strides,
                                  const T* input_data, T* output_data) {
  __shared__ T tile[TILE_DIM * (TILE_DIM + 1)];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;

  tile[threadIdx.y * TILE_DIM + threadIdx.x] = input_data[blockIdx.z * input_strides[0] + y * input_shape[2] + x];
  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x;
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  output_data[blockIdx.z * input_strides[0] + y * input_shape[1] + x] = tile[threadIdx.x * TILE_DIM + threadIdx.y];
}

bool CanDoTranspose3D(int32_t rank,
                      const std::vector<int64_t>& input_dims,
                      const std::vector<size_t>& permutations) {
  if (rank == 3 &&
      // permutation is done in the last two dimensions.
      permutations[rank - 2] == (rank - 1) && permutations[rank - 1] == (rank - 2) &&
      // the last two dimensions are aligned with TILE_DIM.
      input_dims[rank - 2] % TILE_DIM == 0 && input_dims[rank - 1] % TILE_DIM == 0) {
    return true;
  }
  return false;
}

Status Transpose3DImpl(cudaStream_t stream, size_t element_size,
                       const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides,
                       const void* input_data, void* output_data, int64_t N) {
  dim3 block_size(TILE_DIM, TILE_DIM);
  dim3 grid_size(static_cast<unsigned int>(input_shape[2] / TILE_DIM), static_cast<unsigned int>(input_shape[1] / TILE_DIM), static_cast<unsigned int>(input_shape[0]));

  switch (element_size) {
    case sizeof(int8_t):
      Transpose3DKernel<int8_t><<<grid_size, block_size, 0, stream>>>(
          input_shape, input_strides,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data));
      break;
    case sizeof(int16_t):
      Transpose3DKernel<int16_t><<<grid_size, block_size, 0, stream>>>(
          input_shape, input_strides,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data));
      break;
    case sizeof(int32_t):
      Transpose3DKernel<int32_t><<<grid_size, block_size, 0, stream>>>(
          input_shape, input_strides,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data));
      break;
    case sizeof(int64_t):
      Transpose3DKernel<int64_t><<<grid_size, block_size, 0, stream>>>(
          input_shape, input_strides,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }

  return Status::OK();
}

__global__ void Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim(
    const TArray<int64_t> input_strides, const void* input_data,
    const TArray<int64_t> output_strides, void* output_data,
    size_t element_size,
    CUDA_LONG N) {
  // output coordinates will be: blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x
  CUDA_LONG input_index = (blockIdx.y * input_strides[0] +
                           blockIdx.x * input_strides[1] +
                           threadIdx.y * input_strides[2]) /
                              (4 * sizeof(int) / element_size) +
                          threadIdx.x * input_strides[3];

  CUDA_LONG output_index = (blockIdx.y * output_strides[0] +
                            blockIdx.x * output_strides[1] +
                            threadIdx.y * output_strides[2]) /
                               (4 * sizeof(int) / element_size) +
                           threadIdx.x * output_strides[3];

  const int4* v_input = reinterpret_cast<const int4*>(input_data);
  int4* v_output = reinterpret_cast<int4*>(output_data);

  if (input_index < N && output_index < N) {
    v_output[output_index] = v_input[input_index];
  }
}

bool CanDoTranspose4DParallelizeMultipleElementsPerThreadInInnermostDim(const cudaDeviceProp& prop,
                                                                        size_t element_size,
                                                                        int32_t rank,
                                                                        const std::vector<int64_t>& input_dims,
                                                                        const std::vector<size_t>& permutations) {
  if (rank == 4 &&
      // the permutations is not on the last dimension.
      permutations[3] == 3) {
    // The block size will be set based on the outer-most two dimensions of 4D tensor.
    // the number threads per block will be calculated as below.
    unsigned int num_elements_per_thread = 4 * sizeof(int) / static_cast<unsigned int>(element_size);  // int4 is used in the kernel to access data.
    int64_t num_elements_in_last_two_dimensions = input_dims[2] * input_dims[3];
    int64_t num_threads_per_block = num_elements_in_last_two_dimensions / num_elements_per_thread;

    if (((num_elements_in_last_two_dimensions & (num_elements_per_thread - 1)) == 0) &&
        num_threads_per_block <= prop.maxThreadsPerBlock &&
        num_threads_per_block >= prop.warpSize &&
        // num_threads_per_block must be a multiple of warp size (32)
        ((num_threads_per_block & (prop.warpSize - 1)) == 0) &&
        // input_dims[3] must be a multiple of `num_elements_per_thread`
        ((input_dims[3] % num_elements_per_thread) == 0)) {
      return true;
    }
  }
  return false;
}

Status Transpose4DParallelizeMultipleElementsPerThreadInInnermostDim(
    cudaStream_t stream, size_t element_size,
    const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides,
    const void* input_data, const TArray<int64_t>& output_strides,
    void* output_data, int N) {
  if (element_size != sizeof(int8_t) &&
      element_size != sizeof(int16_t) &&
      element_size != sizeof(int32_t) &&
      element_size != sizeof(int64_t)) {
    // User will not hit this as this kernel is for fixed element size tensors only
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                           element_size);
  }

  unsigned int num_elements_per_thread = 4 * sizeof(int) / static_cast<unsigned int>(element_size);  // int4 is used in the kernel to access data.
  dim3 block_size(static_cast<unsigned int>(input_shape[3] / num_elements_per_thread), static_cast<unsigned int>(input_shape[2]));
  dim3 grid_size(static_cast<unsigned int>(input_shape[1]), static_cast<unsigned int>(input_shape[0]));

  Transpose4DKernelParallelizeMultipleElementsPerThreadInInnermostDim<<<grid_size, block_size, 0, stream>>>(
      input_strides, input_data,
      output_strides, output_data, element_size, N / num_elements_per_thread);

  return Status::OK();
}

__global__ void Transpose4DKernelParallelizeOneElementPerThread(
    const TArray<int64_t> input_strides, const int8_t* input_data,
    const TArray<int64_t> output_strides, int8_t* output_data,
    size_t element_size,
    CUDA_LONG N) {
  CUDA_LONG input_index = blockIdx.y * input_strides[0] +
                          blockIdx.x * input_strides[1] +
                          threadIdx.y * input_strides[2] +
                          threadIdx.x * input_strides[3];

  CUDA_LONG output_index = blockIdx.y * output_strides[0] +
                           blockIdx.x * output_strides[1] +
                           threadIdx.y * output_strides[2] +
                           threadIdx.x * output_strides[3];

  if (input_index < N && output_index < N) {
    const int8_t* input_data_to_be_copied = input_data + (input_index * element_size);
    int8_t* output_data_to_be_copied = output_data + (output_index * element_size);

    // copy over the bytes
    memcpy(output_data_to_be_copied, input_data_to_be_copied, element_size);
  }
}

bool CanDoTranspose4DParallelizeOneElementPerThread(const cudaDeviceProp& prop,
                                                    size_t element_size,
                                                    int32_t rank,
                                                    const std::vector<int64_t>& input_dims,
                                                    const std::vector<size_t>& permutations) {
  if (rank == 4) {
    // The block size will be set based on the outer-most two dimensions of 4D tensor.
    // the number threads per block will be calculated as below.
    int64_t number_of_threads_per_block = input_dims[2] * input_dims[3];

    if (number_of_threads_per_block <= prop.maxThreadsPerBlock &&
        number_of_threads_per_block >= prop.warpSize &&
        // num_threads_per_block must be a multiple of warp size (32)
        ((number_of_threads_per_block & (prop.warpSize - 1)) == 0)) {
      return true;
    }
  }
  return false;
}

Status Transpose4DParallelizeOneElementPerThread(
    cudaStream_t stream, size_t element_size,
    const TArray<int64_t>& input_shape, const TArray<int64_t>& input_strides,
    const void* input_data, const TArray<int64_t>& output_strides,
    void* output_data, int N) {
  if (element_size != sizeof(int8_t) &&
      element_size != sizeof(int16_t) &&
      element_size != sizeof(int32_t) &&
      element_size != sizeof(int64_t)) {
    // User will not hit this as this kernel is for fixed element size tensors only
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                           element_size);
  }

  dim3 block_size(static_cast<unsigned int>(input_shape[3]), static_cast<unsigned int>(input_shape[2]));
  dim3 grid_size(static_cast<unsigned int>(input_shape[1]), static_cast<unsigned int>(input_shape[0]));

  Transpose4DKernelParallelizeOneElementPerThread<<<grid_size, block_size, 0, stream>>>(
      input_strides, reinterpret_cast<const int8_t*>(input_data),
      output_strides, reinterpret_cast<int8_t*>(output_data),
      element_size, N);

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
