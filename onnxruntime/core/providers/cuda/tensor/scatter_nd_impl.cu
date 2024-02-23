// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _ScatterNDKernel(
    T* output_data,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const T* updates_data,
    const size_t num_updates_elements) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_indices);

  // Compute the base offset into the output data
  int64_t data_offset = 0;

  size_t indices_start = last_index_dimension * id;
  size_t indices_end = indices_start + last_index_dimension;
  for (size_t i = indices_start; i < indices_end; ++i) {
    int64_t index = indices_data[i];

    int64_t element_count_dim = element_counts_and_input_dims[i - indices_start];
    int64_t dim_value = element_counts_and_input_dims[i - indices_start + last_index_dimension];

    // Clamp the index if out of range
    // This would have been an error in the CPU kernel, but throwing in the CUDA EP
    // is hard. This is the approach taken by other frameworks for out of bound indices
    // in their corresponding GPU backends as well.
    // index >= -dim_value && index < dim_value

    if (index >= 0) {
      if (index >= dim_value) {
        index = dim_value - 1;
      }
    } else {
      if (index < -dim_value) {
        index = 0;
      } else {
        index += dim_value;
      }
    }

    data_offset += (index * element_count_dim);
  }

  const T* updates_data_base = updates_data + num_updates_elements * id;
  T* output_data_base = output_data + data_offset;

  for (size_t i = 0; i < num_updates_elements; ++i) {
    output_data_base[i] = updates_data_base[i];
  }
}

Status ScatterNDImpl(
    cudaStream_t stream,
    void* output_data,
    const size_t element_size,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements) {
  if (num_indices == 0)
    return Status::OK();

  // Parallelize on number of indices
  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(num_indices) / GridDim::maxThreadsPerBlock));

  switch (element_size) {
    case sizeof(int8_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int8_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int8_t*>(updates_data),
          num_updates_elements);
      break;

    case sizeof(int16_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int16_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int16_t*>(updates_data),
          num_updates_elements);
      break;

    case sizeof(int32_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int32_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int32_t*>(updates_data),
          num_updates_elements);
      break;

    case sizeof(int64_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int64_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int64_t*>(updates_data),
          num_updates_elements);
      break;

    default:
      // Shouldn't hit this
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for ScatterND operator");
  }

  return Status::OK();
}

template <class T>
struct FuncAdd {
  __device__ __inline__ void operator()(T* start_addr, T value) const {
    atomic_add(start_addr, value);
  }
};

template <class T>
struct FuncMul {
  __device__ __inline__ void operator()(T* start_addr, T value) const {
    atomic_mul(start_addr, value);
  }
};

template <class T>
struct FuncMax {
  __device__ __inline__ void operator()(T* start_addr, T value) const {
    atomic_max(start_addr, value);
  }
};

template <class T>
struct FuncMin {
  __device__ __inline__ void operator()(T* start_addr, T value) const {
    atomic_min(start_addr, value);
  }
};

template <typename T, typename TFunc>
__global__ void _ScatterNDKernelReduction(
    T* output_data,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const T* updates_data,
    const size_t num_updates_elements,
    const TFunc func) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_indices);

  // Compute the base offset into the output data
  int64_t data_offset = 0;

  size_t indices_start = last_index_dimension * id;
  size_t indices_end = indices_start + last_index_dimension;
  for (size_t i = indices_start; i < indices_end; ++i) {
    int64_t index = indices_data[i];

    int64_t element_count_dim = element_counts_and_input_dims[i - indices_start];
    int64_t dim_value = element_counts_and_input_dims[i - indices_start + last_index_dimension];

    // Clamp the index if out of range
    // This would have been an error in the CPU kernel, but throwing in the CUDA EP
    // is hard. This is the approach taken by other frameworks for out of bound indices
    // in their corresponding GPU backends as well.
    // index >= -dim_value && index < dim_value

    if (index >= 0) {
      if (index >= dim_value) {
        index = dim_value - 1;
      }
    } else {
      if (index < -dim_value) {
        index = 0;
      } else {
        index += dim_value;
      }
    }

    data_offset += (index * element_count_dim);
  }

  const T* updates_data_base = updates_data + num_updates_elements * id;
  T* output_data_base = output_data + data_offset;

  for (size_t i = 0; i < num_updates_elements; ++i) {
    func(output_data_base + i, updates_data_base[i]);
  }
}

Status ScatterNDImplReduction(
    cudaStream_t stream,
    void* output_data,
    const int32_t element_type,
    const size_t num_indices,
    const int64_t* indices_data,
    const int64_t last_index_dimension,
    const int64_t* element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements,
    int reduction_as_int) {
  if (num_indices == 0)
    return Status::OK();

  // Parallelize on number of indices
  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(num_indices) / GridDim::maxThreadsPerBlock));

  switch (reduction_as_int) {
    case 1:  // Reduction::Add
      switch (element_type) {
        case 1:  // ONNXNAMESPACE::TensorProto_DataType_FLOAT: TODO: which header to use?
          _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
              reinterpret_cast<float*>(output_data),
              num_indices,
              indices_data,
              last_index_dimension,
              element_counts_and_input_dims,
              reinterpret_cast<const float*>(updates_data),
              num_updates_elements,
              FuncAdd<float>());
          break;

        case 10:  // ONNXNAMESPACE::TensorProto_DataType_FLOAT16: TODO: which header to use?
          _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
              reinterpret_cast<half*>(output_data),
              num_indices,
              indices_data,
              last_index_dimension,
              element_counts_and_input_dims,
              reinterpret_cast<const half*>(updates_data),
              num_updates_elements,
              FuncAdd<half>());
          break;

        default:
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type ", element_type, " not supported for ScatterND operator.");
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Reduction ", reduction_as_int, " not implemented for ScatterND operator.");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
