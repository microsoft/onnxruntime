// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/tensor/scatter_nd_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/atomic/common.cuh"
#include "core/providers/cuda/tensor/scatter_mean_impl.cuh"

namespace onnxruntime {
namespace cuda {

template <typename T, typename TIndex>
__global__ void _ScatterNDKernel(
    T* output_data,
    const size_t num_indices,
    const TIndex* indices_data,
    const int64_t last_index_dimension,
    ElementCountsAndInputDimsSpanOrGpu element_counts_and_input_dims,
    const T* updates_data,
    const size_t num_updates_elements,
    bool strict_indices) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_indices);

  // Compute the base offset into the output data
  int64_t data_offset = 0;

  size_t indices_start = last_index_dimension * id;
  size_t indices_end = indices_start + last_index_dimension;
  for (size_t i = indices_start; i < indices_end; ++i) {
    int64_t index = indices_data[i];

    int64_t element_count_dim = element_counts_and_input_dims.gpu_ptr == nullptr
                                    ? element_counts_and_input_dims.stack_ptr[i - indices_start]
                                    : element_counts_and_input_dims.gpu_ptr[i - indices_start];
    int64_t dim_value = element_counts_and_input_dims.gpu_ptr == nullptr
                            ? element_counts_and_input_dims.stack_ptr[i - indices_start + last_index_dimension]
                            : element_counts_and_input_dims.gpu_ptr[i - indices_start + last_index_dimension];

    if (strict_indices && (index < -dim_value || index >= dim_value)) {
      printf("ScatterND index %lld is out of bounds; ignoring this update. Use valid indices.\n",
             static_cast<long long>(index));
      return;
    }

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

template <typename TIndex>
Status ScatterNDImplTyped(
    cudaStream_t stream,
    void* output_data,
    const size_t element_size,
    const size_t num_indices,
    const TIndex* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements,
    bool strict_indices) {
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
          num_updates_elements, strict_indices);
      break;

    case sizeof(int16_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int16_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int16_t*>(updates_data),
          num_updates_elements, strict_indices);
      break;

    case sizeof(int32_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int32_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int32_t*>(updates_data),
          num_updates_elements, strict_indices);
      break;

    case sizeof(int64_t):
      _ScatterNDKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<int64_t*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const int64_t*>(updates_data),
          num_updates_elements, strict_indices);
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

template <typename T, typename TIndex, typename TFunc>
__global__ void _ScatterNDKernelReduction(
    T* output_data,
    const size_t num_indices,
    const TIndex* indices_data,
    const int64_t last_index_dimension,
    ElementCountsAndInputDimsSpanOrGpu element_counts_and_input_dims,
    const T* updates_data,
    const size_t num_updates_elements,
    const TFunc func,
    bool strict_indices) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_indices);

  // Compute the base offset into the output data
  int64_t data_offset = 0;

  size_t indices_start = last_index_dimension * id;
  size_t indices_end = indices_start + last_index_dimension;
  for (size_t i = indices_start; i < indices_end; ++i) {
    int64_t index = indices_data[i];

    int64_t element_count_dim = element_counts_and_input_dims.gpu_ptr == nullptr
                                    ? element_counts_and_input_dims.stack_ptr[i - indices_start]
                                    : element_counts_and_input_dims.gpu_ptr[i - indices_start];
    int64_t dim_value = element_counts_and_input_dims.gpu_ptr == nullptr
                            ? element_counts_and_input_dims.stack_ptr[i - indices_start + last_index_dimension]
                            : element_counts_and_input_dims.gpu_ptr[i - indices_start + last_index_dimension];

    if (strict_indices && (index < -dim_value || index >= dim_value)) {
      printf("ScatterND index %lld is out of bounds; ignoring this update. Use valid indices.\n",
             static_cast<long long>(index));
      return;
    }

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

template <typename T, typename TIndex>
Status _ScatterNDType(
    cudaStream_t stream,
    T* output_data,
    const size_t num_indices,
    const TIndex* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const T* updates_data,
    const size_t num_updates_elements,
    ScatterNDReduction reduction,
    bool strict_indices) {
  // Parallelize on number of indices
  int blocksPerGrid = static_cast<int>(ceil(static_cast<float>(num_indices) / GridDim::maxThreadsPerBlock));

  switch (reduction) {
    case ScatterNDReduction::Add:
      _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_data,
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          updates_data,
          num_updates_elements,
          FuncAdd<T>(), strict_indices);
      break;
    case ScatterNDReduction::Mul:
      _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_data,
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          updates_data,
          num_updates_elements,
          FuncMul<T>(), strict_indices);
      break;
    case ScatterNDReduction::Min:
      _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_data,
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          updates_data,
          num_updates_elements,
          FuncMin<T>(), strict_indices);
      break;
    case ScatterNDReduction::Max:
      _ScatterNDKernelReduction<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          output_data,
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          updates_data,
          num_updates_elements,
          FuncMax<T>(), strict_indices);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Reduction ", static_cast<int>(reduction), " not implemented for ScatterND operator.");
  }

  return Status::OK();
}

template <typename TIndex>
Status ScatterNDImplReductionTyped(
    cudaStream_t stream,
    void* output_data,
    const int32_t element_type,
    const size_t num_indices,
    const TIndex* indices_data,
    const int64_t last_index_dimension,
    const ElementCountsAndInputDimsSpanOrGpu& element_counts_and_input_dims,
    const void* updates_data,
    const size_t num_updates_elements,
    ScatterNDReduction reduction,
    bool strict_indices) {
  if (num_indices == 0)
    return Status::OK();

  switch (element_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return _ScatterNDType<float>(
          stream,
          reinterpret_cast<float*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const float*>(updates_data),
          num_updates_elements,
          reduction, strict_indices);
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return _ScatterNDType<half>(
          stream,
          reinterpret_cast<half*>(output_data),
          num_indices,
          indices_data,
          last_index_dimension,
          element_counts_and_input_dims,
          reinterpret_cast<const half*>(updates_data),
          num_updates_elements,
          reduction, strict_indices);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "element_type ", static_cast<int>(element_type), " not implemented for ScatterND operator.");
  }
}

Status ScatterNDImpl(cudaStream_t stream, void* output_data, size_t element_size, size_t num_indices,
                     const void* indices_data, int64_t last_index_dimension,
                     const ElementCountsAndInputDimsSpanOrGpu& dimensions, const void* updates_data,
                     size_t slice_size, size_t index_element_size, bool strict_indices) {
  if (index_element_size == sizeof(int32_t)) {
    return ScatterNDImplTyped(stream, output_data, element_size, num_indices,
                              static_cast<const int32_t*>(indices_data), last_index_dimension,
                              dimensions, updates_data, slice_size, strict_indices);
  }
  return ScatterNDImplTyped(stream, output_data, element_size, num_indices,
                            static_cast<const int64_t*>(indices_data), last_index_dimension,
                            dimensions, updates_data, slice_size, strict_indices);
}

Status ScatterNDImplReduction(cudaStream_t stream, void* output_data, int32_t element_type, size_t num_indices,
                              const void* indices_data, int64_t last_index_dimension,
                              const ElementCountsAndInputDimsSpanOrGpu& dimensions, const void* updates_data,
                              size_t slice_size, ScatterNDReduction reduction, size_t index_element_size,
                              bool strict_indices) {
  if (index_element_size == sizeof(int32_t)) {
    return ScatterNDImplReductionTyped(stream, output_data, element_type, num_indices,
                                       static_cast<const int32_t*>(indices_data), last_index_dimension,
                                       dimensions, updates_data, slice_size, reduction, strict_indices);
  }
  return ScatterNDImplReductionTyped(stream, output_data, element_type, num_indices,
                                     static_cast<const int64_t*>(indices_data), last_index_dimension,
                                     dimensions, updates_data, slice_size, reduction, strict_indices);
}

template <typename TIndex>
__global__ void BuildScatterNDMeanLists(const TIndex* indices, int64_t num_indices, int64_t depth,
                                        ElementCountsAndInputDimsSpanOrGpu dimensions,
                                        int64_t* heads, int64_t* next, int64_t slice_size) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < num_indices; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t offset = 0;
    for (int64_t d = 0; d < depth; ++d) {
      const auto* dims = dimensions.gpu_ptr ? dimensions.gpu_ptr : dimensions.stack_ptr;
      const int64_t limit = dims[d + depth];
      int64_t index = indices[i * depth + d];
      if (index < -limit || index >= limit) {
        printf("ScatterND mean index %lld is outside [-%lld, %lld); ignoring this update. Use valid indices.\n",
               static_cast<long long>(index), static_cast<long long>(limit), static_cast<long long>(limit));
        offset = -1;
        break;
      }
      if (index < 0) index += limit;
      offset += index * dims[d];
    }
    if (offset >= 0) {
      LinkScatterUpdate(heads, next, offset / slice_size, i);
    } else {
      next[i] = -1;
    }
  }
}

Status ScatterNDMeanImpl(cudaStream_t stream, const void* indices, size_t index_element_size,
                         int64_t num_indices, int64_t depth, const ElementCountsAndInputDimsSpanOrGpu& dimensions,
                         int64_t* heads, int64_t* next, int64_t slice_size, bool deterministic) {
  const int blocks = deterministic ? 1 : static_cast<int>(std::min<int64_t>(CeilDiv(num_indices, int64_t{256}), 65535));
  const int threads = deterministic ? 1 : 256;
  if (index_element_size == sizeof(int32_t)) {
    BuildScatterNDMeanLists<<<blocks, threads, 0, stream>>>(
        static_cast<const int32_t*>(indices), num_indices, depth, dimensions, heads, next, slice_size);
  } else {
    BuildScatterNDMeanLists<<<blocks, threads, 0, stream>>>(
        static_cast<const int64_t*>(indices), num_indices, depth, dimensions, heads, next, slice_size);
  }
  return Status::OK();
}

Status ScatterMeanImpl(cudaStream_t stream, int32_t element_type, const void* input, const void* updates,
                       void* output, const int64_t* heads, const int64_t* next, int64_t size, int64_t slice_size) {
  const int blocks = static_cast<int>(std::min<int64_t>(CeilDiv(size, int64_t{256}), 65535));
#define SCATTER_MEAN_CASE(onnx_type, T)                                                        \
  case ONNX_NAMESPACE::TensorProto_DataType_##onnx_type:                                       \
    ScatterMeanKernel<<<blocks, 256, 0, stream>>>(                                             \
        static_cast<const T*>(input), static_cast<const T*>(updates), static_cast<T*>(output), \
        heads, next, size, slice_size);                                                        \
    return Status::OK()
  switch (element_type) {
    SCATTER_MEAN_CASE(INT8, int8_t);
    SCATTER_MEAN_CASE(UINT8, uint8_t);
    SCATTER_MEAN_CASE(INT16, int16_t);
    SCATTER_MEAN_CASE(UINT16, uint16_t);
    SCATTER_MEAN_CASE(INT32, int32_t);
    SCATTER_MEAN_CASE(UINT32, uint32_t);
    SCATTER_MEAN_CASE(INT64, int64_t);
    SCATTER_MEAN_CASE(UINT64, uint64_t);
    SCATTER_MEAN_CASE(FLOAT16, half);
    SCATTER_MEAN_CASE(BFLOAT16, BFloat16);
    SCATTER_MEAN_CASE(FLOAT, float);
    SCATTER_MEAN_CASE(DOUBLE, double);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Scatter mean requires integer or floating-point data; cast data and updates "
                             "to a supported numeric type. Got element type ",
                             element_type, ".");
  }
#undef SCATTER_MEAN_CASE
}

}  // namespace cuda
}  // namespace onnxruntime
