// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/tensor/slice_impl.h"

namespace onnxruntime {
namespace cuda {

template <bool is_grad, int DIMS, int NumThreadsPerBlock, int NumElementsPerThread, typename T>
__global__ void _SliceKernel(const TArray<int64_t> starts,
                             const TArray<int64_t> steps,
                             const TArray<int64_t> input_strides,
                             const TArray<fast_divmod> output_strides,
                             const T* input_data,
                             T* output_data,
                             const CUDA_LONG N) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  CUDA_LONG input_indices[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG input_index = 0;
      int div;
      int mod = id;
      int value = id;
      int dim = 0;
#pragma unroll
      for (; dim < DIMS - 1; ++dim) {
        output_strides[dim].divmod(value, div, mod);
        input_index += (starts[dim] + div * steps[dim]) * input_strides[dim];
        value = mod;
      }
      input_index += starts[dim] + mod * steps[dim];
      input_indices[i] = input_index;
      id += NumThreadsPerBlock;
    }
  }

  if (is_grad) {
    id = start;
#pragma unroll
    for (int i = 0; i < NumElementsPerThread; i++) {
      if (id < N) {
        output_data[input_indices[i]] = input_data[id];
        id += NumThreadsPerBlock;
      }
    }
  } else {
    id = start;
#pragma unroll
    for (int i = 0; i < NumElementsPerThread; i++) {
      if (id < N) {
        output_data[id] = input_data[input_indices[i]];
        id += NumThreadsPerBlock;
      }
    }
  }
}

Status SliceImpl(cudaStream_t stream,
                 const size_t element_size,
                 const int32_t dimension_count,
                 const TArray<int64_t>& starts,
                 const TArray<int64_t>& steps,
                 const TArray<int64_t>& input_strides,
                 const TArray<fast_divmod>& output_strides,
                 const void* input_data,
                 void* output_data,
                 const size_t N) {
  return SliceImplEx<false>(stream, element_size, dimension_count, starts, steps, input_strides, output_strides, input_data,
                            output_data, N);
}

Status SliceImplGrad(cudaStream_t stream,
                     const size_t element_size,
                     const int32_t dimension_count,
                     const TArray<int64_t>& starts,
                     const TArray<int64_t>& steps,
                     const TArray<int64_t>& input_strides,
                     const TArray<fast_divmod>& output_strides,
                     const void* input_data,
                     void* output_data,
                     const size_t N) {
  return SliceImplEx<true>(stream, element_size, dimension_count, starts, steps, input_strides, output_strides, input_data,
                           output_data, N);
}

#define HANDLE_DIMS(ELEMENT_TYPE, DIMS)                                                     \
  case DIMS: {                                                                              \
    _SliceKernel<is_grad, DIMS, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread> \
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(                                \
            starts, steps, input_strides, output_strides,                                   \
            reinterpret_cast<const ToCudaType<ELEMENT_TYPE>::MappedType*>(input_data),      \
            reinterpret_cast<ToCudaType<ELEMENT_TYPE>::MappedType*>(output_data),           \
            (CUDA_LONG)N);                                                                  \
  } break

#define HANDLE_ELEMENT_TYPE(ELEMENT_TYPE) \
  case sizeof(ELEMENT_TYPE): {            \
    switch (dimension_count) {            \
      HANDLE_DIMS(ELEMENT_TYPE, 1);       \
      HANDLE_DIMS(ELEMENT_TYPE, 2);       \
      HANDLE_DIMS(ELEMENT_TYPE, 3);       \
      HANDLE_DIMS(ELEMENT_TYPE, 4);       \
      HANDLE_DIMS(ELEMENT_TYPE, 5);       \
      HANDLE_DIMS(ELEMENT_TYPE, 6);       \
      HANDLE_DIMS(ELEMENT_TYPE, 7);       \
      HANDLE_DIMS(ELEMENT_TYPE, 8);       \
    }                                     \
  } break

template <bool is_grad>
Status SliceImplEx(cudaStream_t stream,
                   const size_t element_size,
                   const int32_t dimension_count,
                   const TArray<int64_t>& starts,
                   const TArray<int64_t>& steps,
                   const TArray<int64_t>& input_strides,
                   const TArray<fast_divmod>& output_strides,
                   const void* input_data,
                   void* output_data,
                   const size_t N) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  switch (element_size) {
    HANDLE_ELEMENT_TYPE(int8_t);
    HANDLE_ELEMENT_TYPE(int16_t);
    HANDLE_ELEMENT_TYPE(int32_t);
    HANDLE_ELEMENT_TYPE(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Slice operator");
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
