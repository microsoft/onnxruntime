// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "expand_impl.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _FillFromDataPtrKernel(T* output_data, const T* input_data, CUDA_LONG N) {
  CUDA_LONG id = NumElementsPerThread * blockDim.x * blockIdx.x + threadIdx.x;
  T val = *input_data;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = val;
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void FillFromDataPtr(cudaStream_t stream, T* output_data, const T* input_data, int64_t count) {
  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _FillFromDataPtrKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(output_data, input_data, N);
}

template <typename T>
__global__ void ExpandKernel2D(
    const int N,
    const T* input_data,
    T* output_data,
    const fast_divmod fdm_output_stride0,
    const int input_view_stride0,
    const int input_view_stride1) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int dim0, dim1;
  fdm_output_stride0.divmod(id, dim0, dim1);
  output_data[id] = input_data[dim0 * input_view_stride0 + dim1 * input_view_stride1];
}

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void ExpandKernel(
    const int rank,
    const int N,
    const T* input_data,
    T* output_data,
    const TArray<fast_divmod> output_strides,
    const TArray<int64_t> input_strides) {
  CUDA_LONG start = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
  T value[NumElementsPerThread];

  CUDA_LONG id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      // compute indexes with broadcasting rules: https://github.com/onnx/onnx/blob/master/docs/Broadcasting.md
      CUDA_LONG index = 0;
      CUDA_LONG offset = id;
#pragma unroll
      for (auto dim = 0; dim < output_strides.Capacity(); dim++) {
        if (dim >= rank) {
          break;
        }

        int q, r;
        output_strides[dim].divmod(offset, q, r);
        index += static_cast<int>(input_strides[dim]) * q;
        offset = r;
      }

      value[i] = input_data[index];
      id += NumThreadsPerBlock;
    }
  }

  id = start;
#pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      output_data[id] = value[i];
      id += NumThreadsPerBlock;
    }
  }
}

Status ExpandByFill(cudaStream_t stream, const size_t element_size, const int N, const void* input_data, void* output_data) {
#define EXPAND_FILL_ON(TYPE)                                   \
  case sizeof(TYPE):                                           \
    FillFromDataPtr(stream,                                    \
                    reinterpret_cast<TYPE*>(output_data),      \
                    reinterpret_cast<const TYPE*>(input_data), \
                    static_cast<int64_t>(N));                  \
    break

  switch (element_size) {
    EXPAND_FILL_ON(int8_t);
    EXPAND_FILL_ON(int16_t);
    EXPAND_FILL_ON(int32_t);
    EXPAND_FILL_ON(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}

Status Expand2D(
    cudaStream_t stream,
    const size_t element_size,
    const int N,
    const void* input_data,
    void* output_data,
    const fast_divmod fdm_output_stride0,
    const int input_view_stride0,
    const int input_view_stride1) {
#define EXPAND2D_ON(TYPE)                                                                   \
  case sizeof(TYPE):                                                                        \
    ExpandKernel2D<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(                      \
        N, reinterpret_cast<const TYPE*>(input_data), reinterpret_cast<TYPE*>(output_data), \
        fdm_output_stride0, input_view_stride0, input_view_stride1);                        \
    break

  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(N, GridDim::maxThreadsPerBlock));
  switch (element_size) {
    EXPAND2D_ON(int8_t);
    EXPAND2D_ON(int16_t);
    EXPAND2D_ON(int32_t);
    EXPAND2D_ON(int64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}

Status ExpandImpl(
    cudaStream_t stream,
    const size_t element_size,
    const int N_output,
    const int N_input,
    const void* input_data,
    void* output_data,
    const TArray<fast_divmod>& output_strides,
    const TArray<int64_t>& input_strides) {
  const int rank = static_cast<int>(output_strides.Size());
  if (rank == 1) {
    if (N_input == N_output) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, N_output * element_size, cudaMemcpyDeviceToDevice, stream));
    } else {  // N_input == 1
      return ExpandByFill(stream, element_size, N_output, input_data, output_data);
    }
  } else if (rank == 2) {
    return Expand2D(stream, element_size, N_output, input_data, output_data,
                    output_strides[0],
                    static_cast<int>(input_strides[0]),
                    static_cast<int>(input_strides[1]));
  }

  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(N_output, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));

#define EXPAND_ON(TYPE)                                                                                      \
  case sizeof(TYPE):                                                                                         \
    ExpandKernel<TYPE, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>                           \
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(                                                 \
            rank, N_output, reinterpret_cast<const TYPE*>(input_data), reinterpret_cast<TYPE*>(output_data), \
            output_strides, input_strides);                                                                  \
    break

  switch (element_size) {
    EXPAND_ON(uint8_t);
    EXPAND_ON(uint16_t);
    EXPAND_ON(uint32_t);
    EXPAND_ON(uint64_t);
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for Expand operator");
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
