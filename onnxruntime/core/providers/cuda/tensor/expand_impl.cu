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
void FillFromDataPtr(T* output_data, const T* input_data, int64_t count) {
  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  CUDA_LONG N = static_cast<CUDA_LONG>(count);
  _FillFromDataPtrKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
      <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(output_data, input_data, N);
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

template <typename T>
__global__ void ExpandKernel(
    const int rank,
    const int N,
    const T* input_data,
    T* output_data,
    const fast_divmod* fdm_output_strides,
    const int64_t* input_view_strides) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int dim, r = id, input_index = 0;
  for (int i = 0; i < rank; ++i) {
    fdm_output_strides[i].divmod(r, dim, r);
    input_index += dim * input_view_strides[i];
  }
  output_data[id] = input_data[input_index];
}

Status ExpandByFill(const size_t element_size, const int N, const void* input_data, void* output_data) {
#define EXPAND_FILL_ON(TYPE)                                   \
  case sizeof(TYPE):                                           \
    FillFromDataPtr(reinterpret_cast<TYPE*>(output_data),      \
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
    const size_t element_size,
    const int N,
    const void* input_data,
    void* output_data,
    const fast_divmod fdm_output_stride0,
    const int input_view_stride0,
    const int input_view_stride1) {
#define EXPAND2D_ON(TYPE)                                                                   \
  case sizeof(TYPE):                                                                        \
    ExpandKernel2D<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(                      \
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
    const size_t element_size,
    const int N_output,
    const int N_input,
    const void* input_data,
    void* output_data,
    CudaKernel::CudaAsyncBuffer<fast_divmod>& fdm_output_strides,
    CudaKernel::CudaAsyncBuffer<int64_t>& input_view_strides) {
  const int rank = static_cast<int>(fdm_output_strides.count());
  if (rank == 1) {
    if (N_input == N_output) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(output_data, input_data, N_output * element_size, cudaMemcpyDeviceToDevice));
    } else {  // N_input == 1
      return ExpandByFill(element_size, N_output, input_data, output_data);
    }
  } else if (rank == 2) {
    return Expand2D(element_size, N_output, input_data, output_data,
                    fdm_output_strides.CpuSpan()[0],
                    static_cast<int>(input_view_strides.CpuSpan()[0]),
                    static_cast<int>(input_view_strides.CpuSpan()[1]));
  }

  int blocksPerGrid = gsl::narrow_cast<int>(CeilDiv(N_output, GridDim::maxThreadsPerBlock));
  fdm_output_strides.CopyToGpu();
  input_view_strides.CopyToGpu();

#define EXPAND_ON(TYPE)                                                                                  \
  case sizeof(TYPE):                                                                                     \
    ExpandKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(                                     \
        rank, N_output, reinterpret_cast<const TYPE*>(input_data), reinterpret_cast<TYPE*>(output_data), \
        fdm_output_strides.GpuPtr(), input_view_strides.GpuPtr());                                       \
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
