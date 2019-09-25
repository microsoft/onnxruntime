// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "transpose_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void _TransposeKernel(size_t shape_rank, const int64_t* input_strides, const size_t* perm,
                                 const T* input_data, const fast_divmod* fdm_output_strides, T* output_data, size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N, NumElementsPerThread);

  #pragma unroll
  for (int i = 0; i < NumElementsPerThread; i++) {
    if (id < N) {
      CUDA_LONG input_index = 0;
      CUDA_LONG output_index = id;
      for (int dim = 0; dim < shape_rank; ++dim) {
        int out_coord, r;
        fdm_output_strides[dim].divmod(output_index, out_coord, r);
        output_index = r;
        input_index += input_strides[perm[dim]] * out_coord;
      }
      output_data[id] = input_data[input_index];
      id += NumThreadsPerBlock;
    }
  }
}

template <typename T>
void TransposeImpl(size_t shape_rank, const int64_t* input_strides, const size_t* perm, const T* input_data,
                   const fast_divmod* fdm_output_strides, T* output_data, size_t N) {
  int blocksPerGrid = static_cast<int>(CeilDiv(N, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
  _TransposeKernel<T, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>\
    <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      shape_rank, input_strides, perm, input_data,
      fdm_output_strides, output_data, N);
}

#define SPECIALIZED_IMPL(T)                                                                                  \
  template void TransposeImpl<T>(size_t shape_rank, const int64_t* input_strides, const size_t* perm,        \
                                 const T* input_data, const fast_divmod* fdm_output_strides, T* output_data, \
                                 size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace onnxruntime
