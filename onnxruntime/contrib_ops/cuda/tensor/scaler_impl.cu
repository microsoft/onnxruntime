// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "image_scaler_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, typename U, int NumThreadsPerBlock, int NumElementsPerThread>
__global__ void ScalerKernel(
    const T* input_data,
    const U* scale,
    U* output_data,
    CUDA_LONG N) {
    CUDA_LONG id = NumElementsPerThread * NumThreadsPerBlock * blockIdx.x + threadIdx.x;
    #pragma unroll
      for (int i = 0; i < NumElementsPerThread; i++) {
        if (id < N) {
          output_data[id] = static_cast<U>(input_data[id]) * (*scale);
          id += NumThreadsPerBlock;
        }
      }
}

template <typename T, typename U>
void ScalerImpl(
    const T* input_data,
    const U* scale,
    U* output_data,
    size_t num_of_element) {
    if (num_of_element <= 0){
      return;
    }
    int blocksPerGrid = static_cast<int>(CeilDiv(num_of_element, GridDim::maxThreadsPerBlock * GridDim::maxElementsPerThread));
    CUDA_LONG N = static_cast<CUDA_LONG>(num_of_element);
    ScalerKernel<T, U, GridDim::maxThreadsPerBlock, GridDim::maxElementsPerThread>
        <<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
            input_data,
            scale,
            output_data,
            N);
}

#define SPECIALIZED_IMPL(T, U) \
  template void ScalerImpl<T>(const T* input_data, const U* scale, U* output_data, size_t N);

SPECIALIZED_IMPL(int32_t, half)
SPECIALIZED_IMPL(int32_t, float)

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
