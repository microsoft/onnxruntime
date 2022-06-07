// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "image_scaler_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, bool batch1>
__global__ void _ImageScalerKernel(
    const T* input_data,
    const float scale,
    const float* bias_data,
    const fast_divmod fdm_C,
    const fast_divmod fdm_HW,
    T* output_data,
    const size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int n, c;
  if (batch1)
    c = fdm_HW.div(id);
  else
    fdm_C.divmod(fdm_HW.div(id), n, c);
  output_data[id] = input_data[id] * (T)scale + (T)bias_data[c];
}

template <typename T>
void ImageScalerImpl(
    cudaStream_t stream,
    const T* input_data,
    const float scale,
    const float* bias_data,
    const int64_t dims[4],  // NCHW
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  fast_divmod fdm_HW((int)(dims[2] * dims[3]));
  fast_divmod fdm_C;
  if (dims[0] == 1) {
    _ImageScalerKernel<T, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        input_data, scale, bias_data, fdm_C, fdm_HW, output_data, N);
  } else {
    fdm_C = fast_divmod((int)dims[1]);
    _ImageScalerKernel<T, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
        input_data, scale, bias_data, fdm_C, fdm_HW, output_data, N);
  }
}

#define SPECIALIZED_IMPL(T) \
  template void ImageScalerImpl<T>(cudaStream_t stream, const T* input_data, const float scale, const float* bias_data, const int64_t dims[4], T* output_data, const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
