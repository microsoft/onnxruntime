// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "image_scaler_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void _CropKernel(
    const T* input_data,
    const int src_start_x,
    const int src_start_y,
    const int src_w,
    const int src_hw,
    const fast_divmod fdm_dst_w,
    const fast_divmod fdm_dst_hw,
    T* output_data,
    const CUDA_LONG N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  int dst_xy, dst_nc;
  fdm_dst_hw.divmod(id, dst_nc, dst_xy);
  int dst_x, dst_y;
  fdm_dst_w.divmod(dst_xy, dst_y, dst_x);
  output_data[id] = input_data[dst_nc * src_hw + (dst_y + src_start_y) * src_w + (dst_x + src_start_x)];
}

template <typename T>
void CropImpl(
    cudaStream_t stream,
    const T* input_data,
    const int src_start_x,
    const int src_start_y,
    const int src_w,
    const int src_hw,
    const fast_divmod& fdm_dst_w,
    const fast_divmod& fdm_dst_hw,
    T* output_data,
    const size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _CropKernel<<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input_data, src_start_x, src_start_y, src_w, src_hw, fdm_dst_w, fdm_dst_hw, output_data, (CUDA_LONG)N);
}

#define SPECIALIZED_IMPL(T) \
  template void CropImpl<T>(cudaStream_t stream, const T* input_data, const int src_start_x, const int src_start_y, const int src_w, const int src_hw, const fast_divmod& fdm_dst_w, const fast_divmod& fdm_dst_hw, T* output_data, const size_t N);

SPECIALIZED_IMPL(float)
SPECIALIZED_IMPL(double)
SPECIALIZED_IMPL(half)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
