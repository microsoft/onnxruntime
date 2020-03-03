// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/math/clip_impl.h"
#include "core/providers/hip/cu_inc/common.cuh"

namespace onnxruntime {
namespace hip {
template <typename T>
__global__ void _Clip(const T* input, T* output, T min, T max, size_t N) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = (input[id] < min) ? min : ((input[id] > max) ? max : input[id]);
}

template <typename T>
void ClipImpl(const T* input_data, T* output_data, T min, T max, size_t count) {
  typedef typename ToHipType<T>::MappedType HipT;

  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  hipLaunchKernelGGL(_Clip<HipT>, dim3(blocksPerGrid), dim3(GridDim::maxThreadsPerBlock), 0, 0, reinterpret_cast<const HipT*>(input_data),
                                                                  reinterpret_cast<HipT*>(output_data),
                                                                  *reinterpret_cast<HipT*>(&min),
                                                                  *reinterpret_cast<HipT*>(&max),
                                                                  count);
}

template void ClipImpl<float>(const float* input_data, float* output_data, float min, float max, size_t count);
template void ClipImpl<double>(const double* input_data, double* output_data, double min, double max, size_t count);
template void ClipImpl<MLFloat16>(const MLFloat16* input_data, MLFloat16* output_data, MLFloat16 min, MLFloat16 max, size_t count);

}  // namespace hip
}  // namespace onnxruntime
