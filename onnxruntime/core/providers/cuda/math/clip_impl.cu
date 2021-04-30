// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/clip_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {
template <typename T>
__global__ void _Clip(const T* input, T* output, const T* min, const T* max, T min_default, T max_default, size_t N) {
  auto min_val = (min) ? *min : min_default; 
  auto max_val = (max) ? *max : max_default; 
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  output[id] = (input[id] < min_val) ? min_val : ((input[id] > max_val) ? max_val : input[id]);
}

template <typename T>
void ClipImpl(cudaStream_t stream, const T* input_data, T* output_data, const T* min, const T* max, T min_default, T max_default, size_t count) {
  typedef typename ToCudaType<T>::MappedType CudaT;

  int blocksPerGrid = (int)(ceil(static_cast<float>(count) / GridDim::maxThreadsPerBlock));
  union alias {
    T t;
    CudaT cudaT;
  };
  _Clip<CudaT><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(reinterpret_cast<const CudaT*>(reinterpret_cast<const union alias*>(input_data)),
                                                                          reinterpret_cast<CudaT*>(reinterpret_cast<union alias*>(output_data)),
                                                                          reinterpret_cast<const CudaT*>(reinterpret_cast<const union alias*>(min)),
                                                                          reinterpret_cast<const CudaT*>(reinterpret_cast<const union alias*>(max)),
                                                                          *reinterpret_cast<CudaT*>(reinterpret_cast<union alias*>(&min_default)),
                                                                          *reinterpret_cast<CudaT*>(reinterpret_cast<union alias*>(&max_default)),
                                                                          count);
}

template void ClipImpl<float>(cudaStream_t stream, const float* input_data, float* output_data, const float* min, const float* max, float min_default, float max_default, size_t count);
template void ClipImpl<double>(cudaStream_t stream, const double* input_data, double* output_data, const double* min, const double* max, double min_default, double max_default, size_t count);
template void ClipImpl<MLFloat16>(cudaStream_t stream, const MLFloat16* input_data, MLFloat16* output_data, const MLFloat16* min, const MLFloat16* max, MLFloat16 min_default, MLFloat16 max_default, size_t count);
template void ClipImpl<int8_t>(cudaStream_t stream, const int8_t* input_data, int8_t* output_data, const int8_t* min, const int8_t* max, int8_t min_default, int8_t max_default, size_t count);
template void ClipImpl<uint8_t>(cudaStream_t stream, const uint8_t* input_data, uint8_t* output_data, const uint8_t* min, const uint8_t* max, uint8_t min_default, uint8_t max_default, size_t count);
template void ClipImpl<int64_t>(cudaStream_t stream, const int64_t* input_data, int64_t* output_data, const int64_t* min, const int64_t* max, int64_t min_default, int64_t max_default, size_t count);
template void ClipImpl<uint64_t>(cudaStream_t stream, const uint64_t* input_data, uint64_t* output_data, const uint64_t* min, const uint64_t* max, uint64_t min_default, uint64_t max_default, size_t count);

}  // namespace cuda
}  // namespace onnxruntime
