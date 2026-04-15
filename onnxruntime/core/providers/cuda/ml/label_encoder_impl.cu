// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <type_traits>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "label_encoder_impl.h"

namespace onnxruntime {
namespace cuda {

// Binary search for a key in a sorted array.
// Returns the index of the key if found, or -1 if not found.
template <typename TKey>
__device__ int64_t BinarySearch(const TKey* keys, int64_t num_keys, TKey target) {
  int64_t lo = 0;
  int64_t hi = num_keys - 1;
  while (lo <= hi) {
    int64_t mid = lo + (hi - lo) / 2;
    if (keys[mid] == target) {
      return mid;
    }
    if (keys[mid] < target) {
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  return -1;
}

template <typename TKey, typename TValue>
__global__ void LabelEncoderKernel(
    const TKey* input,
    TValue* output,
    int64_t num_elements,
    const TKey* keys,
    const TValue* values,
    int64_t num_keys,
    TValue default_value,
    int64_t nan_key_index) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, num_elements);

  TKey val = input[id];

  // For floating-point types, check for NaN
  if constexpr (std::is_floating_point_v<TKey>) {
    if (isnan(val)) {
      output[id] = (nan_key_index >= 0) ? values[nan_key_index] : default_value;
      return;
    }
  }

  int64_t idx = BinarySearch(keys, num_keys, val);
  output[id] = (idx >= 0) ? values[idx] : default_value;
}

template <typename TKey, typename TValue>
void LabelEncoderImpl(
    cudaStream_t stream,
    const TKey* input,
    TValue* output,
    int64_t num_elements,
    const TKey* keys,
    const TValue* values,
    int64_t num_keys,
    TValue default_value,
    int64_t nan_key_index) {
  if (num_elements == 0) return;

  int blocksPerGrid = static_cast<int>(CeilDiv(num_elements, static_cast<int64_t>(GridDim::maxThreadsPerBlock)));

  LabelEncoderKernel<TKey, TValue><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input, output, num_elements, keys, values, num_keys, default_value, nan_key_index);
}

// Explicit template instantiations for all supported type combinations
template void LabelEncoderImpl<int64_t, int64_t>(cudaStream_t, const int64_t*, int64_t*, int64_t,
                                                 const int64_t*, const int64_t*, int64_t,
                                                 int64_t, int64_t);
template void LabelEncoderImpl<int64_t, float>(cudaStream_t, const int64_t*, float*, int64_t,
                                               const int64_t*, const float*, int64_t,
                                               float, int64_t);
template void LabelEncoderImpl<float, int64_t>(cudaStream_t, const float*, int64_t*, int64_t,
                                               const float*, const int64_t*, int64_t,
                                               int64_t, int64_t);
template void LabelEncoderImpl<float, float>(cudaStream_t, const float*, float*, int64_t,
                                             const float*, const float*, int64_t,
                                             float, int64_t);
template void LabelEncoderImpl<double, double>(cudaStream_t, const double*, double*, int64_t,
                                               const double*, const double*, int64_t,
                                               double, int64_t);
template void LabelEncoderImpl<double, int64_t>(cudaStream_t, const double*, int64_t*, int64_t,
                                                const double*, const int64_t*, int64_t,
                                                int64_t, int64_t);
template void LabelEncoderImpl<int64_t, double>(cudaStream_t, const int64_t*, double*, int64_t,
                                                const int64_t*, const double*, int64_t,
                                                double, int64_t);

}  // namespace cuda
}  // namespace onnxruntime
