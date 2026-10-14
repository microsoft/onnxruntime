// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(__CUDACC__)
#define ORT_REDUCTION_HOST_DEVICE __forceinline__ __host__ __device__
#else
#define ORT_REDUCTION_HOST_DEVICE inline
#endif

namespace onnxruntime {
namespace cuda {

ORT_REDUCTION_HOST_DEVICE bool reduction_scan_delta_is_valid(int delta, int remaining) {
  return delta < remaining;
}

ORT_REDUCTION_HOST_DEVICE bool advance_reduction_scan(int limit, int step, int& position) {
  const int remaining = limit - position;
  if (remaining <= step) {
    return false;
  }

  position += step;
  return true;
}

}  // namespace cuda
}  // namespace onnxruntime

#undef ORT_REDUCTION_HOST_DEVICE
