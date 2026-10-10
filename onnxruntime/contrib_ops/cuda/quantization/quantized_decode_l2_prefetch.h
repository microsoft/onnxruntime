// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

namespace onnxruntime::contrib::cuda {

constexpr bool ShouldPrefetchQuantizedDecodeL2(bool enabled, int major, int minor) {
  return enabled && major == 12 && minor == 1;
}

#if defined(__CUDACC__)
__host__ __device__
#endif
    constexpr int64_t
    QuantizedDecodeL2PrefetchOffset(int64_t current, int64_t step, int64_t extent) {
  // Check the remaining extent before adding the two-iteration look-ahead.
  return current >= 0 && current < extent && step > 0 && step <= (extent - 1 - current) / 2
             ? current + 2 * step
             : -1;
}

}  // namespace onnxruntime::contrib::cuda
