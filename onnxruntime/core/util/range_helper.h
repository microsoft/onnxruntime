// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <type_traits>

namespace onnxruntime {

// The caller must reject a zero delta before computing the count.
template <typename T>
uint64_t ComputeIntegerRangeCount(T start, T limit, T delta) {
  static_assert(std::is_integral_v<T> && std::is_signed_v<T> && sizeof(T) <= sizeof(int64_t));
  if ((delta > 0 && limit <= start) || (delta < 0 && limit >= start)) {
    return 0;
  }

  // Convert before subtracting: the span can exceed INT64_MAX, and negating INT64_MIN
  // in a signed type is undefined. Unsigned arithmetic preserves both exact magnitudes.
  const uint64_t span = delta > 0 ? static_cast<uint64_t>(limit) - static_cast<uint64_t>(start)
                                  : static_cast<uint64_t>(start) - static_cast<uint64_t>(limit);
  const uint64_t step = delta > 0 ? static_cast<uint64_t>(delta) : uint64_t{0} - static_cast<uint64_t>(delta);
  return span / step + (span % step != 0);
}

}  // namespace onnxruntime
