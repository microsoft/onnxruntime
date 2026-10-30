// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <optional>

#include "core/common/common.h"

namespace onnxruntime::contrib::cuda {

// Shared by discovery and execution, without CUDA dependencies or allocation.
// Unknown B/S are never substituted: only products with all factors known are checked.
inline Status ValidateCausalConvNativeParameters(std::optional<int64_t> batch, int64_t channels,
                                                 std::optional<int64_t> length, int64_t kernel,
                                                 int64_t dilation, int64_t slots, size_t element_bytes) {
  constexpr auto max_int = std::numeric_limits<int>::max();
  if (length) {
    ORT_RETURN_IF_NOT(*length > 0, "input length must be positive, got ", *length);
  }
  ORT_RETURN_IF_NOT((!batch || (*batch >= 0 && *batch <= max_int)) &&
                        channels > 0 && channels <= max_int &&
                        (!length || *length <= max_int) && kernel > 0 && kernel <= max_int &&
                        dilation > 0 && dilation <= max_int && slots > 0 && slots <= max_int,
                    "CausalConvWithState dimensions exceed native integer range");
  ORT_RETURN_IF_NOT(kernel - 1 <= max_int / dilation,
                    "CausalConvWithState padding exceeds native integer range");
  const int64_t pad = (kernel - 1) * dilation;
  auto fits_bytes = [element_bytes](std::initializer_list<int64_t> factors) {
    int64_t bytes = static_cast<int64_t>(element_bytes);
    for (int64_t factor : factors) {
      if (factor != 0 && bytes > std::numeric_limits<int64_t>::max() / factor) return false;
      bytes *= factor;
    }
    return true;
  };
  ORT_RETURN_IF_NOT((!length || fits_bytes({channels, *length})) &&
                        (!length || !batch || fits_bytes({channels, *length, *batch})) &&
                        fits_bytes({channels, kernel}) && fits_bytes({channels, pad, slots}) &&
                        (!batch || fits_bytes({channels, pad, slots, *batch})),
                    "CausalConvWithState tensor bytes or strides exceed signed 64-bit range");
  return Status::OK();
}

}  // namespace onnxruntime::contrib::cuda
