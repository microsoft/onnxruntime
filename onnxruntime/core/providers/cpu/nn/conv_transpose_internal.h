// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>

#include "core/common/safeint.h"

namespace onnxruntime::conv_transpose_internal {

inline size_t CalculateColBufferSize(size_t element_size, size_t kernel_dim, size_t input_image_size) {
  return SafeInt<size_t>(element_size) * kernel_dim * input_image_size;
}

}  // namespace onnxruntime::conv_transpose_internal
