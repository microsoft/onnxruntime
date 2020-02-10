// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/hip/math/clip.h"
#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/hip_utils.h"

namespace onnxruntime {
namespace hip {
template <typename T>
void ClipImpl(const T* input_data, T* output_data, T min, T max, size_t count);

}  // namespace hip
}  // namespace onnxruntime
