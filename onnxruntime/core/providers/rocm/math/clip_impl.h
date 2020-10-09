// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/rocm/math/clip.h"
#include "core/providers/rocm/hip_common.h"
#include "core/providers/rocm/shared_inc/hip_utils.h"

namespace onnxruntime {
namespace rocm {
template <typename T>
void ClipImpl(const T* input_data, T* output_data, T min, T max, size_t count);

}  // namespace rocm
}  // namespace onnxruntime
