// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
void ClipImpl(const T* input_data, T* output_data, T min, T max, size_t count);

}  // namespace cuda
}  // namespace onnxruntime
