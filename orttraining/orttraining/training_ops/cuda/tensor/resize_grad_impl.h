// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <optional>

namespace onnxruntime::cuda {

template <typename T>
void ResizeGradImpl(cudaStream_t stream, int64_t input_height,
                    int64_t input_width, int64_t output_height,
                    int64_t output_width, int64_t batch_size,
                    int64_t channels, bool align_corners,
                    const std::optional<float>& scale_height,
                    const std::optional<float>& scale_width,
                    const T* dY_data, T* dX_data);

}  // namespace onnxruntime::cuda
