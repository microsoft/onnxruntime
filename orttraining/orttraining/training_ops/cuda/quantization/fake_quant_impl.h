// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>

namespace onnxruntime {
namespace cuda {

template <typename T>
void FakeQuantPerTensor(cudaStream_t stream, const int64_t num_elements, const T* input_data, const T quant_scale,
                        const T quant_zero_point, const int64_t quant_min, const int64_t quant_max,
                        T* fake_quantized_data, bool* quantization_mask_data);

template <typename T>
void FakeQuantGradImpl(cudaStream_t stream, const int64_t num_elements, const T* dY_data,
                       const bool* gradient_mask_data, T* dX_data);

}  // namespace cuda
}  // namespace onnxruntime
