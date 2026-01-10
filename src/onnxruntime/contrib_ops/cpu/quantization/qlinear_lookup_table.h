// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"
#include <functional>
#include <stdint.h>
#include <vector>

// TODO(kreeger): Move this folder to a quantization utils/toolkit folder.
namespace onnxruntime {
namespace contrib {

// function that transform array of input value to array of output value of length
typedef std::function<void(const float* input, float* output, size_t length)> LookupTableArrayTransformer;

// function that transform single value
typedef std::function<float(float)> LookupTableScalarTransformer;

template <typename T>
void QlinearBuildLookupTable(uint8_t* table,
                             const Tensor* tensor_x_scale,
                             const Tensor* tensor_x_zero_point,
                             const Tensor* tensor_y_scale,
                             const Tensor* tensor_y_zero_point,
                             const LookupTableArrayTransformer& array_values_transformer);

template <typename T>
void QlinearBuildLookupTable(uint8_t* table,
                             const Tensor* tensor_x_scale,
                             const Tensor* tensor_x_zero_point,
                             const Tensor* tensor_y_scale,
                             const Tensor* tensor_y_zero_point,
                             const LookupTableScalarTransformer& value_transformer);

template <typename TOutput>
void QLinearLookupTableTransform(const uint8_t* x, const TOutput* table, TOutput* y, size_t n);

}  // namespace contrib
}  // namespace onnxruntime
