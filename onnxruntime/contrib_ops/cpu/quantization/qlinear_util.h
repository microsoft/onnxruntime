// Copyright (c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ONNXRUNTIME_QLINEAR_UTIL_H
#define ONNXRUNTIME_QLINEAR_UTIL_H
#include <cstdint>
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

constexpr int LOOKUP_TABLE_IS_FIXED = 1;
constexpr int LOOKUP_TABLE_IS_COPY = 2;

static inline bool has_same_scale(const Tensor* tensor_x_scale, const Tensor* tensor_y_scale) {
  return *(tensor_x_scale->Data<float>()) == *(tensor_y_scale->Data<float>());
}

static inline bool has_same_zero_point(bool is_signed, const Tensor* tensor_x_zero_point, const Tensor* tensor_y_zero_point) {
  if (is_signed) {
    const int8_t X_zero_point = (tensor_x_zero_point == nullptr) ? static_cast<int8_t>(0) : *(tensor_x_zero_point->Data<int8_t>());
    const int8_t Y_zero_point = (tensor_y_zero_point == nullptr) ? static_cast<int8_t>(0) : *(tensor_y_zero_point->Data<int8_t>());
    return X_zero_point == Y_zero_point;
  }
  const uint8_t X_zero_point = (tensor_x_zero_point == nullptr) ? static_cast<uint8_t>(0) : *(tensor_x_zero_point->Data<uint8_t>());
  const uint8_t Y_zero_point = (tensor_y_zero_point == nullptr) ? static_cast<uint8_t>(0) : *(tensor_y_zero_point->Data<uint8_t>());
  return X_zero_point == Y_zero_point;

}
}  // namespace contrib
}  // namespace onnxruntime
#endif  // ONNXRUNTIME_QLINEAR_UTIL_H
