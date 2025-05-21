// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <type_traits>
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

template <typename... Args>
TensorShape make_shape(Args... args) {
  std::initializer_list<int64_t> dims = {args...};
  return TensorShape(dims);
}

// This assumes the tensor is optional, and check wether its shape is expected.
#define ASSERT_TENSOR_SHAPE(tensor, shape)                                                                       \
  if (tensor != nullptr) {                                                                                       \
    static_assert(std::is_same<decltype(tensor), const Tensor*>::value, "tensor must be a pointer to a Tensor"); \
    static_assert(std::is_same<std::remove_cv_t<std::remove_reference_t<decltype(shape)>>,                       \
                               TensorShape>::value,                                                              \
                  "shape must be or refer to a TensorShape");                                                    \
    if (tensor->Shape() != shape) {                                                                              \
      return ORT_MAKE_STATUS(                                                                                    \
          ONNXRUNTIME, INVALID_ARGUMENT, "Input '" #tensor "' is expected to have shape ", shape,                \
          ", got ", tensor->Shape());                                                                            \
    }                                                                                                            \
  }

// This assumes the tensor is optional, and check wether its shape is shape_1 or shape_2 when it is not null.
#define ASSERT_TENSOR_SHAPE_2(tensor, shape_1, shape_2)                                                          \
  if (tensor != nullptr) {                                                                                       \
    static_assert(std::is_same<decltype(tensor), const Tensor*>::value, "tensor must be a pointer to a Tensor"); \
    static_assert(std::is_same<std::remove_cv_t<std::remove_reference_t<decltype(shape_1)>>,                     \
                               TensorShape>::value,                                                              \
                  "shape_1 must be or refer to a TensorShape");                                                  \
    static_assert(std::is_same<std::remove_cv_t<std::remove_reference_t<decltype(shape_2)>>,                     \
                               TensorShape>::value,                                                              \
                  "shape_2 must be or refer to a TensorShape");                                                  \
    if (tensor->Shape() != shape_1 && tensor->Shape() != shape_2) {                                              \
      return ORT_MAKE_STATUS(                                                                                    \
          ONNXRUNTIME, INVALID_ARGUMENT, "Input '" #tensor "' is expected to have shape ", shape_1,              \
          " or ", shape_2, ", got ", tensor->Shape());                                                           \
    }                                                                                                            \
  }

}  // namespace onnxruntime
