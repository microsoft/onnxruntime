// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor.h"

namespace onnxruntime {
namespace signal {

template <typename T>
static T get_scalar_value_from_tensor(const Tensor* tensor) {
  ORT_ENFORCE(tensor->Shape().Size() == 1, "ratio input should have a single value.");
  return *tensor->Data<T>();
}

}  // namespace signal
}  // namespace onnxruntime
