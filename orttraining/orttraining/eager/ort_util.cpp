// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>

#include "orttraining/eager/ort_backends.h"
#include "orttraining/eager/ort_util.h"

namespace torch_ort {
namespace eager {

using namespace onnxruntime;

onnxruntime::TensorShapeVector GetStrides(gsl::span<const int64_t> shape) {
  onnxruntime::TensorShapeVector strides(shape.size(), 1);
  for (auto i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }
  return strides;
}

}  // namespace eager
}  // namespace torch_ort
