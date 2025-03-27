// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

TensorShape ReduceShapeByComponents(const TensorShape& shape, int64_t components) {
  // Reduce the last dimension by components creating a new tensor shape.
  TensorShapeVector shape_vector = shape.AsShapeVector();
  auto size = shape_vector.size();
  shape_vector[size - 1] = (shape_vector[size - 1] + components - 1) / components;
  return TensorShape(shape_vector);
}

}  // namespace webgpu
}  // namespace onnxruntime
