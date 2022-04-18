// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tensor_shape.h"
#include "data_types.h"

namespace onnxruntime {

struct TensorDesc {
  MLDataType dtype;
  TensorShapeVector shape;
};

}  // namespace onnxruntime
