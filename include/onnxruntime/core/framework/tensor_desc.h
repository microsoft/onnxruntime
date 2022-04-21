// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "tensor_shape.h"

namespace onnxruntime {

// data_types.h and provider_api.h have this
class DataTypeImpl;
using MLDataType = const DataTypeImpl*;

struct TensorDesc {
  MLDataType dtype;
  TensorShapeVector shape;
};

}  // namespace onnxruntime
