// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <numpy/arrayobject.h>
namespace onnxruntime {
namespace python {
constexpr bool IsNumericNumpyType(int npy_type) {
  return npy_type < NPY_OBJECT || npy_type == NPY_HALF;
}
}  // namespace python
}  // namespace onnxruntime