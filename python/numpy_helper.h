// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4127)
#endif
#include <numpy/arrayobject.h>
#ifdef _MSC_VER
#pragma warning(pop)
#endif
namespace onnxruntime {
namespace python {
constexpr bool IsNumericNumpyType(int npy_type) {
  return npy_type < NPY_OBJECT || npy_type == NPY_HALF;
}
}  // namespace python
}  // namespace onnxruntime