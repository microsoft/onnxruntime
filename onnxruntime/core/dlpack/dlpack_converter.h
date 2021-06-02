// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/language_interop_ops/torch/python_common.h"
#include "core/framework/ml_value.h"
#include <dlpack/dlpack.h>

namespace onnxruntime {
namespace dlpack {
// This convertor will take an OrtValue and wrap it as a DLPack tensor
// This may create a new ownership to the underlying tensor in OrtValue,
// so we do pass-by-value here. We don't use pass-by-reference because
// it implies no new ownership.
DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value);

// DLPack uses same config for both bool and unit8. Parameter is_bool_tensor is to
// tell ORT the data type when creating OrtValue.
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack, bool is_bool_tensor = false);

// Allocate a new Capsule object, which takes the ownership of OrtValue.
// Caller is responsible for releasing.
// This function calls OrtValueToDlpack(...).
PyObject* ToDlpack(OrtValue ort_value);

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the conversion.
OrtValue FromDlpack(PyObject* dlpack_tensor, const bool is_bool_tensor);
}  // namespace dlpack
}  // namespace onnxruntime
