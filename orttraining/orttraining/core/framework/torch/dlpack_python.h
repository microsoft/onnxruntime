// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// Python level interface for DLPack conversion.

#pragma once

#include "core/dlpack/dlpack_converter.h"
#include "orttraining/core/framework/torch/python_common.h"

namespace onnxruntime {
namespace training {
namespace framework {
namespace torch {

// Allocate a new Capsule object, which takes the ownership of OrtValue.
// Caller is responsible for releasing.
// This function calls OrtValueToDlpack(...).
PyObject* ToDlpack(OrtValue ort_value);

// Consume a Capsule object and claims the ownership of its underlying tensor to
// create a OrtValue. This function calls DlpackToOrtValue(...) to do the conversion.
OrtValue FromDlpack(PyObject* dlpack_tensor, const bool is_bool_tensor);

}  // namespace torch
}  // namespace framework
}  // namespace training
}  // namespace onnxruntime
