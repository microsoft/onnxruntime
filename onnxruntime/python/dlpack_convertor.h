// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include <dlpack/dlpack.h>

// This convertor will take an OrtValue and wrap it as a DLPack tensor

namespace onnxruntime {
namespace python {

DLManagedTensor* OrtValueToDlpack(const OrtValue& ort_value);

// DLPack uses same config for both bool and unit8. Parameter is_bool_tensor is to
// tell ORT the data type when creating OrtValue.
OrtValue DlpackToOrtValue(const DLManagedTensor* dlpack, bool is_bool_tensor = false);

}  // namespace python
}  // namespace onnxruntime
