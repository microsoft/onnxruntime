// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include <ATen/dlpack.h>

// This convertor will take an OrtValue and wrap it as a DLPack tensor

namespace onnxruntime {
namespace cuda {
namespace torch_wrapper {

DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value);

// DLPack uses same config for both bool and unit8. Parameter is_bool_tensor is to
// tell ORT the data type when creating OrtValue.
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack, bool is_bool_tensor = false);

}   // namespace torch_wrapper
}   // namespace cuda
}   // namespace onnxruntime
