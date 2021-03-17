// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include <dlpack/dlpack.h>

// This convertor will take an OrtValue and wrap it as a DLPack tensor

namespace onnxruntime {
namespace python {

DLManagedTensor* OrtValueToDlpack(const OrtValue& ort_value);
OrtValue DlpackToOrtValue(const DLManagedTensor* dlpack);

}  // namespace python
}  // namespace onnxruntime
