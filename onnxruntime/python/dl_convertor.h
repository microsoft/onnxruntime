// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include "python/dlpack.h"

// this convertor will:
// 1) take a OrtValue object and wrap it in the DLPack tensor

namespace onnxruntime {
namespace python {

DLManagedTensor* ort_value_to_dlpack(const OrtValue& ml_value);
DLDataType get_dlpack_data_type(const OrtValue& ml_value);
DLContext get_dlpack_context(const OrtValue& ml_value, const int64_t& device_id);

}  // namespace python
}  // namespace onnxruntime
