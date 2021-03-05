// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include "python/dlpack.h"

// This convertor will take an OrtValue and wrap it as a DLPack tensor

namespace onnxruntime {
namespace python {

DLManagedTensor* ort_value_to_dlpack(const OrtValue& ort_value);
DLDataType get_dlpack_data_type(const OrtValue& ort_value);
DLContext get_dlpack_context(const OrtValue& ort_value, const int64_t& device_id);

OrtDevice get_ort_device(const DLContext& ctx);
MLDataType get_ort_value_data_type(const DLDataType& dtype);
OrtValue dlpack_to_ort_value(const DLManagedTensor* src);

}  // namespace python
}  // namespace onnxruntime
