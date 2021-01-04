// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/ml_value.h"
#include "python/dlpack.h"

// this convertor will:
// 1) take a OrtValue object and wrap it in the DLPack tensor

namespace onnxruntime {
namespace python {

DLManagedTensor* ortvalue_to_dlpack(const OrtValue& ml_value);
DLDataType get_dlpack_data_type(const OrtValue& ml_value);
DLContext get_dlpack_context(const OrtValue& ml_value, const int64_t& device_id);

onnxruntime::MLDataType get_ortvalue_data_type(const DLDataType& dtype);
OrtValue ortvalue_from_dlpack(const DLManagedTensor* src, AllocatorPtr alloc);

}  // namespace python
}  // namespace onnxruntime
