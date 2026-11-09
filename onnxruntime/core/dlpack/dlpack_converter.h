// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// C++ level interface for DLPack conversion.

#pragma once

#include "core/framework/ort_value.h"
#include <dlpack/dlpack.h>

namespace onnxruntime {
namespace dlpack {

// This convertor will take an OrtValue and wrap it as a DLPack tensor
// This may create a new ownership to the underlying tensor in OrtValue,
// so we do pass-by-value here. We don't use pass-by-reference because
// it implies no new ownership.
DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value);

// Imports a contiguous tensor, preserving its DLPack dtype (including kDLBool).
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack);

// Return the device of an OrtValue.
DLDevice GetDlpackDevice(const OrtValue& ort_value, const int64_t& device_id);

}  // namespace dlpack
}  // namespace onnxruntime
