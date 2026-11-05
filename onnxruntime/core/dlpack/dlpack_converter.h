// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// C++ level interface for DLPack conversion.

#pragma once

#include "core/framework/ort_value.h"
#include <dlpack/dlpack.h>

namespace onnxruntime {
namespace dlpack {

// Wrap an OrtValue as a DLPack tensor, retaining a copy of the OrtValue
// until the consumer calls the DLPack deleter.
DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value);

// Imports a contiguous tensor, preserving its DLPack dtype (including kDLBool).
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack);

// Return the device of an OrtValue.
DLDevice GetDlpackDevice(const OrtValue& ort_value, const int64_t& device_id);

}  // namespace dlpack
}  // namespace onnxruntime
