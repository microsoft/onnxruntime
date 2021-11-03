// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// C++ level interface for DLPack conversion.

#pragma once

#include "core/framework/ort_value.h"
#include <dlpack/dlpack.h>

namespace onnxruntime {
namespace dlpack {

typedef struct {
  OrtValue handle;
  DLManagedTensor tensor;
} OrtDLManagedTensor;

typedef void DlpackDeleterFct(DLManagedTensor* arg);

DlpackDeleterFct* GetDlpackDeleter();

// This convertor will take an OrtValue and wrap it as a DLPack tensor
// This may create a new ownership to the underlying tensor in OrtValue,
// so we do pass-by-value here. We don't use pass-by-reference because
// it implies no new ownership.
DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value);

// This convertor will take an OrtValue and wrap it as a DLPack tensor,
// similar to the previous function but the structure DLManagedTensor
// is already allocated. This function can be used when multiple
// OrtValue are converted in the same function. This saves allocations.
DLManagedTensor* OrtValueToDlpack(OrtValue& ort_value, OrtDLManagedTensor* ort_dlmanaged_tensor, DlpackDeleterFct* deleter);

// DLPack uses same config for both bool and unit8. Parameter is_bool_tensor is to
// tell ORT the data type when creating OrtValue.
OrtValue DlpackToOrtValue(DLManagedTensor* dlpack, bool is_bool_tensor = false);

}  // namespace dlpack
}  // namespace onnxruntime
