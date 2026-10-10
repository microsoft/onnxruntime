// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/ort_value.h"
#include "core/graph/onnx_protobuf.h"

const void* OrtValue::GetOpaqueData(onnxruntime::MLDataType expected_type) const {
  ORT_ENFORCE(expected_type != nullptr && type_ == expected_type, "Opaque value has an unexpected type");
  const auto* proto = type_->GetTypeProto();
  ORT_ENFORCE(type_->IsNonTensorType() && proto != nullptr && proto->has_opaque_type(),
              "Value is not an opaque type");
  ORT_ENFORCE(IsAllocated(), "Opaque value is not allocated");
  return data_.get();
}

void* OrtValue::GetMutableOpaqueData(onnxruntime::MLDataType expected_type) {
  (void)GetOpaqueData(expected_type);
  return data_.get();
}
