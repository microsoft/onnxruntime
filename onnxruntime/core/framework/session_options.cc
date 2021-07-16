// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"
#include "core/framework/ml_value.h"

namespace onnxruntime {

Status SessionOptions::AddInitializer(_In_z_ const char* name, _In_ const OrtValue* val) noexcept {
  // input validation
  if (name == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for name.");
  }

  if (val == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for OrtValue.");
  }

  if (!val->IsTensor()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received OrtValue is not a tensor. Only tensors are supported.");
  }

  if (val->Get<Tensor>().OwnsBuffer()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer containing the initializer must be owned by the user.");
  }

  // now do the actual work
  auto rc = initializers_to_share_map.insert({name, val});
  if (!rc.second) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added.");
  }

  return Status::OK();
}
}  // namespace onnxruntime
