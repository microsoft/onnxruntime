// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/session_options.h"
#include "core/common/logging/logging.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {

namespace {

Status CheckInitializer(const char* name, const OrtValue* val) {
  if (name == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for name");
  }

  if (val == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received nullptr for OrtValue");
  }

  if (!val->IsTensor()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received OrtValue is not a tensor. Only tensors are supported for shared initializers.");
  }
  if (val->Get<Tensor>().OwnsBuffer()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Buffer containing the initializer must be owned by the user.");
  }
  return Status::OK();
}

}  // namespace

Status SessionOptions::AddInitializer(_In_z_ const char* name, _In_ const OrtValue* val) {
  // input validation
  ORT_RETURN_IF_ERROR(CheckInitializer(name, val));
  // now do the actual work
  bool result = initializers_to_share_map.emplace(name, val).second;

  if (!result) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added.");
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
Status SessionOptions::AddExternalInitializers(const char* const* names, const OrtValue* const* values, size_t init_num) {
  external_initializers.reserve(external_initializers.size() + init_num);
  for (size_t i = 0; i < init_num; ++i) {
    const char* name = names[i];
    const OrtValue* val = values[i];
    ORT_RETURN_IF_ERROR(CheckInitializer(name, val));
    bool result = external_initializers.emplace(name, val).second;
    if (!result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, MakeString("An OrtValue for this name has already been added: ", name));
    }
  }
  return Status::OK();
}
#else
Status SessionOptions::AddExternalInitializers(const char* const*, const OrtValue* const*, size_t) {
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Adding external initializers is not supported in minimal builds");
}
#endif  // ORT_MINIMAL_BUILD
}  // namespace onnxruntime
