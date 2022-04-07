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
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Received OrtValue is not a tensor. Only tensors are supported.");
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
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added: ", name);
  }

  return Status::OK();
}

#if !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
Status SessionOptions::AddExternalInitializers(gsl::span<const std::string> names, gsl::span<const OrtValue> values) {
  const auto init_num = names.size();
  ORT_ENFORCE(init_num == values.size(), "Expecting same size spans");
  external_initializers.reserve(external_initializers.size() + init_num);
  for (size_t i = 0; i < init_num; ++i) {
    ORT_RETURN_IF_ERROR(CheckInitializer(names[i].c_str(), &values[i]));
    bool result = external_initializers.emplace(names[i], values[i]).second;
    if (!result) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "An OrtValue for this name has already been added: ", names[i]);
    }
  }
  return Status::OK();
}
#endif  // !defined(ORT_MINIMAL_BUILD) && !defined(DISABLE_EXTERNAL_INITIALIZERS)
}  // namespace onnxruntime
