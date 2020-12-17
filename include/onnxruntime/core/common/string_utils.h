// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/string_utils_base.h"

namespace onnxruntime {

template <typename T>
Status ParseString(const std::string& s, T& value) {
  ORT_RETURN_IF_NOT(TryParseString(s, value), "Failed to parse value: \"", value, "\"");
  return Status::OK();
}

template <typename T>
T ParseString(const std::string& s) {
  T value{};
  ORT_THROW_IF_ERROR(Parse(s, value));
  return value;
}

}  // namespace onnxruntime
