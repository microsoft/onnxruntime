// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

struct OrtTypeInfo;

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtOptionalTypeInfo {
  explicit OrtOptionalTypeInfo(std::unique_ptr<OrtTypeInfo> contained_type) noexcept;
  ~OrtOptionalTypeInfo();

  std::unique_ptr<OrtTypeInfo> contained_type_;

  std::unique_ptr<OrtOptionalTypeInfo> Clone() const;

  static std::unique_ptr<OrtOptionalTypeInfo> FromTypeProto(const ONNX_NAMESPACE::TypeProto&);

  OrtOptionalTypeInfo(const OrtOptionalTypeInfo& other) = delete;
  OrtOptionalTypeInfo& operator=(const OrtOptionalTypeInfo& other) = delete;
};
