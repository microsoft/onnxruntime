// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

#include "core/framework/onnxruntime_typeinfo.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtOptionalTypeInfo {

  using Ptr = std::unique_ptr<OrtOptionalTypeInfo>;

  explicit OrtOptionalTypeInfo(OrtTypeInfo::Ptr contained_type) noexcept;
  ~OrtOptionalTypeInfo();

  OrtTypeInfo::Ptr contained_type_;

  Ptr Clone() const;

  static Ptr FromTypeProto(const ONNX_NAMESPACE::TypeProto&);

  OrtOptionalTypeInfo(const OrtOptionalTypeInfo& other) = delete;
  OrtOptionalTypeInfo& operator=(const OrtOptionalTypeInfo& other) = delete;
};
