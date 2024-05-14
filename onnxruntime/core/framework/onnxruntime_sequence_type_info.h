// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <memory>

#include "core/framework/onnxruntime_typeinfo.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtSequenceTypeInfo {
 public:
  explicit OrtSequenceTypeInfo(std::unique_ptr<OrtTypeInfo> sequence_key_type) noexcept;
  ~OrtSequenceTypeInfo();

  std::unique_ptr<OrtTypeInfo> sequence_key_type_;

  std::unique_ptr<OrtSequenceTypeInfo> Clone() const;

  static std::unique_ptr<OrtSequenceTypeInfo> FromTypeProto(const ONNX_NAMESPACE::TypeProto&);

  OrtSequenceTypeInfo(const OrtSequenceTypeInfo& other) = delete;
  OrtSequenceTypeInfo& operator=(const OrtSequenceTypeInfo& other) = delete;
};
