// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

#include <memory>

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtSequenceTypeInfo {
 public:
  explicit OrtSequenceTypeInfo(OrtTypeInfo* sequence_key_type) noexcept;

  std::unique_ptr<OrtTypeInfo, decltype(OrtApi::ReleaseTypeInfo)> sequence_key_type_;

  OrtStatus* Clone(OrtSequenceTypeInfo** out);

  static OrtStatus* FromTypeProto(const ONNX_NAMESPACE::TypeProto*, OrtSequenceTypeInfo** out);

 private:
  OrtSequenceTypeInfo(const OrtSequenceTypeInfo& other) = delete;
  OrtSequenceTypeInfo& operator=(const OrtSequenceTypeInfo& other) = delete;
};
