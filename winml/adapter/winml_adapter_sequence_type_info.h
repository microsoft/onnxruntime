// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "winml_adapter_c_api.h"

namespace ONNX_NAMESPACE {
class TypeProto;
}

struct OrtSequenceTypeInfo {
 public:
  std::unique_ptr<OrtTypeInfo, void(*)(OrtTypeInfo*)> sequence_key_type_;

  OrtStatus* Clone(OrtSequenceTypeInfo** out);

  static OrtStatus* FromTypeProto(const ONNX_NAMESPACE::TypeProto*, OrtSequenceTypeInfo** out); 

 private:
  OrtSequenceTypeInfo(OrtTypeInfo* sequence_key_type)noexcept;
  OrtSequenceTypeInfo(const OrtSequenceTypeInfo& other) = delete;
  OrtSequenceTypeInfo& operator=(const OrtSequenceTypeInfo& other) = delete;

};
