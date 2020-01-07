// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "winml_adapter_sequence_type_info.h"
#include "onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "ort_apis.h"
#include "../../../winml/adapter/winml_adapter_apis.h"
#include "error_code_helper.h"

namespace winmla = Windows::AI::MachineLearning::Adapter;

OrtSequenceTypeInfo::OrtSequenceTypeInfo(OrtTypeInfo* sequence_key_type) noexcept : sequence_key_type_(sequence_key_type) {  
}

OrtStatus* OrtSequenceTypeInfo::FromTypeProto(const ONNX_NAMESPACE::TypeProto* type_proto, OrtSequenceTypeInfo** out) {
  auto value_case = type_proto->value_case();
  if (value_case != ONNX_NAMESPACE::TypeProto::kSequenceType)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "type_proto is not of type sequence!");;
  }

  auto type_proto_sequence = type_proto->sequence_type();
  OrtTypeInfo* sequence_key_type_info = nullptr;
  if (auto status = OrtTypeInfo::FromTypeProto(&type_proto_sequence->elem_type(), &sequence_key_type_info))
  {
    return status;
  }

  *out = new OrtSequenceTypeInfo(sequence_key_type_info);
  return nullptr;
}

ORT_API_STATUS_IMPL(winmla::GetSequenceElementType, const OrtSequenceTypeInfo* sequence_type_info, OrtTypeInfo** out)NO_EXCEPTION {
  *out = sequence_type_info->sequence_key_type_;
  return nullptr;
}

ORT_API(void, winmla::ReleaseSequenceTypeInfo, OrtSequenceTypeInfo* ptr) {
  delete ptr;
}