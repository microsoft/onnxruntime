// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/onnxruntime_sequence_type_info.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

OrtSequenceTypeInfo::OrtSequenceTypeInfo(OrtTypeInfo* sequence_key_type) noexcept :
	sequence_key_type_(sequence_key_type, &OrtApis::ReleaseTypeInfo) {
}

OrtStatus* OrtSequenceTypeInfo::FromTypeProto(const ONNX_NAMESPACE::TypeProto* type_proto, OrtSequenceTypeInfo** out) {
  auto value_case = type_proto->value_case();
  if (value_case != ONNX_NAMESPACE::TypeProto::kSequenceType)
  {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "type_proto is not of type sequence!");;
  }

  auto type_proto_sequence = type_proto->sequence_type();
  OrtTypeInfo* sequence_key_type_info = nullptr;
  if (auto status = OrtTypeInfo::FromTypeProto(&type_proto_sequence.elem_type(), &sequence_key_type_info))
  {
    return status;
  }

  *out = new OrtSequenceTypeInfo(sequence_key_type_info);
  return nullptr;
}

OrtStatus* OrtSequenceTypeInfo::Clone(OrtSequenceTypeInfo** out) {
  OrtTypeInfo* sequence_key_type_copy = nullptr;
  if (auto status = sequence_key_type_->Clone(&sequence_key_type_copy))
  {
    return status;
  }
  *out = new OrtSequenceTypeInfo(sequence_key_type_copy);
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtApis::GetSequenceElementType, _In_ const OrtSequenceTypeInfo* sequence_type_info,
                    _Outptr_ OrtTypeInfo** out) {
  API_IMPL_BEGIN
  return sequence_type_info->sequence_key_type_->Clone(out);
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseSequenceTypeInfo, _Frees_ptr_opt_ OrtSequenceTypeInfo* ptr) {
  delete ptr;
}