// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/onnxruntime_sequence_type_info.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

OrtSequenceTypeInfo::OrtSequenceTypeInfo(std::unique_ptr<OrtTypeInfo> sequence_key_type) noexcept
    : sequence_key_type_(std::move(sequence_key_type)) {
}

OrtSequenceTypeInfo::~OrtSequenceTypeInfo() = default;

std::unique_ptr<OrtSequenceTypeInfo> OrtSequenceTypeInfo::FromTypeProto(
    const ONNX_NAMESPACE::TypeProto& type_proto) {
  const auto value_case = type_proto.value_case();

  ORT_ENFORCE(value_case == ONNX_NAMESPACE::TypeProto::kSequenceType, "type_proto is not of type sequence!");

  const auto& type_proto_sequence = type_proto.sequence_type();
  auto key_type_info = OrtTypeInfo::FromTypeProto(type_proto_sequence.elem_type());

  return std::make_unique<OrtSequenceTypeInfo>(std::move(key_type_info));
}

std::unique_ptr<OrtSequenceTypeInfo> OrtSequenceTypeInfo::Clone() const {
  auto key_type_copy = sequence_key_type_->Clone();
  return std::make_unique<OrtSequenceTypeInfo>(std::move(key_type_copy));
}

ORT_API_STATUS_IMPL(OrtApis::GetSequenceElementType,
                    _In_ const OrtSequenceTypeInfo* sequence_type_info,
                    _Outptr_ OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto key_type_copy = sequence_type_info->sequence_key_type_->Clone();
  *out = key_type_copy.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseSequenceTypeInfo, _Frees_ptr_opt_ OrtSequenceTypeInfo* ptr) {
  std::unique_ptr<OrtSequenceTypeInfo> p(ptr);
}