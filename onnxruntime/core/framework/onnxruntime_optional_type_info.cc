// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/onnxruntime_optional_type_info.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

OrtOptionalTypeInfo::OrtOptionalTypeInfo(OrtTypeInfo::Ptr contained_type) noexcept
    : contained_type_(std::move(contained_type)) {
}

OrtOptionalTypeInfo::~OrtOptionalTypeInfo() = default;

OrtOptionalTypeInfo::Ptr OrtOptionalTypeInfo::FromTypeProto(const ONNX_NAMESPACE::TypeProto& type_proto) {
  const auto value_case = type_proto.value_case();

  if (value_case != ONNX_NAMESPACE::TypeProto::kOptionalType) {
    ORT_THROW("type_proto is not of optional type");
  }

  const auto& type_proto_optional = type_proto.optional_type();
  auto contained_type_info = OrtTypeInfo::FromTypeProto(type_proto_optional.elem_type());

  return std::make_unique<OrtOptionalTypeInfo>(std::move(contained_type_info));
}

OrtOptionalTypeInfo::Ptr OrtOptionalTypeInfo::Clone() const {
  auto contained_type_copy = contained_type_->Clone();
  return std::make_unique<OrtOptionalTypeInfo>(std::move(contained_type_copy));
}

 ORT_API_STATUS_IMPL(OrtApis::GetOptionalContainedTypeInfo, _In_ const OrtOptionalTypeInfo* optional_type_info,
                     _Outptr_ OrtTypeInfo** out) {
   API_IMPL_BEGIN
   auto type_info = optional_type_info->contained_type_->Clone();
   *out = type_info.release();
   return nullptr;
   API_IMPL_END
 }
