// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/onnxruntime_map_type_info.h"
#include "core/framework/onnxruntime_typeinfo.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/ort_apis.h"
#include "core/framework/error_code_helper.h"

OrtMapTypeInfo::OrtMapTypeInfo(ONNXTensorElementDataType map_key_type,
                               std::unique_ptr<OrtTypeInfo> map_value_type) noexcept
    : map_key_type_(map_key_type), map_value_type_(std::move(map_value_type)) {
}

OrtMapTypeInfo::~OrtMapTypeInfo() = default;

std::unique_ptr<OrtMapTypeInfo> OrtMapTypeInfo::FromTypeProto(
    const ONNX_NAMESPACE::TypeProto& type_proto) {
  auto value_case = type_proto.value_case();

  ORT_ENFORCE(value_case == ONNX_NAMESPACE::TypeProto::kMapType, "type_proto is not of type map!");

  // Get the key type of the map
  const auto& type_proto_map = type_proto.map_type();
  const auto map_key_type = onnxruntime::type_info_internal::ToONNXTensorElementDataType(
      ONNX_NAMESPACE::TensorProto_DataType(type_proto_map.key_type()));

  // Get the value type of the map
  auto map_value_type_info = OrtTypeInfo::FromTypeProto(type_proto_map.value_type());

  return std::make_unique<OrtMapTypeInfo>(map_key_type, std::move(map_value_type_info));
}

std::unique_ptr<OrtMapTypeInfo> OrtMapTypeInfo::Clone() const {
  auto map_value_type_copy = map_value_type_->Clone();
  return std::make_unique<OrtMapTypeInfo>(map_key_type_, std::move(map_value_type_copy));
}

// OrtMapTypeInfo Accessors
ORT_API_STATUS_IMPL(OrtApis::GetMapKeyType, _In_ const OrtMapTypeInfo* map_type_info,
                    _Out_ enum ONNXTensorElementDataType* out) {
  API_IMPL_BEGIN
  *out = map_type_info->map_key_type_;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtApis::GetMapValueType,
                    _In_ const OrtMapTypeInfo* map_type_info, _Outptr_ OrtTypeInfo** out) {
  API_IMPL_BEGIN
  auto clone = map_type_info->map_value_type_->Clone();
  *out = clone.release();
  return nullptr;
  API_IMPL_END
}

ORT_API(void, OrtApis::ReleaseMapTypeInfo, _Frees_ptr_opt_ OrtMapTypeInfo* ptr) {
  std::unique_ptr<OrtMapTypeInfo> p(ptr);
}
