// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./attr_proto.h"

#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>

#include "core/providers/shared_library/provider_api.h"

#include "./vai_assert.h"

namespace vaip {
ONNX_NAMESPACE::AttributeProto* attr_proto_new_int(const std::string& name, int64_t value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INT);
  ret->set_i(value);
  return ret.release();
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_float(const std::string& name, float value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT);
  ret->set_f(value);
  return ret.release();
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_string(const std::string& name, const std::string& value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRING);
  ret->set_s(value);
  return ret.release();
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_tensor(
    const std::string& name, const ONNX_NAMESPACE::TensorProto& value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR);
  *ret->add_tensors() = value;
  return ret.release();
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_ints(const std::string& name, const std::vector<int64_t>& value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_INTS);
  ret->mutable_ints()->Reserve((int)value.size());
  for (auto v : value) {
    ret->add_ints(v);
  }
  return ret.release();
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_floats(
    const std::string& name, const std::vector<float>& value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS);
  ret->mutable_floats()->Reserve((int)value.size());
  for (auto v : value) {
    ret->add_floats(v);
  }
  return ret.release();
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_strings(const std::string& name, const std::vector<std::string>& value) {
  auto ret = ONNX_NAMESPACE::AttributeProto::Create();
  ret->set_name(name);
  ret->set_type(ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS);
  for (auto& v : value) {
    ret->add_strings(v);
  }
  return ret.release();
}
int64_t attr_proto_get_int(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT, attr.name());
  return attr.i();
}
float attr_proto_get_float(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOAT, attr.name());
  return attr.f();
}
const std::string& attr_proto_get_string(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRING, attr.name());
  return attr.s();
}
const ONNX_NAMESPACE::TensorProto& attr_proto_get_tensor(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR, attr.name());
  return attr.t();
}
gsl::span<const int64_t> attr_proto_get_ints(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INTS, attr.name());
  return gsl::span<const int64_t>(attr.ints());
}
gsl::span<const float> attr_proto_get_floats(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_FLOATS, attr.name());
  return gsl::span<const float>(attr.floats());
}
std::vector<std::string> attr_proto_get_strings(const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == ONNX_NAMESPACE::AttributeProto_AttributeType_STRINGS, attr.name());
  std::vector<std::string> ret;
  ret.reserve(attr.strings_size());
  for (int i = 0; i < attr.strings_size(); i++) {
    ret.push_back(attr.strings(i));
  }
  return ret;
}
}  // namespace vaip
