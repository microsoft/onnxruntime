// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "./attr_proto.h"

#include "vaip/vai_assert.h"

#include <cmath>
#include <functional>
#include <string>
#include <unordered_map>

namespace vaip {

ONNX_NAMESPACE::AttributeProto* attr_proto_new_int(const std::string& name,
                                                   int64_t value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_INT);
  ret->set_i(value);
  return ret;
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_float(const std::string& name,
                                                     float value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_FLOAT);
  ret->set_f(value);
  return ret;
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_string(
    const std::string& name, const std::string& value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_STRING);
  ret->set_s(value);
  return ret;
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_tensor(
    const std::string& name, const ONNX_NAMESPACE::TensorProto& value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_TENSOR);
  *ret->mutable_t() = value;
  return ret;
}
ONNX_NAMESPACE::AttributeProto* attr_proto_new_ints(
    const std::string& name, const std::vector<int64_t>& value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_INTS);
  ret->mutable_ints()->Reserve((int)value.size());
  for (auto v : value) {
    ret->add_ints(v);
  }
  return ret;
}

ONNX_NAMESPACE::AttributeProto* attr_proto_new_floats(
    const std::string& name, const std::vector<float>& value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_FLOATS);
  ret->mutable_floats()->Reserve((int)value.size());
  for (auto v : value) {
    ret->add_floats(v);
  }
  return ret;
}

ONNX_NAMESPACE::AttributeProto* attr_proto_new_strings(
    const std::string& name, const std::vector<std::string>& value) {
  auto ret = new onnx::AttributeProto();
  ret->set_name(name);
  ret->set_type(onnx::AttributeProto_AttributeType_STRINGS);
  ret->mutable_strings()->Reserve((int)value.size());
  for (auto& v : value) {
    ret->add_strings(v);
  }
  return ret;
}

int64_t attr_proto_get_int(const onnx::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_INT, attr.DebugString());
  return attr.i();
}

float attr_proto_get_float(const onnx::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_FLOAT, attr.DebugString());
  return attr.f();
}

const std::string& attr_proto_get_string(const onnx::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_STRING, attr.DebugString());
  return attr.s();
}

const ONNX_NAMESPACE::TensorProto& attr_proto_get_tensor(
    const onnx::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_TENSOR, attr.DebugString());
  return attr.t();
}

gsl::span<const int64_t> attr_proto_get_ints(const onnx::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_INTS, attr.DebugString());
  return gsl::span<const int64_t>(attr.ints());
}

gsl::span<const float> attr_proto_get_floats(const onnx::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_FLOATS, attr.DebugString());
  return gsl::span<const float>(attr.floats());
}

std::vector<std::string> attr_proto_get_strings(
    const ONNX_NAMESPACE::AttributeProto& attr) {
  vai_assert(attr.type() == onnx::AttributeProto_AttributeType_STRINGS, attr.DebugString());
  return std::vector<std::string>(attr.strings().begin(), attr.strings().end());
}

ONNX_NAMESPACE::AttributeProto attr_proto_from_i64(const std::string& name,
                                                   int64_t value) {
  ONNX_NAMESPACE::AttributeProto ret;
  ret.set_name(name);
  ret.set_i(value);
  return ret;
}

}  // namespace vaip
