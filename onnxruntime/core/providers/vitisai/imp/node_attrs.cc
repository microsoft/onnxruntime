// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "vaip/node_attrs.h"
#include "vaip/vai_assert.h"

namespace vaip {
static onnx::AttributeProto make_attribute(const std::string& name,
                                           int64_t value) {
  auto ret = onnx::AttributeProto();
  ret.set_name(name);
  ret.set_type(onnx::AttributeProto::INT);
  ret.set_i(value);
  return ret;
}

static onnx::AttributeProto make_attribute(const std::string& name,
                                           const std::vector<int64_t> value) {
  auto ret = onnx::AttributeProto();
  ret.set_name(name);
  ret.set_type(onnx::AttributeProto::INTS);
  for (auto v : value) {
    ret.add_ints(v);
  }
  return ret;
}

static onnx::AttributeProto make_attribute(const std::string& name,
                                           const std::string& value) {
  auto ret = onnx::AttributeProto();
  ret.set_name(name);
  ret.set_type(onnx::AttributeProto::STRING);
  ret.set_s(value);
  return ret;
}
static onnx::AttributeProto make_attribute(
    const std::string& name, const std::vector<std::string>& value) {
  auto ret = onnx::AttributeProto();
  ret.set_name(name);
  ret.set_type(onnx::AttributeProto::STRINGS);
  for (auto v : value) {
    ret.add_strings(v);
  }
  return ret;
}

static onnx::AttributeProto make_attribute(const std::string& name,
                                           const std::vector<float>& value) {
  auto ret = onnx::AttributeProto();
  ret.set_name(name);
  ret.set_type(onnx::AttributeProto::FLOATS);
  for (auto v : value) {
    ret.add_floats(v);
  }
  return ret;
}

static onnx::AttributeProto make_attribute(const std::string& name,
                                           const onnx::TensorProto& value) {
  auto ret = onnx::AttributeProto();
  ret.set_name(name);
  ret.set_type(onnx::AttributeProto::TENSOR);
  *(ret.mutable_t()) = std::move(value);
  return ret;
}  // namespace vaip

NodeAttr::NodeAttr(const std::string& name, int64_t value)
    : attribute_proto_{make_attribute(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const std::vector<int64_t>& value)
    : attribute_proto_{make_attribute(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const std::string& value)
    : attribute_proto_{make_attribute(name, value)} {}

NodeAttr::NodeAttr(const std::string& name,
                   const std::vector<std::string>& value)
    : attribute_proto_{make_attribute(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const std::vector<float>& value)
    : attribute_proto_{make_attribute(name, value)} {}

NodeAttr::NodeAttr(const std::string& name, const onnx::TensorProto& value)
    : attribute_proto_{make_attribute(name, value)} {}

onnx::AttributeProto& NodeAttr::get() { return attribute_proto_; }

NodeAttributesBuiler::NodeAttributesBuiler(size_t capacity) : attrs_{} {
  attrs_.reserve(capacity);
}

NodeAttributes NodeAttributesBuiler::build() {
  auto ret = NodeAttributes();
  ret.reserve(attrs_.size());
  for (auto& node_attr : attrs_) {
    onnx::AttributeProto& attr_proto = node_attr.get();
    auto name = attr_proto.name();
    ret.insert(std::make_pair(name, std::move(attr_proto)));
  }
  attrs_.clear();
  return ret;
}

void NodeAttributesBuiler::merge_into(Node& node) {
  merge_into(node.GetMutableAttributes());
}

void NodeAttributesBuiler::merge_into(NodeAttributes& attrs) {
  for (auto& attr : attrs_) {
    vai_assert(attr.get().has_name(), std::string("attr must has name " + attr.get().DebugString()));
    auto name = attr.get().name();
    attrs.insert_or_assign(std::move(name), std::move(attr.get()));
  }
}
}  // namespace vaip
