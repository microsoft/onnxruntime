// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/graph/model.h>

#include <initializer_list>

#include "core/graph/basic_types.h"
namespace vaip {
using namespace onnxruntime;
class NodeAttr {
 public:
  NodeAttr(const std::string& name, int64_t value);
  NodeAttr(const std::string& name, const std::vector<int64_t>& value);
  NodeAttr(const std::string& name, const std::string& value);
  NodeAttr(const std::string& name, const std::vector<std::string>& value);
  NodeAttr(const std::string& name, const std::vector<float>& value);
  NodeAttr(const std::string& name, const onnx::TensorProto& value);

  onnx::AttributeProto& get();

 private:
  onnx::AttributeProto attribute_proto_;
};

class NodeAttributesBuiler {
 public:
  explicit NodeAttributesBuiler(size_t capacity = 10);
  NodeAttributesBuiler(const NodeAttributesBuiler&) = delete;
  NodeAttributesBuiler(NodeAttributesBuiler&&) = default;
  /// after build, all attrs_ are cleared.
  NodeAttributes build();
  /// for efficiency reason, after merge_into, all attrs_ are moved.
  void merge_into(Node& node);
  void merge_into(NodeAttributes& attrs);
  template <typename T>
  NodeAttributesBuiler& add(const std::string& name, T&& value) {
    attrs_.emplace_back(name, std::forward<T>(value));
    return *this;
  }

 private:
  std::vector<NodeAttr> attrs_;
};
}  // namespace vaip
