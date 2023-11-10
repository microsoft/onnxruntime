
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "utils.h"

#include <core/common/safeint.h>
#include <core/framework/tensorprotoutils.h>
#include <core/graph/graph.h>
#include <core/providers/common.h>
#include "core/providers/shared/node_unit/node_unit.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

bool GetType(const NodeArg& node_arg, int32_t& type, const logging::Logger& logger) {
  type = ONNX_NAMESPACE::TensorProto_DataType_UNDEFINED;
  const auto* type_proto = node_arg.TypeAsProto();
  if (!type_proto || !type_proto->has_tensor_type() || !type_proto->tensor_type().has_elem_type()) {
    LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has no input type";
    return false;
  }

  type = type_proto->tensor_type().elem_type();
  return true;
}

bool GetClipMinMax(const InitializedTensorSet& initializers, const Node& node,
                   float& min, float& max, const logging::Logger& logger) {
  const auto& node_name = node.Name();
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger))
    return false;

  if (input_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    LOGS(logger, VERBOSE) << "GetClipMinMax() only support Clip node with float inputs for now. "
                          << "The node [" << node_name << "] has input 0 type: " << input_type;
    return false;
  }

  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();

  if (node.SinceVersion() < 11) {  // Clip opset 1, 6 is using attributes for min/max
    NodeAttrHelper helper(node);
    min = helper.Get("min", std::numeric_limits<float>::lowest());
    max = helper.Get("max", std::numeric_limits<float>::max());
  } else {
    if (node.InputDefs().size() > 1) {  // we have input min
      const auto& min_name = node.InputDefs()[1]->Name();
      if (!Contains(initializers, min_name)) {
        LOGS(logger, VERBOSE) << "Input min of Clip must be known";
        return false;
      }
      Initializer unpacked_tensor(*initializers.at(min_name));
      min = unpacked_tensor.DataAsSpan<float>()[0];
    }

    if (node.InputDefs().size() > 2) {  // we have input max
      const auto& max_name = node.InputDefs()[2]->Name();
      if (!Contains(initializers, max_name)) {
        LOGS(logger, VERBOSE) << "Input max of Clip must be known";
        return false;
      }
      Initializer unpacked_tensor(*initializers.at(max_name));
      max = unpacked_tensor.DataAsSpan<float>()[0];
    }
  }

  return true;
}

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

NodeAttrHelper::NodeAttrHelper(const NodeUnit& node_unit)
    : node_attributes_(node_unit.GetNode().GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (!HasAttr(key))
    return def_val;

  return node_attributes_.at(key).f();
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (!HasAttr(key))
    return def_val;

  return SafeInt<int32_t>(node_attributes_.at(key).i());
}

uint32_t NodeAttrHelper::Get(const std::string& key, uint32_t def_val) const {
  if (!HasAttr(key))
    return def_val;

  return SafeInt<uint32_t>(node_attributes_.at(key).i());
}

int64_t NodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  if (!HasAttr(key))
    return def_val;

  return node_attributes_.at(key).i();
}

std::string NodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
  if (!HasAttr(key))
    return def_val;

  return node_attributes_.at(key).s();
}

std::vector<int32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& attr(node_attributes_.at(key));
  std::vector<int32_t> v;
  v.reserve(static_cast<size_t>(attr.ints_size()));
  std::transform(attr.ints().cbegin(), attr.ints().cend(), std::back_inserter(v),
                 [](int64_t val) -> int32_t { return SafeInt<int32_t>(val); });
  return v;
}

std::vector<uint32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<uint32_t>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& attr(node_attributes_.at(key));
  std::vector<uint32_t> v;
  v.reserve(static_cast<size_t>(attr.ints_size()));
  std::transform(attr.ints().cbegin(), attr.ints().cend(), std::back_inserter(v),
                 [](int64_t val) -> uint32_t { return SafeInt<uint32_t>(val); });
  return v;
}

std::vector<int64_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& source(node_attributes_.at(key).ints());
  return std::vector<int64_t>{source.cbegin(), source.cend()};
}

std::vector<float> NodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
  if (!HasAttr(key))
    return def_val;

  const auto& source(node_attributes_.at(key).floats());
  return std::vector<float>{source.cbegin(), source.cend()};
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return Contains(node_attributes_, key);
}

}  // namespace onnxruntime
