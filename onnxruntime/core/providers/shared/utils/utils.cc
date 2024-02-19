
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

namespace {
bool GetClipMinMaxImpl(std::function<const ONNX_NAMESPACE::TensorProto*(const std::string&)> get_const_initializer,
                       const Node& node, float& min, float& max, const logging::Logger& logger) {
  const auto& node_name = node.Name();
  int32_t input_type;
  if (!GetType(*node.InputDefs()[0], input_type, logger)) {
    return false;
  }

  min = std::numeric_limits<float>::lowest();
  max = std::numeric_limits<float>::max();

  if (node.SinceVersion() < 11) {  // Clip opset 1, 6 is using attributes for min/max
    NodeAttrHelper helper(node);
    // attributes will be always float
    min = helper.Get("min", std::numeric_limits<float>::lowest());
    max = helper.Get("max", std::numeric_limits<float>::max());
  } else {
    auto get_value =
        [&](const ONNX_NAMESPACE::TensorProto* initializer, std::string_view type, float& value) -> bool {
      if (!initializer) {
        LOGS(logger, VERBOSE) << type << " input of Clip must be a constant initializer";
        return false;
      }

      Initializer unpacked_tensor_min(*initializer);
      switch (input_type) {
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
          value = unpacked_tensor_min.DataAsSpan<float>()[0];
          break;
        case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
          value = unpacked_tensor_min.DataAsSpan<MLFloat16>()[0].ToFloat();
          break;
        default:
          LOGS(logger, VERBOSE) << "GetClipMinMax() only supports float and float16 as min and max inputs for now."
                                << " The node [" << node_name << "] has input type: " << input_type;
          return false;
      }

      return true;
    };

    // min and max are both optional. could have neither, one or both.
    if (node.InputDefs().size() > 1 && node.InputDefs()[1]->Exists()) {
      // we have input min
      const auto& min_name = node.InputDefs()[1]->Name();
      const auto* min_value = get_const_initializer(min_name);
      if (!get_value(min_value, "Min", min)) {
        return false;
      }
    }

    if (node.InputDefs().size() > 2 && node.InputDefs()[2]->Exists()) {
      // we have input max
      const auto& max_name = node.InputDefs()[2]->Name();
      const auto* max_value = get_const_initializer(max_name);
      if (!get_value(max_value, "Max", max)) {
        return false;
      }
    }
  }

  return true;
}
}  // namespace

bool GetClipMinMax(const GraphViewer& graph_viewer, const Node& node, float& min, float& max,
                   const logging::Logger& logger) {
  return GetClipMinMaxImpl(
      [&graph_viewer](const std::string& name) -> const ONNX_NAMESPACE::TensorProto* {
        return graph_viewer.GetConstantInitializer(name);
      },
      node, min, max, logger);
}

// deprecated version that is not able to check if the initializer is constant
bool GetClipMinMax(const InitializedTensorSet& initializers, const Node& node, float& min, float& max,
                   const logging::Logger& logger) {
  return GetClipMinMaxImpl(
      [&initializers](const std::string& name) -> const ONNX_NAMESPACE::TensorProto* {
        auto entry = initializers.find(name);
        return entry == initializers.end() ? nullptr : entry->second;
      },
      node, min, max, logger);
}

NodeAttrHelper::NodeAttrHelper(const onnxruntime::Node& node)
    : node_attributes_(node.GetAttributes()) {}

NodeAttrHelper::NodeAttrHelper(const NodeUnit& node_unit)
    : node_attributes_(node_unit.GetNode().GetAttributes()) {}

float NodeAttrHelper::Get(const std::string& key, float def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return entry->second.f();
  }

  return def_val;
}

int32_t NodeAttrHelper::Get(const std::string& key, int32_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return narrow<int32_t>(entry->second.i());
  }

  return def_val;
}

uint32_t NodeAttrHelper::Get(const std::string& key, uint32_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return narrow<uint32_t>(entry->second.i());
  }

  return def_val;
}

int64_t NodeAttrHelper::Get(const std::string& key, int64_t def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return entry->second.i();
  }

  return def_val;
}

const std::string& NodeAttrHelper::Get(const std::string& key, const std::string& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    return entry->second.s();
  }

  return def_val;
}

std::vector<int32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int32_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& attr = entry->second;
    std::vector<int32_t> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    std::transform(attr.ints().cbegin(), attr.ints().cend(), std::back_inserter(v),
                   [](int64_t val) -> int32_t { return narrow<int32_t>(val); });
    return v;
  }

  return def_val;
}

std::vector<uint32_t> NodeAttrHelper::Get(const std::string& key, const std::vector<uint32_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& attr = entry->second;
    std::vector<uint32_t> v;
    v.reserve(static_cast<size_t>(attr.ints_size()));
    std::transform(attr.ints().cbegin(), attr.ints().cend(), std::back_inserter(v),
                   [](int64_t val) -> uint32_t { return narrow<uint32_t>(val); });
    return v;
  }

  return def_val;
}

std::vector<int64_t> NodeAttrHelper::Get(const std::string& key, const std::vector<int64_t>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = entry->second.ints();
    return std::vector<int64_t>{values.cbegin(), values.cend()};
  }

  return def_val;
}

std::vector<float> NodeAttrHelper::Get(const std::string& key, const std::vector<float>& def_val) const {
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = entry->second.floats();
    return std::vector<float>{values.cbegin(), values.cend()};
  }

  return def_val;
}

std::optional<float> NodeAttrHelper::GetFloat(const std::string& key) const {
  std::optional<float> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = entry->second.f();
  }

  return result;
}

std::optional<int64_t> NodeAttrHelper::GetInt64(const std::string& key) const {
  std::optional<int64_t> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = entry->second.i();
  }

  return result;
}

std::optional<std::vector<float>> NodeAttrHelper::GetFloats(const std::string& key) const {
  std::optional<std::vector<float>> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = entry->second.floats();
    result = std::vector<float>(values.begin(), values.end());
  }

  return result;
}

std::optional<std::vector<int64_t>> NodeAttrHelper::GetInt64s(const std::string& key) const {
  std::optional<std::vector<int64_t>> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    const auto& values = entry->second.ints();
    result = std::vector<int64_t>(values.begin(), values.end());
  }

  return result;
}

std::optional<std::string> NodeAttrHelper::GetString(const std::string& key) const {
  std::optional<std::string> result;
  if (auto entry = node_attributes_.find(key); entry != node_attributes_.end()) {
    result = entry->second.s();
  }

  return result;
}

bool NodeAttrHelper::HasAttr(const std::string& key) const {
  return Contains(node_attributes_, key);
}

}  // namespace onnxruntime
