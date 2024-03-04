// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/label_encoder_fusion.h"
#include <iostream>

#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_kernel_info.h"
#include "onnx/proto_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

#define LABEL_ENCODER_VALID_FOR_FUSING(from, mid, to)                              \
  node.GetAttributes().find("keys_" from) != node.GetAttributes().end() &&         \
      node.GetAttributes().find("values_" mid) != node.GetAttributes().end() &&    \
      next_node.GetAttributes().find("keys_" mid) != node.GetAttributes().end() && \
      next_node.GetAttributes().find("values_" to) != node.GetAttributes().end()

#define KEYS_ATTR_NAME(T) ("keys_" + getTypeNameString<T>() + "s")
#define VALUES_ATTR_NAME(T) ("values_" + getTypeNameString<T>() + "s")
#define DEFAULT_VALUE_ATTR_NAME(T) ("default_" + getTypeNameString<T>())

/**
Transform that fuses two consecutive LabelEncoder nodes
into one LabelEncoder node.
 */
bool LabelEncoderFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(
          node, "LabelEncoder", {4}, "ai.onnx.ml") ||
      node.GetOutputEdgesCount() != 1) {
    return false;
  }

  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "LabelEncoder", {4}, "ai.onnx.ml") ||
      // Make sure the two nodes do not span execution providers.
      next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
    return false;
  }

  if (graph.NodeProducesGraphOutput(node)) {
    return false;
  }

  // Is one of the supported operations
  return LABEL_ENCODER_VALID_FOR_FUSING("strings", "int64s", "strings");
}

template <class T1>
std::string getTypeNameString() {
  if constexpr (std::is_same<T1, int64_t>()) {
    return "int64";
  } else if constexpr (std::is_same<T1, std::string>()) {
    return "string";
  } else if constexpr (std::is_same<T1, float>()) {
    return "float";
  }
}

template <class T1, class T2, class T3>
Status LabelEncoderFusion::ApplyHelper(
    Graph& graph,
    Node& node,
    Node& next_node,
    RewriteRuleEffect& rule_effect) const {
  ProtoHelperNodeContext node_helper_ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> node_helper(&node_helper_ctx);

  ProtoHelperNodeContext next_node_helper_ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> next_node_helper(&next_node_helper_ctx);

  const std::vector<T1> node_keys =
      node_helper.GetAttrsOrDefault<T1>(KEYS_ATTR_NAME(T1));
  const std::vector<T2> node_values =
      node_helper.GetAttrsOrDefault<T2>(VALUES_ATTR_NAME(T2));
  const T2 node_default =
      node_helper.GetAttr<T2>(DEFAULT_VALUE_ATTR_NAME(T2));

  const std::vector<T2> next_node_keys =
      next_node_helper.GetAttrsOrDefault<T2>(KEYS_ATTR_NAME(T2));
  const std::vector<T3> next_node_values =
      next_node_helper.GetAttrsOrDefault<T3>(VALUES_ATTR_NAME(T3));
  const T3 next_node_default =
      next_node_helper.GetAttr<T3>(DEFAULT_VALUE_ATTR_NAME(T3));

  const auto getFromMapDefault = [](const auto& mp, const auto key, const auto def) {
    return (mp.find(key) == mp.end()) ? def : mp.at(key);
  };

  // Perform value propagation through the second label encoder
  std::map<T2, T3> mapping = {};
  for (size_t i = 0; i < next_node_keys.size(); i++) {
    mapping[next_node_keys[i]] = next_node_values[i];
  }

  std::vector<T1> new_node_keys = {};
  std::vector<T3> new_node_values = {};
  const auto new_node_default = getFromMapDefault(mapping, node_default, next_node_default);

  for (const T1& node_key : node_keys) {
    new_node_keys.push_back(node_key);
  }

  for (const T2& node_value : node_values) {
    new_node_values.push_back(getFromMapDefault(mapping, node_value, next_node_default));
  }

  // Remove old attributes
  node.ClearAttribute(KEYS_ATTR_NAME(T1));
  node.ClearAttribute(VALUES_ATTR_NAME(T2));
  node.ClearAttribute(DEFAULT_VALUE_ATTR_NAME(T2));

  node.AddAttribute(KEYS_ATTR_NAME(T1), new_node_keys);
  node.AddAttribute(VALUES_ATTR_NAME(T3), new_node_values);
  node.AddAttribute(DEFAULT_VALUE_ATTR_NAME(T3), new_node_default);

  graph_utils::FinalizeNodeFusion(graph, node, next_node);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

Status LabelEncoderFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& logger) const {
  auto& next_node = *graph.GetNode(node.OutputNodesBegin()->Index());

  if (LABEL_ENCODER_VALID_FOR_FUSING("strings", "int64s", "strings")) {
    return ApplyHelper<std::string, int64_t, std::string>(
        graph, node, next_node, rule_effect);
  }

  return Status::OK();
}

}  // namespace onnxruntime
