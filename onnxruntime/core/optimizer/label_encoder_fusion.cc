// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <unordered_map>
#include <vector>
#include <string>

#include "core/optimizer/label_encoder_fusion.h"
#include "core/framework/op_node_proto_helper.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

#define KEYS_ATTR_NAME(T) ("keys_" + GetTypename<T>() + "s")
#define VALUES_ATTR_NAME(T) ("values_" + GetTypename<T>() + "s")
#define DEFAULT_VALUE_ATTR_NAME(T) ("default_" + GetTypename<T>())

// May be needed somewhere else
// Think about moving into utils
template <typename>
[[maybe_unused]] constexpr bool false_for_T = false;

template <typename T>
std::string GetTypename() {
  if constexpr (std::is_same<T, int64_t>()) {
    return "int64";
  } else if constexpr (std::is_same<T, std::string>()) {
    return "string";
  } else if constexpr (std::is_same<T, float>()) {
    return "float";
  } else {
    static_assert(false_for_T<T>, "Unsupported type");
  }
}

template <typename T>
size_t GetAttributeElementCount(const ONNX_NAMESPACE::AttributeProto& attribute) {
  if constexpr (std::is_same_v<T, int64_t>) {
    return static_cast<size_t>(attribute.ints_size());
  } else if constexpr (std::is_same_v<T, std::string>) {
    return static_cast<size_t>(attribute.strings_size());
  } else if constexpr (std::is_same_v<T, float>) {
    return static_cast<size_t>(attribute.floats_size());
  } else {
    static_assert(false_for_T<T>, "Unsupported type");
  }
}

template <typename T1, typename T2, typename T3>
bool LabelEncoderFusion::IsValidForFusion(const Node& node, const Node& next_node) const {
  const auto& node_attributes = node.GetAttributes();
  const auto node_keys = node_attributes.find(KEYS_ATTR_NAME(T1));
  const auto node_values = node_attributes.find(VALUES_ATTR_NAME(T2));
  const auto& next_node_attributes = next_node.GetAttributes();
  const auto next_node_keys = next_node_attributes.find(KEYS_ATTR_NAME(T2));
  const auto next_node_values = next_node_attributes.find(VALUES_ATTR_NAME(T3));

  if (node_keys == node_attributes.end() ||
      node_values == node_attributes.end() ||
      next_node_keys == next_node_attributes.end() ||
      next_node_values == next_node_attributes.end()) {
    return false;
  }

  const size_t node_key_count = GetAttributeElementCount<T1>(node_keys->second);
  const size_t next_node_key_count = GetAttributeElementCount<T2>(next_node_keys->second);
  return node_key_count > 0 &&
         next_node_key_count > 0 &&
         node_key_count == GetAttributeElementCount<T2>(node_values->second) &&
         next_node_key_count == GetAttributeElementCount<T3>(next_node_values->second);
}

/**
Transform that fuses two consecutive LabelEncoder nodes
into one LabelEncoder node.
 */
bool LabelEncoderFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(
          node, "LabelEncoder", {2, 4}, "ai.onnx.ml") ||
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
  return IsValidForFusion<std::string, std::string, std::string>(node, next_node) ||
         IsValidForFusion<std::string, std::string, int64_t>(node, next_node) ||
         IsValidForFusion<std::string, int64_t, std::string>(node, next_node) ||
         IsValidForFusion<std::string, int64_t, int64_t>(node, next_node) ||
         IsValidForFusion<int64_t, std::string, std::string>(node, next_node) ||
         IsValidForFusion<int64_t, std::string, int64_t>(node, next_node) ||
         IsValidForFusion<int64_t, int64_t, std::string>(node, next_node) ||
         IsValidForFusion<int64_t, int64_t, int64_t>(node, next_node);
}

/**
Since we need to be polymorphic on the datatype
we will dispatch to this method from the main Apply
*/
template <typename T1, typename T2, typename T3>
Status LabelEncoderFusion::ApplyHelper(
    Graph& graph,
    Node& node,
    Node& next_node,
    RewriteRuleEffect& rule_effect) const {
  ProtoHelperNodeContext node_helper_ctx(node);
  OpNodeProtoHelper<ProtoHelperNodeContext> node_helper(&node_helper_ctx);

  ProtoHelperNodeContext next_node_helper_ctx(next_node);
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

  ORT_RETURN_IF_NOT(node_keys.size() == node_values.size(),
                    "LabelEncoder keys and values attributes must have the same number of elements.");
  ORT_RETURN_IF_NOT(next_node_keys.size() == next_node_values.size(),
                    "LabelEncoder keys and values attributes must have the same number of elements.");

  const auto getFromMapDefault = [](const auto& mp, const auto key, const auto def) {
    return (mp.find(key) == mp.end()) ? def : mp.at(key);
  };

  // Perform value propagation through the second label encoder
  std::unordered_map<T2, T3> mapping = {};
  for (size_t i = 0; i < next_node_keys.size(); i++) {
    mapping[next_node_keys[i]] = next_node_values[i];
  }

  std::vector<T3> new_node_values = {};
  const auto new_node_default = getFromMapDefault(mapping, node_default, next_node_default);

  for (const T2& node_value : node_values) {
    new_node_values.push_back(getFromMapDefault(mapping, node_value, next_node_default));
  }

  // Remove old attributes:
  // The keys attribute is correct, we just reroute
  // the values
  node.ClearAttribute(VALUES_ATTR_NAME(T2));
  node.ClearAttribute(DEFAULT_VALUE_ATTR_NAME(T2));

  node.AddAttribute(VALUES_ATTR_NAME(T3), new_node_values);
  node.AddAttribute(DEFAULT_VALUE_ATTR_NAME(T3), new_node_default);

  graph_utils::FinalizeNodeFusion(graph, node, next_node);

  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

#define FUSE_IF_VALID(T1, T2, T3)                      \
  if (IsValidForFusion<T1, T2, T3>(node, next_node)) { \
    return ApplyHelper<T1, T2, T3>(                    \
        graph, node, next_node, rule_effect);          \
  }

Status LabelEncoderFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  auto& next_node = *graph.GetNode(node.OutputNodesBegin()->Index());

  FUSE_IF_VALID(std::string, std::string, std::string);
  FUSE_IF_VALID(std::string, std::string, int64_t);
  FUSE_IF_VALID(std::string, int64_t, std::string);
  FUSE_IF_VALID(std::string, int64_t, int64_t);
  FUSE_IF_VALID(int64_t, std::string, std::string);
  FUSE_IF_VALID(int64_t, std::string, int64_t);
  FUSE_IF_VALID(int64_t, int64_t, std::string);
  FUSE_IF_VALID(int64_t, int64_t, int64_t);

  return Status::OK();
}

}  // namespace onnxruntime
