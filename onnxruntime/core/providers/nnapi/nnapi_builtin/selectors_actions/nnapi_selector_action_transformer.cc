// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nnapi/nnapi_builtin/selectors_actions/nnapi_selector_action_transformer.h"

namespace onnxruntime {

NNAPISelectorActionTransformer::NNAPISelectorActionTransformer(const std::string& name,
                                                               NNAPIQDQSelectorsAndActions&& nnapi_qdq_selectors_and_actions)
    : name_{name},
      nnapi_qdq_selectors_and_actions_{std::move(nnapi_qdq_selectors_and_actions)} {
  // setup a map so we lookup by operator type efficiently
  for (const auto& map_entry : nnapi_qdq_selectors_and_actions_.NNAPIQDQSelectorsAndActionsMap()) {
    for (const auto& op_info : map_entry.second->ops_and_versions) {
      bool inserted = op_type_to_nnapi_qdq_sat_.insert({op_info.first, &*map_entry.second}).second;
      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
}

void NNAPIQDQSelectorsAndActions::RegisterSelector(const std::string& name,
                                                   const NNAPIQDQSelectorAndAction::OpVersionsMap& ops_and_versions_in,
                                                   std::unique_ptr<NNAPIQDQNodeSelector> selector_in) {
  ORT_ENFORCE(nnapi_selectors_and_actions_map_.find(name) == nnapi_selectors_and_actions_map_.cend(),
              "NNAPI: Existing registration with name ", name);

  auto entry = std::make_unique<NNAPIQDQSelectorAndAction>(name,
                                                           ops_and_versions_in,
                                                           std::move(selector_in));

  ORT_IGNORE_RETURN_VALUE(nnapi_selectors_and_actions_map_.emplace(name, std::move(entry)));
}

std::unique_ptr<ConstNodesToOptimize>
NNAPISelectorActionTransformer::Match(const Graph& graph, const Node& node) const {
  std::unique_ptr<ConstNodesToOptimize> node_group;

  // TODO: some nnapi specific checks?
  auto op_rule = op_type_to_nnapi_qdq_sat_.find(node.OpType());
  if (op_rule == op_type_to_nnapi_qdq_sat_.cend()) {
    std::cout << "op_rule is not found:  " << node.OpType() << std::endl;
    return node_group;
  }

  const auto& nnapi_qdq_selector_and_actions = *op_rule->second;

  // check the supported versions if specified
  const auto& versions = nnapi_qdq_selector_and_actions.ops_and_versions.find(node.OpType())->second;
  if (!versions.empty()) {
    if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
      std::cout << "Version is not supported" << std::endl;
      return node_group;
    }
  }

  if (!nnapi_qdq_selector_and_actions.nnapi_selector->Select(graph, node, node_group)) {
    std::cout << "No QDQ group matched: " << node.OpType() << std::endl;
    return node_group;
  }

  std::cout << "Found matched QDQ group with op: " << node.OpType() << "  with name: " << node.Name() << std::endl;

  //TODO: how to return a set of node_groups here?
  return node_group;
}

}  // namespace onnxruntime