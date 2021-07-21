// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

namespace onnxruntime {

SelectorActionTransformer::SelectorActionTransformer(const std::string& name,
                                                     SelectorsAndActions&& selectors_and_actions,
                                                     bool save)
    : GraphTransformer{name},
      selectors_and_actions_{std::move(selectors_and_actions)},
      save_{save} {
#if !defined(ORT_MINIMAL_BUILD)
  // setup a map so we lookup by operator type efficiently
  for (const auto& map_entry : selectors_and_actions_.SelectorsAndActionsMap()) {
    for (const auto& op_info : map_entry.second->ops_and_versions) {
      bool inserted = op_type_to_selector_and_action_.insert({op_info.first, &*map_entry.second}).second;

      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
#endif
}

#if !defined(ORT_MINIMAL_BUILD)
void SelectorsAndActions::RegisterSelectorAndAction(const std::string& name,
                                                    const SelectorAndAction::OpVersionsMap& ops_and_versions_in,
                                                    std::unique_ptr<NodeSelector> selector_in,
                                                    std::unique_ptr<Action> action_in) {
  // currently all registrations are done from internal code with no external inputs,
  // so throw for invalid usage as it should only happen during development.
  ORT_ENFORCE(selectors_and_actions_map_.find(name) == selectors_and_actions_map_.cend(),
              "Existing registration with name ", name);

  auto entry = std::make_unique<SelectorAndAction>(name,
                                                   ops_and_versions_in,
                                                   std::move(selector_in),
                                                   std::move(action_in));

  ORT_IGNORE_RETURN_VALUE(selectors_and_actions_map_.emplace(name, std::move(entry)));
}

// check if the node matches any of the registered operators.
// if it does, run the Selector.
// if that selects nodes, run the Action.
Status SelectorActionTransformer::MatchAndProcess(Graph& graph, Node& node, bool& modified,
                                                  const logging::Logger& logger) const {
  Status status = Status::OK();

  do {
    // TODO: for now this just needs to support ONNX ops. If we ever had a transformer that was going to
    // target non-ONNX ops we'd need to rework a few things to include the op domain in the matches
    if (node.Domain() != kOnnxDomain) {
      break;
    }

    auto op_rule = op_type_to_selector_and_action_.find(node.OpType());
    if (op_rule == op_type_to_selector_and_action_.cend()) {
      break;
    }

    const auto& selector_and_actions = *op_rule->second;

    // check the supported versions if specified
    const auto& versions = selector_and_actions.ops_and_versions.find(node.OpType())->second;
    if (!versions.empty()) {
      if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
        break;
      }
    }

    std::unique_ptr<NodesToOptimize> node_group;
    if (!selector_and_actions.selector->Select(graph, node, node_group)) {
      break;
    }

    LOGS(logger, VERBOSE) << "Matched " << node.OpType();

    if (save_) {
      // TODO: save to Graph using transformer and action name so the node groups and actions are scoped to a
      // specific transformer.
      // e.g. map<transformer name, map<action name, vector<NodesToOptimizeIndexes>>>
      ORT_NOT_IMPLEMENTED("TODO: Save the selected nodes into the Graph.");
    } else {
      status = selector_and_actions.action->Run(graph, *node_group);
      if (!status.IsOK()) {
        break;
      }

      modified = true;
    }
  } while (false);

  return status;
}
#else
void SelectorsAndActions::RegisterAction(const std::string& name,
                                         std::unique_ptr<Action> action) {
  ORT_ENFORCE(actions_map_.find(name) == actions_map_.cend(), "Existing registration with name ", name);

  ORT_IGNORE_RETURN_VALUE(actions_map_.emplace(name, std::move(action)));
}

// TODO: The implementation here is purely an example to give an idea of how it might be done.
//
// The optimization info would be most conveniently stored in the Graph instance under the transformer name
// as that makes de/serialization to ORT format simple (done as part of Graph de/serialization)
// as well as handling subgraphs (values are stored in the current graph, be that the main graph or the subgraph).
struct ActionReplay {
  const std::string action_name;
  std::vector<NodesToOptimizeIndexes> node_groups;
};

Status SelectorActionTransformer::ApplySaved(Graph& graph, bool& modified, const logging::Logger& /*logger*/) const {
  auto fake_get_saved_actions = [](const std::string& /*transfomer_name*/) {
    return std::vector<ActionReplay>();
  };

  // retrieve any actions saved by this transformer in the Graph
  // TODO - setup infra for this to come from the Graph instance. could also have a 'clear all' method to free
  // all the saved actions once we're done replaying them.
  const std::vector<ActionReplay>& saved_actions = fake_get_saved_actions(Name());  // =  graph.GetSavedActions(Name());

  const auto& actions_map = selectors_and_actions_.ActionsMap();
  const auto actions_map_end = actions_map.cend();

  for (const auto entry : saved_actions) {
    auto action_iter = actions_map.find(entry.action_name);
    if (action_iter == actions_map_end) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Missing action ", entry.action_name, " for transformer ", Name());
    }

    const std::unique_ptr<Action>& action = action_iter->second;

    for (const NodesToOptimizeIndexes& node_group : entry.node_groups) {
      NodesToOptimize nodes_to_optimize{graph, node_group};

      // all nodes in the group are still available if IsValid returns true
      if (nodes_to_optimize.IsValid()) {
        ORT_RETURN_IF_ERROR(action->Run(graph, nodes_to_optimize));
        modified = true;
      }
    }
  }

  return Status::OK();
}

#endif

Status SelectorActionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  // TODO: Is there any reason to create a new GraphViewer? Do we need the different topological sort or can we
  // just use graph.GetNodesInTopologicalOrder() and avoid the overhead of re-sorting.
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;  // was removed by this transformer
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

#if !defined(ORT_MINIMAL_BUILD)
    // TODO: use GraphTransformer::GetCompatibleExecutionProviders if we need something more flexible
    if (node->GetExecutionProviderType() == kCpuExecutionProvider) {
      ORT_RETURN_IF_ERROR(MatchAndProcess(graph, *node, modified, logger));
    }
#else
    ORT_RETURN_IF_ERROR(ApplySaved(graph, modified, logger));
#endif
  }

  return Status::OK();
}

}  // namespace onnxruntime
