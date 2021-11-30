// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

#include <cassert>

#include "core/graph/runtime_optimization_record_container.h"

namespace onnxruntime {

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

#else  // !defined(ORT_MINIMAL_BUILD)

void SelectorsAndActions::RegisterAction(const std::string& name,
                                         std::unique_ptr<Action> action) {
  ORT_ENFORCE(actions_map_.find(name) == actions_map_.cend(), "Existing registration with name ", name);

  ORT_IGNORE_RETURN_VALUE(actions_map_.emplace(name, std::move(action)));
}

#endif  // !defined(ORT_MINIMAL_BUILD)

const Action* SelectorsAndActions::LookUpAction(const std::string& name) const {
#if !defined(ORT_MINIMAL_BUILD)

  if (const auto it = selectors_and_actions_map_.find(name);
      it != selectors_and_actions_map_.end() && it->second != nullptr) {
    return &*it->second->action;
  }
  return nullptr;

#else  // !defined(ORT_MINIMAL_BUILD)

  if (const auto it = actions_map_.find(name); it != actions_map_.end()) {
    return &*it->second;
  }
  return nullptr;

#endif  // !defined(ORT_MINIMAL_BUILD)
}

SelectorActionTransformer::SelectorActionTransformer(const std::string& name,
                                                     SelectorsAndActions&& selectors_and_actions,
                                                     const SatApplyContextVariant& apply_context)
    : GraphTransformer{name},
      selectors_and_actions_{std::move(selectors_and_actions)},
      apply_context_{apply_context} {
#if !defined(ORT_MINIMAL_BUILD)
  // setup a map so we lookup by operator type efficiently
  for (const auto& map_entry : selectors_and_actions_.SelectorsAndActionsMap()) {
    for (const auto& op_info : map_entry.second->ops_and_versions) {
      bool inserted = op_type_to_selector_and_action_.insert({op_info.first, &*map_entry.second}).second;

      ORT_ENFORCE(inserted, "Multiple entries for operator is not supported. OpType=", op_info.first);
    }
  }
#endif  // !defined(ORT_MINIMAL_BUILD)
}

#if !defined(ORT_MINIMAL_BUILD)

Status SelectorActionTransformer::MatchAndProcess(
    Graph& graph, const GraphViewer& graph_viewer, Node& node, bool& modified, const logging::Logger& logger,
    const SatRuntimeOptimizationSaveContext* save_context) const {
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

    const auto& selector_and_action = *op_rule->second;

    // check the supported versions if specified
    const auto& versions = selector_and_action.ops_and_versions.find(node.OpType())->second;
    if (!versions.empty()) {
      if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
        break;
      }
    }

    const auto node_selection_opt = selector_and_action.selector->Select(graph_viewer, node);
    if (!node_selection_opt.has_value()) {
      break;
    }
    const auto& node_selection = *node_selection_opt;

    LOGS(logger, VERBOSE) << "Matched " << node.OpType();

    NodesToOptimize node_group(graph, node_selection);

    if (save_context) {
#if defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
      const auto& action = *selector_and_action.action;

      Action::SavedState action_saved_state{};
      status = action.RunForSave(graph, node_group, *save_context, action_saved_state, modified);
      if (!status.IsOK()) {
        break;
      }

      graph.MutableRuntimeOptimizations().AddRecord(
          Name(),
          RuntimeOptimizationRecord{selector_and_action.name,
                                    node_selection,
                                    action_saved_state.produced_nodes});
#else   // defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                               "Saving runtime optimizations is not enabled in this build.");
      break;
#endif  // defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
    } else {
      status = selector_and_action.action->Run(graph, node_group);
      if (!status.IsOK()) {
        break;
      }

      modified = true;
    }
  } while (false);

  return status;
}

Status SelectorActionTransformer::ApplyDirect(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger,
                                              const SatRuntimeOptimizationSaveContext* save_context) const {
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;  // was removed by this transformer
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // TODO: use GraphTransformer::GetCompatibleExecutionProviders if we need something more flexible
    if (node->GetExecutionProviderType() == kCpuExecutionProvider) {
      ORT_RETURN_IF_ERROR(MatchAndProcess(graph, graph_viewer, *node, modified, logger, save_context));
    }
  }

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)

static Status RegisterProducedNodesWithGraph(NodeIndex pre_action_max_num_nodes, NodeIndex post_action_max_num_nodes,
                                             const RuntimeOptimizationRecord& record,
                                             Graph& graph) {
  assert(post_action_max_num_nodes >= pre_action_max_num_nodes);

  const auto num_new_node_indices = post_action_max_num_nodes - pre_action_max_num_nodes;

  auto produced_node_it = record.produced_nodes.begin();
  const auto produced_nodes_end = record.produced_nodes.end();

  std::unordered_map<NodeIndex, HashValue> node_index_to_kernel_def_hash{};

  for (NodeIndex i = 0; i < num_new_node_indices; ++i) {
    const NodeIndex new_node_idx = pre_action_max_num_nodes + i;
    const auto* new_node = graph.GetNode(new_node_idx);

    if (!new_node) continue;

    ORT_RETURN_IF(produced_node_it == produced_nodes_end,
                  "Not enough produced nodes in the runtime optimization record.");

    node_index_to_kernel_def_hash.emplace(new_node_idx, produced_node_it->kernel_def_hash);

    ++produced_node_it;
  }

  ORT_RETURN_IF(produced_node_it != produced_nodes_end, "Too many produced nodes in the runtime optimization record.");

  graph.MutableRuntimeOptimizationReplayCtx().produced_node_index_to_kernel_def_hash.merge(
      node_index_to_kernel_def_hash);

  return Status::OK();
}

Status SelectorActionTransformer::ApplyFromRuntimeOptimizations(
    Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  for (auto& node : graph.Nodes()) {
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  const auto records = graph.MutableRuntimeOptimizations().RemoveRecordsForOptimizer(Name());
  for (const auto& record : records) {
    const auto* action = selectors_and_actions_.LookUpAction(record.action_id);
    if (!action) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Missing action ", record.action_id, " for transformer ", Name());
    }

    NodesToOptimize nodes_to_optimize{graph, record.nodes_to_optimize_indices};

    if (!nodes_to_optimize.IsValid()) {
      continue;
    }

    // all nodes in the group are still available if IsValid returns true

    const NodeIndex pre_action_max_index = graph.MaxNodeIndex();

    ORT_RETURN_IF_ERROR(action->Run(graph, nodes_to_optimize));
    modified = true;

    const NodeIndex post_action_max_index = graph.MaxNodeIndex();

    ORT_RETURN_IF_ERROR(RegisterProducedNodesWithGraph(pre_action_max_index, post_action_max_index,
                                                       record, graph));
  }

  return Status::OK();
}

#endif  // defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)

Status SelectorActionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  if (std::holds_alternative<SatRuntimeOptimizationLoadContext>(apply_context_)) {
#if defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
    return ApplyFromRuntimeOptimizations(graph, modified, graph_level, logger);
#else   // defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Loading runtime optimizations is not enabled in this build.");
#endif  // defined(ORT_ENABLE_ORT_FORMAT_RUNTIME_GRAPH_OPTIMIZATION)
  }

  assert(std::holds_alternative<SatRuntimeOptimizationSaveContext>(apply_context_) ||
         std::holds_alternative<SatDirectApplicationContext>(apply_context_));

#if !defined(ORT_MINIMAL_BUILD)
  const auto* save_context = std::get_if<SatRuntimeOptimizationSaveContext>(&apply_context_);
  return ApplyDirect(graph, modified, graph_level, logger, save_context);
#else   // !defined(ORT_MINIMAL_BUILD)
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "Only loading runtime optimizations is supported in a minimal build.");
#endif  // !defined(ORT_MINIMAL_BUILD)
}

}  // namespace onnxruntime
