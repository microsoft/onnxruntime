// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/selectors_actions/selector_action_transformer.h"

#include <cassert>
#include <algorithm>
#include <iterator>
#include <utility>

#include "core/graph/op_identifier_utils.h"
#include "core/graph/runtime_optimization_record_container.h"

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)

void SelectorActionRegistry::RegisterSelectorAndAction(const std::string& name,
                                                       const OpVersionsMap& ops_and_versions_in,
                                                       std::unique_ptr<NodeSelector> selector_in,
                                                       std::unique_ptr<Action> action_in) {
  // currently all registrations are done from internal code with no external inputs,
  // so throw for invalid usage as it should only happen during development.
  const auto [name_to_entry_it, inserted_in_name_to_entry] =
      name_to_entry_.emplace(name,
                             Entry{name,
                                   ops_and_versions_in,
                                   std::move(selector_in),
                                   std::move(action_in)});
  ORT_ENFORCE(inserted_in_name_to_entry, "Existing registration with name ", name);

  const Entry& entry = name_to_entry_it->second;
  for (const auto& [op_type, versions] : entry.ops_and_versions) {
    ORT_UNUSED_PARAMETER(versions);
    op_type_to_entry_.emplace(op_type, &entry);
  }
}

#else  // !defined(ORT_MINIMAL_BUILD)

void SelectorActionRegistry::RegisterAction(const std::string& name,
                                            std::unique_ptr<Action> action) {
  // currently all registrations are done from internal code with no external inputs,
  // so throw for invalid usage as it should only happen during development.
  const bool inserted_in_name_to_entry = name_to_entry_.emplace(name, Entry{name, std::move(action)}).second;
  ORT_ENFORCE(inserted_in_name_to_entry, "Existing registration with name ", name);
}

#endif  // !defined(ORT_MINIMAL_BUILD)

const SelectorActionRegistry::Entry* SelectorActionRegistry::LookUp(const std::string& name) const {
  if (const auto it = name_to_entry_.find(name); it != name_to_entry_.end()) {
    return &it->second;
  }
  return nullptr;
}

#if !defined(ORT_MINIMAL_BUILD)
auto SelectorActionRegistry::LookUpByOpTypeAndDomain(const std::string& op_type, const std::string& domain) const
    -> std::vector<gsl::not_null<const Entry*>> {
  const auto [range_begin, range_end] = op_type_to_entry_.equal_range(domain == kOnnxDomain ? op_type : (domain + ":" + op_type));
  std::vector<gsl::not_null<const Entry*>> result{};
  result.reserve(std::distance(range_begin, range_end));
  std::transform(range_begin, range_end, std::back_inserter(result),
                 [](const std::pair<std::string, const Entry*> value) { return value.second; });
  return result;
}
#endif  // !defined(ORT_MINIMAL_BUILD)

SelectorActionTransformer::SelectorActionTransformer(const std::string& name,
                                                     SelectorActionRegistry&& selector_action_registry,
                                                     const SatApplyContextVariant& apply_context,
                                                     const InlinedHashSet<std::string_view>& compatible_execution_providers)
    : GraphTransformer{name, compatible_execution_providers},
      selector_action_registry_{std::move(selector_action_registry)},
      apply_context_{apply_context} {}

#if !defined(ORT_MINIMAL_BUILD)

// check if the node matches any of the registered operators.
// if it does, run the Selector.
// if that selects nodes, run or save the Action.
//
// Some part of the MatchAndProcess use a GraphViewer of the given graph,
// we choose to supply both the graph and the graph_viewer to avoid expensive
// and repeatedly construction of the graph_viewer.
// NOTE, the graph must be the same as the graph_viewer's underlying graph
static Status MatchAndProcess(
    Graph& graph, const GraphViewer& graph_viewer, Node& node, bool& modified, const logging::Logger& logger,
    const std::string& transformer_name,
    const SelectorActionRegistry& selector_action_registry,
    const SatRuntimeOptimizationSaveContext* save_context) {
  Status status = Status::OK();

  do {
    std::optional<NodesToOptimizeIndices> node_selection_opt{};
    const SelectorActionRegistry::Entry* selector_action_entry_ptr = nullptr;

    const auto selector_action_entries = selector_action_registry.LookUpByOpTypeAndDomain(node.OpType(), node.Domain());
    const auto& domain = node.Domain();
    std::string domain_optype = (domain == kOnnxDomain) ? node.OpType() : (domain + ":" + node.OpType());
    for (const auto& entry : selector_action_entries) {
      // check the supported versions if specified
      const auto& versions = entry->ops_and_versions.find(domain_optype)->second;
      if (!versions.empty()) {
        if (std::find(versions.cbegin(), versions.cend(), node.SinceVersion()) == versions.cend()) {
          continue;
        }
      }

      auto selection = entry->selector->Select(graph_viewer, node);
      if (!selection.has_value()) {
        continue;
      }

      node_selection_opt = std::move(selection);
      selector_action_entry_ptr = entry.get();
      break;
    }

    if (!selector_action_entry_ptr) {
      break;
    }

    LOGS(logger, VERBOSE) << "Matched " << node.OpType();

    const auto& selector_action_entry = *selector_action_entry_ptr;
    const auto& action = *selector_action_entry.action;
    const auto& node_selection = *node_selection_opt;
    const NodesToOptimize node_group(graph, node_selection);

    if (save_context) {
      // don't save a runtime optimization again if it already exists
      // this might happen if the transformer is run multiple times, e.g., from a graph transformer manager which may
      //   run its transformers in multiple passes
      if (graph.RuntimeOptimizations().RecordExists(transformer_name, selector_action_entry.name, node_selection)) {
        break;
      }

      Action::SavedState action_saved_state{};
      status = action.RunForSave(graph, node_group, *save_context, action_saved_state, modified);
      if (!status.IsOK()) {
        break;
      }

      RuntimeOptimizationRecord::ProducedOpIdVector produced_op_ids{};
      produced_op_ids.reserve(action_saved_state.produced_node_op_schemas.size());

      for (const auto op_schema : action_saved_state.produced_node_op_schemas) {
        produced_op_ids.push_back(utils::MakeOpId(*op_schema));
        if (save_context->record_produced_node_op_schema) {
          status = save_context->record_produced_node_op_schema(*op_schema);
          if (!status.IsOK()) {
            break;
          }
        }
      }

      // handle break out of above for loop on error
      if (!status.IsOK()) {
        break;
      }

      RuntimeOptimizationRecord runtime_optimization_record{selector_action_entry.name,
                                                            node_selection,
                                                            std::move(produced_op_ids)};

      graph.MutableRuntimeOptimizations().AddRecord(transformer_name,
                                                    std::move(runtime_optimization_record));

    } else {
      status = action.Run(graph, node_group);
      if (!status.IsOK()) {
        break;
      }

      modified = true;
    }
  } while (false);

  return status;
}

Status SelectorActionTransformer::ApplySelectorsAndActions(
    Graph& graph, bool& modified, int graph_level,
    const logging::Logger& logger,
    const SatRuntimeOptimizationSaveContext* save_context) const {
  GraphViewer graph_viewer(graph);

  for (auto index : graph_viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;  // was removed by this transformer
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders())) {
      ORT_RETURN_IF_ERROR(MatchAndProcess(graph, graph_viewer, *node, modified, logger,
                                          Name(), selector_action_registry_, save_context));
    }
  }

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

static Status SetOpSinceVersionForProducedNodes(NodeIndex pre_action_max_num_nodes,
                                                NodeIndex post_action_max_num_nodes,
                                                const RuntimeOptimizationRecord& record,
                                                Graph& graph) {
  assert(post_action_max_num_nodes >= pre_action_max_num_nodes);

  const auto num_new_node_indices = post_action_max_num_nodes - pre_action_max_num_nodes;

  auto produced_op_id_it = record.produced_op_ids.begin();
  const auto produced_op_ids_end = record.produced_op_ids.end();

  for (NodeIndex i = 0; i < num_new_node_indices; ++i) {
    const NodeIndex new_node_idx = pre_action_max_num_nodes + i;
    auto* new_node = graph.GetNode(new_node_idx);

    // only account for new nodes that still exist
    // an action could add a temporary node and then remove it
    if (!new_node) {
      continue;
    }

    ORT_RETURN_IF(produced_op_id_it == produced_op_ids_end,
                  "Not enough produced nodes in the runtime optimization record.");

    ORT_RETURN_IF(new_node->Domain() != produced_op_id_it->domain ||
                      new_node->OpType() != produced_op_id_it->op_type,
                  "New node op (", new_node->Domain(), ':', new_node->OpType(),
                  ") does not match produced node op in runtime optimization record (",
                  produced_op_id_it->domain, ':', produced_op_id_it->op_type, ").");

    assert(new_node->SinceVersion() == -1);

    new_node->SetSinceVersion(produced_op_id_it->since_version);

    ++produced_op_id_it;
  }

  ORT_RETURN_IF(produced_op_id_it != produced_op_ids_end, "Too many produced nodes in the runtime optimization record.");

  return Status::OK();
}

Status SelectorActionTransformer::ApplySavedRuntimeOptimizations(
    Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  for (auto& node : graph.Nodes()) {
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));
  }

  const auto records = graph.MutableRuntimeOptimizations().RemoveRecordsForOptimizer(Name());
  for (const auto& record : records) {
    LOGS(logger, VERBOSE) << "Applying runtime optimization action " << record.action_id
                          << " for transformer " << Name();

    const auto* selector_action_entry = selector_action_registry_.LookUp(record.action_id);
    if (!selector_action_entry) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Missing action ", record.action_id, " for transformer ", Name());
    }

    NodesToOptimize nodes_to_optimize{graph, record.nodes_to_optimize_indices};

    if (!nodes_to_optimize.IsValid()) {
      LOGS(logger, VERBOSE) << "Nodes to optimize are not valid, skipping action.";
      continue;
    }

    // all nodes in the group are still available if IsValid returns true

    const NodeIndex pre_action_num_nodes = graph.MaxNodeIndex();

    ORT_RETURN_IF_ERROR(selector_action_entry->action->Run(graph, nodes_to_optimize));
    modified = true;

    const NodeIndex post_action_num_nodes = graph.MaxNodeIndex();

    ORT_RETURN_IF_ERROR(SetOpSinceVersionForProducedNodes(pre_action_num_nodes, post_action_num_nodes,
                                                          record, graph));
  }

  return Status::OK();
}

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

Status SelectorActionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                            const logging::Logger& logger) const {
  if (std::holds_alternative<SatRuntimeOptimizationLoadContext>(apply_context_)) {
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
    return ApplySavedRuntimeOptimizations(graph, modified, graph_level, logger);
#else
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Loading runtime optimizations is not enabled in this build.");
#endif
  }

  assert(std::holds_alternative<SatRuntimeOptimizationSaveContext>(apply_context_) ||
         std::holds_alternative<SatDirectApplicationContext>(apply_context_));

#if !defined(ORT_MINIMAL_BUILD)
  const auto* save_context = std::get_if<SatRuntimeOptimizationSaveContext>(&apply_context_);
  return ApplySelectorsAndActions(graph, modified, graph_level, logger, save_context);
#else
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                         "Running both selectors and actions is not enabled in this build.");
#endif
}

}  // namespace onnxruntime
