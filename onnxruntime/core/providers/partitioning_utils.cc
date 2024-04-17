// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// QDQ models require graph modification at runtime, so we know this infrastructure is not used in a minimal build
#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/providers/partitioning_utils.h"

#include <algorithm>
#include <deque>
#include <iterator>
#include <queue>

#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#include "core/framework/node_unit.h"
#include "core/graph/graph_viewer.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace utils {

// internal helpers
namespace {
#ifndef NDEBUG
std::string NodeDebugString(const Node& node) {
  std::ostringstream oss;
  oss << node.Index() << " '" << node.Name() << "'(" << node.OpType() << ")";
  return oss.str();
}

template <typename Container>
std::string NodeGroupDebugString(const Container& group, bool show_all = false) {
  static_assert(std::is_same_v<typename Container::value_type, const Node*>);

  if (group.empty()) {
    return "<no nodes>";
  }

  std::ostringstream oss;
  oss << "<" << group.size() << (group.size() == 1 ? " node> " : " nodes> ");
  if (show_all) {
    auto node_it = group.begin();
    oss << NodeDebugString(**(node_it++));
    while (node_it != group.end()) {
      oss << ", " << NodeDebugString(**(node_it++));
    }
  } else {
    const Node& start_node = *group.front();
    const Node& end_node = *group.back();
    oss << NodeDebugString(start_node) << " to " << NodeDebugString(end_node);
  }

  return oss.str();
}
#endif

/**
Create partition node groups.

A partition node group (a.k.a. a group) contains supported nodes that will run in a partition.

All nodes in a group can be run together. This means that two nodes with an intervening unsupported node cannot be in
the same group. On the other hand, nodes within the same group do not necessarily have to be connected.

The partitioning algorithm attempts to form the largest possible groups in a greedy fashion. It is a variant of Kahn's
topological sort algorithm that forms the group(s) as it goes.

Conceptually, we consider nodes in a sequence of waves starting from the root nodes. One wave produces at most one
group. A wave flows over nodes in topological order, adding supported nodes to the current group, and stops at the
border of the current group. The next wave starts where the previous wave stopped.

When generating the topological ordering, we maintain a set of nodes that have no inputs produced by unprocessed nodes.
From this set, we select the next node to process.

When selecting the next node to process, we first take:
- a supported node (which will be part of the group)
- an unsupported node that does not consume an output of any node in the group

The remaining unsupported nodes mark the border of the current group so they will be processed later when we consider
the next group.

If node_unit_map is provided, we process NodeUnit instances (a logical 'Node' that can be a single node or a
QDQ node group) instead of individual Node instances. As an EP must take complete NodeUnit instances (i.e. it
must not break up a QDQ node group by taking a subset of nodes in it), this granularity of processing is valid.
It is required to ensure we do not break up a QDQ node unit during partitioning.

@param graph_viewer GraphViewer that IExecutionProvider::GetCapability is called with.
@param is_node_supported_fn Callback to check whether a node is supported.
@param on_group_closed_fn Callback to indicate a completed partition node group.
@param debug_output Print diagnostic output about the partitions and reasons for partition breaks.
                    No-op in a release build.
@return The partition node groups.
*/
std::vector<std::vector<const Node*>> CreateSupportedPartitionNodeGroups(
    const GraphViewer& graph_viewer,
    const IsNodeSupportedFn& is_node_supported_fn,
    const OnGroupClosedFn& on_group_closed_fn,
    const std::string& execution_provider_type,
    const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map,
    bool debug_output) {
#ifdef NDEBUG
  ORT_UNUSED_PARAMETER(debug_output);
#endif

  ORT_ENFORCE(is_node_supported_fn, "Node support test is required.");

  /*
   * NOTE: when making change here PLEASE update the logic that replicates the C++ partitioning in
   * /tools/python/util/mobile_helpers/usability_checker.py:check_partitioning
   */
  std::vector<std::vector<const Node*>> supported_groups{};

  // number of inputs from unprocessed nodes (in-degree) per node
  std::unordered_map<NodeIndex, size_t> in_degree{};
  // nodes that are ready to process
  std::deque<const Node*> nodes_to_process{};
  // nodes that will be processed when considering the next partition node group
  std::deque<const Node*> nodes_to_process_with_next_group{};

  // initialize in-degrees and find root nodes
  for (const auto& node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const auto& node = *graph_viewer.GetNode(node_index);
    auto node_input_edge_count = node.GetInputEdgesCount();

    if (node_unit_map != nullptr) {
      const auto& node_unit = node_unit_map->at(&node);
      if (&node_unit->GetNode() != &node) {
        // only process the target node
        continue;
      }

      node_input_edge_count = node_unit->InputEdgeCount();
    }

    in_degree.insert({node.Index(), node_input_edge_count});
    if (node_input_edge_count == 0) {
      nodes_to_process.push_back(&node);
    }
  }

  std::vector<const Node*> supported_group{};
  // the partition node group's border is the aggregate of its nodes' output nodes
  InlinedHashSet<const Node*> supported_group_border{};

  auto close_group = [&]() {
    if (!supported_group.empty()) {
#ifndef NDEBUG
      if (debug_output) {
        LOGS_DEFAULT(VERBOSE) << "New partition node group.\n"
                              << "Unsupported nodes on group border: "
                              << NodeGroupDebugString(nodes_to_process_with_next_group, true) << "\n"
                              << "Nodes in group: " << NodeGroupDebugString(supported_group);
      }
#endif

      // if no on_group_closed_fn callback was given, keep the partition
      // otherwise, let the callback determine whether to keep it
      const bool keep_partition = !on_group_closed_fn || on_group_closed_fn(supported_group);

      if (keep_partition) {
        supported_groups.emplace_back(std::move(supported_group));
      }
#ifndef NDEBUG
      else {
        LOGS_DEFAULT_IF(debug_output, VERBOSE) << "Discarded partition node group.";
      }
#endif

      supported_group.clear();
      supported_group_border.clear();
    }
  };

  size_t num_nodes_processed = 0;

  while (!nodes_to_process.empty() || !nodes_to_process_with_next_group.empty()) {
    if (nodes_to_process.empty()) {
      // we have processed all the nodes that we can while building this partition node group, start a new one
      close_group();
      nodes_to_process.swap(nodes_to_process_with_next_group);
      continue;
    }

    const Node& node = *nodes_to_process.front();
    nodes_to_process.pop_front();

    const NodeUnit* node_unit = node_unit_map ? node_unit_map->at(&node) : nullptr;
    const bool is_qdq_node_unit = node_unit && node_unit->UnitType() == NodeUnit::Type::QDQGroup;

    // a node that is already assigned to an EP other than current EP is unsupported
    const bool is_node_supported = (node.GetExecutionProviderType().empty() ||
                                    node.GetExecutionProviderType() == execution_provider_type) &&
                                   is_node_supported_fn(node);

    if (!is_node_supported && Contains(supported_group_border, &node)) {
      // an unsupported node on the border will be processed after the current partition node group
      nodes_to_process_with_next_group.push_back(&node);
      continue;
    }

    if (is_node_supported) {
      if (is_qdq_node_unit) {
        // add DQ -> node -> Q for the node unit. must be in topological order
        for (const auto& dq : node_unit->GetDQNodes()) {
          supported_group.push_back(dq);
        }

        supported_group.push_back(&node);

        for (const auto& q : node_unit->GetQNodes()) {
          supported_group.push_back(q);
        }
      } else {
        supported_group.push_back(&node);
      }

      // remove node from the border
      supported_group_border.erase(&node);
    }

    // For each downstream node:
    //   1: add the downstream node to the border if the current node is supported
    //   2: adjust in-degrees of the nodes consuming the current node's outputs, and add any new nodes to process
    const auto process_downstream_node = [&](const Node& downstream_node) {
      if (is_node_supported) {
        supported_group_border.insert(&downstream_node);
      }

      auto& downstream_node_in_degree = in_degree[downstream_node.Index()];
      --downstream_node_in_degree;

      if (downstream_node_in_degree == 0) {
        nodes_to_process.push_back(&downstream_node);
      }
    };

    if (node_unit_map) {
      std::for_each(node_unit->OutputEdgesBegin(), node_unit->OutputEdgesEnd(),
                    [&](const Node::EdgeEnd& edge_end) {
                      const Node& n = edge_end.GetNode();
                      const NodeUnit& downstream_node_unit = *node_unit_map->at(&n);
                      const Node& output = downstream_node_unit.GetNode();

                      process_downstream_node(output);
                    });
    } else {
      std::for_each(node.OutputNodesBegin(), node.OutputNodesEnd(), process_downstream_node);
    }

    ++num_nodes_processed;
  }

  close_group();

  ORT_ENFORCE(num_nodes_processed == in_degree.size(),
              "Processed ", num_nodes_processed, " nodes. Expected to process ", in_degree.size());

  return supported_groups;
}
}  // namespace

InlinedHashSet<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
                                                  const std::unordered_set<std::string>& stop_ops) {
  InlinedHashSet<const Node*> excluded_nodes;

  for (const auto& node : graph_viewer.Nodes()) {
    if (!Contains(excluded_nodes, &node) && Contains(stop_ops, node.OpType())) {
      excluded_nodes.insert(&node);

      // add all the downstream nodes
      std::queue<const Node*> nodes_to_process;
      nodes_to_process.push(&node);

      while (!nodes_to_process.empty()) {
        const Node* cur_node = nodes_to_process.front();
        nodes_to_process.pop();

        std::for_each(cur_node->OutputNodesBegin(), cur_node->OutputNodesEnd(),
                      [&nodes_to_process, &excluded_nodes](const Node& output_node) {
                        nodes_to_process.push(&output_node);
                        excluded_nodes.insert(&output_node);
                      });
      }
    }
  }

  return excluded_nodes;
}

std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                         const std::vector<const Node*>& group,
                                                         const GenerateMetadefNameFn& generate_metadef_name,
                                                         const std::string& execution_provider_name) {
  std::unordered_set<const Node*> node_set;
  node_set.reserve(group.size());
  node_set.insert(group.cbegin(), group.cend());

  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();

  std::unordered_set<const NodeArg*> node_outputs;
  std::unordered_set<const NodeArg*> subgraph_inputs;
  std::unordered_set<const NodeArg*> subgraph_outputs;
  std::vector<const NodeArg*> ordered_subgraph_inputs;
  std::vector<const NodeArg*> ordered_subgraph_outputs;

  const auto& graph_output_list = graph_viewer.GetOutputs();
  std::unordered_set<const NodeArg*> graph_outputs(graph_output_list.cbegin(), graph_output_list.cend());

  for (const Node* node : group) {
    sub_graph->nodes.push_back(node->Index());

    for (const auto* input : node->InputDefs()) {
      if (!input->Exists()) {
        // skip the placeholder inputs
        continue;
      }
      // if the node input was not produced by this subgraph, add it to the subgraph inputs.
      if (!Contains(node_outputs, input)) {
        if (!Contains(subgraph_inputs, input)) {
          subgraph_inputs.insert(input);
          ordered_subgraph_inputs.push_back(input);
        }
      }
    }

    const auto& output_defs = node->OutputDefs();
    for (const auto* output_def : output_defs) {
      node_outputs.insert(output_def);
      // if output is overall graph output we need to produce it.
      if (Contains(graph_outputs, output_def)) {
        ordered_subgraph_outputs.push_back(output_def);
      }
    }

    // if output connects to a node not in this subgraph we need to add it
    // unless it was already added as an overall graph output,
    for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
      if (!Contains(node_set, &it->GetNode())) {
        const auto* output_def = output_defs[it->GetSrcArgIndex()];
        if (!Contains(subgraph_outputs, output_def) && !Contains(graph_outputs, output_def)) {
          subgraph_outputs.insert(output_def);
          ordered_subgraph_outputs.push_back(output_def);
        }
      }
    }
  }

  // Assign inputs and outputs to subgraph's meta_def
  auto meta_def = std::make_unique<::onnxruntime::IndexedSubGraph::MetaDef>();
  meta_def->name = generate_metadef_name();
  meta_def->domain = execution_provider_name;
  meta_def->since_version = 1;
  meta_def->status = ONNX_NAMESPACE::EXPERIMENTAL;

  for (const auto& input : ordered_subgraph_inputs) {
    meta_def->inputs.push_back(input->Name());
  }

  for (const auto& output : ordered_subgraph_outputs) {
    meta_def->outputs.push_back(output->Name());
  }

  sub_graph->SetMetaDef(std::move(meta_def));

  return std::make_unique<ComputeCapability>(std::move(sub_graph));
}

std::vector<std::unique_ptr<ComputeCapability>>
CreateSupportedPartitions(const GraphViewer& graph_viewer,
                          const IsNodeSupportedFn& is_node_supported_fn,
                          const OnGroupClosedFn& on_partition_closed_fn,
                          const GenerateMetadefNameFn& generate_metadef_name_fn,
                          const std::string& execution_provider_name,
                          const std::string& execution_provider_type,
                          const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map,
                          bool debug_output) {
  const auto groups = CreateSupportedPartitionNodeGroups(graph_viewer,
                                                         is_node_supported_fn,
                                                         on_partition_closed_fn,
                                                         execution_provider_type,
                                                         node_unit_map,
                                                         debug_output);

  std::vector<std::unique_ptr<ComputeCapability>> partitions{};
  partitions.reserve(groups.size());

  std::transform(
      groups.begin(), groups.end(),
      std::back_inserter(partitions),
      [&](const auto& supported_partition) {
        return MakeComputeCapability(graph_viewer, supported_partition, generate_metadef_name_fn,
                                     execution_provider_name);
      });

  return partitions;
}

std::vector<std::unique_ptr<ComputeCapability>>
CreateSupportedPartitions(const GraphViewer& graph_viewer,
                          const std::unordered_set<const Node*>& supported_nodes,
                          const std::unordered_set<std::string>& stop_ops,
                          const GenerateMetadefNameFn& generate_metadef_name_fn,
                          const std::string& execution_provider_name,
                          const std::string& execution_provider_type,
                          const std::unordered_map<const Node*, const NodeUnit*>* node_unit_map,
                          bool debug_output) {
  const auto excluded_nodes = CreateExcludedNodeSet(graph_viewer, stop_ops);
  const bool check_excluded_nodes = !excluded_nodes.empty();

  return CreateSupportedPartitions(
      graph_viewer,
      [&](const Node& node) -> bool {
        const bool is_excluded = check_excluded_nodes && Contains(excluded_nodes, &node);
        return !is_excluded && Contains(supported_nodes, &node);
      },
      {},
      generate_metadef_name_fn,
      execution_provider_name,
      execution_provider_type,
      node_unit_map,
      debug_output);
}

}  // namespace utils
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
