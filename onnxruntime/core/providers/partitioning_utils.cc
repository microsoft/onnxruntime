// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/partitioning_utils.h"

#include <queue>

#include "core/framework/compute_capability.h"
#include "core/framework/execution_provider.h"
#include "core/graph/model.h"

namespace onnxruntime {
namespace utils {

// internal helpers
namespace {
// Kahn's topological sort with awareness of whether a node should be in a supported or unsupported partition
std::vector<const Node*> PartitionAwareTopoSort(const GraphViewer& graph_viewer,
                                                const std::unordered_set<const Node*>& supported_nodes) {
  std::queue<const Node*> supported_to_visit, unsupported_to_visit;
  std::unordered_map<NodeIndex, size_t> in_degree;
  std::vector<const Node*> topo_order;

  auto num_nodes = graph_viewer.NumberOfNodes();
  topo_order.reserve(num_nodes);
  in_degree.reserve(num_nodes);

  auto add_to_visit = [&](const Node& node) {
    if (supported_nodes.count(&node)) {
      supported_to_visit.push(&node);
    } else {
      unsupported_to_visit.push(&node);
    }
  };

  // find root nodes
  for (auto& node : graph_viewer.Nodes()) {
    size_t input_edge_count = node.GetInputEdgesCount();
    in_degree.insert({node.Index(), input_edge_count});
    if (input_edge_count == 0) {
      add_to_visit(node);
    }
  }

  // prefer unsupported nodes first. this will increase the number of inputs potentially available to the first
  // partition handled by this EP.
  bool processing_supported_nodes = false;

  while (!supported_to_visit.empty() || !unsupported_to_visit.empty()) {
    const Node* current = nullptr;

    // see if we need to flip
    if ((processing_supported_nodes && supported_to_visit.empty()) ||
        (!processing_supported_nodes && unsupported_to_visit.empty())) {
      processing_supported_nodes = !processing_supported_nodes;
      continue;
    }

    // get next node from same partition
    if (processing_supported_nodes) {
      current = supported_to_visit.front();
      supported_to_visit.pop();
    } else {
      current = unsupported_to_visit.front();
      unsupported_to_visit.pop();
    }

    // when in_degree is zero all the inputs to the node are available
    for (auto node_it = current->OutputNodesBegin(), end = current->OutputNodesEnd(); node_it != end; ++node_it) {
      in_degree[node_it->Index()]--;

      if (in_degree[node_it->Index()] == 0) {
        add_to_visit(*node_it);
      }
    }

    topo_order.push_back(&*current);
  }

  // check we didn't break something
  ORT_ENFORCE(graph_viewer.NumberOfNodes() == static_cast<int>(topo_order.size()),
              "Partition aware topological sort has produced invalid output.");

  return topo_order;
}

std::unordered_set<const Node*> CreateExcludedNodeSet(const GraphViewer& graph_viewer,
                                                      const std::unordered_set<std::string>& stop_ops) {
  std::unordered_set<const Node*> excluded_nodes;
  const auto end_stop_ops = stop_ops.cend();

  for (const NodeIndex node_index : graph_viewer.GetNodesInTopologicalOrder()) {
    const Node& node = *graph_viewer.GetNode(node_index);

    if (excluded_nodes.find(&node) == excluded_nodes.cend() &&
        stop_ops.find(node.OpType()) != end_stop_ops) {
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

}  // namespace

std::unique_ptr<ComputeCapability> MakeComputeCapability(const GraphViewer& graph_viewer,
                                                         const std::vector<const Node*>& group,
                                                         const std::function<std::string()>& generate_metadef_name,
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
      // if the node input was not produced by this subgraph, add it to the subgraph inputs.
      if (node_outputs.count(input) == 0) {
        if (subgraph_inputs.count(input) == 0) {
          subgraph_inputs.insert(input);
          ordered_subgraph_inputs.push_back(input);
        }
      }
    }

    const auto& output_defs = node->OutputDefs();
    for (const auto* output_def : output_defs) {
      node_outputs.insert(output_def);
      // if output is overall graph output we need to produce it.
      if (graph_outputs.count(output_def) != 0) {
        ordered_subgraph_outputs.push_back(output_def);
      }
    }

    // if output connects to a node not in this subgraph we need to add it
    // unless it was already added as an overall graph output,
    for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
      if (node_set.count(&it->GetNode()) == 0) {
        const auto* output_def = output_defs[it->GetSrcArgIndex()];
        if (subgraph_outputs.count(output_def) == 0 && graph_outputs.count(output_def) == 0) {
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
                          const std::unordered_set<const Node*>& supported_nodes,
                          const std::unordered_set<std::string>& stop_ops,
                          const std::function<std::string()>& generate_metadef_name,
                          const std::string& execution_provider_name,
                          bool debug_output) {
  // find any nodes we need to exclude
  std::unordered_set<const Node*> excluded_nodes = CreateExcludedNodeSet(graph_viewer, stop_ops);

#ifndef NDEBUG
  auto node_str = [](const Node& node) {
    std::ostringstream oss;
    oss << node.Index() << " '" << node.Name() << "'(" << node.OpType() << ")";
    return oss.str();
  };

  auto group_str = [&node_str](const std::vector<const Node*>& group) {
    const Node& start_node = *group.front();
    const Node& end_node = *group.back();
    std::ostringstream oss;
    oss << node_str(start_node) << " to " << node_str(end_node) << "\n";
    return oss.str();
  };
#endif

  // partition aware sort. this groups all the nodes we can and can't handle
  const std::vector<const Node*> new_order = PartitionAwareTopoSort(graph_viewer, supported_nodes);

  // create groups using the new sort order
  auto cur_topo_node = new_order.cbegin();
  auto end_topo_nodes = new_order.cend();

  std::queue<const Node*> nodes_to_process;         // supported nodes to process
  std::unordered_set<const Node*> processed_nodes;  // supported nodes we have processed
  std::map<NodeIndex, std::vector<const Node*>> node_groups;
  std::vector<const Node*> cur_group;

  bool check_excluded_nodes = !excluded_nodes.empty();
  const auto excluded_nodes_end = excluded_nodes.cend();

  while (cur_topo_node != end_topo_nodes) {
    const Node* node = *cur_topo_node;
    ++cur_topo_node;

    if (processed_nodes.find(node) != processed_nodes.cend()) {
      continue;
    }

    if (check_excluded_nodes && excluded_nodes.find(node) != excluded_nodes_end) {
      processed_nodes.insert(node);
      continue;
    }

    bool supported = supported_nodes.count(node) != 0;
    bool in_partition = !cur_group.empty();

    // check if end of a partition.
    if (in_partition && !supported) {
#ifndef NDEBUG
      if (debug_output) {
        LOGS_DEFAULT(VERBOSE) << "New partition due to " << node_str(*node)
                              << ". Nodes in old partition: " << cur_group.size() << "\n";
        LOGS_DEFAULT(VERBOSE) << group_str(cur_group) << "\n";
      }
#else
      ORT_UNUSED_PARAMETER(debug_output);
#endif
      node_groups.insert({cur_group.front()->Index(), std::move(cur_group)});
    }

    // add the node and any connected downstream nodes that we can handle if supported.
    // if not mark as processed so we know its inputs are available
    if (supported) {
      nodes_to_process.push(node);
    } else {
      processed_nodes.insert(node);
    }

    while (!nodes_to_process.empty()) {
      node = nodes_to_process.front();
      nodes_to_process.pop();

      if (processed_nodes.find(node) == processed_nodes.cend()) {
        // add to partition if all inputs available
        bool inputs_available = true;
        for (auto cur = node->InputNodesBegin(), end = node->InputNodesEnd(); cur != end; ++cur) {
          if (processed_nodes.find(&*cur) == processed_nodes.cend()) {
            inputs_available = false;
            break;
          }
        }

        if (inputs_available) {
          cur_group.push_back(node);
          processed_nodes.insert(node);

          for (auto cur = node->OutputNodesBegin(), end = node->OutputNodesEnd(); cur != end; ++cur) {
            const Node& downstream_node = *cur;

            // nodes will get added to the queue once per input from a supported node.
            // we need this to happen as they can't be added to the group until all inputs are known to be available.
            if (supported_nodes.count(&downstream_node) != 0) {
              nodes_to_process.push(&downstream_node);
            }
          }
        } else {
          // we need all other nodes providing input to this node to have been processed
          // before it can be added to cur_group.
          //
          // e.g. given A   B  with a topological order of A, B, C.
          //             \  /
          //              C
          //
          // When we process A we add C via the output edge to nodes_to_process. After we finish with A we look at C
          // as the next node in nodes_to_process, but the input from B is missing.
          // There are no more entries in nodes_to_process so we move to the next node in the topological order and
          // process B. Again C is added to nodes_to_process via the output edge. After we finish with B we look at C
          // again as the next node in nodes_to_process.
          // Now all the inputs are available and C is added to the current group.
        }
      }
    }
  }

  if (!cur_group.empty()) {
    node_groups.insert({cur_group.front()->Index(), std::move(cur_group)});
  }

  // create ComputeCapability instances
  std::vector<std::unique_ptr<ComputeCapability>> results;
  results.reserve(node_groups.size());

  for (const auto& idx_to_group : node_groups) {
    results.push_back(
        MakeComputeCapability(graph_viewer, idx_to_group.second, generate_metadef_name, execution_provider_name));
  }

  return results;
}
}  // namespace utils
}  // namespace onnxruntime
