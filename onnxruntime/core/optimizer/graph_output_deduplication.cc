// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// graph_output_deduplication.cc

#include "core/optimizer/graph_output_deduplication.h"
#include <queue>

namespace onnxruntime {

/*
  Deduplication logic for graph outputs:
  if this node produces a graph output which is a duplicate output, introduce an identity
  node in front of each such duplicated output keeping the graph output order consistent
  before and after transformation.

  Example:

    An Add op shown below with duplicate graph outputs as X
                  Add
                /    \
              X       X

    Transforms to:
                  Add
                /    \
          Identity  Identity
            /         \
          X_0         X_1

    onnx models with duplicate output names are a result of exporting PyTorch models
    with same value used multiple times in its return statement. Example `return (out, out)`
 */

namespace {

  std::string IdentityOpName("Identity");
  std::string OpDescription("Identity node to handle duplicate output names.");

} // anonymous namespace

bool GraphOutputDeduplication::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const {
  // Collect graph output occurences
  std::unordered_map<std::string, size_t> graph_output_names_count;
  for (auto& output_node_arg : graph.GetOutputs()) {
    graph_output_names_count[output_node_arg->Name()]++;
  }

  // If any graph output with two or more occurrences is also a node output, return true
  for (auto& node_arg : node.OutputDefs()) {
    auto node_output_is_graph_output_it = graph_output_names_count.find(node_arg->Name());
    if (node_output_is_graph_output_it != graph_output_names_count.end() && node_output_is_graph_output_it->second > 1) {
      return true;
    }
  }

  // This node does not produce any graph output with two or mode occurrences, return false
  return false;
}

Status GraphOutputDeduplication::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  // Collect graph output occurences
  std::unordered_map<std::string, size_t> graph_output_names_count;
  for (auto& output_node_arg : graph.GetOutputs()) {
    graph_output_names_count[output_node_arg->Name()]++;
  }

  // node_args_to_deduplicate is a mapping from the original NodeArg name to a queue of NodeArgs that
  // will replace the original NodeArg as the new graph outputs.
  std::unordered_map<std::string, std::queue<NodeArg*>> node_args_to_deduplicate;

  // For every node output that is also a graph output that occurs multiple times,
  // add an Identity node whose output node arg is a unique name.
  for (auto& node_arg : node.MutableOutputDefs()) {
    auto it = graph_output_names_count.find(node_arg->Name());
    if (it != graph_output_names_count.end() && it->second > 1) {
      for (size_t output_idx_for_this_node = 0; output_idx_for_this_node < it->second; ++output_idx_for_this_node) {
          auto type_proto = node_arg->TypeAsProto();
          auto& output_node_arg = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(node_arg->Name()), type_proto);
          graph.AddNode(graph.GenerateNodeName(node.Name()), IdentityOpName, OpDescription, {node_arg}, {&output_node_arg});
          node_args_to_deduplicate[node_arg->Name()].push(&output_node_arg);
      }
    }
  }

  // Collect the graph output NodeArgs in a vector.
  std::vector<const NodeArg*> graph_outputs;
  for (auto output_node_arg : graph.GetOutputs()) {
    auto node_output_is_graph_output_it = node_args_to_deduplicate.find(output_node_arg->Name());
    if (node_output_is_graph_output_it != node_args_to_deduplicate.end()) {
      // The newly created deduplicated NodeArgs should be picked up from the
      // node_args_to_deduplicate[original_node_arg_name] queue in order.
      auto replaced_node_arg = node_output_is_graph_output_it->second.front();
      node_output_is_graph_output_it->second.pop();
      graph_outputs.push_back(replaced_node_arg);
    } else {
      // If the graph output is not in node_args_to_deduplicate, it means that that output
      // was not a duplicate output or was not an output of this node. Collect it in order.
      graph_outputs.push_back(output_node_arg);
    }
  }

  // Set the new graph outputs and update the rule_effect.
  graph.SetOutputs(graph_outputs);
  rule_effect = RewriteRuleEffect::kModifiedRestOfGraph;

  return Status::OK();
}

}  // namespace onnxruntime
