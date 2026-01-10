// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "selection_and_optimization_func.h"
#include "core/graph/graph_utils.h"
#include "core/framework/compute_capability.h"
#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"

namespace onnxruntime {

std::vector<std::unique_ptr<ComputeCapability>> ConstantFoldingDQFuncs::Select(const GraphViewer& graph_viewer,
                                                                               const KeyValueConfig& /*config*/,
                                                                               const GraphOptimizerRegistry& /*graph_optimizer_registry*/) {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
  const std::vector<NodeIndex>& node_index = graph_viewer.GetNodesInTopologicalOrder(ExecutionOrder::PRIORITY_BASED /*priority-based topological sort*/);
  InitializedTensorSet constant_inputs;
  const InlinedHashSet<std::string> excluded_initializers;

  // Select DequantizeLinear node where all inputs are constant
  for (const auto& index : node_index) {
    const auto& node = graph_viewer.GetNode(index);
    if (node->OpType() != "DequantizeLinear") {
      continue;
    }
    if (!graph_utils::AllNodeInputsAreConstant(graph_viewer.GetGraph(), *node, constant_inputs, excluded_initializers)) {
      continue;
    }
    sub_graph->nodes.push_back(index);
  }

  result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
  result.back()->optimization_func = ConstantFoldingDQFuncs::Optimize;
  return result;
}

Status ConstantFoldingDQFuncs::Optimize(Graph& graph,
                                        const ComputeCapability& optimization_cc,
                                        ComputeCapability& cc_to_update,
                                        const GraphOptimizerRegistry& graph_optimizer_registry) {
  std::string optimizer_name = kConstantFoldingDQ;
  std::unordered_set<std::string> original_initializers_to_remove;
  std::unordered_set<std::string> new_initializers_to_add;
  InlinedHashSet<NodeIndex> dq_node_index_set;

  // iterate the nodes in node_to_optimize to:
  //   1. get original initializers to remove
  //   2. add new initializers
  //   3. create dq node index set
  for (const auto& index : optimization_cc.sub_graph->nodes) {
    auto node = graph.GetNode(index);
    if (node->OpType() != "DequantizeLinear") {
      continue;
    }
    auto input_0 = node->InputDefs()[0];
    auto output_0 = node->OutputDefs()[0];
    original_initializers_to_remove.insert(input_0->Name());
    new_initializers_to_add.insert(output_0->Name());
    dq_node_index_set.insert(index);
  }

  static auto transformer = std::make_unique<ConstantFoldingDQ>(graph_optimizer_registry.GetCpuEp(),
                                                                false /*skip_dequantize_linear*/,
                                                                graph_optimizer_registry.GetSessionOptions().config_options,
                                                                dq_node_index_set);

  bool modified = false;
  ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, *graph_optimizer_registry.GetLogger()));

  // update the overall ComputeCapability
  std::vector<onnxruntime::NodeIndex> updated_nodes;
  for (auto index : cc_to_update.sub_graph->nodes) {
    if (dq_node_index_set.find(index) != dq_node_index_set.end()) {
      continue;
    }
    updated_nodes.push_back(index);
  }
  cc_to_update.sub_graph->nodes = updated_nodes;

  auto meta_def = cc_to_update.sub_graph->GetMutableMetaDef();
  std::vector<std::string> updated_constant_initializers;

  for (auto constant_initializer : meta_def->constant_initializers) {
    if (original_initializers_to_remove.find(constant_initializer) != original_initializers_to_remove.end()) {
      continue;
    }
    updated_constant_initializers.push_back(constant_initializer);
  }

  for (auto constant_initializer : new_initializers_to_add) {
    updated_constant_initializers.push_back(constant_initializer);
  }

  meta_def->constant_initializers = updated_constant_initializers;

  return Status::OK();
}

}  // namespace onnxruntime
