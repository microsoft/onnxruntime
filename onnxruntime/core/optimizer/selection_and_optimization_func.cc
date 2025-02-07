// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "selection_and_optimization_func.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/graph/graph_utils.h"
#include "core/framework/compute_capability.h"
#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"

namespace onnxruntime {

std::vector<std::unique_ptr<ComputeCapability>> ConstantFoldingDQ_selection(const GraphViewer& graph_viewer) {
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
  result.back()->optimization_func = ConstantFoldingDQ_optimization;
  return result;
}

Status ConstantFoldingDQ_optimization(Graph& graph, const ComputeCapability& optimization_cc, ComputeCapability& cc_to_update) {
  std::string optimizer_name = kCONSTANT_FOLDING_DQ;
  auto logger = const_cast<logging::Logger*>(&logging::LoggingManager::DefaultLogger());
  std::unordered_set<std::string> original_initializers_to_remove;
  std::unordered_set<std::string> new_initializers_to_add;
  InlinedHashSet<NodeIndex> dq_node_index_set;

  // iterate node_to_optimize to:
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

  auto optimizer_registry = onnxruntime::GraphOptimizerRegistry::Get();

  // ConstantFoldingDQ optimizer doesn't need the key/value strings.
  std::unordered_map<std::string, std::string> key_value_configs = optimization_cc.optimization_configs;

  // Don't use CreateOptimizer as ConstantFoldingDQ needs dq_node_index_set for instantiation.
  // optimizer_registry->CreateOptimizer(optimizer_name, key_value_configs);

  // Create ConstantFoldingDQ optimizer if it's not existed.
  if (!optimizer_registry->GetTransformerByName(optimizer_name)) {
    auto transformer = std::make_unique<ConstantFoldingDQ>(*optimizer_registry->GetCpuEpReference(),
                                                           false /*skip_dequantize_linear*/,
                                                           optimizer_registry->GetSessionOptionsReference()->config_options,
                                                           dq_node_index_set);
    optimizer_registry->Register(std::move(transformer));
  }

  // apply constant folding on DQ nodes
  optimizer_registry->ApplyTransformer(graph, optimizer_name, *logger);

  // update the overall ComputeCapability
  std::vector<onnxruntime::NodeIndex> updated_nodes;
  for (auto index : cc_to_update.sub_graph->nodes) {
    if (dq_node_index_set.find(index) != dq_node_index_set.end()) {
      continue;
    }
    updated_nodes.push_back(index);
  }
  cc_to_update.sub_graph->nodes = updated_nodes;

  auto original_meta_def = cc_to_update.sub_graph->GetMetaDef();
  std::unique_ptr<IndexedSubGraph::MetaDef> updated_meta_def = std::make_unique<IndexedSubGraph::MetaDef>();
  updated_meta_def->name = original_meta_def->name;
  updated_meta_def->domain = original_meta_def->domain;
  updated_meta_def->since_version = original_meta_def->since_version;
  updated_meta_def->status = original_meta_def->status;
  updated_meta_def->inputs = original_meta_def->inputs;
  updated_meta_def->outputs = original_meta_def->outputs;
  updated_meta_def->attributes = original_meta_def->attributes;
  updated_meta_def->doc_string = original_meta_def->doc_string;
#if !defined(ORT_MINIMAL_BUILD)
  updated_meta_def->type_and_shape_inference_function = original_meta_def->type_and_shape_inference_function;
#endif
  for (auto constant_initializer : original_meta_def->constant_initializers) {
    if (original_initializers_to_remove.find(constant_initializer) != original_initializers_to_remove.end()) {
      continue;
    }
    updated_meta_def->constant_initializers.push_back(constant_initializer);
  }

  for (auto constant_initializer : new_initializers_to_add) {
    updated_meta_def->constant_initializers.push_back(constant_initializer);
  }

  cc_to_update.sub_graph->SetMetaDef(std::move(updated_meta_def));

  return Status::OK();
}

}  // namespace onnxruntime
