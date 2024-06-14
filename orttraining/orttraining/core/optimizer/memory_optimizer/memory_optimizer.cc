// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iomanip>
#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <onnx/defs/attr_proto_util.h>

#include "core/framework/random_seed.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "orttraining/core/graph/recompute_graph_utils.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_optimizer.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/optimization_planner.h"
#include "orttraining/core/optimizer/memory_optimizer/recompute_analysis.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_insight.h"

namespace onnxruntime {

namespace {

constexpr bool IsForwardPassOperator(ptrdiff_t op_order_in_topological_sort,
                                     ptrdiff_t boundary_op_order_in_topological_sort) {
  return op_order_in_topological_sort <= boundary_op_order_in_topological_sort;
}

// Reset seed attribute for the dropout node if the seed is not set.
bool SetSeedForDropoutNode(Node& node) {
  // ONNX Dropout 1, 6, 7, 10 do not have seed attribute, so we remove them from the recompute support.
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Dropout", {12, 13}, kOnnxDomain) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "BitmaskDropout", {1}, kMSDomain) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "BiasDropout", {1}, kMSDomain) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "BitmaskBiasDropout", {1}, kMSDomain) ||
      graph_utils::IsSupportedOptypeVersionAndDomain(node, "BiasSoftmaxDropout", {1}, kMSDomain)) {
    auto& attrs = node.GetAttributes();
    if (attrs.count("seed")) {
      return false;
    }

    int64_t seed = static_cast<int64_t>(utils::GetHashFromString(node.OutputDefs()[0]->Name())) +
                   utils::GetRandomSeed();
    node.AddAttribute("seed", seed);
    return true;
  }

  return false;
}

bool SetTrainingModeForForwardPythonOpNode(Node& node) {
  if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "PythonOp", {1}, kMSDomain)) {
    auto training_mode_attr = graph_utils::GetNodeAttribute(node, "training_mode");
    if (training_mode_attr != nullptr) {
      node.ClearAttribute("training_mode");
    }

    // Let forward node does not maintain information (ctx) for backward.
    node.AddAttribute("training_mode", static_cast<int64_t>(0));
    return true;
  }

  return false;
}

}  // namespace

Status MemoryOptimizer::ParseOptimizationConfigFromString(const std::string& memory_optimization_config_file_path,
                                                          const std::string& recompute_probe_config) {
  optimizer_config_file_path_ = memory_optimization_config_file_path;

  ORT_RETURN_IF_ERROR(optimizer::memory_optimizer::ParseOptimizationConfigFromString(
      memory_optimization_config_file_path,
      pattern_subgraph_to_user_optimizer_config_map_));

  ORT_RETURN_IF_ERROR(optimizer::memory_optimizer::ParseProbeConfigFromString(
      recompute_probe_config,
      recompute_probe_config_));

  return Status::OK();
}

bool MemoryOptimizer::ModifyGraph(Graph& graph,
                                  const InlinedHashMap<NodeIndex, ptrdiff_t>&
                                      node_index_to_its_order_in_topological_sort_map,
                                  const logging::Logger& logger,
                                  ptrdiff_t boundary_op_order_in_topological_sort,
                                  Node* node,
                                  std::shared_ptr<optimizer::memory_optimizer::NodeOptimizationPlanBase>& node_plan,
                                  std::shared_ptr<optimizer::memory_optimizer::ClusterApplyContext>& apply_context)
    const {
  bool graph_is_modified = false;
  int skip_count = (apply_context->requested_count == -1)
                       ? 0
                       : std::max(0, apply_context->total_frequency - apply_context->requested_count);

  apply_context->skip_count += 1;

  if (apply_context->skip_count > skip_count) {
    apply_context->applied_count += 1;
    Node* replacement_node_ptr = nullptr;
    LOGS(logger, INFO) << "Node " << node->Name() << "(" << node->OpType() << ") is applying following optimization:"
                       << "type [" << optimizer::memory_optimizer::OptimizationTypeToString(apply_context->type)
                       << "], request count [" << apply_context->requested_count << "]";
    if (apply_context->type == optimizer::memory_optimizer::OptimizationType::Recompute ||
        apply_context->type == optimizer::memory_optimizer::OptimizationType::RecomputeWithCompromise) {
      optimizer::memory_optimizer::NodeRecomputePlan* recompute_plan =
          dynamic_cast<optimizer::memory_optimizer::NodeRecomputePlan*>(node_plan.get());
      ORT_ENFORCE(recompute_plan != nullptr);
      ORT_ENFORCE(CreateRecomputeGraph(graph, recompute_plan->GetNodesInTopoOrder(), logger, replacement_node_ptr).IsOK());
    } else {
      ORT_THROW("unsupported optimization type found.");
    }

    ORT_ENFORCE(replacement_node_ptr);

    graph_is_modified = true;

    std::vector<graph_utils::GraphEdge> output_edges;
    for (auto it = node->OutputEdgesBegin(), end = node->OutputEdgesEnd(); it != end; ++it) {
      auto tid = node_index_to_its_order_in_topological_sort_map.find(it->GetNode().Index());
      // It is possible the consumer node is newly added as the recompute node, so we need a check here.
      // For those kinds of ops, we can treat them as backward ops.
      if (tid == node_index_to_its_order_in_topological_sort_map.end() ||
          !IsForwardPassOperator(node_index_to_its_order_in_topological_sort_map.at(tid->first),
                                 boundary_op_order_in_topological_sort)) {
        // Ignore the rng state consumer update for the determinstic PythonOp.
        if ((graph_utils::IsSupportedOptypeVersionAndDomain(*node, "PythonOp", {1}, kMSDomain) &&
             (it->GetSrcArgIndex() == 1 || it->GetSrcArgIndex() == 2))) {
          continue;
        }
        // Remove the edge only connecting to backward op.
        output_edges.push_back(graph_utils::GraphEdge::CreateGraphEdge(*node, *it, false));
      }
    }

    if (!output_edges.empty()) {
      // Create connections between the replacement node and the outgoing nodes.
      for (const auto& output_edge : output_edges) {
        // Remove the output edges of the node first
        graph.RemoveEdge(output_edge.src_node,
                         output_edge.dst_node,
                         output_edge.src_arg_index,
                         output_edge.dst_arg_index);

        graph.RemoveConsumerNode(node->MutableOutputDefs()[output_edge.src_arg_index]->Name(), node);

        // Add new edge connecting the input with the output nodes directly.
        // This also updates the destination node's input node args
        graph.AddEdge(replacement_node_ptr->Index(), output_edge.dst_node, static_cast<int>(output_edge.src_arg_index),
                      output_edge.dst_arg_index);
      }
    }
  }

  return graph_is_modified;
}

Status MemoryOptimizer::ApplyImpl(Graph& graph, bool& modified, int /*graph_level*/, const logging::Logger& logger)
    const {
  LOGS(logger, VERBOSE) << "Memory optimization config: " << optimizer_config_file_path_ << ", probe level: "
                        << static_cast<int>(recompute_probe_config_.probe_level)
                        << ", enable_transformer_layer_as_boundary:"
                        << recompute_probe_config_.enable_transformer_layer_as_boundary;

  if (pattern_subgraph_to_user_optimizer_config_map_.empty()) {
    LOGS(logger, VERBOSE) << "No optimization pattern is specified, skip memory optimization.";
    return Status::OK();
  }

  size_t recomputed_node_count = 0;

  ptrdiff_t yield_op_order_in_topological_sort;
  InlinedHashMap<const Node*, InlinedVector<size_t>> candidate_output_args_map;
  InlinedHashMap<NodeIndex, ptrdiff_t> node_index_to_its_order_in_topological_sort_map;

  // The first pass - find the candidate subgraphs.
  GraphViewer graph_viewer(graph);
  optimizer::memory_optimizer::MemoryOptimizationPlanner memory_opt_planner;
  ORT_ENFORCE(optimizer::memory_optimizer::FindORTModuleMemoryOpportunity(
                  graph_viewer,
                  recompute_probe_config_,
                  logger,
                  node_index_to_its_order_in_topological_sort_map,
                  yield_op_order_in_topological_sort,
                  candidate_output_args_map,
                  memory_opt_planner)
                  .IsOK());

  // Finalize the plan according to user config,
  // then create a ClusterApplyContext for each unique cluster (having the same node pattern)
  InlinedHashMap<const Node*, std::shared_ptr<optimizer::memory_optimizer::NodeOptimizationPlanBase>>
      node_to_opt_plan_map;
  optimizer::memory_optimizer::NodeToClusterApplyContextMap node_to_apply_context_map;
  ORT_ENFORCE(memory_opt_planner.FinalizeNodePlansFromUserConfig(pattern_subgraph_to_user_optimizer_config_map_,
                                                                 node_to_opt_plan_map,
                                                                 node_to_apply_context_map)
                  .IsOK());

  // The second pass - apply the transformation.
  const auto& node_ids =
      graph_viewer.GetNodesInTopologicalOrder(optimizer::memory_optimizer::TOPOLOGICAL_SORT_ALGORITHM);
  for (int i = static_cast<int>(node_ids.size()) - 1; i >= 0; --i) {
    Node* p_node = graph.GetNode(node_ids[i]);
    if (p_node == nullptr) {
      continue;
    }

    bool has_been_modified = false;
    if (node_to_opt_plan_map.find(p_node) != node_to_opt_plan_map.end()) {
      has_been_modified = ModifyGraph(graph, node_index_to_its_order_in_topological_sort_map,
                                      logger,
                                      yield_op_order_in_topological_sort,
                                      p_node,
                                      node_to_opt_plan_map[p_node],
                                      node_to_apply_context_map[p_node]);
    }

    if (has_been_modified) {
      recomputed_node_count += 1;
    }

    modified = modified || has_been_modified;
  }

  if (recomputed_node_count > 0) {
    LOGS(logger, INFO) << "Total number of recomputed nodes: " << recomputed_node_count;
  }

  PrintSummary(memory_opt_planner, node_to_apply_context_map, logger);

  return Status::OK();
}

void MemoryOptimizer::PrintSummary(const optimizer::memory_optimizer::MemoryOptimizationPlanner& memory_opt_planner,
                                   const InlinedHashMap<
                                       const Node*,
                                       std::shared_ptr<optimizer::memory_optimizer::ClusterApplyContext>>&
                                       node_to_apply_contexts_map,
                                   const logging::Logger& logger) const {
  std::vector<std::pair<std::string, optimizer::memory_optimizer::MemoryRecord>> records_grouped_by_node_cluster_id;
  optimizer::memory_optimizer::GetMemoryRecordsGroupedByNodeClusterId(memory_opt_planner,
                                                                      node_to_apply_contexts_map,
                                                                      records_grouped_by_node_cluster_id);
  LOGS(logger, INFO) << SerializeMemoryRecords(records_grouped_by_node_cluster_id, optimizer_config_file_path_) << "\n";
}

/******************************************************
 ** Recompute related function implementation starts **
 ******************************************************/

Status MemoryOptimizer::CreateRecomputeGraph(Graph& graph,
                                             const InlinedVector<const Node*>& nodes_in_topological_order,
                                             const logging::Logger& logger,
                                             Node*& new_output_node_ptr) const {
  InlinedHashMap<NodeArg*, NodeArg*> self_contained_outputs_map;
  for (size_t i = 0; i < nodes_in_topological_order.size(); ++i) {
    Node* node_to_duplicate = graph.GetNode(nodes_in_topological_order[i]->Index());

    // Check whether the node has been recomputed/offloaded or not. Simply check the existence of the first output
    // of the node has its corresponding recompute name or not.
    // TODO: if there is more optimization types like offload added, we will add a corresponding check
    // whether the outputs already be offloaded or not.
    if (graph.GetNodeArg(graph_utils::RecomputeName(node_to_duplicate->MutableOutputDefs()[0]->Name())) != nullptr) {
      continue;
    }

    bool seed_reset = SetSeedForDropoutNode(*node_to_duplicate);
    if (seed_reset) {
      LOGS(logger, VERBOSE) << "Set seed for Node " << node_to_duplicate->Name() << "(" << node_to_duplicate->OpType()
                            << ").";
    }

    const bool is_python_op = graph_utils::IsSupportedOptypeVersionAndDomain(*node_to_duplicate, "PythonOp", {1}, kMSDomain);

    InlinedVector<NodeArg*> new_input_args;
    NodeAttributes update_attrs = node_to_duplicate->GetAttributes();

    if (is_python_op) {
      new_input_args.reserve(node_to_duplicate->MutableInputDefs().size() + 2);
      // Ignore the ctx input, and connect the rng output to the recompute node.
      new_input_args.push_back(node_to_duplicate->MutableOutputDefs()[1]);
      new_input_args.push_back(node_to_duplicate->MutableOutputDefs()[2]);
      std::string input_convention = update_attrs.at("input_convention").s();
      input_convention[1] = 'd';  // Update the rng state input to be a tensor.
      input_convention[2] = 'd';
      update_attrs["input_convention"] = ONNX_NAMESPACE::MakeAttribute("input_convention", input_convention);

      const auto& input_pointer_scalars_ints = update_attrs.at("input_pointer_scalars").ints();
      std::vector<int64_t> input_pointer_scalars(input_pointer_scalars_ints.begin(),
                                                 input_pointer_scalars_ints.end());
      // Remove the rng state input.
      input_pointer_scalars.erase(input_pointer_scalars.begin() + 1, input_pointer_scalars.begin() + 3);
      update_attrs["input_pointer_scalars"] = ONNX_NAMESPACE::MakeAttribute("input_pointer_scalars",
                                                                            input_pointer_scalars);

      const auto& input_pointer_scalars_positions_ints = update_attrs.at("input_pointer_scalar_positions").ints();
      std::vector<int64_t> input_pointer_scalar_positions(input_pointer_scalars_positions_ints.begin(),
                                                          input_pointer_scalars_positions_ints.end());
      // Remove the rng state input.
      input_pointer_scalar_positions.erase(input_pointer_scalar_positions.begin() + 1,
                                           input_pointer_scalar_positions.begin() + 3);
      update_attrs["input_pointer_scalar_positions"] = ONNX_NAMESPACE::MakeAttribute("input_pointer_scalar_positions",
                                                                                     input_pointer_scalar_positions);

      const auto& input_tensor_ranks_ints = update_attrs.at("input_tensor_ranks").ints();
      std::vector<int64_t> input_tensor_ranks(input_tensor_ranks_ints.begin(),
                                              input_tensor_ranks_ints.end());
      // Insert the rng state input and cuda rng state input at the beginning.
      input_tensor_ranks.insert(input_tensor_ranks.begin(), {1, 1});

      update_attrs["input_tensor_ranks"] = ONNX_NAMESPACE::MakeAttribute("input_tensor_ranks",
                                                                         input_tensor_ranks);

      const auto& input_tensor_types_ints = update_attrs.at("input_tensor_types").ints();
      std::vector<int64_t> input_tensor_types(input_tensor_types_ints.begin(),
                                              input_tensor_types_ints.end());
      // Insert the uint8 type of rng state and cuda rng state at the beginning.
      input_tensor_types.insert(input_tensor_types.begin(), {ONNX_NAMESPACE::TensorProto_DataType_UINT8,
                                                             ONNX_NAMESPACE::TensorProto_DataType_UINT8});
      update_attrs["input_tensor_types"] = ONNX_NAMESPACE::MakeAttribute("input_tensor_types",
                                                                         input_tensor_types);
    } else {
      new_input_args.reserve(node_to_duplicate->MutableInputDefs().size());
    }

    for (NodeArg* input_arg : node_to_duplicate->MutableInputDefs()) {
      if (self_contained_outputs_map.find(input_arg) == self_contained_outputs_map.end()) {
        NodeArg* recompute_input_arg = graph.GetNodeArg(graph_utils::RecomputeName(input_arg->Name()));
        new_input_args.push_back(recompute_input_arg ? recompute_input_arg : input_arg);
      } else {
        new_input_args.push_back(self_contained_outputs_map[input_arg]);
      }
    }

    InlinedVector<NodeArg*> new_output_args;
    new_output_args.reserve(node_to_duplicate->MutableOutputDefs().size());
    for (size_t k = 0; k < node_to_duplicate->MutableOutputDefs().size(); ++k) {
      const auto& output = node_to_duplicate->MutableOutputDefs()[k];
      new_output_args.push_back(&graph.GetOrCreateNodeArg(graph_utils::RecomputeName(output->Name()),
                                                          output->TypeAsProto()));

      self_contained_outputs_map[output] = new_output_args.back();
    }

    Node& recompute_node = graph.AddNode(node_to_duplicate->Name() + "_recompute",
                                         node_to_duplicate->OpType(),
                                         "Recompute of " + node_to_duplicate->Name(),
                                         new_input_args,
                                         new_output_args,
                                         &update_attrs,
                                         node_to_duplicate->Domain());

    recompute_node.SetExecutionProviderType(node_to_duplicate->GetExecutionProviderType());
    ORT_RETURN_IF_NOT(graph.SetOpSchemaFromRegistryForNode(recompute_node),
                      "Failed to set op schema for added recompute node.");

    new_output_node_ptr = &recompute_node;

    for (size_t j = 0; j < recompute_node.MutableOutputDefs().size(); ++j) {
      graph.UpdateProducerNode(recompute_node.MutableOutputDefs()[j]->Name(), recompute_node.Index());
    }

    // Add the edges from the recompute node to the original node.
    for (size_t j = 0; j < recompute_node.MutableInputDefs().size(); ++j) {
      NodeArg* input_arg = recompute_node.MutableInputDefs()[j];
      const Node* producer_node = graph.GetProducerNode(input_arg->Name());
      if (producer_node == nullptr) {
        // Skip when it is graph input or initializer.
        continue;
      }
      int producer_output_index = optimizer_utils::IndexOfNodeOutput(*producer_node, *input_arg);
      graph.AddEdge(producer_node->Index(), recompute_node.Index(), static_cast<int>(producer_output_index),
                    static_cast<int>(j));

      graph.AddConsumerNode(input_arg->Name(), &recompute_node);
    }

    if (is_python_op) {
      graph.AddConsumerNode(node_to_duplicate->MutableOutputDefs()[1]->Name(), &recompute_node);
      graph.AddConsumerNode(node_to_duplicate->MutableOutputDefs()[2]->Name(), &recompute_node);
    }

    bool training_mode_reset = SetTrainingModeForForwardPythonOpNode(*node_to_duplicate);
    if (training_mode_reset) {
      LOGS(logger, VERBOSE) << "Set training mode for Node " << node_to_duplicate->Name()
                            << "(" << node_to_duplicate->OpType() << ").";
    }
  }

  return Status::OK();
}

/******************************************************
 ** Recompute related function implementation ends   **
 ******************************************************/

}  // namespace onnxruntime
