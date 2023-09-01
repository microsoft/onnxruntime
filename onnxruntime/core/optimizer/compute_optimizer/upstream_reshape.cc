// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"
#include "core/optimizer/compute_optimizer/upstream_reshape.h"

using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

UpStreamReshapeGraphTransformer::UpStreamReshapeGraphTransformer(
    const InlinedHashSet<std::string_view>& compatible_execution_providers) noexcept
    : UpStreamGraphTransformerBase("UpStreamReshapeGraphTransformer", compatible_execution_providers) {
  allowed_passthrough_ops_.insert({
      // Things to consider when more operators are added here:
      // 1. Whether the operator is safe to pass through in terms of computing equivalence.
      //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function.
      // 2. Should all inputs be allowed when tracking back further (bottom-up);
      //    if not, add the input index restriction.
      {GetFullQualifiedOpName("Add", kOnnxDomain),
       OpPassThroughConfig<UpStreamReshapeOperatorActorBase>(
           std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_14_13_7_6_1)},
      {GetFullQualifiedOpName("BiasGelu", kMSDomain),
       OpPassThroughConfig<UpStreamReshapeOperatorActorBase>(
           std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_1)},
      {GetFullQualifiedOpName("Cast", kOnnxDomain),
       OpPassThroughConfig<UpStreamReshapeOperatorActorBase>(
           std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_19_13_9_6_1)},
      {GetFullQualifiedOpName("Dropout", kOnnxDomain),
       OpPassThroughConfig<UpStreamReshapeOperatorActorBase>(
           std::make_shared<SimplePointwiseReshapeActor<true>>(), opset_13_12_10_7_6_1)},
      {// Be noted, this is our own implementation of ONNX domain op.
       GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
       OpPassThroughConfig<UpStreamReshapeOperatorActorBase>(
           std::make_shared<LayerNormalizationReshapeActor>(), opset_1)},
      {GetFullQualifiedOpName("MatMul", kOnnxDomain),
       OpPassThroughConfig<UpStreamReshapeOperatorActorBase>(
           std::make_shared<MatMulReshapeActor>(), opset_13_9_1)},
  });
}

bool UpStreamReshapeGraphTransformer::UpStreamInternal(
    Graph& graph, std::deque<ReshapeInfo>& queue, Node& current_node, ReshapeInfo& info,
    const OpPassThroughConfig<UpStreamReshapeOperatorActorBase>& pass_through_config,
    const logging::Logger& logger) const {
  const std::string op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());

  std::vector<int> propagate_input_indices;
  std::unordered_map<int, std::vector<DimCompare>> all_input_cmp_rets;
  std::function<void(Node & node)> shape_update_func;
  if (!pass_through_config.actor->PreCheck(current_node, info, logger, propagate_input_indices,
                                           all_input_cmp_rets, shape_update_func)) {
    LOG_DEBUG_INFO(logger, "Pre-check failed for " + current_node.Name() + "(" + op_type + ")");
    return false;
  }

  if (propagate_input_indices.empty()) {
    LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                               ") because the requirement is not met.");
    return false;
  }

  for (const int& input_idx : propagate_input_indices) {
    auto candidate_input_shape = current_node.InputDefs()[input_idx]->Shape();
    if (candidate_input_shape->dim_size() != 3) {
      LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                                 ") because not all candidate inputs have rank = 3.");
      return false;
    }

    // For unflatten dims, currently only dim values are supported.
    for (int k = 2; k < candidate_input_shape->dim_size(); ++k) {
      if (!candidate_input_shape->dim(k).has_dim_value()) {
        LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() +
                                   ": non-dim-value are not supported yet for unflatten dims.");
        return false;
      }
    }
  }

  if (!shape_update_func) {
    LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                               ") because the shape update function is not set.");
    return false;
  }

  // Be noted, once we reach this point after PreCheck, graph modification started, any failure after this should
  // be reported as ERROR.

  // Reshape infos that are populated into current_node's inputs.
  std::vector<ReshapeInfo> populated_reshape_infos;
  populated_reshape_infos.reserve(propagate_input_indices.size());
  std::unordered_map<int, ReshapeInfo> new_reshape_infos;
  for (auto input_index : propagate_input_indices) {
    ORT_ENFORCE(all_input_cmp_rets.find(input_index) != all_input_cmp_rets.end(),
                "all_input_cmp_rets should be a superset of propagate_input_indices");

    ReshapeInfo reshape_info = PropagateReshapeForInput(graph, *info.node_ptr, current_node, input_index, info,
                                                        all_input_cmp_rets.at(input_index), logger);

    ORT_ENFORCE(reshape_info.node_ptr, "New added Reshape node should not be null.");
    populated_reshape_infos.push_back(reshape_info);
    new_reshape_infos.insert({{input_index, reshape_info}});
  }

  ORT_ENFORCE(RemoveOriginalReshapeNode(graph, *info.node_ptr, current_node, logger, info).IsOK());

  // Do the shape update for current_node.
  shape_update_func(current_node);

  if (!pass_through_config.actor->PostProcess(graph, current_node, info, logger, propagate_input_indices,
                                              all_input_cmp_rets, new_reshape_infos)) {
    ORT_THROW("Post-process failed for " + current_node.Name() + "(" + op_type + ")");
  }

  queue.insert(queue.end(), populated_reshape_infos.begin(), populated_reshape_infos.end());
  return true;
}

ReshapeInfo UpStreamReshapeGraphTransformer::PropagateReshapeForInput(
    Graph& graph, Node& reshape_node, Node& current_node, int current_node_input_index,
    ReshapeInfo& info, std::vector<DimCompare>& dim_compare_rets, const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "PropagateReshapeForInput for Node " + current_node.Name() + "(" + current_node.OpType() +
                             ") with input index " + std::to_string(current_node_input_index));

  ORT_ENFORCE(dim_compare_rets.size() >= 2, "dim_compare_rets should have at least 2 elements.");

  if (!(dim_compare_rets[0] == DimCompare::Equal && dim_compare_rets[1] == DimCompare::Equal)) {
    // TODO(pengwa): implement input adaptation logic to cover more cases.
    ORT_THROW("Input adaptation is not implemented yet.");
  }

  InlinedVector<NodeArg*> input_args;
  input_args.reserve(reshape_node.InputDefs().size());
  // The first reshape op's data input should be current_node's current_node_input_index-th input.
  input_args.push_back(current_node.MutableInputDefs()[current_node_input_index]);

  // Prepare the target shape initializer. (Currently, only constant target shape is supported.)
  InlinedVector<int64_t> new_shape;
  new_shape.push_back(-1);
  auto input_shape = current_node.MutableInputDefs()[current_node_input_index]->Shape();
  for (int k = 2; k < input_shape->dim_size(); ++k) {
    ORT_ENFORCE(input_shape->dim(k).has_dim_value());
    new_shape.push_back(input_shape->dim(k).dim_value());
  }
  NodeArg* new_shape_arg = CreateInitializerFromVector(graph, {static_cast<int64_t>(new_shape.size())}, new_shape,
                                                       graph.GenerateNodeArgName("new_shape"));

  input_args.push_back(new_shape_arg);

  onnxruntime::NodeAttributes attributes = reshape_node.GetAttributes();

  InlinedVector<NodeArg*> output_args;
  output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(info.entry_reshape_arg_name),
                                current_node.MutableInputDefs()[current_node_input_index]->TypeAsProto()));

  /* new node input index to connect to current_node's input node*/
  int new_reshape_input_index_to_connect = info.GetDataInputIndex();
  /* new node output index to connect to current_node*/
  int new_reshape_output_index_to_connect = info.GetOutputIndex();
  Node* new_reshape_node = InsertIntermediateNodeOnDestInput(
      graph, current_node,
      current_node_input_index,
      new_reshape_input_index_to_connect,
      new_reshape_output_index_to_connect,
      graph.GenerateNodeName(info.entry_reshape_arg_name),
      reshape_node.OpType(),
      "Duplicated Reshape node",
      input_args,
      output_args,
      attributes,
      reshape_node.Domain(),
      logger);

  new_reshape_node->SetExecutionProviderType(reshape_node.GetExecutionProviderType());

  // Set the correct shape for newly created node.
  auto new_reshape_out_arg = new_reshape_node->MutableOutputDefs()[new_reshape_output_index_to_connect];
  new_reshape_out_arg->SetShape(CreateNewShapeWithMergedTwoLeadingDims(new_reshape_out_arg->Shape(),
                                                                       info.last_dim));

  // Notes: current_node's output shapes are not updated here. They will be updated in later.

  auto new_reshape_info = ReshapeInfo(graph, new_reshape_node, false);
  new_reshape_info.entry_node_name = info.entry_node_name;
  new_reshape_info.entry_reshape_arg_name = info.entry_reshape_arg_name;
  return new_reshape_info;
}

Status UpStreamReshapeGraphTransformer::RemoveOriginalReshapeNode(
    Graph& graph, Node& reshape_node, Node& current_node, const logging::Logger& logger, ReshapeInfo& info) const {
  LOG_DEBUG_INFO(logger, "RemoveOriginalReshapeNode target_node " + current_node.Name() + "(" + current_node.OpType() +
                             ") reshape_node " + reshape_node.Name() + "(" + reshape_node.OpType() + ")");

  auto data_input_arg = reshape_node.MutableInputDefs()[info.GetDataInputIndex()];
  int output_index = optimizer_utils::IndexOfNodeOutput(current_node, *data_input_arg);
  auto output_arg = reshape_node.MutableOutputDefs()[info.GetOutputIndex()];

  LOG_DEBUG_INFO(logger, "RemoveOriginalReshapeNode Replace all usage of output " + output_arg->Name() + ":0" +
                             " with " + current_node.MutableOutputDefs()[output_index]->Name() + ":" +
                             std::to_string(output_index));

  graph_utils::ReplaceDownstreamNodeInput(graph, reshape_node, info.GetOutputIndex() /*output_idx*/, current_node,
                                          output_index /*replacement_output_idx*/);
  auto reshape_origin_consumer_nodes = graph.GetConsumerNodes(output_arg->Name());
  std::vector<Node*> reshape_op_consumers;
  reshape_op_consumers.reserve(reshape_origin_consumer_nodes.size());
  for (auto& consumer_node : reshape_origin_consumer_nodes) {
    reshape_op_consumers.push_back(graph.GetNode(consumer_node->Index()));
    LOG_DEBUG_INFO(logger, "RemoveOriginalReshapeNode Reshape's consumer node " + consumer_node->Name() + "(" +
                               consumer_node->OpType() + ")");
  }
  graph.UpdateConsumerNodes(current_node.OutputDefs()[output_index]->Name(), reshape_op_consumers);
  graph.UpdateConsumerNodes(output_arg->Name(), {});
  graph.RemoveNode(reshape_node.Index());

  return Status::OK();
}

std::optional<ReshapeInfo> UpStreamReshapeGraphTransformer::IsSupportedForUpstream(
    Graph& graph, Node& node, const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", {1, 5, 13, 14}, kOnnxDomain)) {
    return std::nullopt;
  }

  auto data_shape = node.MutableInputDefs()[0]->Shape();
  auto reshape_out_shape = node.MutableOutputDefs()[0]->Shape();
  if (data_shape == nullptr || reshape_out_shape == nullptr) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to undefined shape.");
    return std::nullopt;
  }

  const auto data_rank = data_shape->dim_size();
  if (data_rank != 3) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to data rank != 3.");
    return std::nullopt;
  }

  if (!graph_utils::IsConstantInitializer(graph, node.InputDefs()[1]->Name(), /* check_outer_scope */ false)) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to target shape is non-constant initializer.");
    return std::nullopt;
  }

  if (!utils::HasDimValue(data_shape->dim(2))) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to data shape's last dim is not concrete.");
    return std::nullopt;
  }

  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *node.InputDefs()[1], new_shape_const_values, true);
  if (new_shape_const_values.size() != 2) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to target shape is rank 2.");
    return std::nullopt;
  }

  if (new_shape_const_values[1] != data_shape->dim(2).dim_value()) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() +
                               " due to target shape's last dim is not equal to data shape's last dim.");
    return std::nullopt;
  }

  // If the first dim of Reshape output don't have dim_value or dim_param, we can't do the optimization.
  if (!(reshape_out_shape->dim(0).has_dim_value() || reshape_out_shape->dim(0).has_dim_param())) {
    return std::nullopt;
  }

  return ReshapeInfo(graph, &node, true);
}

}  // namespace onnxruntime

#endif
