// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_reshape_actors.h"
#include "core/optimizer/compute_optimizer/upstream_reshape.h"

using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

bool UpStreamReshapeGraphTransformer::UpStreamInternal(
    Graph& graph,
    std::deque<ReshapeInfo>& queue,
    Node& current_node,
    ReshapeInfo& info,
    const OpPassThroughConfig<UpStreamReshapeOperatorActorBase>& pass_through_config,
    const logging::Logger& logger,
    const std::string& entry_node_name) const {
  const std::string op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());

  std::unordered_map<int, std::vector<DimCompareRet>> candidate_input_indices;
  std::function<void(Node & node)> shape_update_func;
  if (!pass_through_config.actor->PreCheck(graph, current_node, info, pass_through_config.input_indices, logger,
                                           candidate_input_indices, shape_update_func)) {
    LOG_DEBUG_INFO(logger, "Pre-check failed for " + current_node.Name() + "(" + op_type + ")");
    return false;
  }

  if (candidate_input_indices.empty()) {
    LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                               ") because the requirement is not met.");
    return false;
  }

  for (auto pair : candidate_input_indices) {
    auto candidate_input_shape = current_node.InputDefs()[pair.first]->Shape();
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

  // Slicing infos that are populated into current_node's inputs.
  std::vector<ReshapeInfo> populated_slicing_infos;
  populated_slicing_infos.reserve(candidate_input_indices.size());
  std::unordered_map<int, ReshapeInfo> new_reshape_infos;
  for (auto pair : candidate_input_indices) {
    auto input_index = pair.first;  // input index of current_node
    if (current_node.InputDefs()[input_index]->Shape()->dim_size() != 3) {
      // If the input is already initialized, we don't need to propagate reshape.
      continue;
    }
    ReshapeInfo reshape_info = PropagateReshapeForInput(graph, *info.node_ptr, current_node, input_index, info,
                                                        pair.second, logger);

    ORT_ENFORCE(reshape_info.node_ptr, "New added gather node should not be null.");
    populated_slicing_infos.push_back(reshape_info);
    new_reshape_infos.insert({{input_index, reshape_info}});
  }

  int index_of_output = optimizer_utils::IndexOfNodeOutput(current_node,
                                                           *info.node_ptr->InputDefs()[info.GetDataInputIndex()]);
  ORT_ENFORCE(RemoveOriginReshapeOp(graph, *info.node_ptr, current_node, logger, info).IsOK());

  shape_update_func(current_node);

  if (!pass_through_config.actor->PostProcess(graph, current_node, index_of_output,
                                              info.output_dim_on_axis,
                                              entry_node_name, new_reshape_infos,
                                              logger)) {
    ORT_THROW("Post-process failed for " + current_node.Name() + "(" + op_type + ")");
  }

  queue.insert(queue.end(), populated_slicing_infos.begin(), populated_slicing_infos.end());
  return true;
}

ReshapeInfo UpStreamReshapeGraphTransformer::PropagateReshapeForInput(
    Graph& graph,
    Node& reshape_node,
    Node& current_node,
    int current_node_input_index,
    ReshapeInfo& info,
    std::vector<DimCompareRet>& dim_compare_rets,
    const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "PropagateReshapeForInput for Node " + current_node.Name() + "(" + current_node.OpType() +
                             ") with input index " + std::to_string(current_node_input_index));

  ORT_ENFORCE(dim_compare_rets.size() >= 2, "dim_compare_rets should have at least 2 elements.");

  if (!(dim_compare_rets[0] == DimCompareRet::Equal && dim_compare_rets[1] == DimCompareRet::Equal)) {
    // TODO(pengwa) implement input adaptation logic
    ORT_THROW("Input adaptation is not implemented yet.");
  }

  InlinedVector<NodeArg*> input_args;
  input_args.reserve(reshape_node.InputDefs().size());
  // The first reshape op's data input should be current_node's current_node_input_index-th input.
  input_args.push_back(current_node.MutableInputDefs()[current_node_input_index]);

  // Prepare the target shape initializer. (Currently only constant target shape is supported.)
  std::vector<int64_t> new_shape;
  new_shape.push_back(-1);
  auto input_shape = current_node.MutableInputDefs()[current_node_input_index]->Shape();
  for (int k = 2; k < input_shape->dim_size(); ++k) {
    ORT_ENFORCE(input_shape->dim(k).has_dim_value());
    new_shape.push_back(input_shape->dim(k).dim_value());
  }
  ONNX_NAMESPACE::TensorProto new_shape_const_tensor;
  new_shape_const_tensor.set_name(graph.GenerateNodeArgName("new_shape"));
  new_shape_const_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  new_shape_const_tensor.add_dims(new_shape.size());
  new_shape_const_tensor.set_raw_data(new_shape.data(), new_shape.size() * sizeof(int64_t));
  NodeArg* new_shape_arg = &graph_utils::AddInitializer(graph, new_shape_const_tensor);

  input_args.push_back(new_shape_arg);

  onnxruntime::NodeAttributes attributes = reshape_node.GetAttributes();

  InlinedVector<NodeArg*> output_args;
  output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(info.entry_slice_arg_name),
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
      graph.GenerateNodeName(info.entry_slice_arg_name),
      reshape_node.OpType(),
      "Duplicated Reshape node",
      input_args,
      output_args,
      attributes,
      reshape_node.Domain(),
      logger);

  new_reshape_node->SetExecutionProviderType(reshape_node.GetExecutionProviderType());

  // Set correct shape for new created node.
  auto new_reshape_out_arg = new_reshape_node->MutableOutputDefs()[new_reshape_output_index_to_connect];
  new_reshape_out_arg->SetShape(CreateNewShapeWithMergedTwoLeadingDims(new_reshape_out_arg->Shape(), info.output_dim_on_axis));
  auto new_reshape_info = ReshapeInfo(new_reshape_node, false);
  new_reshape_info.entry_slice_arg_name = info.entry_slice_arg_name;
  return new_reshape_info;
}

Status UpStreamReshapeGraphTransformer::RemoveOriginReshapeOp(Graph& graph,
                                                              Node& reshape_node,
                                                              Node& current_node,
                                                              const logging::Logger& logger,
                                                              ReshapeInfo& info) const {
  LOG_DEBUG_INFO(logger, "RemoveOriginReshapeOp target_node " + current_node.Name() + "(" + current_node.OpType() +
                             ") reshape_node " + reshape_node.Name() + "(" + reshape_node.OpType() + ")");

  auto slice_input_arg = reshape_node.MutableInputDefs()[info.GetDataInputIndex()];
  // int slice_input_rank = slice_input_arg->Shape()->dim_size();
  int output_index = optimizer_utils::IndexOfNodeOutput(current_node, *slice_input_arg);
  auto slice_op_output_arg = reshape_node.MutableOutputDefs()[info.GetOutputIndex()];

  LOG_DEBUG_INFO(logger, "RemoveOriginReshapeOp Replace all usage of output " + slice_op_output_arg->Name() + ":0" +
                             " with " + current_node.MutableOutputDefs()[output_index]->Name() + ":" +
                             std::to_string(output_index));

  graph_utils::ReplaceDownstreamNodeInput(graph, reshape_node, info.GetOutputIndex() /*output_idx*/, current_node,
                                          output_index /*replacement_output_idx*/);
  auto gather_origin_consumer_nodes = graph.GetConsumerNodes(slice_op_output_arg->Name());
  std::vector<Node*> slice_op_consumers;
  slice_op_consumers.reserve(gather_origin_consumer_nodes.size());
  for (auto& consumer_node : gather_origin_consumer_nodes) {
    slice_op_consumers.push_back(graph.GetNode(consumer_node->Index()));
    LOG_DEBUG_INFO(logger, "RemoveOriginReshapeOp Gather's consumer node " + consumer_node->Name() + "(" +
                               consumer_node->OpType() + ")");
  }
  graph.UpdateConsumerNodes(current_node.OutputDefs()[output_index]->Name(), slice_op_consumers);
  graph.UpdateConsumerNodes(slice_op_output_arg->Name(), {});
  graph.RemoveNode(reshape_node.Index());

  return Status::OK();
}

std::optional<ReshapeInfo> UpStreamReshapeGraphTransformer::IsSupportedForUpstream(
    Graph& graph,
    Node& node,
    const logging::Logger& logger) const {
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

  InlinedVector<int64_t> new_shape_const_values;
  optimizer_utils::AppendTensorFromInitializer(graph, *node.InputDefs()[1], new_shape_const_values, true);
  if (new_shape_const_values.size() != 2 || new_shape_const_values[0] != -1) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to target shape is not merging first two dims.");
    return std::nullopt;
  }

  if (!utils::HasDimValue(data_shape->dim(2))) {
    LOG_DEBUG_INFO(logger, "Skip Reshape node " + node.Name() + " due to the last dim size is not concrete value.");
    return std::nullopt;
  }

  return ReshapeInfo(&node, data_shape->dim(2).dim_value(), true);
}

}  // namespace onnxruntime

#endif
