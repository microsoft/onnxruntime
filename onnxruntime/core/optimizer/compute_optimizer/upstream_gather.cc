// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING

#include <onnx/defs/attr_proto_util.h>
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/upstream_gather_actors.h"
#include "core/optimizer/compute_optimizer/upstream_gather.h"
#include "core/optimizer/compute_optimizer/upstream_transformer_base.h"

using namespace onnxruntime::optimizer::compute_optimizer;

namespace onnxruntime {

UpStreamGatherGraphTransformer::UpStreamGatherGraphTransformer(
    const InlinedHashSet<std::string_view>& compatible_execution_providers) noexcept
    : UpStreamGraphTransformerBase("UpStreamGatherGraphTransformer", compatible_execution_providers) {
  allowed_passthrough_ops_.insert({
      // Things to consider when more operators are added here:
      // 1. Whether the operator is safe to pass through in terms of computing equivalence.
      //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function
      //    (as LayerNormalization did).
      // 2. Whether the outputs have the same dim changes if the Gather node moves before that operator.
      // 3. Should all inputs be allowed when tracking back further (bottom-up);
      //    if not, add the input index restriction as MatMul did.
      {GetFullQualifiedOpName("Add", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SimplePointwiseGatherActor<true>>(),
                                                            opset_14_13_7_6_1)},
      {GetFullQualifiedOpName("BiasGelu", kMSDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SimplePointwiseGatherActor<true>>(), opset_1)},

      {GetFullQualifiedOpName("Cast", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SimplePointwiseGatherActor<true>>(),
                                                            opset_19_13_9_6_1)},
      {GetFullQualifiedOpName("Div", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SimplePointwiseGatherActor<true>>(),
                                                            opset_14_13_7_6_1)},
      {GetFullQualifiedOpName("Dropout", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SimplePointwiseGatherActor<true>>(),
                                                            opset_13_12_10_7_6_1)},
      {GetFullQualifiedOpName("Gelu", kMSDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SimplePointwiseGatherActor<true>>(),
                                                            opset_1)},
      {// Be noted, this is our own implementation of ONNX domain op.
       GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<LayerNormalizationGatherActor>(),
                                                            opset_1)},
      {GetFullQualifiedOpName("MatMul", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<MatMulGatherActor>(),
                                                            opset_13_9_1)},
      {GetFullQualifiedOpName("Reshape", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<ReshapeGatherActor>(),
                                                            opset_19_14_13_5_1)},
      {GetFullQualifiedOpName("Softmax", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<SoftmaxGatherActor>(),
                                                            opset_13_11_1)},
      {GetFullQualifiedOpName("Transpose", kOnnxDomain),
       OpPassThroughConfig<UpStreamGatherOperatorActorBase>(std::make_shared<TransposeGatherActor>(),
                                                            opset_13_1)},
  });
}

bool UpStreamGatherGraphTransformer::UpStreamInternal(
    Graph& graph, std::deque<SliceInfo>& queue,
    Node& current_node, SliceInfo& info,
    const OpPassThroughConfig<UpStreamGatherOperatorActorBase>& pass_through_config,
    const logging::Logger& logger) const {
  Node& slice_node = *info.node_ptr;
  const std::string op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());

  std::unordered_map<int, int> propagate_input_indices;
  std::unordered_map<int, std::vector<DimCompare>> all_input_cmp_rets;
  std::function<void(Node & node)> shape_update_func;

  if (!pass_through_config.actor->PreCheck(graph, current_node, info, logger,
                                           propagate_input_indices, all_input_cmp_rets, shape_update_func)) {
    LOG_DEBUG_INFO(logger, "Pre-check failed for " + current_node.Name() + "(" + op_type + ")");
    return false;
  }

  if (propagate_input_indices.empty()) {
    LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                               ") because the requirement is not met.");
    return false;
  }

  if (!shape_update_func) {
    LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                               ") because the shape update function is not set.");
    return false;
  }

  // Be noted, once we reach this point after PreCheck, graph modification started, any failure after this should
  // be reported as ERROR.

  // Slicing infos that are populated into current_node's inputs.
  std::vector<SliceInfo> populated_slicing_infos;
  populated_slicing_infos.reserve(propagate_input_indices.size());
  std::unordered_map<int, SliceInfo> new_gather_infos;
  for (auto pair : propagate_input_indices) {
    auto input_index = pair.first;  // input index of current_node
    int new_axis = pair.second;     // new axis of current_node's input to be sliced
    SliceInfo gather_info = PropagateSlicingForInput(graph, slice_node, current_node, input_index, info, new_axis,
                                                     logger);

    ORT_ENFORCE(gather_info.node_ptr, "New added gather node should not be null.");
    populated_slicing_infos.push_back(gather_info);
    new_gather_infos.insert({{input_index, gather_info}});
  }

  ORT_ENFORCE(RemoveOriginSlicingOp(graph, slice_node, current_node, logger, info).IsOK());

  // Update shapes
  shape_update_func(current_node);

  if (!pass_through_config.actor->PostProcess(graph, current_node, info, logger, propagate_input_indices,
                                              all_input_cmp_rets, new_gather_infos)) {
    ORT_THROW("Post-process failed for " + current_node.Name() + "(" + op_type + ")");
  }

  queue.insert(queue.end(), populated_slicing_infos.begin(), populated_slicing_infos.end());
  return true;
}

SliceInfo UpStreamGatherGraphTransformer::PropagateSlicingForInput(
    Graph& graph,
    Node& slice_node,
    Node& current_node,
    int current_node_input_index,
    SliceInfo& info,
    int new_axis,
    const logging::Logger& logger) const {
  LOG_DEBUG_INFO(logger, "PropagateSlicingForInput for Node " + slice_node.Name() + "(" + slice_node.OpType() +
                             ") with input index " + std::to_string(current_node_input_index) + ", keep_dim = " +
                             std::to_string(!info.is_scalar_slice));

  InlinedVector<NodeArg*> input_args;
  input_args.resize(slice_node.InputDefs().size());

  int axis_input_index = -1;  // -1 means axis is passed in attribute.
  if (std::holds_alternative<int>(info.axis_attr_name_or_input_index)) {
    axis_input_index = std::get<int>(info.axis_attr_name_or_input_index);
  }

  auto create_axes_input = [&info, new_axis, &graph]() -> NodeArg* {
    InlinedVector<int64_t> dims;
    if (info.rank_of_axis_value == 1) {
      dims.push_back(1);
    }
    return CreateInitializerFromVector(graph, dims, {new_axis}, graph.GenerateNodeArgName("axes"));
  };

  // The first slice op's data input should be current_node's current_node_input_index-th input.
  // For some cases when rank changes, slice op's slice input should also be adapted.
  int i = 0;
  for (; i < static_cast<int>(slice_node.InputDefs().size()); ++i) {
    if (i == info.GetDataInputIndex()) {
      input_args[i] = current_node.MutableInputDefs()[current_node_input_index];
    } else if (axis_input_index != -1 && i == axis_input_index) {
      if (info.non_negative_axis == new_axis) {
        input_args[i] = slice_node.MutableInputDefs()[i];
      } else {
        input_args[i] = create_axes_input();
      }
    } else {
      input_args[i] = slice_node.MutableInputDefs()[i];
    }
  }

  // It is possible axes input is null.
  if (axis_input_index != -1 && info.non_negative_axis != new_axis) {
    for (; i <= axis_input_index; ++i) {
      if (i == axis_input_index) {
        input_args.push_back(create_axes_input());
      } else {
        NodeArg& empty_input = graph.GetOrCreateNodeArg("", nullptr);
        input_args.push_back(&empty_input);
      }
    }
  }

  // Update the axis attribute if new_axis is not the same as the original slicing axis (which happens when data
  // layout got changed by Transpose or Reshape ops)
  onnxruntime::NodeAttributes attributes = slice_node.GetAttributes();

  if (axis_input_index == -1 && info.non_negative_axis != new_axis) {
    std::string attr_name = std::get<std::string>(info.axis_attr_name_or_input_index);
    if (info.rank_of_axis_value == 0) {
      attributes[attr_name] =
          ONNX_NAMESPACE::MakeAttribute(attr_name, static_cast<int64_t>(new_axis));
    } else if (info.rank_of_axis_value == 1) {
      attributes[attr_name] =
          ONNX_NAMESPACE::MakeAttribute(attr_name, std::vector<int64_t>{static_cast<int64_t>(new_axis)});
    } else {
      ORT_THROW("Unexpected rank of axis attribute value: " + std::to_string(info.rank_of_axis_value));
    }
  }

  InlinedVector<NodeArg*> output_args;
  output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(info.entry_slice_arg_name),
                                current_node.MutableInputDefs()[current_node_input_index]->TypeAsProto()));

  /* new node input index to connect to current_node's input node*/
  int new_slice_input_index_to_connect = info.GetDataInputIndex();
  /* new node output index to connect to current_node*/
  int new_slice_output_index_to_connect = info.GetOutputIndex();
  Node* new_slice_node = InsertIntermediateNodeOnDestInput(
      graph, current_node,
      current_node_input_index,
      new_slice_input_index_to_connect,
      new_slice_output_index_to_connect,
      graph.GenerateNodeName(info.entry_slice_arg_name),
      slice_node.OpType(),
      "Duplicated Gather node",
      input_args,
      output_args,
      attributes,
      slice_node.Domain(),
      logger);

  new_slice_node->SetExecutionProviderType(slice_node.GetExecutionProviderType());

  // Set the correct shape for the newly created node.
  auto new_slice_out_arg = new_slice_node->MutableOutputDefs()[new_slice_output_index_to_connect];
  UpdateSliceOutputShape(*new_slice_out_arg, new_axis, info.output_dim_on_axis);

  auto new_slice_info = SliceInfo(graph, new_slice_node, info.is_scalar_slice, info.axis_attr_name_or_input_index,
                                  new_axis, info.rank_of_axis_value);
  new_slice_info.entry_node_name = info.entry_node_name;
  new_slice_info.entry_slice_arg_name = info.entry_slice_arg_name;
  return new_slice_info;
}

Status UpStreamGatherGraphTransformer::RemoveOriginSlicingOp(
    Graph& graph, Node& slice_node, Node& current_node,
    const logging::Logger& logger, SliceInfo& info) const {
  LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp target_node " + current_node.Name() + "(" + current_node.OpType() +
                             ") slice_node " + slice_node.Name() + "(" + slice_node.OpType() + "), keep_dim = " +
                             std::to_string(!(info.is_scalar_slice)));

  auto slice_input_arg = slice_node.MutableInputDefs()[info.GetDataInputIndex()];
  int output_index = optimizer_utils::IndexOfNodeOutput(current_node, *slice_input_arg);
  auto slice_op_output_arg = slice_node.MutableOutputDefs()[info.GetOutputIndex()];

  LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Replace all usage of output " + slice_op_output_arg->Name() + ":0" +
                             " with " + current_node.MutableOutputDefs()[output_index]->Name() + ":" +
                             std::to_string(output_index));

  graph_utils::ReplaceDownstreamNodeInput(graph, slice_node, info.GetOutputIndex() /*output_idx*/, current_node,
                                          output_index /*replacement_output_idx*/);
  auto gather_origin_consumer_nodes = graph.GetConsumerNodes(slice_op_output_arg->Name());
  std::vector<Node*> slice_op_consumers;
  slice_op_consumers.reserve(gather_origin_consumer_nodes.size());
  for (auto& consumer_node : gather_origin_consumer_nodes) {
    slice_op_consumers.push_back(graph.GetNode(consumer_node->Index()));
    LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Gather's consumer node " + consumer_node->Name() + "(" +
                               consumer_node->OpType() + ")");
  }
  graph.UpdateConsumerNodes(current_node.OutputDefs()[output_index]->Name(), slice_op_consumers);
  graph.UpdateConsumerNodes(slice_op_output_arg->Name(), {});
  graph.RemoveNode(slice_node.Index());

  return Status::OK();
}

namespace {

std::optional<SliceInfo> IsSupportedGatherND(Graph& graph, Node& node,
                                             const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                             const logging::Logger& logger) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GatherND", {1, 12, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, compatible_execution_providers)) {
    return std::nullopt;
  }

  auto data_shape = node.MutableInputDefs()[0]->Shape();
  auto indices_shape = node.MutableInputDefs()[1]->Shape();
  auto gather_out_shape = node.MutableOutputDefs()[0]->Shape();
  if (data_shape == nullptr || indices_shape == nullptr || gather_out_shape == nullptr) {
    LOG_DEBUG_INFO(logger, "Skip GatherND node " + node.Name() + " due to undefined shape.");
    return std::nullopt;
  }

  const auto data_rank = data_shape->dim_size();
  const auto indices_rank = indices_shape->dim_size();

  // batch_dims is an integer indicating the number of batch dimensions,
  // i.e the leading b number of dimensions of data tensor and indices are representing the batches,
  // and the gather starts from the b+1 dimension.
  auto batch_dims = static_cast<int64_t>(node.GetAttributes().at("batch_dims").i());
  ORT_ENFORCE(batch_dims >= 0 && batch_dims < indices_rank && batch_dims < data_rank,
              "batch_dims must be in the range [0, min(indices_rank, data_rank)):" + std::to_string(batch_dims) +
                  " indices_rank:" + std::to_string(indices_rank) + " data_rank:" + std::to_string(data_rank));

  // Since GatherND is assumed to have `batch_dims=1` , if the input data's shape is [batch, sequence, ..., ... ],
  // limiting indices_rank=3 will make sure produced output is in shape [batch, sliced_sequence, ..., ...]
  // and the rank did not change.
  // TODO: release the constraint here.
  if (data_rank != 3 || indices_rank != 3 || batch_dims != 1) {
    return std::nullopt;
  }

  auto& indices_last_dim = indices_shape->dim(indices_rank - 1);
  if (!(indices_last_dim.has_dim_value() && indices_last_dim.dim_value() == 1)) {
    return std::nullopt;
  }

  return SliceInfo(graph, &node, false, "batch_dims", static_cast<int>(batch_dims),
                   0 /* rank of axis attribute value */, true);
}

std::optional<SliceInfo> IsSupportedGather(Graph& graph, Node& node,
                                           const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                           const logging::Logger& logger) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gather", {1, 11, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, compatible_execution_providers)) {
    return std::nullopt;
  }

  auto data_shape = node.MutableInputDefs()[0]->Shape();
  auto indices_shape = node.MutableInputDefs()[1]->Shape();
  auto gather_out_shape = node.MutableOutputDefs()[0]->Shape();
  if (data_shape == nullptr || indices_shape == nullptr || gather_out_shape == nullptr) {
    LOG_DEBUG_INFO(logger, "Skip Gather node " + node.Name() + " due to undefined shape.");
    return std::nullopt;
  }

  const auto data_rank = data_shape->dim_size();
  if (data_rank <= 1) {
    LOG_DEBUG_INFO(logger, "Skip Gather node " + node.Name() + " due to rank <= 1.");
    return std::nullopt;
  }

  auto axis = static_cast<int>(node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + data_rank : axis;
  size_t dim_size = static_cast<size_t>(indices_shape->dim_size());
  bool is_single_value_1d_tensor = dim_size != 0 && (dim_size == 1 && utils::HasDimValue(indices_shape->dim(0)) &&
                                                     indices_shape->dim(0).dim_value() == 1);
  if (dim_size != 0 && !is_single_value_1d_tensor) {
    if (dim_size == 1 && utils::HasDimValue(data_shape->dim(axis)) &&
        data_shape->dim(axis).dim_value() > indices_shape->dim(0).dim_value()) {
      // Can support.
    } else {
      LOG_DEBUG_INFO(logger, "Skip Gather node " + node.Name() + " due to unsupported dim size: " +
                                 std::to_string(dim_size));
      return std::nullopt;
    }
  }

  return SliceInfo(graph, &node, dim_size == 0, "axis", axis, 0 /* rank of axis attribute value */, true);
}

std::optional<SliceInfo> IsSupportedShrunkenGather(Graph& graph, Node& node,
                                                   const InlinedHashSet<std::string_view>&
                                                       compatible_execution_providers,
                                                   const logging::Logger& logger) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "ShrunkenGather", {1}, kMSDomain) ||
      !graph_utils::IsSupportedProvider(node, compatible_execution_providers)) {
    return std::nullopt;
  }

  auto data_shape = node.MutableInputDefs()[0]->Shape();
  auto indices_shape = node.MutableInputDefs()[1]->Shape();
  auto gather_out_shape = node.MutableOutputDefs()[0]->Shape();
  if (data_shape == nullptr || indices_shape == nullptr || gather_out_shape == nullptr) {
    LOG_DEBUG_INFO(logger, "Skip ShrunkenGather node " + node.Name() + " due to undefined shape." +
                               std::to_string(data_shape == nullptr) + std::to_string(indices_shape == nullptr) +
                               std::to_string(gather_out_shape == nullptr));
    return std::nullopt;
  }

  const int data_rank = data_shape->dim_size();
  if (data_rank <= 1) {
    LOG_DEBUG_INFO(logger, "Skip ShrunkenGather node " + node.Name() + " due to data rank <= 1.");
    return std::nullopt;
  }

  int axis = static_cast<int>(node.GetAttributes().at("axis").i());
  axis = axis < 0 ? axis + data_rank : axis;
  int dim_size = indices_shape->dim_size();

  if (dim_size == 0) {
    LOG_DEBUG_INFO(logger, "Skip ShrunkenGather node " + node.Name() + " due to unsupported dim size: " +
                               std::to_string(dim_size));
    return std::nullopt;
  }

  return SliceInfo(graph, &node, false /*is_slice_scalar*/, "axis", axis, 0 /* rank of axis attribute value */, true);
}

/**
 * @brief Check if the Slice node can be up-streamed to the previous node.
 *
 * If "Slice" node is operating on one single axis, then it is supported.
 * @return std::optional<SliceInfo>
 */
std::optional<SliceInfo> IsSupportedSlice(Graph& graph, Node& node,
                                          const InlinedHashSet<std::string_view>&
                                              compatible_execution_providers,
                                          const logging::Logger& logger) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Slice", {10, 11, 13}) ||
      !graph_utils::IsSupportedProvider(node, compatible_execution_providers)) {
    return std::nullopt;
  }

  const NodeArg* data_input = node.InputDefs()[0];
  const NodeArg* starts_input = node.InputDefs()[1];
  const NodeArg* ends_input = node.InputDefs()[2];
  const NodeArg* axes_input = node.InputDefs().size() > 3 ? node.InputDefs()[3] : nullptr;

  if (data_input->Shape() == nullptr || starts_input->Shape() == nullptr || ends_input->Shape() == nullptr ||
      (axes_input && axes_input->Exists() && axes_input->Shape() == nullptr)) {
    LOG_DEBUG_INFO(logger, "Skip Slice node " + node.Name() + " due to undefined shape.");
    return std::nullopt;
  }

  // Make sure starts/ends/axes/steps are all 1D tensors, since we only support single-dimension slicing.
  if (starts_input->Shape()->dim_size() != 1 || ends_input->Shape()->dim_size() != 1 ||
      (axes_input && axes_input->Exists() && axes_input->Shape()->dim_size() != 1)) {
    LOG_DEBUG_INFO(logger, "Skip Slice node " + node.Name() + " due to unsupported dim size: " +
                               std::to_string(starts_input->Shape()->dim_size()) + ", " +
                               std::to_string(ends_input->Shape()->dim_size()) + ", " +
                               std::to_string(axes_input && axes_input->Exists() ? axes_input->Shape()->dim_size() : 0));
    return std::nullopt;
  }

  // Try to parse the 'axes' value.
  int axis = 0;
  if (axes_input && axes_input->Exists()) {
    InlinedVector<int64_t> axes_values;
    if (!graph_utils::IsConstantInitializer(graph, axes_input->Name()) ||
        !optimizer_utils::AppendTensorFromInitializer(graph, *axes_input, axes_values, true) ||
        axes_values.size() != 1) {
      LOG_DEBUG_INFO(logger, "Skip Slice node " + node.Name() + " due to unsupported axes value.");
      return std::nullopt;
    }
    axis = static_cast<int>(axes_values[0]);
  } else {
    // If 'axes' is not specified, then it is [0, .., r-1], so we force data rank to be 1.
    if (data_input->Shape()->dim_size() != 1) {
      LOG_DEBUG_INFO(logger, "Skip Slice node " + node.Name() + " due to unsupported data rank: " +
                                 std::to_string(data_input->Shape()->dim_size()));
      return std::nullopt;
    }
  }

  if (axis < 0)
    axis += data_input->Shape()->dim_size();

  return SliceInfo(graph, &node, false /*is_slice_scalar*/, 3 /* axis input index */, axis,
                   1 /* rank of axes value */, true);
}

}  // namespace

std::optional<SliceInfo> UpStreamGatherGraphTransformer::IsSupportedForUpstream(
    Graph& graph, Node& node, const logging::Logger& logger) const {
  std::optional<SliceInfo> gather_info;
  // Same ideas might apply to GatherElements, Slice, Split, etc.
  gather_info = IsSupportedGatherND(graph, node, GetCompatibleExecutionProviders(), logger);
  if (!gather_info.has_value()) {
    gather_info = IsSupportedGather(graph, node, GetCompatibleExecutionProviders(), logger);
  }
  if (!gather_info.has_value()) {
    gather_info = IsSupportedShrunkenGather(graph, node, GetCompatibleExecutionProviders(), logger);
  }
  if (!gather_info.has_value()) {
    gather_info = IsSupportedSlice(graph, node, GetCompatibleExecutionProviders(), logger);
  }
  return gather_info;
}

}  // namespace onnxruntime

#endif
