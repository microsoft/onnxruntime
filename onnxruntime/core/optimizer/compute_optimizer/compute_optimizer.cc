// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/passthrough_actors.h"
#include "core/optimizer/compute_optimizer/compute_optimizer.h"

using namespace ::onnxruntime::common;
using SliceInfo = onnxruntime::optimizer::compute_optimizer::SliceInfo;
using SliceOperationReorderHandle = onnxruntime::optimizer::compute_optimizer::SliceOperationReorderHandle;

namespace onnxruntime {
namespace optimizer {
namespace compute_optimizer {

static constexpr int kSliceDataInputIndex = 0;

bool EnforceNodeAllInputOutputHaveShapes(const Node& node) {
  for (const auto* input_def : node.InputDefs()) {
    if (!input_def->Shape()) {
      return false;
    }
  }

  for (const auto* output_def : node.OutputDefs()) {
    if (!output_def->Shape()) {
      return false;
    }
  }
  return true;
}

bool SliceOperationReorderHandle::operator()(Graph& graph, Node& current_node,
                                             SliceInfo& info,
                                             const logging::Logger& logger,
                                             std::deque<SliceInfo>& queue) {
  Node& slice_node = *info.GetNode();
  const std::string& op_type = GetFullQualifiedOpName(current_node.OpType(), current_node.Domain());
  if (GetOpPassThroughConfigMap().count(op_type)) {
    auto& pass_through_config = GetOpPassThroughConfigMap().at(op_type);
    LOG_DEBUG_INFO(logger, "Enter reorder handle for node " + current_node.Name() + "(" + op_type + ")");

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(current_node, current_node.OpType(),
                                                        pass_through_config.opsets, current_node.Domain())) {
      LOG_DEBUG_INFO(logger, "Unsupported opset for " + current_node.Name() + "(" + op_type + ") since version: " +
                                 std::to_string(current_node.SinceVersion()));
      return false;
    }

    if (!EnforceNodeAllInputOutputHaveShapes(current_node)) {
      LOG_DEBUG_INFO(logger, "Some inputs/outputs' shape not found for node " + current_node.Name() + "(" +
                                 op_type + ")");
      return false;
    }

    std::unordered_map<int, int> candidate_input_indices;
    bool input_has_dim_1_for_axis = false;
    if (!pass_through_config.actor->PreCheck(graph, current_node, info, candidate_input_indices,
                                             pass_through_config.input_indices,
                                             input_has_dim_1_for_axis, logger)) {
      LOG_DEBUG_INFO(logger, "Pre-check failed for " + current_node.Name() + "(" + op_type + ")");
      return false;
    }

    if (candidate_input_indices.empty()) {
      LOG_DEBUG_INFO(logger, "Skip handling current node " + current_node.Name() + "(" + op_type +
                                 ") because the requirement is not met.");
      return false;
    }

    // Be noted, once we reach this point after PreCheck, graph modification started, any failure after this should
    // be reported as ERROR.
    std::vector<SliceInfo> populated_slicing_infos;  // Slicing infos that are populated into current_node's inputs.
    populated_slicing_infos.reserve(candidate_input_indices.size());
    std::unordered_map<int, SliceInfo> new_gather_infos;
    for (auto pair : candidate_input_indices) {
      auto input_index = pair.first;  // input index of current_node
      int new_axis = pair.second;     // new axis of current_node's input to be sliced
      SliceInfo gather_info = PropagateSlicingForInput(graph, slice_node, current_node, input_index, info, new_axis,
                                                       logger);

      ORT_ENFORCE(gather_info.GetNode(), "New added gather node should not be null.");
      populated_slicing_infos.push_back(gather_info);
      new_gather_infos[input_index] = gather_info;
    }

    int index_of_output =
        optimizer_utils::IndexOfNodeOutput(current_node, *slice_node.InputDefs()[kSliceDataInputIndex]);
    ORT_ENFORCE(RemoveOriginSlicingOp(graph, slice_node, current_node, logger, info).IsOK());
    if (!pass_through_config.actor->PostProcess(graph, current_node, index_of_output, info.GetAxis(),
                                                entry_node_name_, new_gather_infos, input_has_dim_1_for_axis,
                                                info.IsSliceScalar(), info.GetInputRank(), info.GetOutputDimOnAxis(),
                                                logger)) {
      ORT_THROW("Post-process failed for " + current_node.Name() + "(" + op_type + ")");
    }

    queue.insert(queue.end(), populated_slicing_infos.begin(), populated_slicing_infos.end());
    return true;
  } else {
    LOG_DEBUG_INFO(logger, "op_type not supported for " + current_node.Name() + "(" + op_type + ")");
    return false;
  }
}

SliceInfo SliceOperationReorderHandle::PropagateSlicingForInput(Graph& graph,
                                                                Node& slice_node,
                                                                Node& current_node,
                                                                int current_node_input_index,
                                                                SliceInfo& info,
                                                                int new_axis,
                                                                const logging::Logger& logger) {
  LOG_DEBUG_INFO(logger, "PropagateSlicingForInput for Node " + slice_node.Name() + "(" + slice_node.OpType() +
                             ") with input index " + std::to_string(current_node_input_index) + ", keep_dim = " +
                             std::to_string(!info.IsSliceScalar()));

  InlinedVector<NodeArg*> input_args;
  input_args.reserve(slice_node.InputDefs().size());
  // The first slice op's data input should be current_node's current_node_input_index-th input.
  // For some cases when rank changes, slice op's slice input should also be adapted.
  input_args.push_back(current_node.MutableInputDefs()[current_node_input_index]);
  for (size_t i = 1; i < slice_node.InputDefs().size(); ++i) {
    input_args.push_back(slice_node.MutableInputDefs()[i]);
  }

  // Update the axis attribute if new_axis is not same with the original slicing axis (which happens when data
  // layerout got changed by Transpose or Reshape ops)
  onnxruntime::NodeAttributes attributes = slice_node.GetAttributes();
  if (info.GetAxis() != new_axis) {
    attributes[info.GetAxisAttrName()] =
        ONNX_NAMESPACE::MakeAttribute(info.GetAxisAttrName(), static_cast<int64_t>(new_axis));
  }

  InlinedVector<NodeArg*> output_args;
  output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(info.GetEntrySliceArgName()),
                                current_node.MutableInputDefs()[current_node_input_index]->TypeAsProto()));

  int new_slice_input_index_to_connect = 0;  /* new node input index to connect to current_node's input node*/
  int new_slice_output_index_to_connect = 0; /* new node output index to connect to current_node*/
  Node* new_slice_node = InsertItermediateNodeOnDestInput(graph, current_node,
                                                          current_node_input_index,
                                                          new_slice_input_index_to_connect,
                                                          new_slice_output_index_to_connect,
                                                          graph.GenerateNodeName(info.GetEntrySliceArgName()),
                                                          slice_node.OpType(),
                                                          "Duplicated Gather node",
                                                          input_args,
                                                          output_args,
                                                          attributes,
                                                          slice_node.Domain(),
                                                          logger);

  new_slice_node->SetExecutionProviderType(slice_node.GetExecutionProviderType());

  // Set correct shape for new created node.
  auto new_slice_out_arg = new_slice_node->MutableOutputDefs()[0];
  int reversed_axis = new_axis - new_slice_out_arg->Shape()->dim_size();
  UpdateSliceOutputShape(*new_slice_out_arg, reversed_axis, info.GetOutputDimOnAxis());
  auto new_slice_info = SliceInfo(new_slice_node, info.IsSliceScalar(), info.GetAxisAttrName(), new_axis);
  new_slice_info.UpdateEntrySliceArgName(info.GetEntrySliceArgName());
  return new_slice_info;
}

Status SliceOperationReorderHandle::RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
                                                          const logging::Logger& logger, SliceInfo& info) {
  LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp target_node " + current_node.Name() + "(" + current_node.OpType() +
                             ") slice_node " + slice_node.Name() + "(" + slice_node.OpType() + "), keep_dim = " +
                             std::to_string(!(info.IsSliceScalar())));

  auto slice_input_arg = slice_node.MutableInputDefs()[kSliceDataInputIndex];
  int slice_input_rank = slice_input_arg->Shape()->dim_size();
  int output_index = optimizer_utils::IndexOfNodeOutput(current_node, *slice_input_arg);
  auto slice_op_output_arg = slice_node.MutableOutputDefs()[0];

  // Loop all outputs of target node, update the shape accordingly.
  // For elementwise ops like (LayerNorm/Dropout/Add), we should handle all outputs.
  // If some output rank is lower than sliced axis, we should just ignore it (the correctness is guaranteed by devs
  // who adds more operator coverage in the pass through).
  for (size_t i = 0; i < current_node.MutableOutputDefs().size(); ++i) {
    UpdateSliceOutputShape(*current_node.MutableOutputDefs()[i], info.GetAxis() - slice_input_rank,
                           info.GetOutputDimOnAxis());
  }
  LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Replace all usage of output " + slice_op_output_arg->Name() + ":0" +
                             " with " + current_node.MutableOutputDefs()[output_index]->Name() + ":" +
                             std::to_string(output_index));

  for (auto it = slice_node.OutputEdgesBegin(), end = slice_node.OutputEdgesEnd(); it != end; ++it) {
    if (static_cast<size_t>(it->GetSrcArgIndex()) == 0) {
      LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Gather's output edge " + it->GetNode().Name() + "(" +
                                 it->GetNode().OpType() + ") input index " + std::to_string(it->GetDstArgIndex()));
    }
  }

  graph_utils::ReplaceDownstreamNodeInput(graph, slice_node, 0 /*output_idx*/, current_node,
                                          output_index /*replacement_output_idx*/);
  auto gather_origin_consumer_nodes = graph.GetConsumerNodes(slice_op_output_arg->Name());
  std::vector<Node*> gathernd_consumer_nodes;
  gathernd_consumer_nodes.reserve(gather_origin_consumer_nodes.size());
  for (auto& consumer_node : gather_origin_consumer_nodes) {
    gathernd_consumer_nodes.push_back(graph.GetNode(consumer_node->Index()));
    LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp Gather's consumer node " + consumer_node->Name() + "(" +
                               consumer_node->OpType() + ")");
  }
  graph.UpdateConsumerNodes(current_node.OutputDefs()[output_index]->Name(), gathernd_consumer_nodes);

  graph.UpdateConsumerNodes(slice_op_output_arg->Name(), {});
  graph.RemoveNode(slice_node.Index());

  return Status::OK();
}
}  // namespace compute_optimizer
}  // namespace optimizer

std::optional<SliceInfo> ComputeOptimizer::IsSupportedGatherND(Graph& /*graph*/, Node& node,
                                                               const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GatherND", {1, 12, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
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

  // Since GatherND is assumed to have batch_dims=1, if the input data's shape is [batch, sequence, ..., ... ],
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

  return SliceInfo(&node, false, "batch_dims", static_cast<int>(batch_dims), true);
}

std::optional<SliceInfo> ComputeOptimizer::IsSupportedGather(Graph& /*graph*/, Node& node,
                                                             const logging::Logger& logger) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Gather", {1, 11, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders())) {
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

  return SliceInfo(&node, dim_size == 0, "axis", axis, true);
}

Status ComputeOptimizer::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger)
    const {
  LOG_DEBUG_INFO(logger, "Enter ComputeOptimizer");
  bool reordered = false;
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  const auto& graph_outputs = graph.GetOutputs();
  size_t reordered_node_count = 0;  // For summary
  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed.
      continue;

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    std::optional<SliceInfo> gather_info;
    // Same ideas might apply for GatherElements, Slice, Split, etc..
    gather_info = IsSupportedGatherND(graph, node, logger);
    if (!gather_info.has_value()) {
      gather_info = IsSupportedGather(graph, node, logger);
    }

    if (!gather_info.has_value()) {
      continue;
    }

    auto& output_arg = node.MutableOutputDefs()[0];
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output_arg) != graph_outputs.end()) {
      continue;
    }

    std::string node_name = node.Name();
    std::string node_type = node.OpType();
    std::deque<SliceInfo> gather_queue;
    gather_queue.push_back(gather_info.value());

    std::string log_prefix = "Entry node " + node_name + " (" + node_type + ") with axis " +
                             std::to_string(gather_info.value().GetAxis());
    LOG_DEBUG_INFO(logger, log_prefix + " starts re-ordering check");

    SliceOperationReorderHandle handle(node_name);

    // DON'T operate on `node` once this loop starts, as it may be removed from the graph.
    while (!gather_queue.empty()) {
      SliceInfo info = gather_queue.front();
      Node* gather_node = info.GetNode();
      gather_queue.pop_front();
      const Node* gathernd_data_producer = graph.GetProducerNode(gather_node->MutableInputDefs()[0]->Name());
      if (gathernd_data_producer == nullptr) {
        break;
      }
      Node* input_node = const_cast<Node*>(gathernd_data_producer);
      if (graph.GetConsumerNodes(input_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOG_DEBUG_INFO(logger, log_prefix + " stops at node " + input_node->Name() + " since multiple consumer found");
        continue;
      }

      auto ret = handle(graph, *input_node, info, logger, gather_queue);
      if (ret) {
        LOG_DEBUG_INFO(logger, log_prefix + " moves up across node " + input_node->Name());
        modified = true;
        reordered = true;
      } else {
        LOG_DEBUG_INFO(logger, log_prefix + " stops when handling " + input_node->Name());
      }
    }

    if (reordered) {
      ++reordered_node_count;
    }
  }

  if (modified) {
    graph.SetGraphResolveNeeded();
  }
  LOGS(logger, INFO) << "Exit ComputeOptimizer with summary - reorderd_node_count:" << reordered_node_count
                     << " nodes.";
  return Status::OK();
}

}  // namespace onnxruntime
