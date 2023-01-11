// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_TRAINING_CORE
#include <onnx/defs/attr_proto_util.h>

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/compute_optimizer/passthrough_actors.h"
#include "core/optimizer/compute_optimizer/compute_optimizer.h"

using SliceInfo = onnxruntime::optimizer::compute_optimizer::SliceInfo;
using namespace onnxruntime::optimizer::compute_optimizer;
namespace onnxruntime {

namespace {

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

using OPSET_VERSION_LIST = std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>;
const OPSET_VERSION_LIST opset_1{1};
const OPSET_VERSION_LIST opset_13_1{13, 1};
const OPSET_VERSION_LIST opset_13_9_1{13, 9, 1};
const OPSET_VERSION_LIST opset_13_11_1{13, 11, 1};
const OPSET_VERSION_LIST opset_13_9_6_1{13, 9, 6, 1};
const OPSET_VERSION_LIST opset_14_13_5_1{14, 13, 5, 1};
const OPSET_VERSION_LIST opset_14_13_7_6_1{14, 13, 7, 6, 1};
const OPSET_VERSION_LIST opset_13_12_10_7_6_1{13, 12, 10, 7, 6, 1};

/**
 * @brief Functor to trigger the optimization search for a given slicing node
 *   (for example Gather/GatherND node).
 */
struct SliceOperationReorderHandle {
  /**
   * @brief Pass through configuration for specific operator.
   *
   * For each operator:
   * > `input_indices` can be used to explicitly specify the input indices that Slicing op can be passed through.
   *   This could be helpful if some inputs are not applicable for pass through. If not specified, all inputs
   *   are considered (but there will be checks to ignore those inputs that are not affected by the slicing axis).
   * > `actor` will be used to perform the actual pass through, including both pre-check stage and post process
   *   stage.
   */
  struct OpPassThroughConfig {
    OpPassThroughConfig(const std::vector<int>& input_indices,
                        std::shared_ptr<OperatorPassThroughActorBase> actor,
                        const OPSET_VERSION_LIST& opset_list)
        : input_indices(input_indices), actor(actor), opsets(opset_list) {
    }

    std::vector<int> input_indices;
    std::shared_ptr<OperatorPassThroughActorBase> actor;
    const OPSET_VERSION_LIST& opsets;
  };

  static std::string GetFullQualifiedOpName(const std::string& op_type, const std::string& domain) {
    return domain + "::" + op_type;
  }

  static std::unordered_map<std::string, OpPassThroughConfig>& GetOpPassThroughConfigMap() {
    static std::unordered_map<std::string, OpPassThroughConfig> allowed_passthrough_ops;
    static std::once_flag allowed_ops_init;
    std::call_once(allowed_ops_init, []() {
      allowed_passthrough_ops.insert({
          // Things to consider when more operators are added here:
          // 1. Whether the operator is safe to pass through in term of compute equivalence.
          //    If optype is not enough to guarantee the equivalence, we need to add a customized pre-check function
          //    (as LayerNormalization did).
          // 2. Whether the outputs have the same dim changes if Gather node moves before that operator.
          // 3. Should all inputs be allowed when track back further (bottom-up);
          //    if not, add the input index restriction as MatMul did.
          {GetFullQualifiedOpName("Add", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
          {GetFullQualifiedOpName("BiasGelu", kMSDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
          {GetFullQualifiedOpName("BitmaskBiasDropout", kMSDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
          {GetFullQualifiedOpName("Cast", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_13_9_6_1)},
          {GetFullQualifiedOpName("Div", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_14_13_7_6_1)},
          {GetFullQualifiedOpName("Dropout", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_13_12_10_7_6_1)},
          {GetFullQualifiedOpName("Gelu", kMSDomain),
           OpPassThroughConfig({}, std::make_shared<SimplePassThroughActor>(), opset_1)},
          {// Be noted, this is our own implementation of ONNX domain op.
           GetFullQualifiedOpName("LayerNormalization", kOnnxDomain),
           OpPassThroughConfig({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_1)},
          {GetFullQualifiedOpName("MatMul", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<MatMulPassThroughActor>(), opset_13_9_1)},
          {GetFullQualifiedOpName("Reshape", kOnnxDomain),
           OpPassThroughConfig({0}, std::make_shared<ReshapePassThroughActor>(), opset_14_13_5_1)},
          {GetFullQualifiedOpName("Softmax", kOnnxDomain),
           OpPassThroughConfig({0}, std::make_shared<ReductionOpPassThroughActor>(), opset_13_11_1)},
          {GetFullQualifiedOpName("Transpose", kOnnxDomain),
           OpPassThroughConfig({}, std::make_shared<TransposePassThroughActor>(), opset_13_1)},
      });
    });

    return allowed_passthrough_ops;
  }

  SliceOperationReorderHandle(const std::string& node_name) : entry_node_name_(node_name) {
  }

  bool operator()(Graph& graph, Node& current_node, SliceInfo& info, const logging::Logger& logger,
                  std::deque<SliceInfo>& queue);

 private:
  /**
   * @brief Pass through Slicing op from current_node's output to its specific input.
   *
   * Propagate the slicing operation into current_node's current_input_index-th input, e.g. a slicing op is inserted
   * between current_node's current_input_index-th input and current_node. For example, if current_node is Add,
   * and slice_node is a Gather(axis=1, indices=[1]):
   *
   *    input_0 [M, N, K]    input_1 [M, N, K]
   *                \        /
   *                Add [M, N, K]
   *                     |
   *            Gather0(axis=1, indices=[1])
   *                     |
   *              output [M, 1, K]
   *
   * After the pass through, the graph will be:
   *
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, N, K]
   *                               |
   *                       Gather0(axis=1, indices=[1])
   *                               |
   *                         output [M, 1, K]
   *
   * Be noted: Gather1 and Gather2 are inserted on Add's two inputs.
   * Gather0's removal and Add's output shape update is done in RemoveOriginSlicingOp.
   *
   *
   * @param graph Graph to iterate.
   * @param slice_node Slicing op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param current_node_input_index The current_node_input_index-th input to propagate the Slice op pass through.
   * @param info slice_node's SliceInfo.
   * @param logger Logger.
   * @param new_axis The new axis (for the new Slice op) upon current_node's original current_node_input_index-th input.
   * @return  SliceInfo for new created slicing op.
   */
  SliceInfo PropagateSlicingForInput(Graph& graph, Node& slice_node, Node& current_node, int current_node_input_index,
                                     SliceInfo& info, int new_axis, const logging::Logger& logger);

  /**
   * @brief Remove the origin slicing op (for example Gather/GatherND) and update shapes.
   *
   * In the above example, the graph will be cleaned up to:
   *   input_0 [M, N, K]                      input_1 [M, N, K]
   *                \                                /
   *     Gather1(axis=1, indices=[1])       Gather2(axis=1, indices=[1])
   *                     \                       /
   *                       \                   /
   *                          \             /
   *                           Add [M, 1, K]
   *                               |
   *                               |
   *                         output [M, 1, K]
   *
   * Be noted: Gather0 is removed, Add's output shape is updated.
   *
   * @param graph Graph to iterate.
   * @param slice_node Slicing op node the takes current_node's output as input.
   * @param current_node Current node.
   * @param logger Logger.
   * @param info slice_node's SliceInfo.
   * @return
   */
  Status RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
                               const logging::Logger& logger, SliceInfo& info);

  std::string entry_node_name_;
};

bool SliceOperationReorderHandle::operator()(Graph& graph, Node& current_node,
                                             SliceInfo& info,
                                             const logging::Logger& logger,
                                             std::deque<SliceInfo>& queue) {
  Node& slice_node = *info.node_ptr;
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
    if (!pass_through_config.actor->PreCheck(graph, current_node, info, pass_through_config.input_indices, logger,
                                             candidate_input_indices, input_has_dim_1_for_axis)) {
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

      ORT_ENFORCE(gather_info.node_ptr, "New added gather node should not be null.");
      populated_slicing_infos.push_back(gather_info);
      new_gather_infos.insert({{input_index, gather_info}});
    }

    int index_of_output =
        optimizer_utils::IndexOfNodeOutput(current_node, *slice_node.InputDefs()[info.GetDataInputIndex()]);
    ORT_ENFORCE(RemoveOriginSlicingOp(graph, slice_node, current_node, logger, info).IsOK());
    if (!pass_through_config.actor->PostProcess(graph, current_node, index_of_output, info.non_negative_axis,
                                                info.is_scalar_slice, input_has_dim_1_for_axis,
                                                info.output_dim_on_axis,
                                                entry_node_name_, new_gather_infos,
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
                             std::to_string(!info.is_scalar_slice));

  InlinedVector<NodeArg*> input_args;
  input_args.reserve(slice_node.InputDefs().size());
  // The first slice op's data input should be current_node's current_node_input_index-th input.
  // For some cases when rank changes, slice op's slice input should also be adapted.
  input_args.push_back(current_node.MutableInputDefs()[current_node_input_index]);
  for (size_t i = 1; i < slice_node.InputDefs().size(); ++i) {
    input_args.push_back(slice_node.MutableInputDefs()[i]);
  }

  // Update the axis attribute if new_axis is not same with the original slicing axis (which happens when data
  // layout got changed by Transpose or Reshape ops)
  onnxruntime::NodeAttributes attributes = slice_node.GetAttributes();
  if (info.non_negative_axis != new_axis) {
    attributes[info.axis_attr_name] =
        ONNX_NAMESPACE::MakeAttribute(info.axis_attr_name, static_cast<int64_t>(new_axis));
  }

  InlinedVector<NodeArg*> output_args;
  output_args.push_back(
      &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(info.entry_slice_arg_name),
                                current_node.MutableInputDefs()[current_node_input_index]->TypeAsProto()));

  /* new node input index to connect to current_node's input node*/
  int new_slice_input_index_to_connect = info.GetDataInputIndex();
  /* new node output index to connect to current_node*/
  int new_slice_output_index_to_connect = info.GetOutputIndex();
  Node* new_slice_node = InsertIntermediateNodeOnDestInput(graph, current_node,
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

  // Set correct shape for new created node.
  auto new_slice_out_arg = new_slice_node->MutableOutputDefs()[new_slice_output_index_to_connect];
  int reversed_axis = new_axis - new_slice_out_arg->Shape()->dim_size();
  UpdateSliceOutputShape(*new_slice_out_arg, reversed_axis, info.output_dim_on_axis);
  auto new_slice_info = SliceInfo(new_slice_node, info.is_scalar_slice, info.axis_attr_name, new_axis);
  new_slice_info.entry_slice_arg_name = info.entry_slice_arg_name;
  return new_slice_info;
}

Status SliceOperationReorderHandle::RemoveOriginSlicingOp(Graph& graph, Node& slice_node, Node& current_node,
                                                          const logging::Logger& logger, SliceInfo& info) {
  LOG_DEBUG_INFO(logger, "RemoveOriginSlicingOp target_node " + current_node.Name() + "(" + current_node.OpType() +
                             ") slice_node " + slice_node.Name() + "(" + slice_node.OpType() + "), keep_dim = " +
                             std::to_string(!(info.is_scalar_slice)));

  auto slice_input_arg = slice_node.MutableInputDefs()[info.GetDataInputIndex()];
  int slice_input_rank = slice_input_arg->Shape()->dim_size();
  int output_index = optimizer_utils::IndexOfNodeOutput(current_node, *slice_input_arg);
  auto slice_op_output_arg = slice_node.MutableOutputDefs()[info.GetOutputIndex()];

  // Loop all outputs of target node, update the shape accordingly.
  // For elementwise ops like (LayerNorm/Dropout/Add), we should handle all outputs.
  // If some output rank is lower than sliced axis, we should just ignore it (the correctness is guaranteed by devs
  // who adds more operator coverage in the pass through).
  for (size_t i = 0; i < current_node.MutableOutputDefs().size(); ++i) {
    UpdateSliceOutputShape(*current_node.MutableOutputDefs()[i], info.non_negative_axis - slice_input_rank,
                           info.output_dim_on_axis);
  }
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

}  // namespace

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
                             std::to_string(gather_info.value().non_negative_axis);
    LOG_DEBUG_INFO(logger, log_prefix + " starts re-ordering check");

    SliceOperationReorderHandle handle(node_name);

    // DON'T operate on `node` once this loop starts, as it may be removed from the graph.
    while (!gather_queue.empty()) {
      SliceInfo info = gather_queue.front();
      Node* gather_node = info.node_ptr;
      gather_queue.pop_front();
      Node* slice_input_data_producer =
          graph.GetMutableProducerNode(gather_node->MutableInputDefs()[0]->Name());
      if (slice_input_data_producer == nullptr) {
        break;
      }
      Node* input_node = slice_input_data_producer;
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

  LOGS(logger, INFO) << "Exit ComputeOptimizer with summary - reorderd_node_count:" << reordered_node_count
                     << " nodes.";
  return Status::OK();
}

}  // namespace onnxruntime

#endif
