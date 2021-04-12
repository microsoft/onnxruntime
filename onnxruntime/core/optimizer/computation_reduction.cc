// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/computation_reduction.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
typedef std::function<Status(Graph&, Node&, Node&)> Handler;

constexpr int GATHERND_BATCH_DIM = 1;

static bool IsLeadingDimsEqual(const TensorShapeProto* input_shape, const TensorShapeProto* output_shape,
                               const int num_dim_to_check) {
  ORT_ENFORCE(output_shape->dim_size() >= num_dim_to_check && input_shape->dim_size() >= num_dim_to_check);

  for (int i = 0; i < num_dim_to_check; ++i) {
    auto& output_dim = output_shape->dim(i);
    auto& input_dim = input_shape->dim(i);
    if (output_dim.has_dim_value() && input_dim.has_dim_value()) {
      if (output_dim.dim_value() != input_dim.dim_value()) {
        return false;
      }
    } else if (output_dim.has_dim_param() && input_dim.has_dim_param()) {
      if (output_dim.dim_param() != input_dim.dim_param()) {
        return false;
      }
    } else {
      return false;
    }
  }

  return true;
}

static int GetValidInputForGatherND(const Node& target_node) {
  // target_node is the producer of GatherND's input.
  // If target_node's some input tensors have exactly same shape with
  // target_node output tensor shape, then it is safe to gather using
  // original slice ranges.
  int candidate_input_index = -1;
  auto output_shape = target_node.OutputDefs()[0]->Shape();
  const int output_rank = output_shape->dim_size();
  for (size_t i = 0; i < target_node.InputDefs().size(); ++i) {
    auto input_shape = target_node.InputDefs()[i]->Shape();
    const int input_rank = input_shape->dim_size();
    if (input_rank != output_rank) {
      continue;
    }

    if (IsLeadingDimsEqual(input_shape, output_shape, GATHERND_BATCH_DIM + 1)) {
      candidate_input_index = SafeInt<int>(i);
      break;
    }
  }

  return candidate_input_index;
}

static TensorShapeProto ReplaceSymbolicDimValue(const TensorShapeProto* shape, const int replacement_axis,
                                                const std::string& replacement_dim_value) {
  ORT_ENFORCE(replacement_axis >= 0 && replacement_axis < shape->dim_size());
  TensorShapeProto output_shape;
  for (int i = 0; i < shape->dim_size(); ++i) {
    auto& dim = shape->dim(i);
    if (i == replacement_axis) {
      output_shape.add_dim()->set_dim_param(replacement_dim_value);
      continue;
    }

    if (dim.has_dim_value()) {
      output_shape.add_dim()->set_dim_value(dim.dim_value());
    } else if (dim.has_dim_param()) {
      output_shape.add_dim()->set_dim_param(dim.dim_param());
    } else {
      ORT_THROW("Invalid dim found in ReplaceSymbolicDimValue");
    }
  }

  return output_shape;
}

static Status SwapGatherNDWithTargetNode(Graph& graph, Node& gathernd_node, Node& target_node,
                                         const int target_node_input_index = 0) {
  auto new_input_arg_for_gathernd = target_node.MutableInputDefs()[target_node_input_index];
  auto target_node_out_arg = target_node.MutableOutputDefs()[0];
  auto gathernd_out_arg = gathernd_node.MutableOutputDefs()[0];
  auto gathernd_old_consumers = graph.GetConsumerNodes(gathernd_out_arg->Name());
  const auto& graph_outputs = graph.GetOutputs();
  bool need_update_graph_output = false;
  if (std::find(graph_outputs.begin(), graph_outputs.end(), gathernd_out_arg) != graph_outputs.end()) {
    need_update_graph_output = true;
  }

  const std::string& gathered_dim_param = gathernd_out_arg->Shape()->dim(GATHERND_BATCH_DIM).dim_param();
  TensorShapeProto new_output_shape_for_gathernd =
      ReplaceSymbolicDimValue(new_input_arg_for_gathernd->Shape(), GATHERND_BATCH_DIM, gathered_dim_param);

  TensorShapeProto new_output_shape_for_target_node =
      ReplaceSymbolicDimValue(target_node_out_arg->Shape(), GATHERND_BATCH_DIM, gathered_dim_param);

  // update input/output definitions.
  int output_index = optimizer_utils::IndexOfNodeOutput(target_node, *gathernd_node.MutableInputDefs()[0]);
  graph.RemoveEdge(target_node.Index(), gathernd_node.Index(), output_index, 0);
  const Node* target_node_input_node = graph.GetProducerNode(new_input_arg_for_gathernd->Name());
  if (target_node_input_node != nullptr) {
    output_index = optimizer_utils::IndexOfNodeOutput(*target_node_input_node, *new_input_arg_for_gathernd);
    graph.AddEdge(target_node_input_node->Index(), gathernd_node.Index(), output_index, 0);
  } else {
    // new_input_arg_for_gathernd is graph input
    graph_utils::ReplaceNodeInput(gathernd_node, 0, *new_input_arg_for_gathernd);
  }

  graph_utils::ReplaceDownstreamNodeInput(graph, gathernd_node, 0 /*output_idx*/,
                                          target_node, 0 /*replacement_output_idx*/);

  if (target_node_input_node != nullptr) {
    graph.RemoveEdge(target_node_input_node->Index(), target_node.Index(), output_index, target_node_input_index);
  }
  graph.AddEdge(gathernd_node.Index(), target_node.Index(), 0, target_node_input_index);

  // update consumer relation ship
  if (!gathernd_old_consumers.empty()) {
    graph.UpdateConsumerNodes(target_node_out_arg->Name(), {const_cast<Node*>(gathernd_old_consumers[0])});
  }
  graph.UpdateConsumerNodes(gathernd_out_arg->Name(), {&target_node});

  // update shapes
  gathernd_out_arg->SetShape(new_output_shape_for_gathernd);
  target_node_out_arg->SetShape(new_output_shape_for_target_node);

  if (need_update_graph_output) {
    std::vector<const NodeArg*> graph_new_outputs;
    for (auto out_arg : graph_outputs) {
      if (out_arg->Name().compare(gathernd_out_arg->Name()) == 0) {
        graph_new_outputs.push_back(target_node_out_arg);
      } else {
        graph_new_outputs.push_back(out_arg);
      }
    }
    graph.SetOutputs(graph_new_outputs);
    graph.SetGraphResolveNeeded();
    graph.SetGraphProtoSyncNeeded();
  }

  return Status::OK();
}

static Status SimpleHandler(Graph& graph, Node& gathernd_node, Node& target_node) {
  return SwapGatherNDWithTargetNode(graph, gathernd_node, target_node, 0);
}

/*
  This handler change the graphs this way:
    Before:
      input_1[b,s,h]   weight_2[h]
                  \         /
                    Add[b,s,h]    indices[b,p_s,1]
                      |            /
                  GatherND[b,p_s,h]
                      |
                <subsquent graphs>
    After :
               input_1[b,s,h]      indices[b,p_s,1]
                       |            /
                    GatherND[b,p_s,h]    weight_2[h]
                              \              /
                              Add[b,p_s,h]
                                    |
                            <subsquent graphs>
  Note: b: batch, s: sequence_length, h: hidden_size, p_s: dynamic_prediction_count
*/
static Status BinaryElementwiseHandler(Graph& graph, Node& gathernd_node, Node& target_node) {
  int target_node_input_index = GetValidInputForGatherND(target_node);
  ORT_RETURN_IF(target_node_input_index == -1, "Invalid target node index");
  return SwapGatherNDWithTargetNode(graph, gathernd_node, target_node, target_node_input_index);
}

/*
  This handler change the graphs this way:
    Before:
      input_1[b,s,h]    weight_2[h, 2h]
                  \         /
                  MatMul[b,s,2h]    indices[b,p_s,1]
                      |            /
                  GatherND[b,p_s,2h]
                      |
                <subsquent graphs>
    After :
               input_1[b,s,h]      indices[b,p_s,1]
                       |            /
                    GatherND[b,p_s,h]    weight_2[h,2h]
                              \              /
                              MatMul[b,p_s,2h]
                                    |
                            <subsquent graphs>
  Note: b: batch, s: sequence_length, h: hidden_size, p_s: dynamic_prediction_count
*/
static Status MatMulHandler(Graph& graph, Node& gathernd_node, Node& target_node) {
  int target_node_input_index = GetValidInputForGatherND(target_node);
  ORT_RETURN_IF_NOT(target_node_input_index == 0, "target_node_input_index != 0");
  return SwapGatherNDWithTargetNode(graph, gathernd_node, target_node, target_node_input_index);
}

static std::unordered_map<std::string, Handler> handlers = {
    {"Add", BinaryElementwiseHandler},
    {"Div", BinaryElementwiseHandler},
    {"Gelu", SimpleHandler},
    //{"LayerNormalization", SimpleHandler},
    {"MatMul", MatMulHandler}};

static Status Delegate(Graph& graph, Node& gathernd_node, Node& target_node) {
  const std::string& op_type = target_node.OpType();
  if (handlers.count(op_type)) {
    return handlers[op_type](graph, gathernd_node, target_node);
  } else {
    return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED, op_type + " handler is not implemented");
  }
}

Status ComputationReductionTransformer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                                  const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  for (auto index : order) {
    auto* node_ptr = graph.GetNode(index);
    if (!node_ptr)
      // node was removed, this should not happen since we are not removing nodes.
      continue;

    auto& node = *node_ptr;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Same ideas might apply for Gather, GatherElements, Slice, Split, etc.
    // Todo: let's review the real cases to make the logic more generic.
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GatherND", {1, 12, 13}, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() > 1) {  // allow GatherND have no out edges in case it is graph output.
      continue;
    }

    auto batch_dims = static_cast<int64_t>(node.GetAttributes().at("batch_dims").i());
    if (batch_dims != GATHERND_BATCH_DIM) {
      continue;
    }

    auto indices_shape = node.MutableInputDefs()[1]->Shape();
    if (indices_shape == nullptr) {
      continue;
    }

    const auto indices_rank = indices_shape->dim_size();
    auto& indices_last_dim = indices_shape->dim(indices_rank - 1);
    // Since GatherND is assumed to have batch_dims=1, if the input data's shape is [batch, sequence, ..., ... ],
    // limiting indices_rank=3 will make sure produced output is in shape [batch, sliced_sequence, ..., ...]
    // and the rank did not change.
    if (!(indices_last_dim.has_dim_value() && indices_last_dim.dim_value() == 1 && indices_rank == 3)) {
      continue;
    }

    // Todo: check whether we want to move GatherND up, for example, if GatherND's outputs are larger
    // than inputs, we should NOT probably bring it ahead.
    bool stop = false;
    while (!stop) {
      const Node* gathernd_data_producer = graph.GetProducerNode(node.MutableInputDefs()[0]->Name());
      if (gathernd_data_producer == nullptr) {
        break;
      }
      Node* input_node = const_cast<Node*>(gathernd_data_producer);
      if (graph.GetConsumerNodes(input_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOGS_DEFAULT(WARNING) << "node " << node.Name() << " stopped at node "
                              << input_node->Name();
        break;
      }

      auto ret = Delegate(graph, node, *input_node);
      if (ret.IsOK()) {
        LOGS_DEFAULT(WARNING) << "node " << node.Name() << " up across node "
                              << input_node->Name() << std::endl;
        modified = true;
      } else if (ret.Code() == common::NOT_IMPLEMENTED) {
        LOGS_DEFAULT(WARNING) << "node " << node.Name() << " stopped at node "
                              << input_node->Name();
        break;
      } else {
        LOGS_DEFAULT(WARNING) << " terminate due to unexpected error, node names:" << node.Name()
                              << ", " << input_node->Name() << ", error " << ret.ErrorMessage() << std::endl;
        stop = true;
      }
    }
  }

  if (modified) {
    graph.SetGraphResolveNeeded();
  }
  return Status::OK();
}

}  // namespace onnxruntime
