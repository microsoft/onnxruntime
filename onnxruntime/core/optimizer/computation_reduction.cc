// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/computation_reduction.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
typedef std::function<Status(Graph&, Node&, Node&)> Handler;

static const int GATHERND_BATCH_DIM = 1;

static bool IsLeadingDimsEqual(const TensorShapeProto* input_shape, const TensorShapeProto* output_shape,
                               int num_dim_to_check) {
  if (output_shape->dim_size() < num_dim_to_check || input_shape->dim_size() < num_dim_to_check) {
    return false;
  }

  for (int i = 0; i < num_dim_to_check; i++) {
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

static int GetInputNodeToReplace(const Node& gathernd_input_node) {
  // If gathernd_input_node's some input tensor have exactly same shape with
  // gathernd_input_node output tensor shape, then it is safe to gather using
  // original slice ranges.
  int gathernd_input_index = -1;
  auto output_shape = gathernd_input_node.OutputDefs()[0]->Shape();
  const int output_rank = output_shape->dim_size();
  for (size_t i = 0; i < gathernd_input_node.InputDefs().size(); i++) {
    auto input_shape = gathernd_input_node.InputDefs()[i]->Shape();
    const int input_rank = input_shape->dim_size();
    if (input_rank != output_rank) {
      continue;
    }

    // We compare the first GATHERND_BATCH_DIM + 1 to make sure this input node
    // (of gathernd_input_node) can be input of GatherND.
    if (IsLeadingDimsEqual(input_shape, output_shape, GATHERND_BATCH_DIM + 1)) {
      gathernd_input_index = SafeInt<int>(i);
      break;
    }
  }

  return gathernd_input_index;
}

static Status SwapGatherNDWithInputNode(Graph& graph, Node& gathernd_node, Node& gathernd_input_node,
                                        int gathernd_input_index = 0) {
  auto gathernd_input_arg = gathernd_input_node.MutableInputDefs()[gathernd_input_index];
  const auto old_gathernd_input_shape = *(gathernd_input_arg->Shape());
  graph_utils::ReplaceDownstreamNodeInput(graph, gathernd_node, 0, gathernd_input_node, 0);
  graph_utils::ReplaceNodeInput(gathernd_node, 0, *gathernd_input_arg);
  graph_utils::ReplaceNodeInput(gathernd_input_node, gathernd_input_index, *gathernd_node.MutableOutputDefs()[0]);
  gathernd_node.MutableOutputDefs()[0]->ClearShape();
  gathernd_input_node.MutableOutputDefs()[0]->ClearShape();

  // Todo in this PR: reduce Resolve as less as possible.
  auto ret = graph.Resolve();
  ORT_ENFORCE(ret.IsOK());
  return Status::OK();
}

static Status SimpleHandler(Graph& graph, Node& gathernd_node, Node& gathernd_input_node) {
  return SwapGatherNDWithInputNode(graph, gathernd_node, gathernd_input_node, 0);
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
static Status BinaryElementwiseHandler(Graph& graph, Node& gathernd_node, Node& gathernd_input_node) {
  int gathernd_input_index = GetInputNodeToReplace(gathernd_input_node);
  ORT_RETURN_IF_NOT(gathernd_input_index != -1);
  return SwapGatherNDWithInputNode(graph, gathernd_node, gathernd_input_node, gathernd_input_index);
}

/*
  This handler change the graphs this way:
    Before:
      input_1[b,s,h]    weight_2[h, 2h]
                  \         /
                  Gemm[b,s,2h]    indices[b,p_s,1]
                      |            /
                  GatherND[b,p_s,2h]
                      |
                <subsquent graphs>
    After :
               input_1[b,s,h]      indices[b,p_s,1]
                       |            /
                    GatherND[b,p_s,h]    weight_2[h,2h]
                              \              /
                              Gemm[b,p_s,2h]
                                    |
                            <subsquent graphs>
  Note: b: batch, s: sequence_length, h: hidden_size, p_s: dynamic_prediction_count
*/
static Status GemmHandler(Graph& graph, Node& gathernd_node, Node& gathernd_input_node) {
  int gathernd_input_index = GetInputNodeToReplace(gathernd_input_node);
  ORT_RETURN_IF_NOT(gathernd_input_index == 0);
  return SwapGatherNDWithInputNode(graph, gathernd_node, gathernd_input_node, gathernd_input_index);
}

static std::unordered_map<std::string, Handler> handlers = {
    {"Add", BinaryElementwiseHandler},
    {"Div", BinaryElementwiseHandler},
    {"Gelu", SimpleHandler},
    {"Gemm", GemmHandler},
    {"LayerNormalization", SimpleHandler},
    {"MatMul", GemmHandler}};

static Status Delegate(std::string op_type, Graph& graph, Node& gathernd_node, Node& gathernd_input_node) {
  if (handlers.count(op_type)) {
    return handlers[op_type](graph, gathernd_node, gathernd_input_node);
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
        node.GetOutputEdgesCount() != 1) {
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
      Node* input_node = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
      if (graph.GetConsumerNodes(input_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOGS_DEFAULT(WARNING) << "node " << node.Name() << " stopped at node "
                              << input_node->Name();
        break;
      }

      auto ret = Delegate(input_node->OpType(), graph, node, *input_node);
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
