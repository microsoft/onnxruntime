// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/computation_reduction.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
typedef std::function<Status(Graph&, Node*, Node*)> Handler;

static Status SwapGatherNDWithInputNode(Graph& graph, Node* gathernd_node, Node* gathernd_input_node, int producer_input_index = 0) {
  graph_utils::ReplaceDownstreamNodeInput(graph, *gathernd_node, 0, *gathernd_input_node, 0);
  graph_utils::ReplaceNodeInput(*gathernd_node, 0, *gathernd_input_node->MutableInputDefs()[producer_input_index]);
  graph_utils::ReplaceNodeInput(*gathernd_input_node, producer_input_index, *gathernd_node->MutableOutputDefs()[0]);
  gathernd_node->MutableOutputDefs()[0]->ClearShape();
  gathernd_input_node->MutableOutputDefs()[0]->ClearShape();
  auto ret = graph.Resolve();
  ORT_ENFORCE(ret.IsOK());
  return Status::OK();
}

static Status SimpleHandler(Graph& graph, Node* gathernd_node, Node* gathernd_input_node) {
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
static Status AddHandler(Graph& graph, Node* gathernd_node, Node* gathernd_input_node) {
  int non_weight_input_index = -1;
  for (size_t i = 0; i < gathernd_input_node->MutableInputDefs().size(); i++) {
    if (!graph_utils::IsGraphInput(graph, gathernd_input_node->MutableInputDefs()[i]) &&
        !graph_utils::IsInitializer(graph, gathernd_input_node->MutableInputDefs()[i]->Name(), false)) {
      non_weight_input_index = i;
    }
  }

  ORT_RETURN_IF_NOT(non_weight_input_index != -1);
  auto output_shape = gathernd_input_node->MutableOutputDefs()[0]->Shape();
  auto input_shape = gathernd_input_node->MutableInputDefs()[non_weight_input_index]->Shape();

  ORT_RETURN_IF_NOT(output_shape != nullptr && input_shape != nullptr);
  const int output_rank = output_shape->dim_size();
  const int input_rank = input_shape->dim_size();
  ORT_RETURN_IF_NOT(input_rank == output_rank);

  // Only hande cases where input share exactly equal with output shape.
  // otherwise, it might not be safe to move GatherND upward.
  for (int i = 0; i < input_rank; i++) {
    auto& output_dim = output_shape->dim(output_rank - 1 - i);
    auto& input_dim = input_shape->dim(input_rank - 1 - i);
    if (output_dim.has_dim_value() && input_dim.has_dim_value()) {
      ORT_RETURN_IF_NOT(output_dim.dim_value() == input_dim.dim_value());
    } else if (output_dim.has_dim_param() && input_dim.has_dim_param()) {
      ORT_RETURN_IF_NOT(output_dim.dim_param() == input_dim.dim_param());
    }
  }

  return SwapGatherNDWithInputNode(graph, gathernd_node, gathernd_input_node, non_weight_input_index);
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
                              Add[b,p_s,2h]
                                    |
                            <subsquent graphs>
  Note: b: batch, s: sequence_length, h: hidden_size, p_s: dynamic_prediction_count
*/
static Status MatMulHandler(Graph& graph, Node* gathernd_node, Node* gathernd_input_node) {
  int non_weight_input_index = -1;
  for (size_t i = 0; i < gathernd_input_node->MutableInputDefs().size(); i++) {
    if (!graph_utils::IsGraphInput(graph, gathernd_input_node->MutableInputDefs()[i]) &&
        !graph_utils::IsInitializer(graph, gathernd_input_node->MutableInputDefs()[i]->Name(), false)) {
      non_weight_input_index = i;
    }
  }

  ORT_RETURN_IF_NOT(non_weight_input_index == 0);
  auto output_shape = gathernd_input_node->MutableOutputDefs()[0]->Shape();
  auto input_shape = gathernd_input_node->MutableInputDefs()[non_weight_input_index]->Shape();

  ORT_RETURN_IF_NOT(output_shape != nullptr && input_shape != nullptr);
  const int output_rank = output_shape->dim_size();
  const int input_rank = input_shape->dim_size();
  ORT_RETURN_IF_NOT(input_rank == output_rank);
  // Only hande cases where input share exactly equal with output shape on the first two dims (
  // because GatherND's batch_dims be 1).
  // otherwise, it might not be safe to move GatherND upward.
  for (int i = 0; i < 2; i++) {
    auto& output_dim = output_shape->dim(i);
    auto& input_dim = input_shape->dim(i);
    if (output_dim.has_dim_value() && input_dim.has_dim_value()) {
      ORT_RETURN_IF_NOT(output_dim.dim_value() == input_dim.dim_value());
    } else if (output_dim.has_dim_param() && input_dim.has_dim_param()) {
      ORT_RETURN_IF_NOT(output_dim.dim_param() == input_dim.dim_param());
    }
  }
  return SwapGatherNDWithInputNode(graph, gathernd_node, gathernd_input_node, 0);
}

static std::unordered_map<std::string, Handler> handlers = {
    {"Add", AddHandler},
    {"Gelu", SimpleHandler},
    {"LayerNormalization", SimpleHandler},
    {"MatMul", MatMulHandler},
};

static Status Delegate(std::string op_type, Graph& graph, Node* gathernd_node, Node* gathernd_input_node) {
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

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "GatherND", {1}, kOnnxDomain) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto batch_dims = static_cast<int64_t>(node.GetAttributes().at("batch_dims").i());
    if (batch_dims != 1) {
      continue;
    }

    auto indices_shape = node.MutableInputDefs()[1]->Shape();
    if (indices_shape == nullptr) {
      continue;
    }

    const auto indices_rank = indices_shape->dim_size();
    auto& indices_last_dim = indices_shape->dim(indices_rank - 1);
    // Since GatherND is assumed to have batch_dims=1, if the input data's shape is [batch, sequence, ..., ... ],
    // limiting indices_rank = 3 will make sure produced output is in shape [batch, sliced_sequence, ..., ...]
    // and the rank did not change.
    if (!(indices_last_dim.has_dim_value() && indices_last_dim.dim_value() == 1 && indices_rank == 3)) {
      continue;
    }

    // Todo: check whether we want to move GatherND up, for example, if GatherND's outputs are larger
    // than inputs, we should probably bring it ahead.
    bool stop = false;
    while (!stop) {
      Node* input_node = const_cast<Node*>(graph.GetProducerNode(node.MutableInputDefs()[0]->Name()));
      if (graph.GetConsumerNodes(input_node->MutableOutputDefs()[0]->Name()).size() > 1) {
        LOGS_DEFAULT(WARNING) << "node " << node.Name() << " stopped at node "
                              << input_node->Name();
        break;
      }

      auto ret = Delegate(input_node->OpType(), graph, &node, input_node);
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

  return Status::OK();
}

}  // namespace onnxruntime
