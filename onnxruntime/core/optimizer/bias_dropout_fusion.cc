// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/bias_dropout_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static bool IsSameShape(const TensorShapeProto& shape1, const TensorShapeProto& shape2) {
  int rank1 = shape1.dim_size();
  if (rank1 != shape2.dim_size()) {
    return false;
  }
  bool same_shape = true;
  for (int i = 0; i < rank1; ++i) {
    same_shape &= ONNX_NAMESPACE::operator==(shape1.dim(i), shape2.dim(i));
  }
  return same_shape;
}

void FuseResidualAddIfAny(Graph& graph, const Node& dropout_node, InlinedVector<NodeArg*>& dropout_input,
                          InlinedVector<NodeArg*>& dropout_output,
                          InlinedVector<std::reference_wrapper<Node>>& nodes_to_fuse) {
  bool has_residual_add = false;

  int dropout_consumers_count = 0;
  for (auto edge_itr = dropout_node.OutputEdgesBegin(); edge_itr != dropout_node.OutputEdgesEnd(); ++edge_itr) {
    if (edge_itr->GetSrcArgIndex() == 0) {
      ++dropout_consumers_count;
    }
  }
  // To be able to fuse the residual Add,
  // the Dropout's output must not be a graph output and
  // there must be only one consumer of the Dropout's first output.
  if (dropout_consumers_count < 2 && !graph.NodeProducesGraphOutput(dropout_node)) {
    for (auto last_node_itr = dropout_node.OutputNodesBegin(); last_node_itr != dropout_node.OutputNodesEnd();
         ++last_node_itr) {
      const Node& last_node = (*last_node_itr);

      if (graph_utils::IsSupportedOptypeVersionAndDomain(last_node, "Add", {7, 13, 14}) &&
          last_node.GetExecutionProviderType() == dropout_node.GetExecutionProviderType()) {
        const TensorShapeProto* input1_shape = last_node.InputDefs()[0]->Shape();
        const TensorShapeProto* input2_shape = last_node.InputDefs()[1]->Shape();

        if (!input1_shape || !input2_shape || input1_shape->dim_size() < 1 || input2_shape->dim_size() < 1) {
          continue;
        }

        // Inputs of Residual Add must match in shape
        if (!IsSameShape(*input1_shape, *input2_shape)) {
          continue;
        }

        Node& residual_add_node = *graph.GetNode(last_node.Index());
        const std::string& dropout_output_name = dropout_node.OutputDefs()[0]->Name();
        if (dropout_output_name == residual_add_node.InputDefs()[0]->Name()) {
          dropout_input.push_back(residual_add_node.MutableInputDefs()[1]);  // residual
        } else if (dropout_output_name == residual_add_node.InputDefs()[1]->Name()) {
          dropout_input.push_back(residual_add_node.MutableInputDefs()[0]);  // residual
        }

        dropout_output[0] = residual_add_node.MutableOutputDefs()[0];

        nodes_to_fuse.push_back(residual_add_node);
        has_residual_add = true;
        break;
      }
    }
  }

  if (!has_residual_add) {
    NodeArg& dummy = graph.GetOrCreateNodeArg("", nullptr);
    dropout_input.push_back(&dummy);  // add a dummy residual
  }
}

Status BiasDropoutFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    // Matching for bias Add node.
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Add", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) || node.GetOutputEdgesCount() != 1 ||
        graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    const TensorShapeProto* input1_shape = node.MutableInputDefs()[0]->Shape();
    const TensorShapeProto* input2_shape = node.MutableInputDefs()[1]->Shape();

    if (!input1_shape || !input2_shape || input1_shape->dim_size() < 1 || input2_shape->dim_size() < 1) {
      continue;
    }

    InlinedVector<NodeArg*> dropout_input;

    if (IsSameShape(*input1_shape, *input2_shape)) {
      dropout_input.push_back(node.MutableInputDefs()[0]);  // dropout input
      dropout_input.push_back(node.MutableInputDefs()[1]);  // bias
    } else {
      const int last_dim_shape1 = input1_shape->dim_size() - 1;
      const int last_dim_shape2 = input2_shape->dim_size() - 1;
      if (!(utils::HasDimValue(input1_shape->dim(last_dim_shape1)) &&
            utils::HasDimValue(input2_shape->dim(last_dim_shape2)) &&
            input1_shape->dim(last_dim_shape1).dim_value() == input2_shape->dim(last_dim_shape2).dim_value()) &&
          !(utils::HasDimParam(input1_shape->dim(last_dim_shape1)) &&
            utils::HasDimParam(input2_shape->dim(last_dim_shape2)) &&
            input1_shape->dim(last_dim_shape1).dim_param() == input2_shape->dim(last_dim_shape2).dim_param())) {
        continue;  // continue if no same DimValue && no same DimParam
      }
      if (input1_shape->dim_size() == 1) {
        dropout_input.push_back(node.MutableInputDefs()[1]);  // dropout input
        dropout_input.push_back(node.MutableInputDefs()[0]);  // bias
      } else if (input2_shape->dim_size() == 1) {
        dropout_input.push_back(node.MutableInputDefs()[0]);  // dropout input
        dropout_input.push_back(node.MutableInputDefs()[1]);  // bias
      } else {
        continue;
      }
    }

    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse;

    Node& add_node = node;
    nodes_to_fuse.push_back(add_node);

    // matching for Dropout node
    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if ((!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Dropout", {12, 13}, kOnnxDomain) &&
         !graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "BitmaskDropout", {1}, kMSDomain)) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    Node& dropout_node = *graph.GetNode(next_node.Index());
    nodes_to_fuse.push_back(dropout_node);

    InlinedVector<NodeArg*> dropout_output;
    for (size_t i = 0; i < dropout_node.OutputDefs().size(); ++i) {
      dropout_output.push_back(dropout_node.MutableOutputDefs()[i]);
    }

    FuseResidualAddIfAny(graph, dropout_node, dropout_input, dropout_output, nodes_to_fuse);

    for (size_t i = 1; i < dropout_node.InputDefs().size(); ++i) {
      dropout_input.push_back(dropout_node.MutableInputDefs()[i]);  // ratio and training_mode if exist.
    }

    std::string op_type = dropout_node.OpType() == "Dropout" ? "BiasDropout" : "BitmaskBiasDropout";
    Node& dropout_add_fusion_node =
        graph.AddNode(graph.GenerateNodeName(op_type), op_type, "fused Add-Dropout-(Add) for " + dropout_node.Name(),
                      dropout_input, dropout_output, &dropout_node.GetAttributes(), kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    dropout_add_fusion_node.SetExecutionProviderType(dropout_node.GetExecutionProviderType());

    // Delete bias_add_node, dropout_node and optionally residual_add_node.
    for (Node& n : nodes_to_fuse) {
      graph_utils::RemoveNodeOutputEdges(graph, n);
      graph.RemoveNode(n.Index());
    }

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
