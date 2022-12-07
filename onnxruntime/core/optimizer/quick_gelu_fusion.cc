// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/quick_gelu_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

/**
Rewrite x*sigmoid(alpha*x) or x*sigmoid(x) to QuickGelu.
*/
Status QuickGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (!p_node) continue;

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    InlinedVector<std::reference_wrapper<Node>> nodes_to_fuse;

    int alpha_index = -1;
    float alpha = 1.0f;
    if (graph_utils::IsSupportedOptypeVersionAndDomain(node, "Mul", {7, 13, 14}) &&
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) && node.GetOutputEdgesCount() == 1) {
      for (int i = 0; i < static_cast<int>(node.InputDefs().size()); ++i) {
        const NodeArg& input_arg = *(node.InputDefs()[i]);
        if (!optimizer_utils::IsScalar(input_arg)) continue;
        const TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_arg.Name());
        if (!tensor_proto) continue;
        Initializer init_const{*tensor_proto, graph.ModelPath()};
        const auto data_type = tensor_proto->data_type();
        if (data_type == TensorProto_DataType_FLOAT) {
          alpha = *(init_const.data<float>());
          alpha_index = i;
          break;
        } else if (data_type == TensorProto_DataType_DOUBLE) {
          alpha = static_cast<float>(*(init_const.data<double>()));
          alpha_index = i;
          break;
        } else if (data_type == TensorProto_DataType_FLOAT16) {
          alpha = math::halfToFloat(init_const.data<MLFloat16>()->val);
          alpha_index = i;
          break;
        }
      }
    }

    NodeArg* quick_gelu_input_arg = nullptr;
    Node* p_sigmoid_node = p_node;
    // If alpha_index is not -1, it means the node is Mul node and it has a scalar input.
    // We expect the output of Mul node is consumed by a Sigmoid node.
    // If alpha_index is -1, it means current node is expected to be a Sigmoid node.
    if (alpha_index != -1) {
      quick_gelu_input_arg = node.MutableInputDefs()[(alpha_index + 1) % 2];
      nodes_to_fuse.emplace_back(node);
      p_sigmoid_node = graph.GetNode(node.OutputNodesBegin()->Index());
    }

    Node& sigmoid_node = *p_sigmoid_node;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(sigmoid_node, "Sigmoid", {6, 13}) ||
        !graph_utils::IsSupportedProvider(sigmoid_node, GetCompatibleExecutionProviders()) ||
        sigmoid_node.GetOutputEdgesCount() != 1) {
      continue;
    }
    nodes_to_fuse.emplace_back(sigmoid_node);
    if (!quick_gelu_input_arg) {
      quick_gelu_input_arg = sigmoid_node.MutableInputDefs()[0];
    }

    Node& mul_node = *graph.GetNode(sigmoid_node.OutputNodesBegin()->Index());
    int sigmoid_output_index = optimizer_utils::IndexOfNodeInput(mul_node, *sigmoid_node.MutableOutputDefs()[0]);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(mul_node, GetCompatibleExecutionProviders()) ||
        mul_node.MutableInputDefs()[(sigmoid_output_index + 1) % 2]->Name() != quick_gelu_input_arg->Name()) {
      continue;
    }
    nodes_to_fuse.emplace_back(mul_node);

    NodeArg* quick_gelu_output_arg = mul_node.MutableOutputDefs()[0];
    Node& quick_gelu_node =
        graph.AddNode(graph.GenerateNodeName("QuickGelu"), "QuickGelu", "QuickGelu", std::array{quick_gelu_input_arg},
                      std::array{quick_gelu_output_arg}, {}, kMSDomain);
    quick_gelu_node.AddAttribute("alpha", alpha);
    quick_gelu_node.SetExecutionProviderType(node.GetExecutionProviderType());
    graph_utils::FinalizeNodeFusion(graph, nodes_to_fuse, quick_gelu_node);
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
