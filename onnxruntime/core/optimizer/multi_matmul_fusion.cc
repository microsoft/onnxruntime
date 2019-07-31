// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/multi_matmul_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

//class MultiMatMulFusionImpl
//{
//public:
//   MultiMatMulFusionImpl(Graph& graph) noexcept: graph_(graph){}
//
//};

Status MultiMatMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;

  for (auto node_index : node_topology_list) {
    Node& node = *(graph.GetNode(node_index));
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    std::vector<Node*> matmuls;
    for (Node::NodeConstIterator output_node_itr = node.OutputNodesBegin(); output_node_itr != node.OutputNodesEnd(); ++output_node_itr) {
      if (graph_utils::IsSupportedOptypeVersionAndDomain(*output_node_itr, "MatMul", {1, 9}) &&
          graph_utils::IsSupportedProvider(*output_node_itr, GetCompatibleExecutionProviders())) {
        matmuls.push_back(graph.GetNode((*output_node_itr).Index()));
      }
    }

    if (matmuls.size() <= 1) continue;

    bool is_mergeable = true;
    for (std::vector<Node*>::iterator iter_matmul = matmuls.begin(); iter_matmul != matmuls.end(); iter_matmul++) {
      Node* matmul_node = *iter_matmul;
      //if (!graph_utils::NodeArgIsConstant(graph, *matmul_node->InputDefs()[1])) {
      //  is_mergeable = false;
      //}
      auto matmul_b_shape = matmul_node->InputDefs()[1]->Shape();
      if (2 != matmul_b_shape->dim_size()) {
        is_mergeable = false;
        break;
      }
    }
    if (!is_mergeable && matmuls.size() != 3) continue;

    std::vector<NodeArg*> split_output_node_arg;
    for (std::vector<Node*>::iterator iter_matmul = matmuls.begin(); iter_matmul != matmuls.end(); iter_matmul++) {
      Node* matmul_node = *iter_matmul;
      removed_nodes.push_back((matmul_node->Index()));
      split_output_node_arg.push_back((matmul_node->MutableOutputDefs())[0]);
    }

    std::vector<float> multi_matmul_b(768 * 768 * matmuls.size());
    ONNX_NAMESPACE::TensorProto multi_matmul_tensor_proto;

    multi_matmul_tensor_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    multi_matmul_tensor_proto.set_name(graph.GenerateNodeArgName("multi_matmul"));
    multi_matmul_tensor_proto.set_raw_data(multi_matmul_b.data(), 768 * 768 * matmuls.size() * sizeof(float));

    multi_matmul_tensor_proto.add_dims(768);
    multi_matmul_tensor_proto.add_dims(768 * matmuls.size());
    graph.AddInitializedTensor(multi_matmul_tensor_proto);

    TypeProto fused_matmul_tensor_type;
    fused_matmul_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
    //auto* mutable_dim = fused_matmul_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim();
    //if( include_dim_values )
    //{
    //   mutable_dim->set_dim_value( 1 );
    //}
    //else if( sym_dim_zero )
    //{
    //   mutable_dim->set_dim_param( "symbolic" );
    //}

    // outer scope values
    std::vector<NodeArg*> multi_matmul_inputs{matmuls[0]->MutableInputDefs()[0], &graph.GetOrCreateNodeArg(multi_matmul_tensor_proto.name(), nullptr)};
    std::vector<NodeArg*> fused_multi_matmul_outputs{&graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("fused_multi_matmul"), &fused_matmul_tensor_type)};

    graph.AddNode(graph.GenerateNodeName("gemm"),
                  "MatMul",
                  "fused Matmul and Add ",
                  multi_matmul_inputs,
                  fused_multi_matmul_outputs)
        .SetExecutionProviderType(node.GetExecutionProviderType());

    Node& split_node = graph.AddNode(graph.GenerateNodeName("split"),
                                     "Split",
                                     "Split Op ",
                                     fused_multi_matmul_outputs,
                                     split_output_node_arg);
    AttributeProto attr;
    attr.set_name("axis");
    attr.set_type(AttributeProto_AttributeType_INT);
    attr.set_i(-1);
    split_node.AddAttribute("axis", attr);
    split_node.SetExecutionProviderType(node.GetExecutionProviderType());
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
