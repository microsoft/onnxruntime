// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/graph/graph_utils.h"
#include "core/framework/tensorprotoutils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status MatMulAddFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr)
      continue;  // node was removed

    auto& node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "MatMul", {1, 9, 13}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }

    if (graph.NodeProducesGraphOutput(node)) {
      continue;
    }

    auto next_node_itr = node.OutputNodesBegin();
    if (next_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& next_node = (*next_node_itr);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(next_node, "Add", {7, 13, 14}) ||
        next_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    Node& matmul_node = node;
    Node& add_node = const_cast<Node&>(next_node);
    std::vector<NodeArg> input_args;
    std::vector<NodeArg> output_args;
    auto matmul_input_defs = matmul_node.MutableInputDefs();
    auto add_input_defs = add_node.MutableInputDefs();

    // Gemm requires that inputs be the same data type and both floating point (float32/float16).
    auto matmul_type = matmul_input_defs[0]->Type();
    auto add_type = add_input_defs[0]->Type();
    if ((*matmul_type) != (*add_type)) {
      continue;
    }
    if ((*matmul_type) != "tensor(float)" && (*matmul_type) != "tensor(float16)" && (*matmul_type) != "tensor(bfloat16)") {
      continue;
    }

    // Gemm only support Matrix, need to check the shape of MatMul and Add
    auto matmul_a_shape = matmul_input_defs[0]->Shape();
    auto matmul_b_shape = matmul_input_defs[1]->Shape();
    if (nullptr == matmul_a_shape || nullptr == matmul_b_shape || matmul_b_shape->dim_size() != 2) {
      continue;
    }

    bool need_reshape = matmul_a_shape->dim_size() != 2;
    const auto& dim_n = matmul_b_shape->dim(1);
    std::vector<int64_t> shape_values;
    int64_t m = 0, k = 0, n = 0;
    if (need_reshape) {
      // Logically we can use Shape-Concat to produce shape input for Reshape, to keep it simple, we require
      // both inputs have concrete shape for now, we can add dynamic shape support in future.
      bool is_concrete_shape = true;
      for (int i = 0; i < matmul_a_shape->dim_size(); ++i) {
        const auto& dim = matmul_a_shape->dim(i);
        if (!utils::HasDimValue(dim)) {
          is_concrete_shape = false;
          break;
        }
        shape_values.emplace_back(dim.dim_value());
      }
      if (!is_concrete_shape) {
        continue;
      }
      const auto& dim_k = matmul_b_shape->dim(0);
      if (!utils::HasDimValue(dim_k) || !utils::HasDimValue(dim_n)) {
        continue;
      }
      k = dim_k.dim_value();
      n = dim_n.dim_value();
      ORT_ENFORCE(shape_values.back() == k);
      m = std::accumulate(shape_values.begin(), shape_values.end() - 1, 1, std::multiplies<int64_t>());
    }

    const auto& matmul_output = *matmul_node.OutputDefs()[0];

    auto matmul_output_name = matmul_output.Name();
    auto gemm_input_defs = matmul_input_defs;
    if (matmul_output_name == add_input_defs[0]->Name()) {
      // matmul output as Add_A, should use Add_B as input C for gemm
      gemm_input_defs.push_back(add_input_defs[1]);
    } else {
      // matmul output as Add_B, should use Add_A as input C for gemm
      gemm_input_defs.push_back(add_input_defs[0]);
    }

    // valid bias_shapes are (N) or (1, N) or (M, 1) or (M, N) as
    // GEMM only supports unidirectional broadcast on the bias input C
    if (!gemm_input_defs.back()->Shape()) {
      continue;
    }
    const auto& bias_shape = *gemm_input_defs.back()->Shape();
    auto dim_has_value_1 = [](const TensorShapeProto_Dimension& dim) {
      return dim.has_dim_value() && dim.dim_value() == 1;
    };

    bool valid = ((bias_shape.dim_size() == 1 && bias_shape.dim(0) == dim_n) ||
                  (bias_shape.dim_size() == 2 && dim_has_value_1(bias_shape.dim(0)) && bias_shape.dim(1) == dim_n) ||
                  (bias_shape.dim_size() == 2 &&
                   ((!need_reshape && bias_shape.dim(0) == matmul_a_shape->dim(0)) ||
                    (need_reshape && utils::HasDimValue(bias_shape.dim(0)) && bias_shape.dim(0).dim_value() == m)) &&
                   (dim_has_value_1(bias_shape.dim(1)) || bias_shape.dim(1) == dim_n)));
    if (!valid) {
      continue;
    }

    auto gemm_output_defs = add_node.MutableOutputDefs();
    if (need_reshape) {
      auto add_reshape = [&](const std::vector<int64_t>& shape, Graph& graph, bool is_input) {
        const std::string name = is_input ? "gemm_input" : "gemm_output";
        ONNX_NAMESPACE::TensorProto shape_initializer_proto;
        shape_initializer_proto.set_name(graph.GenerateNodeName(name + "_shape"));
        shape_initializer_proto.add_dims(static_cast<int64_t>(shape.size()));
        shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        shape_initializer_proto.set_raw_data(shape.data(), shape.size() * sizeof(int64_t));
        NodeArg* shape_arg = &graph_utils::AddInitializer(graph, shape_initializer_proto);
        ONNX_NAMESPACE::TypeProto new_arg_type;
        const ONNX_NAMESPACE::TensorProto_DataType element_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(
            gemm_input_defs[0]->TypeAsProto()->tensor_type().elem_type());
        new_arg_type.mutable_tensor_type()->set_elem_type(element_type);
        new_arg_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(m);
        new_arg_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(is_input ? k : n);
        NodeArg* new_arg = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(name + "_reshape_arg"), &new_arg_type);
        Node& reshape_node = graph.AddNode(graph.GenerateNodeName(name + "_reshape"), "Reshape", "Reshape for " + name,
                                           {is_input ? gemm_input_defs[0] : new_arg, shape_arg},
                                           {is_input ? new_arg : gemm_output_defs[0]});
        reshape_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());
        return new_arg;
      };

      gemm_input_defs[0] = add_reshape({m, k}, graph, true);
      shape_values.back() = n;
      gemm_output_defs[0] = add_reshape(shape_values, graph, false);
    }

    Node& gemm_node = graph.AddNode(graph.GenerateNodeName(matmul_node.Name() + "/MatMulAddFusion/"), "Gemm",
                                    "fused Matmul and Add", gemm_input_defs, gemm_output_defs);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gemm_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());

    graph_utils::RemoveNodeOutputEdges(graph, matmul_node);
    graph.RemoveNode(matmul_node.Index());
    graph_utils::RemoveNodeOutputEdges(graph, add_node);
    graph.RemoveNode(add_node.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
