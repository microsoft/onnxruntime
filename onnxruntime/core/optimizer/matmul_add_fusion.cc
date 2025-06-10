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

namespace {

// Attention subgraph has 4 MatMul-Add pairs, that we want to skip here because AttentionFusion will handle it.
// In such case, 3 of MatMul-Add pairs are following LN, the other one produces output which is added with LN's output.
// Use two sets to remember such patterns we already met during the graph iteration so that we can skip them directly
// if we go to other MatMul-Add pairs in the same pattern.
struct AttentionPatternCache {
  bool IsAttentionPattern(const Graph& graph, const Node& matmul_node, const Node& add_node) {
    const Node* parent_node = graph.GetProducerNode(matmul_node.InputDefs()[0]->Name());
    if (attn_ln_nodes.count(parent_node) > 0 || attn_add_nodes.count(&add_node) > 0) {
      return true;
    }

    if (parent_node && parent_node->OpType() == "LayerNormalization") {
      unsigned int add_count = 0;
      unsigned int matmul_count = 0;
      unsigned int shape_count = 0;
      const Node* ln_add_node = nullptr;
      for (auto it = parent_node->OutputNodesBegin(); it != parent_node->OutputNodesEnd(); ++it) {
        std::string op_type = (*it).OpType();
        if (op_type == "Add") {
          ln_add_node = &(*it);
          add_count++;
        } else if (op_type == "MatMul") {
          matmul_count++;
        } else if (op_type == "Shape") {
          shape_count++;
        }
      }

      if (add_count == 1 && matmul_count == 3 && shape_count == parent_node->GetOutputEdgesCount() - 4) {
        size_t index = ln_add_node->InputDefs()[0]->Name() == parent_node->OutputDefs()[0]->Name() ? 1 : 0;
        const Node* attn_add_node = graph.GetProducerNode(ln_add_node->InputDefs()[index]->Name());
        if (attn_add_node && attn_add_node->OpType() == "Add") {
          attn_ln_nodes.insert(parent_node);
          attn_add_nodes.insert(attn_add_node);
          return true;
        }
      }
    }

    return false;
  }

  std::unordered_set<const Node*> attn_ln_nodes;
  std::unordered_set<const Node*> attn_add_nodes;
};

}  // namespace

Status MatMulAddFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // Cache for skipping Attention subgraph pattern.
  AttentionPatternCache attn_pattern_cache;

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
    InlinedVector<int64_t> shape_values;
    int64_t m = 0, k = 0, n = 0;
    if (need_reshape) {
      // Only check and skip Attention pattern here because normally input to Attention is 4D.
      if (attn_pattern_cache.IsAttentionPattern(graph, matmul_node, add_node)) {
        continue;
      }

      // Logically we can use Shape-Concat to produce shape input for Reshape, to keep it simple, we require
      // both inputs have concrete shape for now, we can add dynamic shape support in future.
      auto a_shape = utils::GetTensorShapeFromTensorShapeProto(*matmul_a_shape);
      if (a_shape.Size() == -1) {
        continue;
      }

      const auto& dim_k = matmul_b_shape->dim(0);
      if (!utils::HasDimValue(dim_k) || !utils::HasDimValue(dim_n)) {
        continue;
      }

      shape_values = a_shape.AsShapeVector();
      // If a_shape is 1D, m is 1 from SizeToDimension() with empty dimension interval.
      m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
      k = dim_k.dim_value();
      n = dim_n.dim_value();
    }

    const auto& matmul_output = *matmul_node.OutputDefs()[0];

    auto matmul_output_name = matmul_output.Name();
    auto gemm_input_defs = matmul_input_defs;
    int bias_idx = matmul_output_name == add_input_defs[0]->Name() ? 1 : 0;
    gemm_input_defs.push_back(add_input_defs[bias_idx]);

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
                  (!need_reshape && bias_shape.dim_size() == 2 && dim_has_value_1(bias_shape.dim(0)) &&
                   bias_shape.dim(1) == dim_n) ||
                  (!need_reshape && bias_shape.dim_size() == 2 && bias_shape.dim(0) == matmul_a_shape->dim(0) &&
                   (dim_has_value_1(bias_shape.dim(1)) || bias_shape.dim(1) == dim_n)));
    if (!valid) {
      continue;
    }

    auto gemm_output_defs = add_node.MutableOutputDefs();
    Node* input_node = nullptr;
    Node* output_node = nullptr;
    if (need_reshape) {
      auto add_reshape = [&](const InlinedVector<int64_t>& shape, Graph& graph, bool is_input) -> Node* {
        const std::string name = is_input ? "gemm_input" : "gemm_output";
        ONNX_NAMESPACE::TensorProto shape_initializer_proto;
        shape_initializer_proto.set_name(graph.GenerateNodeName(name + "_shape"));
        shape_initializer_proto.add_dims(static_cast<int64_t>(shape.size()));
        shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
        utils::SetRawDataInTensorProto(shape_initializer_proto, shape.data(), shape.size() * sizeof(int64_t));
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
        return &reshape_node;
      };

      input_node = add_reshape({m, k}, graph, true);
      gemm_input_defs[0] = input_node->MutableOutputDefs()[0];
      shape_values.back() = n;
      output_node = add_reshape(shape_values, graph, false);
      gemm_output_defs[0] = output_node->MutableInputDefs()[0];
    }

    Node& gemm_node = graph.AddNode(graph.GenerateNodeName(matmul_node.Name() + "/MatMulAddFusion"), "Gemm",
                                    "fused Matmul and Add", gemm_input_defs, gemm_output_defs);
    gemm_node.SetExecutionProviderType(matmul_node.GetExecutionProviderType());

    if (need_reshape) {
      graph.AddEdge(input_node->Index(), gemm_node.Index(), 0, 0);
      graph.AddEdge(gemm_node.Index(), output_node->Index(), 0, 0);
    } else {
      input_node = &gemm_node;
      output_node = &gemm_node;
    }

    auto matmul_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(matmul_node);
    for (auto cur = matmul_input_edges.cbegin(), end = matmul_input_edges.cend(); cur != end; ++cur) {
      if (cur->dst_arg_index == 0) {
        graph.AddEdge(cur->src_node, input_node->Index(), cur->src_arg_index, 0);
      } else if (cur->dst_arg_index == 1) {
        graph.AddEdge(cur->src_node, gemm_node.Index(), cur->src_arg_index, 1);
      }
    }

    graph_utils::GraphEdge::RemoveGraphEdges(graph, matmul_input_edges);
    auto add_input_edges = graph_utils::GraphEdge::GetNodeInputEdges(add_node);
    for (auto cur = add_input_edges.cbegin(), end = add_input_edges.cend(); cur != end; ++cur) {
      if (cur->dst_arg_index == bias_idx) {
        graph.AddEdge(cur->src_node, gemm_node.Index(), cur->src_arg_index, 2);
        break;
      }
    }

    graph_utils::GraphEdge::RemoveGraphEdges(graph, add_input_edges);
    graph_utils::RemoveNodeOutputEdges(graph, matmul_node);
    graph_utils::ReplaceDownstreamNodeInput(graph, add_node, 0, *output_node, 0);
    graph.RemoveNode(matmul_node.Index());
    graph.RemoveNode(add_node.Index());

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
