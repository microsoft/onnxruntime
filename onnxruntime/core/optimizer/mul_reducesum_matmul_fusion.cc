// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/mul_reducesum_matmul_fusion.h"

#include <algorithm>
#include <optional>
#include <string>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

namespace {

// Returns the rank of `arg`, or std::nullopt if its shape is unknown.
std::optional<int64_t> RankOf(const NodeArg& arg) {
  const auto* shape = arg.Shape();
  if (shape == nullptr) {
    return std::nullopt;
  }
  return static_cast<int64_t>(shape->dim_size());
}

// Returns true if both dimensions are known to be the same: the same value, or the same symbolic name.
bool SameDim(const TensorShapeProto_Dimension& a, const TensorShapeProto_Dimension& b) {
  if (utils::HasDimValue(a) && utils::HasDimValue(b)) {
    return a.dim_value() == b.dim_value();
  }
  if (utils::HasDimParam(a) && utils::HasDimParam(b)) {
    return a.dim_param() == b.dim_param();
  }
  return false;
}

// Returns the single axis that `node` (Unsqueeze, Squeeze or ReduceSum) reads from its "axes" attribute (before
// opset 13) or from its constant second input, or std::nullopt if it doesn't have exactly one.
std::optional<int64_t> SingleAxis(const Graph& graph, const Node& node) {
  InlinedVector<int64_t> axes;
  const auto& inputs = node.InputDefs();
  if (inputs.size() > 1 && inputs[1]->Exists()) {
    if (!optimizer_utils::AppendTensorFromInitializer(graph, *inputs[1], axes, true)) {
      return std::nullopt;
    }
  } else {
    const auto* attr = graph_utils::GetNodeAttribute(node, "axes");
    if (attr == nullptr) {
      return std::nullopt;
    }
    axes.assign(attr->ints().begin(), attr->ints().end());
  }
  if (axes.size() != 1) {
    return std::nullopt;
  }
  return axes[0];
}

bool IsFloatTensor(const NodeArg& arg) {
  const auto* type = arg.TypeAsProto();
  return type != nullptr && type->has_tensor_type() &&
         type->tensor_type().elem_type() == TensorProto_DataType_FLOAT;
}

// Adds an Unsqueeze or Squeeze of `input` along axis -2, the form its opset takes.
Node& AddAxisNode(Graph& graph, const std::string& op_type, NodeArg& input, NodeArg& output, int onnx_opset,
                  const std::string& provider) {
  Node* node = nullptr;
  if (onnx_opset < 13) {
    node = &graph.AddNode(graph.GenerateNodeName(op_type), op_type, "MulReduceSumToMatMulFusion", {&input}, {&output});
    node->AddAttribute("axes", std::vector<int64_t>{-2});
  } else {
    TensorProto axes;
    axes.set_name(graph.GenerateNodeArgName(op_type + "_axes"));
    axes.set_data_type(TensorProto_DataType_INT64);
    axes.add_dims(1);
    axes.add_int64_data(-2);
    NodeArg& axes_arg = graph_utils::AddInitializerWithOrtValue(graph, axes);
    node = &graph.AddNode(graph.GenerateNodeName(op_type), op_type, "MulReduceSumToMatMulFusion", {&input, &axes_arg},
                          {&output});
  }
  node->SetExecutionProviderType(provider);
  return *node;
}

// Connects input `dst_arg` of `dst` to whatever produced input `src_input` of `old_dst`, if a node did.
void CopyInputEdge(Graph& graph, const Node& old_dst, int src_input, Node& dst, int dst_arg) {
  const auto* edge = graph_utils::GetInputEdge(old_dst, src_input);
  if (edge != nullptr) {
    graph.AddEdge(edge->GetNode().Index(), dst.Index(), edge->GetSrcArgIndex(), dst_arg);
  }
}

}  // namespace

Status MulReduceSumToMatMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                             const logging::Logger& logger) const {
  const auto& opset_map = graph.DomainToVersionMap();
  const auto opset_it = opset_map.find(kOnnxDomain);
  const int onnx_opset = opset_it != opset_map.end() ? opset_it->second : -1;

  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();
  for (auto index : order) {
    Node* reduce_ptr = graph.GetNode(index);
    if (reduce_ptr == nullptr) {
      continue;  // removed
    }
    Node& reduce = *reduce_ptr;
    ORT_RETURN_IF_ERROR(Recurse(reduce, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(reduce, "ReduceSum", {1, 11, 13}) ||
        !graph_utils::IsSupportedProvider(reduce, GetCompatibleExecutionProviders())) {
      continue;
    }
    const auto* keepdims_attr = graph_utils::GetNodeAttribute(reduce, "keepdims");
    const bool keepdims = keepdims_attr == nullptr || keepdims_attr->i() != 0;

    // The product: a Mul read only by the ReduceSum.
    const Node* mul_ptr = graph_utils::GetInputNode(reduce, 0);
    if (mul_ptr == nullptr ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*mul_ptr, "Mul", {7, 13, 14}) ||
        mul_ptr->GetExecutionProviderType() != reduce.GetExecutionProviderType() ||
        mul_ptr->GetOutputEdgesCount() != 1 || graph.NodeProducesGraphOutput(*mul_ptr)) {
      continue;
    }
    Node& mul = *graph.GetNode(mul_ptr->Index());

    // One factor is a column Unsqueeze(v, axes=[-1]); the other is the matrix X.
    int column_input = -1;
    const Node* column_ptr = nullptr;
    for (int i = 0; i < 2; ++i) {
      const Node* input = graph_utils::GetInputNode(mul, i);
      if (input != nullptr &&
          graph_utils::IsSupportedOptypeVersionAndDomain(*input, "Unsqueeze", {1, 11, 13, 21, 23, 24, 25})) {
        const auto axis = SingleAxis(graph, *input);
        const auto rank = RankOf(*input->OutputDefs()[0]);
        if (axis.has_value() && rank.has_value() && (*axis == -1 || *axis == *rank - 1)) {
          column_input = i;
          column_ptr = input;
          break;
        }
      }
    }
    if (column_ptr == nullptr) {
      continue;
    }
    NodeArg& v = *graph.GetNode(column_ptr->Index())->MutableInputDefs()[0];
    NodeArg& x = *mul.MutableInputDefs()[1 - column_input];

    // Exact shapes: X is [..., K, N] and v is [..., K], with the same K.
    const auto x_rank = RankOf(x);
    const auto v_rank = RankOf(v);
    const auto product_rank = RankOf(*mul.OutputDefs()[0]);
    if (!x_rank.has_value() || !v_rank.has_value() || !product_rank.has_value() || *x_rank < 2 || *v_rank < 1 ||
        *product_rank != std::max(*x_rank, *v_rank + 1) || !IsFloatTensor(x) || !IsFloatTensor(v) ||
        !SameDim(x.Shape()->dim(static_cast<int>(*x_rank - 2)), v.Shape()->dim(static_cast<int>(*v_rank - 1)))) {
      continue;
    }

    // The sum runs over the product's second-to-last axis only.
    const auto reduce_axis = SingleAxis(graph, reduce);
    if (!reduce_axis.has_value() || (*reduce_axis != -2 && *reduce_axis != *product_rank - 2)) {
      continue;
    }

    const std::string& provider = reduce.GetExecutionProviderType();
    NodeArg& row = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("matvec_row"), nullptr);
    Node& row_node = AddAxisNode(graph, "Unsqueeze", v, row, onnx_opset, provider);
    CopyInputEdge(graph, *column_ptr, 0, row_node, 0);

    NodeArg& matmul_out = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("matvec_product"), nullptr);
    Node& matmul = graph.AddNode(graph.GenerateNodeName("MatMul"), "MatMul", "MulReduceSumToMatMulFusion",
                                 {&row, &x}, {&matmul_out});
    matmul.SetExecutionProviderType(provider);
    graph.AddEdge(row_node.Index(), matmul.Index(), 0, 0);
    CopyInputEdge(graph, mul, 1 - column_input, matmul, 1);

    Node* last = &matmul;
    if (!keepdims) {
      NodeArg& squeezed = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("matvec_squeezed"), nullptr);
      last = &AddAxisNode(graph, "Squeeze", matmul_out, squeezed, onnx_opset, provider);
      graph.AddEdge(matmul.Index(), last->Index(), 0, 0);
    }

    // The last new node takes over the ReduceSum's output; the Mul and, if nothing else reads it, the column go.
    const NodeIndex column_index = column_ptr->Index();
    graph_utils::FinalizeNodeFusion(graph, *last, reduce);
    graph_utils::RemoveNodeOutputEdges(graph, mul);
    graph.RemoveNode(mul.Index());
    Node& column = *graph.GetNode(column_index);
    if (column.GetOutputEdgesCount() == 0 && !graph.NodeProducesGraphOutput(column)) {
      graph.RemoveNode(column_index);
    }

    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime
