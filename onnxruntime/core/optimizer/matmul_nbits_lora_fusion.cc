// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_lora_fusion.h"

#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_nbits_fusion_utils.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace {

const Node* Producer(const Graph& graph, const Node& node, int input) {
  const auto* edge = graph_utils::GetInputEdge(node, input);
  return edge ? graph.GetNode(edge->GetNode().Index()) : nullptr;
}

bool SingleConsumer(const Graph& graph, const Node& node) {
  return !graph.NodeProducesGraphOutput(node) && optimizer_utils::CheckOutputEdges(graph, node, 1);
}

bool IsMatMul(const Node* node) {
  return node && graph_utils::IsSupportedOptypeVersionAndDomain(*node, "MatMul", {9, 13}) &&
         node->GetExecutionProviderType() == kWebGpuExecutionProvider;
}

bool MatrixShape(const NodeArg& arg, int64_t rows, int64_t cols) {
  const auto* s = arg.Shape();
  return s && s->dim_size() == 2 && s->dim(0).has_dim_value() && s->dim(1).has_dim_value() &&
         s->dim(0).dim_value() == rows && s->dim(1).dim_value() == cols;
}

const Node* AdapterTranspose(const Graph& graph, const Node& matmul) {
  const auto* node = Producer(graph, matmul, 1);
  if (!node || !graph_utils::IsSupportedOptypeVersionAndDomain(*node, "Transpose", {1, 13, 21, 23, 24}) ||
      node->GetExecutionProviderType() != kWebGpuExecutionProvider || !SingleConsumer(graph, *node)) return nullptr;
  const auto* shape = node->InputDefs()[0]->Shape();
  if (!shape || shape->dim_size() != 2) return nullptr;
  auto perm = node->GetAttributes().find("perm");
  if (perm != node->GetAttributes().end() &&
      (perm->second.ints_size() != 2 || perm->second.ints(0) != 1 || perm->second.ints(1) != 0)) return nullptr;
  return node;
}

bool ZeroDefault(const Graph& graph, const NodeArg& arg) {
  if (!graph.IsInputsIncludingInitializers(&arg) || graph.IsOutput(&arg)) {
    return false;
  }
  const ONNX_NAMESPACE::TensorProto* proto = nullptr;
  if (!graph.GetInitializedTensor(arg.Name(), proto)) {
    return false;
  }
  // Only rewrite a private adapter parameter: another consumer may depend on
  // its original dimensions or on IEEE zero-times-infinity semantics.
  const auto consumers = graph.GetConsumerNodes(arg.Name());
  if (consumers.size() != 1) {
    return false;
  }
  Initializer data(graph, *proto, graph.ModelPath());
  auto bytes = data.DataAsByteSpan();
  return std::all_of(bytes.begin(), bytes.end(), [](uint8_t v) { return v == 0; });
}

void EmptyDefault(Graph& graph, NodeArg& arg, int rank_axis) {
  const ONNX_NAMESPACE::TensorProto* original = nullptr;
  ORT_ENFORCE(graph.GetInitializedTensor(arg.Name(), original));
  ONNX_NAMESPACE::TensorProto empty;
  empty.set_name(original->name());
  empty.set_data_type(original->data_type());
  for (int i = 0; i < original->dims_size(); ++i) {
    empty.add_dims(i == rank_axis ? 0 : original->dims(i));
  }
  auto shape = *arg.Shape();
  shape.mutable_dim(rank_axis)->Clear();
  arg.SetShape(shape);
  graph.RemoveInitializedTensor(arg.Name());
  graph.AddInitializedTensor(empty);
}

}  // namespace

Status MatMulNBitsLoRAFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                        const logging::Logger& logger) const {
  GraphViewer viewer(graph);
  for (auto index : viewer.GetNodesInTopologicalOrder()) {
    auto* add = graph.GetNode(index);
    if (!add) continue;
    ORT_RETURN_IF_ERROR(Recurse(*add, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*add, "Add", {7, 13, 14}) ||
        add->GetExecutionProviderType() != kWebGpuExecutionProvider) continue;
    const Node* base = Producer(graph, *add, 0);
    const Node* up = Producer(graph, *add, 1);
    if (base && base->OpType() != "MatMulNBits") std::swap(base, up);
    if (!base || !graph_utils::IsSupportedOptypeVersionAndDomain(*base, "MatMulNBits", {1}, kMSDomain) ||
        base->GetExecutionProviderType() != kWebGpuExecutionProvider || !IsMatMul(up)) continue;
    const Node* down = Producer(graph, *up, 0);
    if (!IsMatMul(down) || down->InputDefs()[0] != base->InputDefs()[0] ||
        !SingleConsumer(graph, *base) || !SingleConsumer(graph, *up) || !SingleConsumer(graph, *down)) continue;

    using namespace matmul_nbits_fusion_utils;
    const int64_t k = GetIntAttr(*base, "K", -1, true);
    const int64_t n = GetIntAttr(*base, "N", -1, true);
    const auto* shape = down->InputDefs()[1]->Shape();
    if (!shape || shape->dim_size() != 2 || !shape->dim(1).has_dim_value()) continue;
    const int64_t rank = shape->dim(1).dim_value();
    if (k <= 0 || n <= 0 || rank != 32 || k % 32 != 0 || n % 4 != 0 ||
        GetIntAttr(*base, "bits", 4) != 4 || GetIntAttr(*base, "block_size", -1, true) != 32 ||
        GetIntAttr(*base, "weight_prepacked", 0) != 0 || HasInput(*base, 3) || HasInput(*base, 4) ||
        HasInput(*base, 5) || !MatrixShape(*down->InputDefs()[1], k, rank) ||
        !MatrixShape(*up->InputDefs()[1], rank, n) ||
        !base->InputDefs()[0]->Shape() || base->InputDefs()[0]->Shape()->dim_size() < 2) continue;

    const auto* x_type = base->InputDefs()[0]->TypeAsProto();
    if (!x_type || !x_type->has_tensor_type()) continue;
    const auto float_type = x_type->tensor_type().elem_type();
    if (float_type != ONNX_NAMESPACE::TensorProto::FLOAT16 && float_type != ONNX_NAMESPACE::TensorProto::FLOAT)
      continue;
    bool types_ok = true;
    for (const auto* arg : {base->InputDefs()[2], down->InputDefs()[1], up->InputDefs()[1]}) {
      const auto* type = arg->TypeAsProto();
      types_ok &= type && type->has_tensor_type() && type->tensor_type().elem_type() == float_type;
    }
    if (!types_ok) continue;

    // MatMulNBits also accepts legacy flattened weights. Leave those graphs
    // alone until the fused kernel supports that representation.
    const auto* weight_shape = base->InputDefs()[1]->Shape();
    if (!weight_shape || weight_shape->dim_size() != 3 ||
        !weight_shape->dim(0).has_dim_value() || weight_shape->dim(0).dim_value() != n ||
        !weight_shape->dim(1).has_dim_value() || weight_shape->dim(1).dim_value() != k / 32 ||
        !weight_shape->dim(2).has_dim_value() || weight_shape->dim(2).dim_value() != 16) continue;
    const auto* scale_shape = base->InputDefs()[2]->Shape();
    const bool flat_scales = scale_shape && scale_shape->dim_size() == 1 &&
                             scale_shape->dim(0).has_dim_value() &&
                             scale_shape->dim(0).dim_value() / n == k / 32 &&
                             scale_shape->dim(0).dim_value() % n == 0;
    if (!flat_scales && !MatrixShape(*base->InputDefs()[2], n, k / 32)) continue;

    auto* a = const_cast<NodeArg*>(down->InputDefs()[1]);
    auto* b = const_cast<NodeArg*>(up->InputDefs()[1]);
    const auto* transpose_a = AdapterTranspose(graph, *down);
    const auto* transpose_b = AdapterTranspose(graph, *up);
    const bool transposed = transpose_a && transpose_b &&
                            MatrixShape(*transpose_a->InputDefs()[0], rank, k) &&
                            MatrixShape(*transpose_b->InputDefs()[0], n, rank);
    if (transposed) {
      a = const_cast<NodeArg*>(transpose_a->InputDefs()[0]);
      b = const_cast<NodeArg*>(transpose_b->InputDefs()[0]);
    }
    const bool empty_defaults = ZeroDefault(graph, *a) && ZeroDefault(graph, *b);
    InlinedVector<NodeArg*> inputs{const_cast<NodeArg*>(base->InputDefs()[0]),
                                   const_cast<NodeArg*>(base->InputDefs()[1]),
                                   const_cast<NodeArg*>(base->InputDefs()[2]), a, b};
    auto attrs = base->GetAttributes();
    attrs.erase("weight_prepacked");
    utils::SetNodeAttribute(utils::MakeAttribute("adapter_transposed", int64_t{transposed}), attrs);
    Node& fused = graph.AddNode(graph.GenerateNodeName("MatMulNBitsLoRA"), "MatMulNBitsLoRA",
                                "fused unmerged rank-32 LoRA projection", inputs,
                                add->MutableOutputDefs(), &attrs, kMSDomain);
    fused.SetExecutionProviderType(kWebGpuExecutionProvider);
    // All input producer edges and final consumers must be preserved before
    // removal, including weights supplied by upstream nodes.
    const Node* sources[] = {base, base, base, transposed ? transpose_a : down, transposed ? transpose_b : up};
    const int slots[] = {0, 1, 2, transposed ? 0 : 1, transposed ? 0 : 1};
    for (int i = 0; i < 5; ++i) {
      if (const auto* edge = graph_utils::GetInputEdge(*sources[i], slots[i])) {
        graph.AddEdge(edge->GetNode().Index(), fused.Index(), edge->GetSrcArgIndex(), i);
      }
    }
    const auto outputs = graph_utils::GraphEdge::GetNodeOutputEdges(*add);
    for (const Node* node : {base, down, up, static_cast<const Node*>(add)}) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(node->Index()));
    }
    const NodeIndex indices[] = {base->Index(), down->Index(), up->Index(), add->Index()};
    if (transposed) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(transpose_a->Index()));
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(transpose_b->Index()));
    }
    for (auto idx : indices) graph.RemoveNode(idx);
    if (transposed) {
      graph.RemoveNode(transpose_a->Index());
      graph.RemoveNode(transpose_b->Index());
    }
    for (const auto& edge : outputs) {
      graph.AddEdge(fused.Index(), edge.dst_node, 0, edge.dst_arg_index);
    }
    if (empty_defaults) {
      EmptyDefault(graph, *a, transposed ? 0 : 1);
      EmptyDefault(graph, *b, transposed ? 1 : 0);
    }
    LOGS(logger, VERBOSE) << "MatMulNBitsLoRAFusion: " << fused.Name() << " empty defaults=" << empty_defaults;
    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
