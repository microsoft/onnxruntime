// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/matmul_nbits_qlora_fusion.h"

#include <cmath>

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

bool Private(const Graph& graph, const Node& node) {
  return !graph.NodeProducesGraphOutput(node) && optimizer_utils::CheckOutputEdges(graph, node, 1);
}

bool Shape(const NodeArg& arg, std::initializer_list<int64_t> expected, int empty_axis = -1) {
  const auto* shape = arg.Shape();
  if (!shape || shape->dim_size() != static_cast<int>(expected.size())) return false;
  int index = 0;
  for (auto value : expected) {
    const auto& dim = shape->dim(index);
    if (index++ == empty_axis && (!dim.has_dim_value() || dim.dim_value() == 0)) continue;
    if (!dim.has_dim_value() || dim.dim_value() != value) return false;
  }
  return true;
}

bool AdapterShape(const Graph& graph, const NodeArg& arg, std::initializer_list<int64_t> expected, int empty_axis) {
  // K/N attributes establish the logical rank even when an overridable default
  // is empty. Only the axis used by the QLoRA empty-adapter convention may vary.
  if (!Shape(arg, expected, empty_axis)) return false;
  const ONNX_NAMESPACE::TensorProto* init = nullptr;
  if (graph.GetInitializedTensor(arg.Name(), init)) {
    if (init->dims_size() != static_cast<int>(expected.size())) return false;
    int index = 0;
    for (auto value : expected) {
      const auto dim = init->dims(index);
      if (dim != value && !(index == empty_axis && dim == 0)) return false;
      ++index;
    }
  }
  return true;
}

bool Projection(const Node* node, int64_t bits) {
  using namespace matmul_nbits_fusion_utils;
  return node && graph_utils::IsSupportedOptypeVersionAndDomain(*node, "MatMulNBits", {1}, kMSDomain) &&
         node->GetExecutionProviderType() == kWebGpuExecutionProvider && GetIntAttr(*node, "bits", 4) == bits &&
         GetIntAttr(*node, "block_size", -1, true) == 32 && GetIntAttr(*node, "weight_prepacked", 0) == 0 &&
         !HasInput(*node, 3) && !HasInput(*node, 4) && !HasInput(*node, 5) &&
         (bits != 8 || GetIntAttr(*node, "accuracy_level", 0) == 0);
}

bool ZeroDefault(const Graph& graph, const NodeArg& q, const NodeArg& scales) {
  if (!graph.IsInputsIncludingInitializers(&q) || graph.IsOutput(&q) || graph.GetConsumerNodes(q.Name()).size() != 1)
    return false;
  const ONNX_NAMESPACE::TensorProto* qp = nullptr;
  const ONNX_NAMESPACE::TensorProto* sp = nullptr;
  if (!graph.GetInitializedTensor(q.Name(), qp) || !graph.GetInitializedTensor(scales.Name(), sp)) return false;
  Initializer qdata(graph, *qp, graph.ModelPath());
  Initializer sdata(graph, *sp, graph.ModelPath());
  const auto bytes = qdata.DataAsByteSpan();
  if (!bytes.empty() && sdata.DataAsByteSpan().empty()) return false;
  if (!std::all_of(bytes.begin(), bytes.end(), [](uint8_t value) { return value == 128; })) return false;
  if (sp->data_type() == ONNX_NAMESPACE::TensorProto::FLOAT) {
    const auto values = sdata.DataAsSpan<float>();
    return std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); });
  }
  const auto values = sdata.DataAsSpan<MLFloat16>();
  return std::all_of(values.begin(), values.end(), [](MLFloat16 value) { return std::isfinite(value.ToFloat()); });
}

void EmptyDefault(Graph& graph, NodeArg& arg, int axis) {
  const ONNX_NAMESPACE::TensorProto* original = nullptr;
  ORT_ENFORCE(graph.GetInitializedTensor(arg.Name(), original));
  ONNX_NAMESPACE::TensorProto empty;
  empty.set_name(arg.Name());
  empty.set_data_type(original->data_type());
  for (int i = 0; i < original->dims_size(); ++i) empty.add_dims(i == axis ? 0 : original->dims(i));
  auto shape = *arg.Shape();
  shape.mutable_dim(axis)->Clear();
  arg.SetShape(shape);
  graph.RemoveInitializedTensor(arg.Name());
  graph.AddInitializedTensor(empty);
}

}  // namespace

Status MatMulNBitsQLoRAFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                         const logging::Logger& logger) const {
  using namespace matmul_nbits_fusion_utils;
  GraphViewer viewer(graph);
  for (auto index : viewer.GetNodesInTopologicalOrder()) {
    auto* add = graph.GetNode(index);
    if (!add) continue;
    ORT_RETURN_IF_ERROR(Recurse(*add, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*add, "Add", {7, 13, 14}) ||
        add->GetExecutionProviderType() != kWebGpuExecutionProvider)
      continue;
    auto* base = Producer(graph, *add, 0);
    auto* up = Producer(graph, *add, 1);
    if (!Projection(base, 4)) std::swap(base, up);
    if (!Projection(base, 4) || !Projection(up, 8)) continue;
    auto* down = Producer(graph, *up, 0);
    if (!Projection(down, 8) || down->InputDefs()[0] != base->InputDefs()[0] || !Private(graph, *base) ||
        !Private(graph, *down) || !Private(graph, *up))
      continue;
    const int64_t k = GetIntAttr(*base, "K", -1, true);
    const int64_t n = GetIntAttr(*base, "N", -1, true);
    if (k <= 0 || k % 32 != 0 || n <= 0 || GetIntAttr(*down, "K", -1, true) != k ||
        GetIntAttr(*down, "N", -1, true) != 32 || GetIntAttr(*up, "K", -1, true) != 32 ||
        GetIntAttr(*up, "N", -1, true) != n)
      continue;
    InlinedVector<NodeArg*> inputs{
        const_cast<NodeArg*>(base->InputDefs()[0]), const_cast<NodeArg*>(base->InputDefs()[1]),
        const_cast<NodeArg*>(base->InputDefs()[2]), const_cast<NodeArg*>(down->InputDefs()[1]),
        const_cast<NodeArg*>(down->InputDefs()[2]), const_cast<NodeArg*>(up->InputDefs()[1]),
        const_cast<NodeArg*>(up->InputDefs()[2])};
    const auto* x_type = inputs[0]->TypeAsProto();
    if (!x_type || !x_type->has_tensor_type()) continue;
    const auto float_type = x_type->tensor_type().elem_type();
    if (float_type != ONNX_NAMESPACE::TensorProto::FLOAT16 && float_type != ONNX_NAMESPACE::TensorProto::FLOAT)
      continue;
    bool types_ok = true;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto* type = inputs[i]->TypeAsProto();
      const auto expected =
          (i == 1 || i == 3 || i == 5) ? ONNX_NAMESPACE::TensorProto::UINT8 : float_type;
      types_ok &= type && type->has_tensor_type() && type->tensor_type().elem_type() == expected;
    }
    if (!types_ok || !inputs[0]->Shape() || inputs[0]->Shape()->dim_size() < 2 || !Shape(*inputs[1], {n, k / 32, 16}) ||
        !AdapterShape(graph, *inputs[3], {32, k / 32, 32}, 0) ||
        !AdapterShape(graph, *inputs[4], {32, k / 32}, 0) ||
        !AdapterShape(graph, *inputs[5], {n, 1, 32}, 1) || !AdapterShape(graph, *inputs[6], {n, 1}, 1))
      continue;
    const auto* s = inputs[2]->Shape();
    const bool flat_scales = s && s->dim_size() == 1 && s->dim(0).has_dim_value() &&
                             s->dim(0).dim_value() / n == k / 32 && s->dim(0).dim_value() % n == 0;
    if (!flat_scales && !Shape(*inputs[2], {n, k / 32})) continue;
    const bool empty_defaults =
        ZeroDefault(graph, *inputs[3], *inputs[4]) && ZeroDefault(graph, *inputs[5], *inputs[6]);
    auto attrs = base->GetAttributes();
    attrs.erase("weight_prepacked");
    utils::SetNodeAttribute(utils::MakeAttribute("lora_bits", int64_t{8}), attrs);
    utils::SetNodeAttribute(utils::MakeAttribute("lora_block_size", int64_t{32}), attrs);
    auto& fused = graph.AddNode(graph.GenerateNodeName("MatMulNBitsQLoRA"), "MatMulNBitsQLoRA",
                                "fused unmerged rank-32 INT8 LoRA projection", inputs, add->MutableOutputDefs(), &attrs,
                                kMSDomain);
    fused.SetExecutionProviderType(kWebGpuExecutionProvider);
    const Node* sources[] = {base, base, base, down, down, up, up};
    const int slots[] = {0, 1, 2, 1, 2, 1, 2};
    for (int i = 0; i < 7; ++i) {
      if (const auto* edge = graph_utils::GetInputEdge(*sources[i], slots[i])) {
        graph.AddEdge(edge->GetNode().Index(), fused.Index(), edge->GetSrcArgIndex(), i);
      }
    }
    const auto outputs = graph_utils::GraphEdge::GetNodeOutputEdges(*add);
    for (const Node* node : {base, down, up, static_cast<const Node*>(add)}) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(node->Index()));
    }
    const NodeIndex removed[] = {base->Index(), down->Index(), up->Index(), add->Index()};
    for (auto node : removed) graph.RemoveNode(node);
    for (const auto& edge : outputs) graph.AddEdge(fused.Index(), edge.dst_node, 0, edge.dst_arg_index);
    if (empty_defaults) {
      // Keep scale defaults: overriding only a packed weight must still use its original scale.
      EmptyDefault(graph, *inputs[3], 0);
      EmptyDefault(graph, *inputs[5], 1);
    }
    modified = true;
    LOGS(logger, VERBOSE) << "MatMulNBitsQLoRAFusion: " << fused.Name() << " empty defaults=" << empty_defaults;
  }
  return Status::OK();
}

}  // namespace onnxruntime
