// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_mask_reformatting_to_graph_capture.h"

#include <string>
#include <vector>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {

namespace {

const Node* GetInputNode(const Graph& graph, const Node& node, int input_index) {
  const auto* edge = graph_utils::GetInputEdge(node, input_index);
  return edge == nullptr ? nullptr : graph.GetNode(edge->GetNode().Index());
}

bool IsCastToInt32(const Node& node) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Cast", {9, 13, 19, 21, 23, 24})) {
    return false;
  }
  const auto* to_attr = graph_utils::GetNodeAttribute(node, "to");
  return to_attr != nullptr && to_attr->i() == ONNX_NAMESPACE::TensorProto_DataType_INT32;
}

bool IsKeepDimsZero(const Node& node) {
  const auto* attr = graph_utils::GetNodeAttribute(node, "keepdims");
  return attr != nullptr && attr->i() == 0;
}

bool IsInitializerWithInt64Value(const Graph& graph, const NodeArg* arg, int64_t expected) {
  if (arg == nullptr) return false;
  const ONNX_NAMESPACE::TensorProto* tp = nullptr;
  if (!graph.GetInitializedTensor(arg->Name(), tp) || tp == nullptr) return false;
  if (tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) return false;
  Initializer init(*tp);
  return init.size() == 1 && init.DataAsSpan<int64_t>()[0] == expected;
}

bool IsConstantInt(const Graph& graph, const Node& consumer, int input_index, int64_t expected) {
  if (input_index >= static_cast<int>(consumer.InputDefs().size())) return false;
  const NodeArg* arg = consumer.InputDefs()[input_index];
  if (arg == nullptr) return false;
  // Check graph initializer first.
  if (IsInitializerWithInt64Value(graph, arg, expected)) return true;
  // Check int32 initializer.
  const ONNX_NAMESPACE::TensorProto* tp = nullptr;
  if (graph.GetInitializedTensor(arg->Name(), tp) && tp != nullptr &&
      tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    Initializer init(*tp);
    if (init.size() == 1 && static_cast<int64_t>(init.DataAsSpan<int32_t>()[0]) == expected) return true;
  }
  // Check Constant node.
  const Node* producer = GetInputNode(graph, consumer, input_index);
  if (producer == nullptr || producer->OpType() != "Constant") return false;
  const auto* vi = graph_utils::GetNodeAttribute(*producer, "value_int");
  if (vi != nullptr && vi->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
    return vi->i() == expected;
  }
  return false;
}

NodeArg& AddVec1Initializer(Graph& graph, const std::string& name_hint,
                             int data_type, const void* data, size_t byte_len) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(graph.GenerateNodeArgName(name_hint));
  tp.set_data_type(data_type);
  tp.add_dims(1);
  utils::SetRawDataInTensorProto(tp, data, byte_len);
  return graph_utils::AddInitializerWithOrtValue(graph, tp);
}

void RemoveChain(Graph& graph, std::initializer_list<const Node*> nodes) {
  std::vector<NodeIndex> upstream;
  for (const Node* n : nodes) {
    if (n == nullptr) continue;
    for (auto it = n->InputEdgesBegin(); it != n->InputEdgesEnd(); ++it) {
      upstream.push_back(it->GetNode().Index());
    }
  }
  for (const Node* n : nodes) {
    if (n == nullptr) continue;
    graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(n->Index()));
    graph.RemoveNode(n->Index());
  }
  // Reap orphaned upstream nodes (Constant producers, etc.).
  for (NodeIndex idx : upstream) {
    const Node* u = graph.GetNode(idx);
    if (u != nullptr && u->GetOutputEdgesCount() == 0 && !graph.NodeProducesGraphOutput(*u)) {
      graph_utils::RemoveNodeOutputEdges(graph, *graph.GetNode(idx));
      graph.RemoveNode(idx);
    }
  }
}

}  // namespace

Status GqaMaskReformattingToGraphCapture::ApplyImpl(Graph& graph,
                                                    bool& modified,
                                                    int /*graph_level*/,
                                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& topo = graph_viewer.GetNodesInTopologicalOrder();

  for (NodeIndex idx : topo) {
    Node* gqa = graph.GetNode(idx);
    if (gqa == nullptr) continue;
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*gqa, "GroupQueryAttention", {1}, kMSDomain)) continue;
    if (!gqa->GetExecutionProviderType().empty() &&
        gqa->GetExecutionProviderType() != kWebGpuExecutionProvider) {
      continue;
    }
    if (gqa->InputDefs().size() < 7) continue;

    const NodeArg* seqlens_k_arg = gqa->InputDefs()[5];
    const NodeArg* total_seq_len_arg = gqa->InputDefs()[6];
    if (seqlens_k_arg == nullptr || total_seq_len_arg == nullptr) continue;

    // --- Match seqlens_k chain: Cast(INT32) <- Sub(_, 1) <- ReduceSum(mask, axes=[1], keepdims=0) ---
    const Node* sk_cast = graph.GetProducerNode(seqlens_k_arg->Name());
    if (sk_cast == nullptr || !IsCastToInt32(*sk_cast)) continue;
    const Node* sk_sub = GetInputNode(graph, *sk_cast, 0);
    if (sk_sub == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*sk_sub, "Sub", {7, 13, 14})) continue;
    if (!IsConstantInt(graph, *sk_sub, 1, 1)) continue;
    const Node* sk_rs = GetInputNode(graph, *sk_sub, 0);
    if (sk_rs == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*sk_rs, "ReduceSum", {13, 18, 20})) continue;
    if (!IsKeepDimsZero(*sk_rs)) continue;
    if (sk_rs->InputDefs().size() < 2 || !IsInitializerWithInt64Value(graph, sk_rs->InputDefs()[1], 1)) continue;

    // --- Match total_seq_len chain: Cast(INT32) <- Gather(_, idx=1, axis=0) <- Shape(mask) ---
    const Node* tsl_cast = graph.GetProducerNode(total_seq_len_arg->Name());
    if (tsl_cast == nullptr || !IsCastToInt32(*tsl_cast)) continue;
    const Node* tsl_gather = GetInputNode(graph, *tsl_cast, 0);
    if (tsl_gather == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*tsl_gather, "Gather", {1, 11, 13})) continue;
    const auto* axis_attr = graph_utils::GetNodeAttribute(*tsl_gather, "axis");
    if (axis_attr != nullptr && axis_attr->i() != 0) continue;
    if (!IsConstantInt(graph, *tsl_gather, 1, 1)) continue;
    const Node* tsl_shape = GetInputNode(graph, *tsl_gather, 0);
    if (tsl_shape == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*tsl_shape, "Shape", {1, 13, 15, 19, 21, 23, 24})) continue;

    // Both chains must consume the same attention_mask.
    const NodeArg* sk_mask = sk_rs->InputDefs()[0];
    const NodeArg* tsl_mask = tsl_shape->InputDefs()[0];
    if (sk_mask != tsl_mask) continue;
    NodeArg* mask_arg = graph.GetNodeArg(sk_mask->Name());
    if (mask_arg == nullptr) continue;

    LOGS(logger, INFO) << "GqaMaskReformattingToGraphCapture: rewriting CPU-bound mask subgraph"
                       << " (mask='" << mask_arg->Name() << "')";

    // --- Build replacement subgraph ---
    //   Cast(INT32) -> ReduceSum(axes=[1]) -+-> Sub(_, 1) -> seqlens_k
    //                                       +-> ReduceMax   -> total_seq_len

    NodeArg& cast_out = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName("gqa_gc_cast"), nullptr);
    {
      NodeAttributes attrs;
      utils::SetNodeAttribute(
          utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32)), attrs);
      graph.AddNode(graph.GenerateNodeName("gqa_gc_cast"), "Cast", "",
                    {mask_arg}, {&cast_out}, &attrs, kOnnxDomain)
          .SetExecutionProviderType(kWebGpuExecutionProvider);
    }

    int64_t one_i64 = 1;
    NodeArg& rs_axes = AddVec1Initializer(graph, "gqa_gc_rs_axes",
                                          ONNX_NAMESPACE::TensorProto_DataType_INT64, &one_i64, sizeof(int64_t));
    NodeArg& rs_out = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName("gqa_gc_rs"), nullptr);
    {
      NodeAttributes attrs;
      utils::SetNodeAttribute(utils::MakeAttribute("keepdims", static_cast<int64_t>(0)), attrs);
      graph.AddNode(graph.GenerateNodeName("gqa_gc_rs"), "ReduceSum", "",
                    {&cast_out, &rs_axes}, {&rs_out}, &attrs, kOnnxDomain)
          .SetExecutionProviderType(kWebGpuExecutionProvider);
    }

    // Capture downstream edges, remove old chains, then add replacement nodes
    // producing the same output NodeArgs.
    auto sk_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*sk_cast);
    auto tsl_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*tsl_cast);

    NodeArg* sk_out = graph.GetNodeArg(sk_cast->OutputDefs()[0]->Name());
    NodeArg* tsl_out = graph.GetNodeArg(tsl_cast->OutputDefs()[0]->Name());

    RemoveChain(graph, {sk_cast, sk_sub, sk_rs, tsl_cast, tsl_gather, tsl_shape});

    // Sub: seqlens_k = ReduceSum - 1
    int32_t one_i32 = 1;
    NodeArg& sk_one = AddVec1Initializer(graph, "gqa_gc_sk_one",
                                         ONNX_NAMESPACE::TensorProto_DataType_INT32, &one_i32, sizeof(int32_t));
    {
      Node& sub = graph.AddNode(graph.GenerateNodeName("gqa_gc_sub"), "Sub", "",
                                {&rs_out, &sk_one}, {sk_out}, nullptr, kOnnxDomain);
      sub.SetExecutionProviderType(kWebGpuExecutionProvider);
      for (const auto& e : sk_edges) {
        graph.AddEdge(sub.Index(), e.dst_node, 0, e.dst_arg_index);
      }
    }

    // ReduceMax: total_seq_len = max(ReduceSum)
    int64_t zero_i64 = 0;
    NodeArg& rmax_axes = AddVec1Initializer(graph, "gqa_gc_rmax_axes",
                                            ONNX_NAMESPACE::TensorProto_DataType_INT64, &zero_i64, sizeof(int64_t));
    {
      NodeAttributes attrs;
      utils::SetNodeAttribute(utils::MakeAttribute("keepdims", static_cast<int64_t>(0)), attrs);
      Node& rmax = graph.AddNode(graph.GenerateNodeName("gqa_gc_rmax"), "ReduceMax", "",
                                 {&rs_out, &rmax_axes}, {tsl_out}, &attrs, kOnnxDomain);
      rmax.SetExecutionProviderType(kWebGpuExecutionProvider);
      for (const auto& e : tsl_edges) {
        graph.AddEdge(rmax.Index(), e.dst_node, 0, e.dst_arg_index);
      }
    }

    modified = true;
    break;
  }

  return Status::OK();
}

}  // namespace onnxruntime
