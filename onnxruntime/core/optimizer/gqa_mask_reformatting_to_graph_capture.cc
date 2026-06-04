// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/gqa_mask_reformatting_to_graph_capture.h"

#include <string>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/node_attr_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

const Node* GetInputNode(const Graph& graph, const Node& node, int input_index) {
  const auto* edge = graph_utils::GetInputEdge(node, input_index);
  return edge == nullptr ? nullptr : graph.GetNode(edge->GetNode().Index());
}

// Match a node that produces a scalar/1-element int constant with value `expected`.
// Accepts either a graph initializer (referenced via NodeArg) or a `Constant` node
// with `value`, `value_int`, or `value_ints` attribute.
bool IsConstantIntValue(const Graph& graph, const NodeArg& input_arg, int64_t expected) {
  return optimizer_utils::IsInitializerWithExpectedValue(graph, input_arg, expected, true);
}

bool IsConstantNodeWithIntValue(const Node& node, int64_t expected) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Constant", {9, 11, 12, 13, 19, 20, 21, 23})) {
    return false;
  }
  const auto* value_int = graph_utils::GetNodeAttribute(node, "value_int");
  if (value_int != nullptr && value_int->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
    return value_int->i() == expected;
  }
  const auto* value = graph_utils::GetNodeAttribute(node, "value");
  if (value != nullptr && value->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
    const auto& tp = value->t();
    if (tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
        tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      return false;
    }
    Initializer init(tp);
    if (init.size() != 1) {
      return false;
    }
    if (tp.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64) {
      return init.DataAsSpan<int64_t>()[0] == expected;
    }
    return static_cast<int64_t>(init.DataAsSpan<int32_t>()[0]) == expected;
  }
  return false;
}

// Resolves `arg` (an input NodeArg on `consumer`) to an int constant value if
// possible. Returns true and writes the value to `out` if the producer is a
// graph initializer or a `Constant` node with the requisite attribute.
bool TryResolveConstantInt(const Graph& graph, const Node& consumer, int input_index, int64_t& out) {
  const NodeArg* arg = consumer.InputDefs()[input_index];
  if (arg == nullptr) {
    return false;
  }
  if (optimizer_utils::IsInitializerWithExpectedValue(graph, *arg, static_cast<int64_t>(0), true) ||
      optimizer_utils::IsInitializerWithExpectedValue(graph, *arg, static_cast<int64_t>(1), true)) {
    // Cheap probe — fall through to the actual unpack below if it matched.
  }
  const ONNX_NAMESPACE::TensorProto* tp = nullptr;
  if (graph.GetInitializedTensor(arg->Name(), tp) && tp != nullptr) {
    if (tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
        tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      return false;
    }
    Initializer init(*tp);
    if (init.size() != 1) {
      return false;
    }
    out = tp->data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64
              ? init.DataAsSpan<int64_t>()[0]
              : static_cast<int64_t>(init.DataAsSpan<int32_t>()[0]);
    return true;
  }
  const Node* producer = GetInputNode(graph, consumer, input_index);
  if (producer == nullptr || producer->OpType() != "Constant") {
    return false;
  }
  const auto* value_int = graph_utils::GetNodeAttribute(*producer, "value_int");
  if (value_int != nullptr && value_int->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_INT) {
    out = value_int->i();
    return true;
  }
  const auto* value = graph_utils::GetNodeAttribute(*producer, "value");
  if (value != nullptr && value->type() == ONNX_NAMESPACE::AttributeProto_AttributeType_TENSOR) {
    const auto& v_tp = value->t();
    if (v_tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
        v_tp.data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
      return false;
    }
    Initializer init(v_tp);
    if (init.size() != 1) {
      return false;
    }
    out = v_tp.data_type() == ONNX_NAMESPACE::TensorProto_DataType_INT64
              ? init.DataAsSpan<int64_t>()[0]
              : static_cast<int64_t>(init.DataAsSpan<int32_t>()[0]);
    return true;
  }
  return false;
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
  // ONNX default is keepdims=1; we require explicit 0.
  return attr != nullptr && attr->i() == 0;
}

// Match: input_arg is produced by `Cast(to=INT32)(Sub(ReduceSum(mask, axes=[1], keepdims=0), 1))`.
// Returns the three intermediate nodes (cast, sub, reduce_sum) and the mask NodeArg.
struct SeqlensKChain {
  const Node* cast;
  const Node* sub;
  const Node* reduce_sum;
  const NodeArg* mask;
};

bool MatchSeqlensKChain(const Graph& graph, const NodeArg& seqlens_k, SeqlensKChain& out) {
  const Node* cast = graph.GetProducerNode(seqlens_k.Name());
  if (cast == nullptr || !IsCastToInt32(*cast)) {
    return false;
  }
  const Node* sub = GetInputNode(graph, *cast, 0);
  if (sub == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*sub, "Sub", {7, 13, 14})) {
    return false;
  }
  int64_t one_value = 0;
  if (!TryResolveConstantInt(graph, *sub, 1, one_value) || one_value != 1) {
    return false;
  }
  const Node* reduce_sum = GetInputNode(graph, *sub, 0);
  if (reduce_sum == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*reduce_sum, "ReduceSum", {13, 18, 20})) {
    return false;
  }
  if (!IsKeepDimsZero(*reduce_sum)) {
    return false;
  }
  // axes input must be [1]
  int64_t axes_value = 0;
  if (reduce_sum->InputDefs().size() < 2) {
    return false;
  }
  const NodeArg* axes_arg = reduce_sum->InputDefs()[1];
  const ONNX_NAMESPACE::TensorProto* axes_tp = nullptr;
  if (!graph.GetInitializedTensor(axes_arg->Name(), axes_tp) || axes_tp == nullptr) {
    return false;
  }
  if (axes_tp->data_type() != ONNX_NAMESPACE::TensorProto_DataType_INT64) {
    return false;
  }
  Initializer axes_init(*axes_tp);
  if (axes_init.size() != 1) {
    return false;
  }
  axes_value = axes_init.DataAsSpan<int64_t>()[0];
  if (axes_value != 1) {
    return false;
  }
  out.cast = cast;
  out.sub = sub;
  out.reduce_sum = reduce_sum;
  out.mask = reduce_sum->InputDefs()[0];
  return true;
}

struct TotalSeqLenChain {
  const Node* cast;
  const Node* gather;
  const Node* shape;
  const NodeArg* mask;
};

bool MatchTotalSeqLenChain(const Graph& graph, const NodeArg& total_seq_len, TotalSeqLenChain& out) {
  const Node* cast = graph.GetProducerNode(total_seq_len.Name());
  if (cast == nullptr || !IsCastToInt32(*cast)) {
    return false;
  }
  const Node* gather = GetInputNode(graph, *cast, 0);
  if (gather == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*gather, "Gather", {1, 11, 13})) {
    return false;
  }
  const auto* axis_attr = graph_utils::GetNodeAttribute(*gather, "axis");
  const int64_t axis = axis_attr != nullptr ? axis_attr->i() : 0;
  if (axis != 0) {
    return false;
  }
  int64_t idx_value = 0;
  if (!TryResolveConstantInt(graph, *gather, 1, idx_value) || idx_value != 1) {
    return false;
  }
  const Node* shape = GetInputNode(graph, *gather, 0);
  if (shape == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*shape, "Shape", {1, 13, 15, 19, 21, 23, 24})) {
    return false;
  }
  out.cast = cast;
  out.gather = gather;
  out.shape = shape;
  out.mask = shape->InputDefs()[0];
  return true;
}

NodeArg& AddInt64ScalarInitializer(Graph& graph, const std::string& name_hint, int64_t value) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(graph.GenerateNodeArgName(name_hint));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  utils::SetRawDataInTensorProto(tp, &value, sizeof(int64_t));
  return graph_utils::AddInitializerWithOrtValue(graph, tp);
}

NodeArg& AddInt64Vec1Initializer(Graph& graph, const std::string& name_hint, int64_t value) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(graph.GenerateNodeArgName(name_hint));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  tp.add_dims(1);
  utils::SetRawDataInTensorProto(tp, &value, sizeof(int64_t));
  return graph_utils::AddInitializerWithOrtValue(graph, tp);
}

NodeArg& AddInt32Vec1Initializer(Graph& graph, const std::string& name_hint, int32_t value) {
  ONNX_NAMESPACE::TensorProto tp;
  tp.set_name(graph.GenerateNodeArgName(name_hint));
  tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT32);
  tp.add_dims(1);
  utils::SetRawDataInTensorProto(tp, &value, sizeof(int32_t));
  return graph_utils::AddInitializerWithOrtValue(graph, tp);
}

void TryRemoveIfOrphan(Graph& graph, const Node* node) {
  if (node == nullptr) {
    return;
  }
  if (graph.NodeProducesGraphOutput(*node)) {
    return;
  }
  if (node->GetOutputEdgesCount() != 0) {
    return;
  }
  // Recurse upstream after removal so orphan Constant / initializer-producing
  // nodes are also reaped.
  std::vector<NodeIndex> upstream;
  upstream.reserve(node->GetInputEdgesCount());
  for (auto it = node->InputEdgesBegin(); it != node->InputEdgesEnd(); ++it) {
    upstream.push_back(it->GetNode().Index());
  }
  graph_utils::RemoveNodeOutputEdges(graph, *const_cast<Node*>(node));
  graph.RemoveNode(node->Index());
  for (NodeIndex idx : upstream) {
    TryRemoveIfOrphan(graph, graph.GetNode(idx));
  }
}

}  // namespace

Status GqaMaskReformattingToGraphCapture::ApplyImpl(Graph& graph,
                                                    bool& modified,
                                                    int /*graph_level*/,
                                                    const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& topo = graph_viewer.GetNodesInTopologicalOrder();

  // Find a GQA node whose seqlens_k & total_seq_len both match the standard
  // CPU-bound subgraph. Because all GQAs in a model typically share the same
  // producer chain (gemma4 has 35 GQAs sharing one chain), the first match is
  // sufficient — rewriting it rewires every consumer at once.
  for (NodeIndex idx : topo) {
    Node* gqa = graph.GetNode(idx);
    if (gqa == nullptr) {
      continue;
    }
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*gqa, "GroupQueryAttention", {1}, kMSDomain)) {
      continue;
    }
    if (!gqa->GetExecutionProviderType().empty() &&
        gqa->GetExecutionProviderType() != kWebGpuExecutionProvider) {
      continue;
    }
    if (gqa->InputDefs().size() < 7) {
      continue;
    }

    const NodeArg* seqlens_k_arg = gqa->InputDefs()[5];
    const NodeArg* total_seq_len_arg = gqa->InputDefs()[6];
    if (seqlens_k_arg == nullptr || total_seq_len_arg == nullptr) {
      continue;
    }

    SeqlensKChain sk_chain{};
    TotalSeqLenChain tsl_chain{};
    if (!MatchSeqlensKChain(graph, *seqlens_k_arg, sk_chain)) {
      continue;
    }
    if (!MatchTotalSeqLenChain(graph, *total_seq_len_arg, tsl_chain)) {
      continue;
    }
    if (sk_chain.mask != tsl_chain.mask) {
      continue;
    }

    NodeArg* mask_arg = graph.GetNodeArg(sk_chain.mask->Name());
    if (mask_arg == nullptr) {
      continue;
    }

    LOGS(logger, INFO) << "GqaMaskReformattingToGraphCapture: rewriting CPU-bound mask reformatting subgraph "
                       << "(seqlens_k chain anchor='" << sk_chain.cast->Name()
                       << "', total_seq_len chain anchor='" << tsl_chain.cast->Name()
                       << "', mask input='" << mask_arg->Name() << "')";

    // ---- Build the replacement subgraph: ----
    //   cast32  = Cast(mask, to=INT32)
    //   rs      = ReduceSum(cast32, axes=[1], keepdims=0)  (int32, shape [B])
    //   sk      = Sub(rs, [1])  → seqlens_k (replaces sk_chain.cast output)
    //   tsl_red = ReduceMax(rs, axes=[0], keepdims=0)  (int32, scalar) → total_seq_len
    // ReduceMax over the only remaining axis collapses to a scalar, matching
    // the original total_sequence_length tensor (scalar int32).
    NodeArg& cast32_out = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName("gqa_mask_gc_cast_i32"), nullptr);
    {
      NodeAttributes attrs;
      utils::SetNodeAttribute(
          utils::MakeAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_INT32)),
          attrs);
      graph.AddNode(graph.GenerateNodeName("gqa_mask_gc_cast"),
                    "Cast",
                    "Cast attention_mask to int32 for GC-friendly reformatting",
                    {mask_arg},
                    {&cast32_out},
                    &attrs,
                    kOnnxDomain)
          .SetExecutionProviderType(kWebGpuExecutionProvider);
    }

    NodeArg& reduce_sum_axes = AddInt64Vec1Initializer(graph, "gqa_mask_gc_rs_axes", 1);
    NodeArg& reduce_sum_out = graph.GetOrCreateNodeArg(
        graph.GenerateNodeArgName("gqa_mask_gc_rs"), nullptr);
    {
      NodeAttributes attrs;
      utils::SetNodeAttribute(utils::MakeAttribute("keepdims", static_cast<int64_t>(0)), attrs);
      graph.AddNode(graph.GenerateNodeName("gqa_mask_gc_reducesum"),
                    "ReduceSum",
                    "Per-row valid-token count for GQA seqlens_k / total_seq_len",
                    {&cast32_out, &reduce_sum_axes},
                    {&reduce_sum_out},
                    &attrs,
                    kOnnxDomain)
          .SetExecutionProviderType(kWebGpuExecutionProvider);
    }

    // New seqlens_k = ReduceSum(...) - 1, reusing the original Cast-to-int32
    // output NodeArg so every downstream GQA edge is preserved automatically.
    NodeArg& sk_one = AddInt32Vec1Initializer(graph, "gqa_mask_gc_sk_one", 1);
    NodeArg* sk_out_arg = graph.GetNodeArg(sk_chain.cast->OutputDefs()[0]->Name());

    // New total_seq_len = ReduceMax(rs).
    NodeArg& reduce_max_axes = AddInt64Vec1Initializer(graph, "gqa_mask_gc_rmax_axes", 0);
    NodeArg* tsl_out_arg = graph.GetNodeArg(tsl_chain.cast->OutputDefs()[0]->Name());

    // Capture downstream consumers BEFORE we remove the original nodes.
    auto sk_out_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*sk_chain.cast);
    auto tsl_out_edges = graph_utils::GraphEdge::GetNodeOutputEdges(*tsl_chain.cast);

    // Remove the original chains. Recursive orphan reaping cleans up the
    // shared Shape/ReduceSum/Constant/initializer producers as their last
    // consumers vanish.
    Node* sk_cast_mut = graph.GetNode(sk_chain.cast->Index());
    Node* sk_sub_mut = graph.GetNode(sk_chain.sub->Index());
    Node* sk_rs_mut = graph.GetNode(sk_chain.reduce_sum->Index());
    Node* tsl_cast_mut = graph.GetNode(tsl_chain.cast->Index());
    Node* tsl_gather_mut = graph.GetNode(tsl_chain.gather->Index());
    Node* tsl_shape_mut = graph.GetNode(tsl_chain.shape->Index());

    auto safe_remove_top = [&](Node* node) {
      if (node == nullptr) {
        return;
      }
      graph_utils::RemoveNodeOutputEdges(graph, *node);
      graph.RemoveNode(node->Index());
    };

    // Snapshot upstream indices so we can reap orphans after removal.
    auto collect_upstream = [&](const Node* node, std::vector<NodeIndex>& out_idxs) {
      if (node == nullptr) {
        return;
      }
      for (auto it = node->InputEdgesBegin(); it != node->InputEdgesEnd(); ++it) {
        out_idxs.push_back(it->GetNode().Index());
      }
    };

    std::vector<NodeIndex> upstream_idxs;
    collect_upstream(sk_chain.cast, upstream_idxs);
    collect_upstream(sk_chain.sub, upstream_idxs);
    collect_upstream(sk_chain.reduce_sum, upstream_idxs);
    collect_upstream(tsl_chain.cast, upstream_idxs);
    collect_upstream(tsl_chain.gather, upstream_idxs);
    collect_upstream(tsl_chain.shape, upstream_idxs);

    safe_remove_top(sk_cast_mut);
    safe_remove_top(sk_sub_mut);
    safe_remove_top(sk_rs_mut);
    safe_remove_top(tsl_cast_mut);
    safe_remove_top(tsl_gather_mut);
    safe_remove_top(tsl_shape_mut);

    // The Cast output NodeArgs now have no producer — re-attach the new ones.
    // Add Sub (seqlens_k) producing sk_out_arg (same name as original).
    {
      Node& sub_node = graph.AddNode(graph.GenerateNodeName("gqa_mask_gc_sub"),
                                     "Sub",
                                     "GC-friendly seqlens_k = ReduceSum(mask) - 1",
                                     {&reduce_sum_out, &sk_one},
                                     {sk_out_arg},
                                     nullptr,
                                     kOnnxDomain);
      sub_node.SetExecutionProviderType(kWebGpuExecutionProvider);
      // Rewire edges to consumers.
      for (const auto& e : sk_out_edges) {
        graph.AddEdge(sub_node.Index(), e.dst_node, 0, e.dst_arg_index);
      }
    }

    // Add ReduceMax (total_seq_len) producing tsl_out_arg (same name as original).
    {
      NodeAttributes attrs;
      utils::SetNodeAttribute(utils::MakeAttribute("keepdims", static_cast<int64_t>(0)), attrs);
      Node& rmax_node = graph.AddNode(graph.GenerateNodeName("gqa_mask_gc_reducemax"),
                                      "ReduceMax",
                                      "GC-friendly total_seq_len = ReduceMax(ReduceSum(mask))",
                                      {&reduce_sum_out, &reduce_max_axes},
                                      {tsl_out_arg},
                                      &attrs,
                                      kOnnxDomain);
      rmax_node.SetExecutionProviderType(kWebGpuExecutionProvider);
      for (const auto& e : tsl_out_edges) {
        graph.AddEdge(rmax_node.Index(), e.dst_node, 0, e.dst_arg_index);
      }
    }

    // Reap orphaned upstream nodes (Constant producers, leftover Shape input
    // chains that only this subgraph consumed).
    for (NodeIndex u : upstream_idxs) {
      TryRemoveIfOrphan(graph, graph.GetNode(u));
    }

    modified = true;
    // One rewrite covers all GQAs sharing the chain; stop after the first hit.
    break;
  }

  return Status::OK();
}

}  // namespace onnxruntime
