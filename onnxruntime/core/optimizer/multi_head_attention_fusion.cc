// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/multi_head_attention_fusion.h"

#include <array>
#include <cmath>
#include <optional>

#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"

namespace onnxruntime {

namespace {

constexpr std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> kTransposeVersions = {1, 13, 21, 23, 24, 25};
constexpr std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> kReshapeVersions = {5, 13, 14, 19, 21, 23, 24, 25};
constexpr std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> kMatMulVersions = {1, 9, 13};
constexpr std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion> kBinaryVersions = {7, 13, 14};

bool IsTransposeWithPerm(const Node& node, const std::vector<int64_t>& perm) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "Transpose", kTransposeVersions) &&
         optimizer_utils::IsAttributeWithExpectedValues(node, "perm", perm);
}

bool IsReshapeWithoutAllowZero(const Node& node) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, "Reshape", kReshapeVersions) &&
         node.InputDefs().size() == 2 &&
         !optimizer_utils::IsAttributeWithExpectedValue(node, "allowzero", static_cast<int64_t>(1));
}

bool GetConstantShape(const Graph& graph, const Node& reshape, InlinedVector<int64_t>& shape) {
  return optimizer_utils::AppendTensorFromInitializer(graph, *reshape.InputDefs()[1], shape, true);
}

// Reads a scalar float/float16 constant.
std::optional<float> GetScalarConstant(const Graph& graph, const NodeArg& arg) {
  if (!optimizer_utils::IsScalar(arg)) {
    return std::nullopt;
  }
  const auto* tensor_proto = graph_utils::GetConstantInitializer(graph, arg.Name());
  if (tensor_proto == nullptr) {
    return std::nullopt;
  }
  Initializer init{graph, *tensor_proto, graph.ModelPath()};
  if (init.size() != 1) {
    return std::nullopt;
  }
  switch (tensor_proto->data_type()) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      return *init.data<float>();
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      return static_cast<float>(*init.data<MLFloat16>());
    default:
      return std::nullopt;
  }
}

// Matches `Reshape[*,*,N,H] -> Transpose(perm)` feeding `consumer` at `input_index`.
const Node* MatchHeadSplit(const Graph& graph, const Node& consumer, int input_index, const std::vector<int64_t>& perm,
                           int64_t& num_heads, int64_t& head_size, const Node*& reshape) {
  const Node* transpose = graph_utils::GetInputNode(consumer, input_index);
  if (transpose == nullptr || !IsTransposeWithPerm(*transpose, perm) ||
      !optimizer_utils::CheckOutputEdges(graph, *transpose, 1)) {
    return nullptr;
  }

  reshape = graph_utils::GetInputNode(*transpose, 0);
  if (reshape == nullptr || !IsReshapeWithoutAllowZero(*reshape)) {
    return nullptr;
  }

  InlinedVector<int64_t> shape;
  if (!GetConstantShape(graph, *reshape, shape) || shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0) {
    return nullptr;
  }

  num_heads = shape[2];
  head_size = shape[3];
  return transpose;
}

bool HasRankAtMost(const NodeArg& arg, int rank) {
  const auto* shape = arg.Shape();
  return shape != nullptr && shape->dim_size() <= rank;
}

// Walks up from a head-split Reshape through an optional bias Add and a MatMul to the tensor the projection reads.
// Only projections that keep the leading dims of that tensor are followed: a 2D constant weight and a bias of rank 1
// or less, since higher-rank constants could broadcast new leading dims. Gemm projections are not followed: the
// MatMulAddFusion that produces them requires static shapes, whose batch dims are compared directly.
const NodeArg* GetProjectionSource(const Graph& graph, const Node& head_split_reshape) {
  const Node* node = graph_utils::GetInputNode(head_split_reshape, 0);
  if (node != nullptr && node->OpType() == "Add" && node->InputDefs().size() == 2) {
    const int data_index = graph_utils::IsConstantInitializer(graph, node->InputDefs()[1]->Name(), true) ? 0 : 1;
    const NodeArg& bias = *node->InputDefs()[1 - data_index];
    if (!graph_utils::IsConstantInitializer(graph, bias.Name(), true) || !HasRankAtMost(bias, 1)) {
      return nullptr;
    }
    node = graph_utils::GetInputNode(*node, data_index);
  }
  if (node == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*node, "MatMul", kMatMulVersions)) {
    return nullptr;
  }
  const NodeArg& weight = *node->InputDefs()[1];
  if (!graph_utils::IsConstantInitializer(graph, weight.Name(), true) || !weight.Shape() ||
      weight.Shape()->dim_size() != 2) {
    return nullptr;
  }
  return node->InputDefs()[0];
}

// Whether two head-split Reshapes produce the same batch dim. Statically comparable inferred dims decide it. When
// they are unknown (dynamic axes), it still holds if both project the same tensor through leading-dim-preserving
// projections and keep the same leading reshape dims, as in self-attention.
bool HaveSameBatchDim(const Graph& graph, const Node& a, const Node& b) {
  const auto* a_shape = a.OutputDefs()[0]->Shape();
  const auto* b_shape = b.OutputDefs()[0]->Shape();
  if (a_shape != nullptr && b_shape != nullptr && a_shape->dim_size() == 4 && b_shape->dim_size() == 4) {
    const auto& a_dim = a_shape->dim(0);
    const auto& b_dim = b_shape->dim(0);
    if (utils::HasDimValue(a_dim) && utils::HasDimValue(b_dim)) {
      return a_dim.dim_value() == b_dim.dim_value();
    }
    if (utils::HasDimParam(a_dim) && utils::HasDimParam(b_dim) && a_dim.dim_param() == b_dim.dim_param()) {
      return true;
    }
  }

  InlinedVector<int64_t> a_dims, b_dims;
  const NodeArg* a_source = GetProjectionSource(graph, a);
  return a_source != nullptr && a_source == GetProjectionSource(graph, b) && GetConstantShape(graph, a, a_dims) &&
         GetConstantShape(graph, b, b_dims) && a_dims[0] == b_dims[0] && a_dims[1] == b_dims[1];
}

bool TryFuse(Graph& graph, Node& softmax, const InlinedHashSet<std::string_view>& compatible_eps,
             NodeArg*& bsd_shape_arg, const logging::Logger& logger) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(softmax, "Softmax", {1, 11, 13}) ||
      !graph_utils::IsSupportedProvider(softmax, compatible_eps) ||
      !optimizer_utils::CheckOutputEdges(graph, softmax, 1)) {
    return false;
  }

  // Before opset 13 Softmax coerces the input to 2D at `axis` (default 1), so only the last axis of the 4D score
  // tensor matches MultiHeadAttention.
  const auto* axis_attr = graph_utils::GetNodeAttribute(softmax, "axis");
  const int64_t axis = axis_attr != nullptr ? axis_attr->i() : (softmax.SinceVersion() >= 13 ? -1 : 1);
  if (axis != -1 && axis != 3) {
    return false;
  }

  float scale = 1.0f;
  const Node* scale_node = nullptr;
  const Node* qk_matmul = graph_utils::GetInputNode(softmax, 0);
  if (qk_matmul != nullptr &&
      (graph_utils::IsSupportedOptypeVersionAndDomain(*qk_matmul, "Div", kBinaryVersions) ||
       graph_utils::IsSupportedOptypeVersionAndDomain(*qk_matmul, "Mul", kBinaryVersions))) {
    scale_node = qk_matmul;
    const bool is_div = scale_node->OpType() == "Div";
    if (!optimizer_utils::CheckOutputEdges(graph, *scale_node, 1)) {
      return false;
    }
    // For Mul the scalar may be either operand; for Div it must be the divisor.
    int scalar_index = 1;
    std::optional<float> value = GetScalarConstant(graph, *scale_node->InputDefs()[1]);
    if (!value.has_value() && !is_div) {
      scalar_index = 0;
      value = GetScalarConstant(graph, *scale_node->InputDefs()[0]);
    }
    if (!value.has_value() || *value == 0.0f) {
      return false;
    }
    scale = is_div ? 1.0f / *value : *value;
    // Also catches a tiny divisor whose reciprocal overflows.
    if (!std::isfinite(scale)) {
      return false;
    }
    qk_matmul = graph_utils::GetInputNode(*scale_node, 1 - scalar_index);
  }

  if (qk_matmul == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*qk_matmul, "MatMul", kMatMulVersions) ||
      !optimizer_utils::CheckOutputEdges(graph, *qk_matmul, 1)) {
    return false;
  }

  Node* pv_matmul = graph.GetNode(softmax.OutputNodesBegin()->Index());
  if (pv_matmul == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*pv_matmul, "MatMul", kMatMulVersions) ||
      pv_matmul->InputDefs()[0] != softmax.OutputDefs()[0] ||
      !optimizer_utils::CheckOutputEdges(graph, *pv_matmul, 1)) {
    return false;
  }

  Node* out_transpose = graph.GetNode(pv_matmul->OutputNodesBegin()->Index());
  if (out_transpose == nullptr || !IsTransposeWithPerm(*out_transpose, {0, 2, 1, 3}) ||
      !optimizer_utils::CheckOutputEdges(graph, *out_transpose, 1)) {
    return false;
  }

  // The output Reshape is kept and now reads the [B, S, N*H] MultiHeadAttention output instead of the
  // [B, S, N, H] Transpose output. Both tensors hold the same elements in the same order, so explicit dims and -1
  // keep their meaning, but a 0 copies the input dim at that index, which only matches for the two leading dims.
  // The shape therefore has to be a constant without zeros past index 1.
  const Node* out_reshape = graph.GetNode(out_transpose->OutputNodesBegin()->Index());
  InlinedVector<int64_t> out_shape;
  if (out_reshape == nullptr || !IsReshapeWithoutAllowZero(*out_reshape) ||
      out_reshape->InputDefs()[0] != out_transpose->OutputDefs()[0] ||
      !GetConstantShape(graph, *out_reshape, out_shape)) {
    return false;
  }
  for (size_t i = 2; i < out_shape.size(); ++i) {
    if (out_shape[i] == 0) {
      return false;
    }
  }

  int64_t q_heads = 0, q_head_size = 0, k_heads = 0, k_head_size = 0, v_heads = 0, v_head_size = 0;
  const Node *q_reshape = nullptr, *k_reshape = nullptr, *v_reshape = nullptr;
  const Node* q_transpose = MatchHeadSplit(graph, *qk_matmul, 0, {0, 2, 1, 3}, q_heads, q_head_size, q_reshape);
  const Node* k_transpose = MatchHeadSplit(graph, *qk_matmul, 1, {0, 2, 3, 1}, k_heads, k_head_size, k_reshape);
  const Node* v_transpose = MatchHeadSplit(graph, *pv_matmul, 1, {0, 2, 1, 3}, v_heads, v_head_size, v_reshape);
  if (q_transpose == nullptr || k_transpose == nullptr || v_transpose == nullptr) {
    return false;
  }
  // Unequal head counts also cover multi-query attention, where K/V have one head that MatMul broadcasts.
  if (q_heads != k_heads || q_heads != v_heads || q_head_size != k_head_size || q_head_size != v_head_size) {
    LOGS(logger, VERBOSE) << "MultiHeadAttentionFusion[" << softmax.Name() << "]: mismatched Q/K/V head shapes";
    return false;
  }
  // MatMul broadcasts the batch dim, but MultiHeadAttention requires Q, K and V to share it.
  if (!HaveSameBatchDim(graph, *q_reshape, *k_reshape) || !HaveSameBatchDim(graph, *q_reshape, *v_reshape)) {
    return false;
  }

  const auto* q_type = q_reshape->OutputDefs()[0]->TypeAsProto();
  if (q_type == nullptr || !q_type->has_tensor_type() ||
      (q_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
       q_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16)) {
    return false;
  }

  // The remaining checks keep the fused node on the WebGPU kernel's dense FlashAttention prefill path. Its fallbacks
  // are slower than the unfused graph, and the non-flash one does not support graph capture. They mirror kernel logic
  // in contrib_ops/webgpu/bert/flash_attention.cc and contrib_ops/cpu/bert/multihead_attention_helper.h, so keep them
  // in sync with the functions named below.
  // multihead_attention_helper::CheckInputs rejects more heads than maxComputeInvocationsPerWorkgroup, whose
  // guaranteed minimum is 256.
  if (q_heads > 256) {
    return false;
  }
  // CanApplyFlashAttention requires a head size that is a multiple of 8 on every adapter.
  if (q_head_size % 8 != 0) {
    return false;
  }
  // CanApplyDensePrefill needs K/V tiles of 16 keys to fit in the 16 KB of workgroup storage every adapter
  // guarantees (DensePrefillFitsWorkgroupStorage, SelectDensePrefillMaxKStep). That matches the kernel except on
  // Qualcomm, whose extra output tile can exceed 16 KB on 16 KB adapters; the adapter is
  // unknown here, and missing dense prefill only costs speed, so the bound is not tightened for that case.
  const int64_t element_size =
      q_type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16 ? 2 : 4;
  if (2 * 16 * q_head_size * element_size > 16384) {
    return false;
  }
  // CanApplyDensePrefill sends queries shorter than 32 to the split-reduce decode kernel, and they leave only a small
  // score tensor to avoid.
  // A symbolic query length cannot be checked here, so it is fused.
  const auto* q_bnsh_shape = q_transpose->OutputDefs()[0]->Shape();
  if (q_bnsh_shape != nullptr && q_bnsh_shape->dim_size() == 4 && utils::HasDimValue(q_bnsh_shape->dim(2)) &&
      q_bnsh_shape->dim(2).dim_value() < 32) {
    return false;
  }

  const auto& provider = softmax.GetExecutionProviderType();
  for (const Node* node : {qk_matmul, scale_node, static_cast<const Node*>(pv_matmul),
                           static_cast<const Node*>(out_transpose), q_transpose, k_transpose, v_transpose,
                           q_reshape, k_reshape, v_reshape, out_reshape}) {
    if (node != nullptr && node->GetExecutionProviderType() != provider) {
      return false;
    }
  }

  if (bsd_shape_arg == nullptr) {
    ONNX_NAMESPACE::TensorProto shape_proto;
    shape_proto.set_name(graph.GenerateNodeArgName("mha_fusion_bsd_shape"));
    shape_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    shape_proto.add_dims(3);
    const std::array<int64_t, 3> bsd_shape{0, 0, -1};
    utils::SetRawDataInTensorProto(shape_proto, bsd_shape.data(), bsd_shape.size() * sizeof(int64_t));
    bsd_shape_arg = &graph_utils::AddInitializerWithOrtValue(graph, shape_proto);
  }

  // Only the element type is set; shape inference fills in the shapes when the graph is resolved.
  ONNX_NAMESPACE::TypeProto tensor_type;
  tensor_type.mutable_tensor_type()->set_elem_type(q_type->tensor_type().elem_type());

  auto add_to_bsd = [&](const Node& reshape) -> Node& {
    NodeArg* bsnh = graph.GetNode(reshape.Index())->MutableOutputDefs()[0];
    NodeArg& bsd = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(bsnh->Name() + "_bsd"), &tensor_type);
    Node& node = graph.AddNode(graph.GenerateNodeName(reshape.Name() + "/MhaBsd"), "Reshape",
                               "Fold heads into the hidden dim for MultiHeadAttention", {bsnh, bsd_shape_arg}, {&bsd});
    node.SetExecutionProviderType(provider);
    graph.AddEdge(reshape.Index(), node.Index(), 0, 0);
    return node;
  };
  Node& q_bsd = add_to_bsd(*q_reshape);
  Node& k_bsd = add_to_bsd(*k_reshape);
  Node& v_bsd = add_to_bsd(*v_reshape);

  NodeArg& mha_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("mha_fusion_output"), &tensor_type);
  Node& mha = graph.AddNode(graph.GenerateNodeName(softmax.Name() + "/MultiHeadAttention"), "MultiHeadAttention",
                            "Fused scaled dot-product attention",
                            {q_bsd.MutableOutputDefs()[0], k_bsd.MutableOutputDefs()[0], v_bsd.MutableOutputDefs()[0]},
                            {&mha_output}, nullptr, kMSDomain);
  mha.AddAttribute("num_heads", q_heads);
  mha.AddAttribute("scale", scale);
  mha.SetExecutionProviderType(provider);
  graph.AddEdge(q_bsd.Index(), mha.Index(), 0, 0);
  graph.AddEdge(k_bsd.Index(), mha.Index(), 0, 1);
  graph.AddEdge(v_bsd.Index(), mha.Index(), 0, 2);

  graph_utils::ReplaceDownstreamNodeInput(graph, *out_transpose, 0, mha, 0);

  InlinedVector<NodeIndex> nodes_to_remove{q_transpose->Index(), k_transpose->Index(), v_transpose->Index(),
                                           qk_matmul->Index(), softmax.Index(), pv_matmul->Index(),
                                           out_transpose->Index()};
  if (scale_node != nullptr) {
    nodes_to_remove.push_back(scale_node->Index());
  }
  graph.NotifyNodeReplacement(nodes_to_remove, mha.Index());
  for (NodeIndex index : nodes_to_remove) {
    Node* node = graph.GetNode(index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(index);
  }

  LOGS(logger, VERBOSE) << "MultiHeadAttentionFusion: fused attention at " << mha.Name();
  return true;
}

}  // namespace

Status MultiHeadAttentionFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                           const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  NodeArg* bsd_shape_arg = nullptr;
  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    Node* node = graph.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (TryFuse(graph, *node, GetCompatibleExecutionProviders(), bsd_shape_arg, logger)) {
      ++fused_count;
      modified = true;
    }
  }

  if (fused_count > 0) {
    LOGS(logger, INFO) << "Total fused MultiHeadAttention node count: " << fused_count;
  }

  return Status::OK();
}

}  // namespace onnxruntime
