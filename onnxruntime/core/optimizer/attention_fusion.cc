// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/attention_fusion_helper.h"
#include <cmath>
#include <optional>

namespace onnxruntime {

static bool ValidateMatMulInitializer(const Graph& graph, const Node& matmul, int64_t hidden_size);

namespace {

static bool ValidateAddBiasInitializerEitherInput(const Graph& graph, const Node& add, int64_t hidden_size) {
  if (add.InputDefs().size() < 2) {
    return false;
  }

  const NodeArg& input_0 = *(add.InputDefs()[0]);
  const NodeArg& input_1 = *(add.InputDefs()[1]);
  const bool input_0_is_bias = graph_utils::IsInitializer(graph, input_0.Name(), true) &&
                               optimizer_utils::ValidateShape(input_0, {hidden_size});
  const bool input_1_is_bias = graph_utils::IsInitializer(graph, input_1.Name(), true) &&
                               optimizer_utils::ValidateShape(input_1, {hidden_size});
  return input_0_is_bias || input_1_is_bias;
}

static bool ValidateProjectionGemmInitializer(const Graph& graph, const Node& gemm, int64_t hidden_size) {
  if (gemm.InputDefs().size() < 3) {
    return false;
  }

  if (const auto* alpha_attr = graph_utils::GetNodeAttribute(gemm, "alpha");
      alpha_attr && std::abs(alpha_attr->f() - 1.0f) > 1e-6f) {
    return false;
  }

  if (const auto* beta_attr = graph_utils::GetNodeAttribute(gemm, "beta");
      beta_attr && std::abs(beta_attr->f() - 1.0f) > 1e-6f) {
    return false;
  }

  if (const auto* trans_a_attr = graph_utils::GetNodeAttribute(gemm, "transA");
      trans_a_attr && trans_a_attr->i() != 0) {
    return false;
  }

  if (const auto* trans_b_attr = graph_utils::GetNodeAttribute(gemm, "transB");
      trans_b_attr && trans_b_attr->i() != 0) {
    return false;
  }

  const NodeArg& input_b = *(gemm.InputDefs()[1]);
  const NodeArg& input_c = *(gemm.InputDefs()[2]);
  if (!graph_utils::IsInitializer(graph, input_b.Name(), true) ||
      !graph_utils::IsInitializer(graph, input_c.Name(), true)) {
    return false;
  }

  return optimizer_utils::ValidateShape(input_b, {hidden_size, hidden_size}) &&
         optimizer_utils::ValidateShape(input_c, {hidden_size});
}

// Most attention fusions require all matched nodes to already be assigned to an execution provider
// that supports the fused op. MobileClipMHA is also matched before partitioning in graph-transform
// tests, so nodes may still be unassigned here. Accept nodes that are either unassigned or already
// assigned to a compatible provider, and preserve the original provider string on the fused nodes
// once the pattern is rewritten.
static bool IsSupportedOrUnassignedNode(const Node& node,
                                        const InlinedHashSet<std::string_view>& compatible_execution_providers) {
  return node.GetExecutionProviderType().empty() ||
         graph_utils::IsSupportedProvider(node, compatible_execution_providers);
}

static bool IsSupportedOrUnassignedNode(const Node& node,
                                        std::string_view required_execution_provider) {
  const auto& execution_provider = node.GetExecutionProviderType();
  return execution_provider.empty() ||
         execution_provider == required_execution_provider;
}

static bool AreSupportedOrUnassignedNodes(
    const Node& anchor_node,
    const std::initializer_list<const Node*>& nodes,
    const InlinedHashSet<std::string_view>& compatible_execution_providers) {
  if (!IsSupportedOrUnassignedNode(anchor_node, compatible_execution_providers)) {
    return false;
  }

  const auto& required_execution_provider = anchor_node.GetExecutionProviderType();
  for (const Node* node : nodes) {
    if (node == nullptr) {
      continue;
    }

    if (!IsSupportedOrUnassignedNode(*node, required_execution_provider)) {
      return false;
    }
  }

  return true;
}

static bool HasExpectedPerm(const Node& node, const std::initializer_list<int64_t>& expected_perm) {
  return optimizer_utils::IsAttributeWithExpectedValues(node, "perm", std::vector<int64_t>(expected_perm));
}

static bool HasExpectedAxesInput(const Graph& graph, const Node& node, const std::initializer_list<int64_t>& expected_axes) {
  if (node.InputDefs().size() < 2) {
    return false;
  }

  InlinedVector<int64_t> axes;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *node.InputDefs()[1], axes, true)) {
    return false;
  }

  return axes == InlinedVector<int64_t>(expected_axes.begin(), expected_axes.end());
}

static bool TryGetMobileClipQkvReshapeInfo(const Graph& graph, const Node& qkv_reshape,
                                           int64_t& num_heads, int64_t& head_size, int64_t& hidden_size) {
  if (qkv_reshape.InputDefs().size() < 2) {
    return false;
  }

  InlinedVector<int64_t> reshape_dims;
  if (!optimizer_utils::AppendTensorFromInitializer(graph, *qkv_reshape.InputDefs()[1], reshape_dims, true)) {
    return false;
  }

  if (reshape_dims.size() != 5 || reshape_dims[2] != 3 || reshape_dims[3] <= 0 || reshape_dims[4] <= 0) {
    return false;
  }

  num_heads = reshape_dims[3];
  head_size = reshape_dims[4];

  try {
    hidden_size = SafeInt<int64_t>(num_heads) * head_size;
  } catch (const OnnxRuntimeException&) {
    return false;
  }

  return hidden_size > 0;
}

static std::optional<ONNX_NAMESPACE::TypeProto> TryCreateMobileClipMhaOutputType(const NodeArg& qkv_output,
                                                                                 int64_t hidden_size) {
  const auto* qkv_output_type = qkv_output.TypeAsProto();
  if (qkv_output_type == nullptr || !qkv_output_type->has_tensor_type()) {
    return std::nullopt;
  }

  ONNX_NAMESPACE::TypeProto mha_output_type(*qkv_output_type);
  auto* shape = mha_output_type.mutable_tensor_type()->mutable_shape();
  if (shape->dim_size() > 0) {
    auto* last_dim = shape->mutable_dim(shape->dim_size() - 1);
    last_dim->clear_dim_param();
    last_dim->set_dim_value(hidden_size);
  }

  return mha_output_type;
}

static Node* GetOnlyChildByOutputIndex(Graph& graph, const Node& node, size_t output_index, const char* child_op_type) {
  const auto output_edges = graph_utils::GraphEdge::GetNodeOutputEdges(node, output_index);
  if (output_edges.size() != 1) {
    return nullptr;
  }

  Node* child = graph.GetNode(output_edges[0].dst_node);
  if (child == nullptr || child->OpType() != child_op_type) {
    return nullptr;
  }

  return child;
}

static bool TryCreateNormalizedProjectionGemm(Graph& graph,
                                              NodeArg& projection_input,
                                              const NodeArg& original_projection_input,
                                              const NodeArg& proj_weight,
                                              const NodeArg& proj_bias,
                                              NodeArg& projection_output,
                                              const std::string& base_name,
                                              const std::string& provider_type) {
  const auto* proj_input_shape = original_projection_input.Shape();
  const auto* proj_weight_shape = proj_weight.Shape();
  if (proj_input_shape == nullptr || proj_weight_shape == nullptr || proj_weight_shape->dim_size() != 2) {
    return false;
  }

  auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*proj_input_shape);
  if (input_shape.Size() == -1 || input_shape.NumDimensions() < 2) {
    return false;
  }

  const auto& dim_k = proj_weight_shape->dim(0);
  const auto& dim_n = proj_weight_shape->dim(1);
  if (!utils::HasDimValue(dim_k) || !utils::HasDimValue(dim_n)) {
    return false;
  }

  const int64_t m = input_shape.SizeToDimension(input_shape.NumDimensions() - 1);
  if (m <= 0) {
    return false;
  }

  const int64_t k = dim_k.dim_value();
  const int64_t n = dim_n.dim_value();
  if (input_shape[input_shape.NumDimensions() - 1] != k) {
    return false;
  }

  const auto* bias_shape = proj_bias.Shape();
  if (bias_shape == nullptr || bias_shape->dim_size() != 1 || !utils::HasDimValue(bias_shape->dim(0)) ||
      bias_shape->dim(0).dim_value() != n) {
    return false;
  }

  const auto* input_type = original_projection_input.TypeAsProto();
  if (input_type == nullptr || !input_type->has_tensor_type()) {
    return false;
  }

  const auto element_type = static_cast<ONNX_NAMESPACE::TensorProto_DataType>(input_type->tensor_type().elem_type());

  auto add_shape_initializer = [&](const std::string& name, const InlinedVector<int64_t>& shape) -> NodeArg& {
    ONNX_NAMESPACE::TensorProto shape_initializer_proto;
    shape_initializer_proto.set_name(graph.GenerateNodeArgName(name));
    shape_initializer_proto.add_dims(static_cast<int64_t>(shape.size()));
    shape_initializer_proto.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
    const size_t shape_bytes = SafeInt<size_t>(shape.size()) * sizeof(int64_t);
    utils::SetRawDataInTensorProto(shape_initializer_proto, shape.data(), shape_bytes);
    return graph_utils::AddInitializerWithOrtValue(graph, shape_initializer_proto);
  };

  auto make_tensor_arg = [&](const std::string& name, const InlinedVector<int64_t>& shape) -> NodeArg* {
    ONNX_NAMESPACE::TypeProto type_proto;
    type_proto.mutable_tensor_type()->set_elem_type(element_type);
    for (int64_t dim_value : shape) {
      type_proto.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dim_value);
    }

    return &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName(name), &type_proto);
  };

  InlinedVector<int64_t> gemm_input_shape{m, k};
  InlinedVector<int64_t> gemm_output_shape{m, n};
  InlinedVector<int64_t> output_shape_values = input_shape.AsShapeVector();
  output_shape_values.back() = n;

  NodeArg* gemm_input_arg = make_tensor_arg("mobileclip_proj_gemm_input", gemm_input_shape);
  NodeArg* gemm_output_arg = make_tensor_arg("mobileclip_proj_gemm_output", gemm_output_shape);
  NodeArg& gemm_input_shape_arg = add_shape_initializer("mobileclip_proj_gemm_input_shape", gemm_input_shape);
  NodeArg& gemm_output_shape_arg = add_shape_initializer("mobileclip_proj_gemm_output_shape", output_shape_values);

  Node& input_reshape = graph.AddNode(
      graph.GenerateNodeName("MobileClipProjGemmInputReshape"),
      "Reshape",
      "Reshape MobileCLIP projection input for Gemm",
      {&projection_input, &gemm_input_shape_arg},
      {gemm_input_arg});
  input_reshape.SetExecutionProviderType(provider_type);

  Node& gemm_node = graph.AddNode(
      graph.GenerateNodeName(base_name + "/MobileClipProjectionGemm"),
      "Gemm",
      "Normalized MobileCLIP projection Gemm",
      {gemm_input_arg, const_cast<NodeArg*>(&proj_weight), const_cast<NodeArg*>(&proj_bias)},
      {gemm_output_arg});
  gemm_node.SetExecutionProviderType(provider_type);

  Node& output_reshape = graph.AddNode(
      graph.GenerateNodeName("MobileClipProjGemmOutputReshape"),
      "Reshape",
      "Restore MobileCLIP projection output shape after Gemm",
      {gemm_output_arg, &gemm_output_shape_arg},
      {&projection_output});
  output_reshape.SetExecutionProviderType(provider_type);

  return true;
}

static bool TryRewriteProjectionMatMulAddToGemm(Graph& graph,
                                                NodeArg& projection_input,
                                                Node& proj_matmul,
                                                Node& proj_add) {
  if (proj_matmul.InputDefs().size() < 2 || proj_add.InputDefs().size() < 2) {
    return false;
  }

  const int bias_idx = proj_matmul.OutputDefs()[0]->Name() == proj_add.InputDefs()[0]->Name() ? 1 : 0;
  return TryCreateNormalizedProjectionGemm(graph,
                                           projection_input,
                                           *proj_matmul.InputDefs()[0],
                                           *proj_matmul.InputDefs()[1],
                                           *proj_add.InputDefs()[bias_idx],
                                           *proj_add.MutableOutputDefs()[0],
                                           proj_matmul.Name(),
                                           proj_matmul.GetExecutionProviderType());
}

static bool TryRewriteProjectionGemm(Graph& graph,
                                     NodeArg& projection_input,
                                     Node& proj_gemm) {
  if (proj_gemm.InputDefs().size() < 3 || proj_gemm.OutputDefs().empty()) {
    return false;
  }

  return TryCreateNormalizedProjectionGemm(graph,
                                           projection_input,
                                           *proj_gemm.InputDefs()[0],
                                           *proj_gemm.InputDefs()[1],
                                           *proj_gemm.InputDefs()[2],
                                           *proj_gemm.MutableOutputDefs()[0],
                                           proj_gemm.Name(),
                                           proj_gemm.GetExecutionProviderType());
}

static bool TryFuseMobileClipMHA(Node& qkv_matmul,
                                 Graph& graph,
                                 const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                 const logging::Logger& logger) {
  const auto fail = [&](const char* message) {
    LOGS(logger, VERBOSE) << "MobileClipMHA[" << qkv_matmul.Name() << "]: fusion skipped: " << message;
    return false;
  };

  if (!graph_utils::IsSupportedOptypeVersionAndDomain(qkv_matmul, "MatMul", {1, 9, 13}, kOnnxDomain)) {
    return false;
  }

  if (!IsSupportedOrUnassignedNode(qkv_matmul, compatible_execution_providers)) {
    return false;
  }

  if (!optimizer_utils::CheckOutputEdges(graph, qkv_matmul, 1) || qkv_matmul.InputDefs().size() < 2 ||
      !graph_utils::IsInitializer(graph, qkv_matmul.InputDefs()[1]->Name(), true)) {
    return fail("qkv MatMul output count or weight initializer check failed");
  }

  const Node* sequence_transpose = graph_utils::GetInputNode(qkv_matmul, 0);
  if (sequence_transpose == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*sequence_transpose, "Transpose", {1, 13}, kOnnxDomain) ||
      !HasExpectedPerm(*sequence_transpose, {0, 2, 1}) ||
      !optimizer_utils::CheckOutputEdges(graph, *sequence_transpose, 1)) {
    return false;
  }

  const Node* input_reshape = graph_utils::GetInputNode(*sequence_transpose, 0);
  if (input_reshape == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*input_reshape, "Reshape", {5, 13, 14}, kOnnxDomain) ||
      !optimizer_utils::CheckOutputEdges(graph, *input_reshape, 1)) {
    return fail("missing input Reshape before sequence transpose");
  }

  Node* qkv_reshape = GetOnlyChildByOutputIndex(graph, qkv_matmul, 0, "Reshape");
  if (qkv_reshape == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*qkv_reshape, "Reshape", {5, 13, 14}, kOnnxDomain) ||
      !optimizer_utils::CheckOutputEdges(graph, *qkv_reshape, 1)) {
    return fail("qkv Reshape after MatMul not matched");
  }

  Node* split = GetOnlyChildByOutputIndex(graph, *qkv_reshape, 0, "Split");
  if (split == nullptr || !graph_utils::IsSupportedOptypeVersionAndDomain(*split, "Split", {13, 18}, kOnnxDomain) ||
      split->OutputDefs().size() != 3 || !optimizer_utils::IsAttributeWithExpectedValue(*split, "axis", static_cast<int64_t>(2))) {
    return fail("qkv Split(axis=2, outputs=3) not matched");
  }

  Node* q_transpose = GetOnlyChildByOutputIndex(graph, *split, 0, "Transpose");
  Node* k_squeeze = GetOnlyChildByOutputIndex(graph, *split, 1, "Squeeze");
  Node* v_transpose = GetOnlyChildByOutputIndex(graph, *split, 2, "Transpose");
  if (q_transpose == nullptr || k_squeeze == nullptr || v_transpose == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*q_transpose, "Transpose", {1, 13}, kOnnxDomain) ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*k_squeeze, "Squeeze", {13}, kOnnxDomain) ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*v_transpose, "Transpose", {1, 13}, kOnnxDomain) ||
      !HasExpectedPerm(*q_transpose, {2, 0, 3, 1, 4}) ||
      !HasExpectedPerm(*v_transpose, {2, 0, 3, 1, 4}) ||
      !HasExpectedAxesInput(graph, *k_squeeze, {2})) {
    return fail("q/k/v branch entry pattern after Split not matched");
  }

  Node* q_squeeze = GetOnlyChildByOutputIndex(graph, *q_transpose, 0, "Squeeze");
  Node* v_squeeze = GetOnlyChildByOutputIndex(graph, *v_transpose, 0, "Squeeze");
  if (q_squeeze == nullptr || v_squeeze == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*q_squeeze, "Squeeze", {13}, kOnnxDomain) ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*v_squeeze, "Squeeze", {13}, kOnnxDomain) ||
      !HasExpectedAxesInput(graph, *q_squeeze, {0}) ||
      !HasExpectedAxesInput(graph, *v_squeeze, {0})) {
    return fail("q/v squeeze pattern not matched");
  }

  Node* q_scale_mul = GetOnlyChildByOutputIndex(graph, *q_squeeze, 0, "Mul");
  Node* k_transpose = GetOnlyChildByOutputIndex(graph, *k_squeeze, 0, "Transpose");
  if (q_scale_mul == nullptr || k_transpose == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*q_scale_mul, "Mul", {7, 13, 14}, kOnnxDomain) ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*k_transpose, "Transpose", {1, 13}, kOnnxDomain) ||
      !HasExpectedPerm(*k_transpose, {0, 2, 3, 1})) {
    return fail("q scale Mul or k Transpose(0,2,3,1) not matched");
  }

  float scale = 0.0f;
  if (q_scale_mul->InputDefs().size() < 2) {
    return fail("q scale constant not found");
  }

  const NodeArg* q_squeeze_output = q_squeeze->OutputDefs()[0];
  const NodeArg* mul_input_0 = q_scale_mul->InputDefs()[0];
  const NodeArg* mul_input_1 = q_scale_mul->InputDefs()[1];
  const bool input_0_is_q_squeeze = mul_input_0 != nullptr && q_squeeze_output != nullptr &&
                                    mul_input_0->Name() == q_squeeze_output->Name();
  const bool input_1_is_q_squeeze = mul_input_1 != nullptr && q_squeeze_output != nullptr &&
                                    mul_input_1->Name() == q_squeeze_output->Name();

  const NodeArg* scale_input = nullptr;
  if (input_0_is_q_squeeze && !input_1_is_q_squeeze) {
    scale_input = mul_input_1;
  } else if (input_1_is_q_squeeze && !input_0_is_q_squeeze) {
    scale_input = mul_input_0;
  }

  if (scale_input == nullptr ||
      !optimizer_utils::GetScalarInitializerValue<float>(graph, *scale_input, scale, true)) {
    return fail("q scale constant not found");
  }

  Node* qk_matmul = GetOnlyChildByOutputIndex(graph, *q_scale_mul, 0, "MatMul");
  if (qk_matmul == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*qk_matmul, "MatMul", {1, 9, 13}, kOnnxDomain) ||
      graph_utils::GetInputNode(*qk_matmul, 1) == nullptr ||
      graph_utils::GetInputNode(*qk_matmul, 1)->Index() != k_transpose->Index() ||
      !optimizer_utils::CheckOutputEdges(graph, *qk_matmul, 1)) {
    return fail("qk MatMul not matched");
  }

  Node* softmax = GetOnlyChildByOutputIndex(graph, *qk_matmul, 0, "Softmax");
  if (softmax == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*softmax, "Softmax", {1, 11, 13}, kOnnxDomain) ||
      !optimizer_utils::IsAttributeWithExpectedValue(*softmax, "axis", static_cast<int64_t>(-1)) ||
      !optimizer_utils::CheckOutputEdges(graph, *softmax, 1)) {
    return fail("Softmax(axis=-1) not matched");
  }

  Node* qkv_matmul_1 = GetOnlyChildByOutputIndex(graph, *softmax, 0, "MatMul");
  if (qkv_matmul_1 == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*qkv_matmul_1, "MatMul", {1, 9, 13}, kOnnxDomain) ||
      graph_utils::GetInputNode(*qkv_matmul_1, 1) == nullptr ||
      graph_utils::GetInputNode(*qkv_matmul_1, 1)->Index() != v_squeeze->Index() ||
      !optimizer_utils::CheckOutputEdges(graph, *qkv_matmul_1, 1)) {
    return fail("attention-value MatMul not matched");
  }

  Node* transpose_3 = GetOnlyChildByOutputIndex(graph, *qkv_matmul_1, 0, "Transpose");
  if (transpose_3 == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*transpose_3, "Transpose", {1, 13}, kOnnxDomain) ||
      !HasExpectedPerm(*transpose_3, {0, 2, 1, 3}) ||
      !optimizer_utils::CheckOutputEdges(graph, *transpose_3, 1)) {
    return fail("output Transpose(0,2,1,3) not matched");
  }

  Node* reshape_2 = GetOnlyChildByOutputIndex(graph, *transpose_3, 0, "Reshape");
  if (reshape_2 == nullptr ||
      !graph_utils::IsSupportedOptypeVersionAndDomain(*reshape_2, "Reshape", {5, 13, 14}, kOnnxDomain) ||
      !optimizer_utils::CheckOutputEdges(graph, *reshape_2, 1)) {
    return fail("output Reshape not matched");
  }

  Node* proj_matmul = GetOnlyChildByOutputIndex(graph, *reshape_2, 0, "MatMul");
  Node* proj_gemm = proj_matmul == nullptr ? GetOnlyChildByOutputIndex(graph, *reshape_2, 0, "Gemm") : nullptr;
  Node* proj_gemm_input_reshape = nullptr;
  Node* proj_gemm_output_reshape = nullptr;
  Node* proj_add = nullptr;

  if (proj_matmul != nullptr) {
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*proj_matmul, "MatMul", {1, 9, 13}, kOnnxDomain) ||
        proj_matmul->InputDefs().size() < 2 ||
        !graph_utils::IsInitializer(graph, proj_matmul->InputDefs()[1]->Name(), true) ||
        !optimizer_utils::CheckOutputEdges(graph, *proj_matmul, 1)) {
      return fail("projection MatMul not matched");
    }

    proj_add = GetOnlyChildByOutputIndex(graph, *proj_matmul, 0, "Add");
    if (proj_add == nullptr ||
        !graph_utils::IsSupportedOptypeVersionAndDomain(*proj_add, "Add", {7, 13, 14}, kOnnxDomain) ||
        !optimizer_utils::CheckOutputEdges(graph, *proj_add, 1)) {
      return fail("projection Add not matched");
    }
  } else {
    if (proj_gemm == nullptr) {
      proj_gemm_input_reshape = GetOnlyChildByOutputIndex(graph, *reshape_2, 0, "Reshape");
      if (proj_gemm_input_reshape == nullptr ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*proj_gemm_input_reshape, "Reshape", {5, 13, 14}, kOnnxDomain) ||
          !optimizer_utils::CheckOutputEdges(graph, *proj_gemm_input_reshape, 1)) {
        return fail("projection MatMul/Gemm not matched");
      }

      proj_gemm = GetOnlyChildByOutputIndex(graph, *proj_gemm_input_reshape, 0, "Gemm");
      if (proj_gemm == nullptr ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*proj_gemm, "Gemm", {7, 9, 11, 13}, kOnnxDomain) ||
          !optimizer_utils::CheckOutputEdges(graph, *proj_gemm, 1)) {
        return fail("projection MatMul/Gemm not matched");
      }

      proj_gemm_output_reshape = GetOnlyChildByOutputIndex(graph, *proj_gemm, 0, "Reshape");
      if (proj_gemm_output_reshape == nullptr ||
          !graph_utils::IsSupportedOptypeVersionAndDomain(*proj_gemm_output_reshape, "Reshape", {5, 13, 14}, kOnnxDomain) ||
          !optimizer_utils::CheckOutputEdges(graph, *proj_gemm_output_reshape, 1)) {
        return fail("normalized projection Gemm output Reshape not matched");
      }
    } else if (!graph_utils::IsSupportedOptypeVersionAndDomain(*proj_gemm, "Gemm", {7, 9, 11, 13}, kOnnxDomain) ||
               !optimizer_utils::CheckOutputEdges(graph, *proj_gemm, 1)) {
      return fail("projection MatMul/Gemm not matched");
    }
  }

  int64_t num_heads = 0;
  int64_t head_size = 0;
  int64_t hidden_size = 0;
  if (!TryGetMobileClipQkvReshapeInfo(graph, *qkv_reshape, num_heads, head_size, hidden_size)) {
    return fail("unable to derive num_heads/head_size from qkv reshape initializer");
  }

  if (proj_matmul != nullptr) {
    if (!ValidateMatMulInitializer(graph, *proj_matmul, hidden_size) ||
        !ValidateAddBiasInitializerEitherInput(graph, *proj_add, hidden_size)) {
      return fail("projection weight/bias shape validation failed");
    }
  } else {
    if (!ValidateProjectionGemmInitializer(graph, *proj_gemm, hidden_size)) {
      return fail("projection Gemm weight/bias shape validation failed");
    }
  }

  const NodeArg& qkv_weight = *qkv_matmul.InputDefs()[1];
  if (!optimizer_utils::ValidateShape(qkv_weight, {hidden_size, 3 * hidden_size})) {
    return fail("qkv weight shape is not [hidden, 3*hidden]");
  }

  if (!AreSupportedOrUnassignedNodes(
          qkv_matmul,
          {sequence_transpose,
           input_reshape,
           qkv_reshape,
           split,
           q_transpose,
           k_squeeze,
           v_transpose,
           q_squeeze,
           v_squeeze,
           q_scale_mul,
           k_transpose,
           qk_matmul,
           softmax,
           qkv_matmul_1,
           transpose_3,
           reshape_2,
           proj_matmul,
           proj_add,
           proj_gemm_input_reshape,
           proj_gemm,
           proj_gemm_output_reshape},
          compatible_execution_providers)) {
    return fail("matched nodes are assigned to incompatible execution providers");
  }

  auto mha_output_type = TryCreateMobileClipMhaOutputType(*qkv_matmul.OutputDefs()[0], hidden_size);
  auto* mha_output = &graph.GetOrCreateNodeArg(
      graph.GenerateNodeArgName("mobileclip_mha_output"),
      mha_output_type ? &*mha_output_type : nullptr);

  if (proj_matmul != nullptr) {
    if (!TryRewriteProjectionMatMulAddToGemm(graph, *mha_output, *proj_matmul, *proj_add)) {
      return fail("projection MatMul/Add could not be rewritten to Gemm");
    }
  } else if (proj_gemm_input_reshape == nullptr) {
    if (!TryRewriteProjectionGemm(graph, *mha_output, *proj_gemm)) {
      return fail("projection Gemm could not be normalized");
    }
  }

  ONNX_NAMESPACE::TensorProto split_sizes_tensor;
  split_sizes_tensor.set_name(graph.GenerateNodeArgName("mobileclip_mha_split_sizes"));
  split_sizes_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  split_sizes_tensor.add_dims(3);
  const std::array<int64_t, 3> split_sizes{hidden_size, hidden_size, hidden_size};
  utils::SetRawDataInTensorProto(split_sizes_tensor, split_sizes.data(), split_sizes.size() * sizeof(int64_t));
  NodeArg& split_sizes_arg = graph_utils::AddInitializerWithOrtValue(graph, split_sizes_tensor);

  auto* mha_q = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("mobileclip_mha_q"), nullptr);
  auto* mha_k = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("mobileclip_mha_k"), nullptr);
  auto* mha_v = &graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("mobileclip_mha_v"), nullptr);

  Node& split_for_mha = graph.AddNode(
      graph.GenerateNodeName("MobileClipSplitForMHA"),
      "Split",
      "Split packed MobileCLIP QKV for MultiHeadAttention",
      {qkv_matmul.MutableOutputDefs()[0], &split_sizes_arg},
      {mha_q, mha_k, mha_v},
      nullptr,
      kOnnxDomain);
  split_for_mha.AddAttribute("axis", static_cast<int64_t>(2));

  Node& mha_node = graph.AddNode(
      graph.GenerateNodeName("MobileClipMultiHeadAttention"),
      "MultiHeadAttention",
      "Fused MobileCLIP attention subgraph",
      {mha_q, mha_k, mha_v},
      {mha_output},
      nullptr,
      kMSDomain);
  mha_node.AddAttribute("num_heads", num_heads);
  mha_node.AddAttribute("scale", scale);

  const auto& provider = qkv_matmul.GetExecutionProviderType();
  split_for_mha.SetExecutionProviderType(provider);
  mha_node.SetExecutionProviderType(provider);

  if (proj_gemm_input_reshape != nullptr) {
    graph_utils::ReplaceDownstreamNodeInput(graph, *reshape_2, 0, mha_node, 0);
  }

  std::vector<NodeIndex> nodes_to_remove{
      qkv_reshape->Index(),
      split->Index(),
      q_transpose->Index(),
      q_squeeze->Index(),
      q_scale_mul->Index(),
      k_squeeze->Index(),
      k_transpose->Index(),
      qk_matmul->Index(),
      softmax->Index(),
      v_transpose->Index(),
      v_squeeze->Index(),
      qkv_matmul_1->Index(),
      transpose_3->Index(),
      reshape_2->Index(),
  };

  if (proj_matmul != nullptr) {
    nodes_to_remove.push_back(proj_matmul->Index());
    nodes_to_remove.push_back(proj_add->Index());
  } else if (proj_gemm_input_reshape == nullptr) {
    nodes_to_remove.push_back(proj_gemm->Index());
  }

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    if (node == nullptr) {
      continue;
    }

    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node_index);
  }

  LOGS(logger, VERBOSE) << "MobileClipMHA[" << qkv_matmul.Name()
                        << "]: fused MobileCLIP attention subgraph to MultiHeadAttention";

  return true;
}

}  // namespace

static bool ValidateMatMulInitializer(const Graph& graph, const Node& matmul, int64_t hidden_size) {
  const NodeArg& input_b = *(matmul.InputDefs()[1]);
  if (!graph_utils::IsInitializer(graph, input_b.Name(), true)) {
    return false;
  }

  return optimizer_utils::ValidateShape(input_b, {hidden_size, hidden_size});
}

static bool ValidateAddBiasInitializer(const Graph& graph, const Node& add, int64_t hidden_size) {
  const NodeArg& input_b = *(add.InputDefs()[1]);
  if (!graph_utils::IsInitializer(graph, input_b.Name(), true)) {
    return false;
  }

  return optimizer_utils::ValidateShape(input_b, {hidden_size});
}

// Load q, k and v weights, and validate their data types.
static bool LoadQkvWeights(
    Graph& graph,
    const Node& q, const Node& k, const Node& v,
    const ONNX_NAMESPACE::TensorProto*& q_tensor,
    const ONNX_NAMESPACE::TensorProto*& k_tensor,
    const ONNX_NAMESPACE::TensorProto*& v_tensor) {
  if (!graph.GetInitializedTensor(q.InputDefs()[1]->Name(), q_tensor)) {
    return false;
  }

  // Attention Op requires float or float16 weights.
  const auto data_type = q_tensor->data_type();
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT16) {
    return false;
  }

  if (!graph.GetInitializedTensor(k.InputDefs()[1]->Name(), k_tensor) ||
      data_type != k_tensor->data_type()) {
    return false;
  }

  if (!graph.GetInitializedTensor(v.InputDefs()[1]->Name(), v_tensor) ||
      data_type != v_tensor->data_type()) {
    return false;
  }

  return true;
}

// Merge the weights of Q, K and V inputs for MatMul or Add (bias) into one input.
static NodeArg& MergeQkvWeights(Graph& graph, int64_t hidden_size,
                                const ONNX_NAMESPACE::TensorProto* q_tensor,
                                const ONNX_NAMESPACE::TensorProto* k_tensor,
                                const ONNX_NAMESPACE::TensorProto* v_tensor,
                                bool is_matmul) {
  assert(nullptr != q_tensor);
  assert(nullptr != k_tensor);
  assert(nullptr != v_tensor);
  Initializer q_initializer(graph, *q_tensor, graph.ModelPath());
  Initializer k_initializer(graph, *k_tensor, graph.ModelPath());
  Initializer v_initializer(graph, *v_tensor, graph.ModelPath());
  auto data_type = q_tensor->data_type();

  ONNX_NAMESPACE::TensorProto initializer;
  initializer.set_name(graph.GenerateNodeArgName(is_matmul ? "qkv_weights" : "qkv_bias"));
  // Shape of weights for MatMul is (hidden_size, 3 * hidden_size)
  // Shape of weights for Add bias is (3 * hidden_size)
  if (is_matmul) {
    initializer.add_dims(hidden_size);
  }
  initializer.add_dims(3 * hidden_size);
  initializer.set_data_type(data_type);
  const int64_t element_count = 3 * hidden_size * (is_matmul ? hidden_size : 1);

  if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    const float* q_weight = q_initializer.data<float>();
    const float* k_weight = k_initializer.data<float>();
    const float* v_weight = v_initializer.data<float>();
    std::vector<float> result;
    result.reserve(gsl::narrow<size_t>(element_count));
    if (is_matmul) {
      optimizer_utils::MergeMatMulWeightsByRow<float>(q_weight, k_weight, v_weight, result, hidden_size, hidden_size, hidden_size);
    } else {
      optimizer_utils::MergeWeights1d<float>(q_weight, k_weight, v_weight, result, hidden_size, hidden_size);
    }
    utils::SetRawDataInTensorProto(initializer, result.data(), gsl::narrow<size_t>(element_count) * sizeof(float));
  } else {  // data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT16
    const MLFloat16* q_weight = q_initializer.data<MLFloat16>();
    const MLFloat16* k_weight = k_initializer.data<MLFloat16>();
    const MLFloat16* v_weight = v_initializer.data<MLFloat16>();
    std::vector<MLFloat16> result;
    result.reserve(gsl::narrow<size_t>(element_count));
    if (is_matmul) {
      optimizer_utils::MergeMatMulWeightsByRow<MLFloat16>(q_weight, k_weight, v_weight, result, hidden_size, hidden_size, hidden_size);
    } else {
      optimizer_utils::MergeWeights1d<MLFloat16>(q_weight, k_weight, v_weight, result, hidden_size, hidden_size);
    }
    utils::SetRawDataInTensorProto(initializer, result.data(), gsl::narrow<size_t>(element_count) * sizeof(MLFloat16));
  }

  return graph_utils::AddInitializerWithOrtValue(graph, initializer);
}

static NodeArg* ConvertMaskToInt32(Graph& graph, NodeArg* mask_input, ProviderType provider_type,
                                   const logging::Logger& logger) {
  // Validate mask input shape (batch_size, sequence_length) and data type.
  // Note that batch_size and sequence_length could be symbolic.
  const TensorShapeProto* mask_shape = mask_input->Shape();
  if (mask_shape == nullptr || mask_shape->dim_size() != 2 || mask_input->Type() == nullptr) {
    DEBUG_LOG("Mask shape is unknown or not 2D, or data type unknown");
    return nullptr;
  }

  auto data_type = mask_input->TypeAsProto()->tensor_type().elem_type();
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT64 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32 &&
      data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    DEBUG_LOG("Mask data type is not int32 or int64 or float32");
    return nullptr;
  }

  NodeArg* mask_int32 = mask_input;
  if (data_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    NodeArg& cast_int32 = AttentionFusionHelper::CastMaskToInt32(graph, mask_input, provider_type);
    mask_int32 = &cast_int32;
  }

  return mask_int32;
}

static NodeArg* ConvertMaskToInt32(
    Graph& graph,
    NodeArg* mask_input,
    std::map<std::string, NodeArg*>& mask_int32_map,
    ProviderType provider_type,
    const logging::Logger& logger) {
  // Lookup in map, and return the converted mask.
  auto search = mask_int32_map.find(mask_input->Name());
  if (search != mask_int32_map.end()) {
    return search->second;
  }

  NodeArg* output = ConvertMaskToInt32(graph, mask_input, provider_type, logger);
  if (nullptr == output) {
    return nullptr;
  }

  // Add it to map for lookup later.
  mask_int32_map.insert(std::pair<std::string, NodeArg*>(mask_input->Name(), output));
  return output;
}

Status AttentionFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  // A map from mask input arg name to the one casted to int32
  std::map<std::string, NodeArg*> mask_int32_map;

  int fused_count = 0;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    if (TryFuseMobileClipMHA(node, graph, GetCompatibleExecutionProviders(), logger)) {
      fused_count++;
      modified = true;
      continue;
    }

    // Add node.GetOutputEdgesCount() == 5/6 for distilbert
    if ((node.GetOutputEdgesCount() >= 2 && node.GetOutputEdgesCount() <= 6) &&
        graph_utils::IsSupportedOptypeVersionAndDomain(node, "LayerNormalization", {1, 17}, kOnnxDomain) &&
        graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) &&
        node.InputDefs().size() > 2 && node.InputDefs()[2]->Exists()) {  // Bias is an optional input for LayerNorm
      // Get hidden size from layer norm bias tensor shape.
      const NodeArg& layer_norm_bias = *(node.InputDefs()[2]);
      if (!optimizer_utils::IsShapeKnownOnAllDims(layer_norm_bias, 1)) {
        DEBUG_LOG("shape of layer norm bias tensor not expected");
        continue;
      }
      int64_t hidden_size = layer_norm_bias.Shape()->dim(0).dim_value();

      const Node* add_node = nullptr;
      unsigned int add_count = 0;
      unsigned int matmul_count = 0;
      unsigned int shape_count = 0;
      unsigned int reshape_count = 0;
      for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
        if ((*it).OpType().compare("Add") == 0) {
          add_count++;
          add_node = &(*it);
        } else if ((*it).OpType().compare("MatMul") == 0) {
          matmul_count++;
        } else if ((*it).OpType().compare("Shape") == 0) {
          shape_count++;
        } else if ((*it).OpType().compare("Reshape") == 0) {
          reshape_count++;
        }
      }

      if (add_count == 1 && matmul_count == 3 && shape_count == node.GetOutputEdgesCount() - 4) {  // BERT or DistilBert
        if (AttentionFusion::FuseSubGraph(node, *add_node, graph, hidden_size, mask_int32_map, logger)) {
          fused_count++;
          modified = true;
        }
      } else if (reshape_count == 1 && (shape_count == 1 || shape_count == 3) &&
                 (static_cast<size_t>(reshape_count) + shape_count) == node.GetOutputEdgesCount()) {  // GPT
        if (AttentionFusionHelper::FuseGptAttention(node, graph, hidden_size, mask_int32_map, shape_count == 1,
                                                    logger)) {
          fused_count++;
          modified = true;
        }
      }
    }
  }

  if (fused_count > 0) {
    LOGS(logger, INFO) << "Total fused Attention node count: " << fused_count;
  }

  return Status::OK();
}

static bool FuseSubGraphQKImpl(Node& layer_norm,
                               Graph& graph,
                               std::vector<std::reference_wrapper<const Node>>& parent_path_nodes,
                               NodeArg* mask_input,
                               std::map<std::string, NodeArg*>& mask_int32_map,
                               std::vector<const Node::EdgeEnd*>& edges,
                               std::vector<NodeIndex>& nodes_to_remove,
                               int64_t hidden_size,
                               int64_t num_heads,
                               int64_t head_size,
                               const float mask_filter_value,
                               const logging::Logger& logger) {
  InlinedVector<std::reference_wrapper<const Node>> pivot_nodes;
  if (edges.size() == 2) {
    const Node& qk_div = (edges[0]->GetNode().OpType() == "Div") ? edges[0]->GetNode() : edges[1]->GetNode();
    const Node& qk_matmul = (edges[1]->GetNode().OpType() == "MatMul") ? edges[1]->GetNode() : edges[0]->GetNode();
    pivot_nodes.push_back(qk_matmul);
    pivot_nodes.push_back(qk_div);
  } else {
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> q_path{
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1}, kOnnxDomain}};
  if (!graph_utils::FindPath(edges[edges.size() - 1]->GetNode(), true, q_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  const Node& q_transpose = edges[0]->GetNode();
  const Node& q_reshape = edges[1]->GetNode();
  const Node& q_add = edges[2]->GetNode();
  const Node& q_matmul = edges[3]->GetNode();
  const Node& q_root = edges[4]->GetNode();
  if (q_root.Index() != layer_norm.Index()) {
    DEBUG_LOG("q root should be layer normalization");
    return false;
  }

  if (!AttentionFusionHelper::CheckNodesInPathQ(graph, pivot_nodes[1].get(),
                                                q_reshape, q_transpose, num_heads, head_size, logger)) {
    DEBUG_LOG("CheckNodesInPathQ returns false");
    return false;
  }

  if (!(ValidateAddBiasInitializer(graph, q_add, hidden_size) &&
        ValidateMatMulInitializer(graph, q_matmul, hidden_size))) {
    DEBUG_LOG("q_matmul and q_add shape not matched");
    return false;
  }

  std::vector<graph_utils::EdgeEndToMatch> k_path{
      {0, 1, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1, 17}, kOnnxDomain}};

  if (!graph_utils::FindPath(pivot_nodes[0].get(), true, k_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for k");
    return false;
  }

  const Node& k_transpose = edges[0]->GetNode();
  const Node& k_reshape = edges[1]->GetNode();
  const Node& k_add = edges[2]->GetNode();
  const Node& k_matmul = edges[3]->GetNode();
  const Node& k_root = edges[4]->GetNode();
  if (k_root.Index() != layer_norm.Index()) {
    DEBUG_LOG("k root is not layer norm");
    return false;
  }
  if (!AttentionFusionHelper::CheckNodesInPathK(graph, k_reshape, k_transpose, num_heads,
                                                head_size, /*transpose_optimized_pattern*/ false, logger)) {
    DEBUG_LOG("CheckNodesInPathK returns false");
    return false;
  }

  if (!(ValidateAddBiasInitializer(graph, k_add, hidden_size) &&
        ValidateMatMulInitializer(graph, k_matmul, hidden_size))) {
    DEBUG_LOG("k_matmul and k_add shape not matched");
    return false;
  }

  const Node& v_matmul = parent_path_nodes[6];
  // Load q, k and v weights
  const ONNX_NAMESPACE::TensorProto* q_weight_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* k_weight_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* v_weight_tensor = nullptr;
  if (!LoadQkvWeights(graph, q_matmul, k_matmul, v_matmul, q_weight_tensor, k_weight_tensor, v_weight_tensor)) {
    DEBUG_LOG("Failed to load Q, K and V weights, or data type is not float or float16.");
    return false;
  }

  const Node& v_add = parent_path_nodes[5];
  const ONNX_NAMESPACE::TensorProto* q_bias_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* k_bias_tensor = nullptr;
  const ONNX_NAMESPACE::TensorProto* v_bias_tensor = nullptr;
  if (!LoadQkvWeights(graph, q_add, k_add, v_add, q_bias_tensor, k_bias_tensor, v_bias_tensor)) {
    DEBUG_LOG("Failed to load Q, K and V bias tensors, or data type is not float or float16.");
    return false;
  }

  // Now everything is ready, we will start fusing subgraph.
  NodeArg* mask_int32 = ConvertMaskToInt32(graph, mask_input, mask_int32_map, layer_norm.GetExecutionProviderType(),
                                           logger);
  if (nullptr == mask_int32) {
    DEBUG_LOG("Failed to convert mask to int32");
    return false;
  }

  // Merge Q, K and V weights
  NodeArg& qkv_weights = MergeQkvWeights(graph, hidden_size, q_weight_tensor, k_weight_tensor, v_weight_tensor, true);
  NodeArg& qkv_bias = MergeQkvWeights(graph, hidden_size, q_bias_tensor, k_bias_tensor, v_bias_tensor, false);
  // Create Attention Node.
  const Node& reshape = parent_path_nodes[0];
  const std::array input_defs{layer_norm.MutableOutputDefs()[0], &qkv_weights, &qkv_bias, mask_int32};
  const std::array output_defs{graph.GetNode(reshape.Index())->MutableOutputDefs()[0]};
  Node& attention_node = graph.AddNode(
      graph.GenerateNodeName("Attention"),
      "Attention",
      "Fused Attention subgraphs ",
      input_defs,
      output_defs,
      nullptr,
      kMSDomain);
  attention_node.AddAttribute("num_heads", num_heads);
  attention_node.AddAttribute("mask_filter_value", mask_filter_value);

  // Assign provider to this new node.
  attention_node.SetExecutionProviderType(layer_norm.GetExecutionProviderType());

  // Remove nodes that are not used anymore.
  parent_path_nodes.insert(parent_path_nodes.end(), pivot_nodes.begin(), pivot_nodes.end());

  std::transform(parent_path_nodes.begin(),
                 parent_path_nodes.end(),
                 std::back_inserter(nodes_to_remove),
                 [](std::reference_wrapper<const Node> node_ref_wrapper) -> NodeIndex {
                   return node_ref_wrapper.get().Index();
                 });

  std::vector<NodeIndex> nodes_to_remove_temp{
      q_transpose.Index(),
      q_reshape.Index(),
      q_add.Index(),
      q_matmul.Index(),
      k_transpose.Index(),
      k_reshape.Index(),
      k_add.Index(),
      k_matmul.Index()};

  nodes_to_remove.insert(nodes_to_remove.end(), nodes_to_remove_temp.begin(), nodes_to_remove_temp.end());

  return true;
}

static bool FuseSubGraphQK(Node& layer_norm,
                           Graph& graph,
                           AttentionFusionHelper::AttentionMaskNodes& mask_nodes,
                           NodeArg* mask_input,
                           std::vector<std::reference_wrapper<const Node>>& parent_path_nodes,
                           int64_t hidden_size,
                           int64_t num_heads,
                           int64_t head_size,
                           std::map<std::string, NodeArg*>& mask_int32_map,
                           const logging::Logger& logger) {
  // path to q
  std::vector<graph_utils::EdgeEndToMatch> q_varience_path{
      {0, 0, "Div", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(*(mask_nodes.add), true, q_varience_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  std::vector<NodeIndex> nodes_to_remove;
  if (!FuseSubGraphQKImpl(layer_norm, graph, parent_path_nodes,
                          mask_input, mask_int32_map, edges, nodes_to_remove, hidden_size,
                          num_heads, head_size, mask_nodes.mask_filter_value, logger)) {
    return false;
  }

  AttentionFusionHelper::SetMaskNodesToRemove(graph, mask_nodes, nodes_to_remove);

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  DEBUG_LOG("Fused an attention node.");

  return true;
}

/** DistilBert's attention is a bit different here
@remark add_after_layer_norm is the Add node in the bottom of sub-graph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               B and S could be symbolic.
    Graph before Fusion (q_, k_, v_, qk_, qkv_ and mask_ prefix is added before Operator type):
                  [Input](BxSxW)
                        |
                LayerNormalization ---------------------------------------------
            /       |        |     \     [Weights](WxW)                         |
           /        |        |      \    /                                    Shape
          |   q_MatMul    k_MatMul  v_MatMul  [Bias](W)                     /       \
          |         |        |        |   /                                /         \
          |     q_Add     k_Add     v_Add     [Shape=0,-1,N,H]    Gather(indices:0)   Gather(indices:1)
          |         |        |        |      /                          |               |
          | q_Reshape   k_Reshape   v_Reshape                           |               |
          |         |        |        |                             Unsqueeze        Unsqueeze
          |q_Transpose  k_Transpose v_Transpose                         |   \          /
          |  (0,2,1,3)  (0,2,3,1)    (perm=0,2,1,3)                     |    \        /
          |         |       |         |                                 |     \      /
          |        q_Div   /                                            |      Concat [_, 1, 1, _]
          |           |  /            |                                 |         |
          |        qk_MatMul          |                                 |         |         --------- AttentionMask
          |           |    \          |                                 |         |        /
          |           |      \        |                                 |         |       /
          |           |     Shape     |                                 |         |     Equal (B = 0)
          |           |       |       |                                 |         |     /
          |           |    Expand-----|-----------------------------------------Reshape
          |            \   /          |                                 |
          |         Where             /                                 |
          |             |           /                                   |
          |          Softmax       /                                    |
          |             \         /                                     |
          |              \       /                                      |
          |            qkv_MatMul                                       |
          |                   |                                         |
          |                Transpose (perm=0,2,1,3)                     |
          |                   |                                         |
          |                Reshape-----------------------------------Concat [Shape=_,-1,W]
          |                   |
          |                 MatMul----[Weights](WxW)
          |                   |
          |                  Add----[Bias](W)
          +-------------------|---+
                              |   |
                               Add

A change compared with first version attention fusion for distilbert:
There were two Shape nodes after LayerNormalization, gets fused into one before ORT 1.5.0 release

However, the first version of attention fusion for distilbert is still supported for now.
*/
static bool FuseSubGraphQKDistilBert(Node& layer_norm,
                                     Graph& graph,
                                     AttentionFusionHelper::AttentionMaskNodesDistilBert& mask_nodes,
                                     NodeArg* mask_input,
                                     std::vector<std::reference_wrapper<const Node>>& parent_path_nodes,
                                     int64_t hidden_size,
                                     int64_t num_heads,
                                     int64_t head_size,
                                     std::map<std::string, NodeArg*>& mask_int32_map,
                                     const logging::Logger& logger) {
  // path to q
  std::vector<graph_utils::EdgeEndToMatch> q_varience_path{
      {0, 2, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "Div", {7, 13}, kOnnxDomain}};
  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(*(mask_nodes.where), true, q_varience_path, edges, logger)) {
    DEBUG_LOG("Failed to find path for q");
    return false;
  }

  std::vector<NodeIndex> nodes_to_remove;
  if (!FuseSubGraphQKImpl(layer_norm, graph, parent_path_nodes,
                          mask_input, mask_int32_map, edges, nodes_to_remove, hidden_size,
                          num_heads, head_size, mask_nodes.mask_filter_value, logger)) {
    return false;
  }

  const Node& reshape_1 = parent_path_nodes[0];
  const Node& reshape_2 = *(mask_nodes.reshape);

  const Node* p_concat_1 = graph_utils::GetInputNode(reshape_1, 1);
  const Node* p_concat_2 = graph_utils::GetInputNode(reshape_2, 1);
  if (p_concat_1 != nullptr && p_concat_2 != nullptr) {
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *p_concat_1);
    graph_utils::RemoveNodesWithOneOutputBottomUp(graph, *p_concat_2);
  } else {
    return false;
  }

  AttentionFusionHelper::SetMaskNodesToRemove(graph, mask_nodes, nodes_to_remove);

  for (const auto& node_index : nodes_to_remove) {
    Node* node = graph.GetNode(node_index);
    graph_utils::RemoveNodeOutputEdges(graph, *node);
    graph.RemoveNode(node->Index());
  }

  DEBUG_LOG("Fused an attention node.");

  return true;
}

/** Fuse Attention SubGraph.
@remark add_after_layer_norm is the Add node in the bottom of sub-graph.
 Abbreviatios: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               B and S could be symbolic.
    Graph before Fusion (q_, k_, v_, qk_, qkv_ and mask_ prefix is added before Operator type):
                  [Input](BxSxW)
                        |
                LayerNormalization
            /       |        |     \     [Weights](WxW)
           /        |        |      \    /
          |   q_MatMul    k_MatMul  v_MatMul  [Bias](W)
          |         |        |        |   /
          |     q_Add     k_Add     v_Add     [Shape=0,0,N,H]
          |         |        |        |      /
          | q_Reshape   k_Reshape   v_Reshape                [Mask] (BxS)
          |         |        |        |                          |
          |q_Transpose  k_Transpose v_Transpose            mask_Unsqueeze(axes=1)
          |  (0,2,1,3)  (0,2,3,1)    (perm=0,2,1,3)              |
          |         \       /         |                    mask_Unsqueeze(axes=2)
          |      qk_MatMul            |                          |
          |           |    [B=2]      |              ([A=1.0] mask_Cast(to=1))
          |           |   /           |                   \     /
          |        qk_Div             |                 mask_Sub   [B=-10000.0 or value of mask_filter_value]
          |            \              |                        \   /
          |       mask_Add <-------- /---------------------mask_Mul
          |             |           /
          |          Softmax       /
          |             \         /
          |              \       /
          |            qkv_MatMul
          |                   |
          |                Transpose (perm=0,2,1,3)
          |                   |
          |                Reshape---[shape=0,0,W]
          |                   |
          |                 MatMul----[Weights](WxW)
          |                   |
          |                  Add----[Bias](W)
          +-------------------|---+
                              |   |
                               Add

After Fusion:
  LayerNormalization  [Weights](Wx3W)   Mask
      |        \      /   [Bias](3W)     |
      |         \    /   /               |
      |         Attention <------------ReduceSum
      \          |
       \        MatMul
        \        |
         \      Add
          +------|---+
                 |   |
                  Add
*/
bool AttentionFusion::FuseSubGraph(Node& layer_norm,
                                   const Node& add_after_layer_norm,
                                   Graph& graph,
                                   int64_t hidden_size,
                                   std::map<std::string, NodeArg*>& mask_int32_map,
                                   const logging::Logger& logger) {
  std::vector<graph_utils::EdgeEndToMatch> parent_path{
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 1, "Transpose", {1, 13}, kOnnxDomain},
      {0, 0, "Reshape", {5, 13}, kOnnxDomain},
      {0, 0, "Add", {7, 13}, kOnnxDomain},
      {0, 0, "MatMul", {1, 9, 13}, kOnnxDomain},
      {0, 0, "LayerNormalization", {1, 17}, kOnnxDomain}};

  std::vector<const Node::EdgeEnd*> edges;
  if (!graph_utils::FindPath(add_after_layer_norm, true, parent_path, edges, logger)) {
    DEBUG_LOG("Faild to find path v");
    return false;
  }

  const Node& add = edges[0]->GetNode();
  const Node& matmul = edges[1]->GetNode();
  const Node& reshape = edges[2]->GetNode();
  const Node& transpose = edges[3]->GetNode();
  const Node& qkv_matmul = edges[4]->GetNode();
  const Node& v_transpose = edges[5]->GetNode();
  const Node& v_reshape = edges[6]->GetNode();
  const Node& v_add = edges[7]->GetNode();
  const Node& v_matmul = edges[8]->GetNode();
  const Node& v_root = edges[9]->GetNode();
  if (v_root.Index() != layer_norm.Index()) {
    return false;
  }

  if (!optimizer_utils::CheckOutputEdges(graph, v_add, 1) ||
      !optimizer_utils::CheckOutputEdges(graph, v_matmul, 1)) {
    DEBUG_LOG("Output edge count not expected for Add or MatMul in path v");
    return false;
  }

  int64_t num_heads = 0;          // will be updated in CheckNodesInPathV
  int64_t head_size = 0;          // will be updated in CheckNodesInPathV
  NodeIndex record_node_idx = 0;  // will be updated in CheckNodesInPathV if it's distilbert model
  if (!AttentionFusionHelper::CheckNodesInPathV(graph, reshape, transpose,
                                                qkv_matmul, v_transpose, v_reshape, num_heads,
                                                head_size, hidden_size, record_node_idx, logger)) {
    DEBUG_LOG("CheckNodesInPathV return false");
    return false;
  }

  // Validate the input shape of MatMul and Add according to hidden_size.
  if (!(ValidateAddBiasInitializer(graph, add, hidden_size) &&
        ValidateMatMulInitializer(graph, matmul, hidden_size) &&
        ValidateAddBiasInitializer(graph, v_add, hidden_size) &&
        ValidateMatMulInitializer(graph, v_matmul, hidden_size))) {
    DEBUG_LOG("Failed in match v_matmul and v_add input shape");
    return false;
  }

  // store parent path
  std::vector<std::reference_wrapper<const Node>> parent_path_nodes{
      reshape, transpose, qkv_matmul, v_transpose, v_reshape, v_add, v_matmul};

  // Find mask nodes: Unsqueeze -> Unsqueeze -> (Cast) -> Sub -> Mul -> Add -> Softmax --> [MatMul]
  // The "Cast" node in parentheses is optional.
  AttentionFusionHelper::AttentionMaskNodes mask_nodes;
  AttentionFusionHelper::AttentionMaskNodesDistilBert mask_nodes_distilbert;

  if (AttentionFusionHelper::MatchInputMaskSubgraph(graph, qkv_matmul, mask_nodes, logger, false)) {
    NodeArg* mask_input = graph.GetNode(mask_nodes.unsqueeze_1->Index())->MutableInputDefs()[0];
    return FuseSubGraphQK(layer_norm, graph, mask_nodes, mask_input,
                          parent_path_nodes, hidden_size, num_heads, head_size, mask_int32_map, logger);
  } else if (AttentionFusionHelper::MatchInputMaskSubgraph(graph, layer_norm, qkv_matmul,
                                                           mask_nodes_distilbert, record_node_idx, logger)) {
    NodeArg* mask_input = graph.GetNode(mask_nodes_distilbert.equal->Index())->MutableInputDefs()[0];
    return FuseSubGraphQKDistilBert(layer_norm, graph, mask_nodes_distilbert, mask_input,
                                    parent_path_nodes, hidden_size, num_heads, head_size, mask_int32_map, logger);
  } else {
    DEBUG_LOG("Failed in match input mask subgraph");
    return false;
  }

  return true;
}

}  // namespace onnxruntime
