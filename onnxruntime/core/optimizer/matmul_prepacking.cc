// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "matmul_prepacking.h"
#include "core/graph/graph_utils.h"

#include "core/mlas/inc/mlas.h"

#include "onnx/defs/shape_inference.h"

namespace onnxruntime {

template<typename TIntegral>
void AddIntegralAttr(const std::string& name, TIntegral value, NodeAttributes& attributes) {
  ONNX_NAMESPACE::AttributeProto proto;
  proto.set_name(name);
  proto.set_type(onnx::AttributeProto_AttributeType_INT);
  proto.set_i(static_cast<int64_t>(value));
  attributes.emplace(name, std::move(proto));
}

NodeAttributes GemmParamsToNodeAttributes(const MLAS_GEMM_PARAMETERS& params) {
  NodeAttributes attributes;
  AddIntegralAttr("N", params.N, attributes);
  AddIntegralAttr("K", params.K, attributes);
  AddIntegralAttr("PackedSize", params.PackedSize, attributes);
  AddIntegralAttr("PackedStrideN", params.PackedStrideN, attributes);
  AddIntegralAttr("PackedStrideK", params.PackedStrideK, attributes);

  return attributes;
}

template<typename Impl, typename TIntegral>
Status GetIntegralAttr(const OpNodeProtoHelper<Impl>& node_context, const std::string& name, TIntegral* enum_value) {
  int64_t integer_representation;
  ORT_RETURN_IF_ERROR(node_context.GetAttr(name, &integer_representation));
  *enum_value = static_cast<TIntegral>(integer_representation);
  return Status::OK();
}


template<typename Impl>
Status GemmParamsFromNodeAttributes(const OpNodeProtoHelper<Impl>& node_context, MLAS_GEMM_PARAMETERS& params) {
  ORT_RETURN_IF_ERROR(GetIntegralAttr(node_context, "K", &params.K));
  ORT_RETURN_IF_ERROR(GetIntegralAttr(node_context, "N", &params.N));
  ORT_RETURN_IF_ERROR(GetIntegralAttr(node_context, "PackedSize", &params.PackedSize));
  ORT_RETURN_IF_ERROR(GetIntegralAttr(node_context, "PackedStrideN", &params.PackedStrideN));
  ORT_RETURN_IF_ERROR(GetIntegralAttr(node_context, "PackedStrideK", &params.PackedStrideK));

  return Status::OK();
}

template
Status GemmParamsFromNodeAttributes(const OpNodeProtoHelper<ProtoHelperNodeContext>& node_context, MLAS_GEMM_PARAMETERS& params);

template
Status GemmParamsFromNodeAttributes(const OpNodeProtoHelper<ONNX_NAMESPACE::InferenceContext>& node_context, MLAS_GEMM_PARAMETERS& params);

bool MatMulPrepacking::SatisfyCondition(const Graph&, const Node& node, const logging::Logger&) const {
  return node.Domain() == kOnnxDomain && node.GetExecutionProviderType() == kCpuExecutionProvider;
}

Status MatMulPrepacking::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger& /*logger*/) const {
  rule_effect = RewriteRuleEffect::kNone;
  NodeArg* A = node.MutableInputDefs()[0];
  NodeArg* B = node.MutableInputDefs()[1];

  if (!graph_utils::NodeArgIsConstant(graph, *B)) {
    return Status::OK();
  }

  const ONNX_NAMESPACE::TensorShapeProto* Ashape = A->Shape();
  const ONNX_NAMESPACE::TensorShapeProto* Bshape = B->Shape();
  if (Bshape == nullptr) {
    return Status::OK();
  }

  if (Bshape->dim_size() < 2) {
    return Status::OK();
  }

  std::size_t M;
  if (Ashape == nullptr) {
    M = 0; // dynamic shape
  } else if (Ashape->dim_size() < 2) {
    M = 1;
  } else if (Ashape->dim(Ashape->dim_size() - 2).has_dim_value()) {
    M = static_cast<size_t>(Ashape->dim(Ashape->dim_size() - 2).dim_value());
    if (M == 0) {
      return Status::OK(); // MatMul of empty tensor, nothing to optimize here.
    }
  } else {
    M = 0; // symbolic dimension
  }

  const auto K = static_cast<size_t>(Bshape->dim(Bshape->dim_size() - 2).dim_value());
  const auto N = static_cast<size_t>(Bshape->dim(Bshape->dim_size() - 1).dim_value());

  MLAS_GEMM_PARAMETERS gemm_params = MlasGemmPrepare(CblasNoTrans, CblasNoTrans, M, N, K, max_num_threads_, true);
  if (!gemm_params.PrePackedBPossiblyUsed) {
    return Status::OK();
  }

  NodeAttributes attributes = GemmParamsToNodeAttributes(gemm_params);

  NodeArg& Bprepack_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("Bprepack_output"), nullptr);

  Node& Bprepack = graph.AddNode(graph.GenerateNodeName("Bprepack " + node.Name()),
    "PackForGemm", "Pack constant B matrix in " + node.Name(), {B} ,
    {&Bprepack_output}, &attributes, kOnnxRuntimeDomain);
  Bprepack.SetExecutionProviderType(kCpuExecutionProvider);

  std::vector<NodeArg*> matmul_inputs{A, &Bprepack_output};
  if (gemm_params.OriginalBPossiblyUsed) {
    matmul_inputs.push_back(B);
  }

  Node& matmul_prepacked = graph.AddNode(graph.GenerateNodeName("MatMulPrepacked " + node.Name()),
    "MatMulPrepacked", "Prepacked MatMul in " + node.Name(), matmul_inputs,
    node.MutableOutputDefs(), &attributes, kOnnxRuntimeDomain);
  matmul_prepacked.SetExecutionProviderType(kCpuExecutionProvider);

  graph_utils::ReplaceDownstreamNodeInput(graph, node, 0, matmul_prepacked, 0);
  graph.RemoveNode(node.Index());
  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  return Status::OK();
}

}

