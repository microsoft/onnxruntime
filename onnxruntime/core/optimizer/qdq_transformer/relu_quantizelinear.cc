// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/qdq_transformer/relu_quantizelinear.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

bool ReluQuantFusion::SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger& /*logger*/) const {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Relu", {6, 13, 14}) ||
      !graph_utils::IsSupportedProvider(node, {kCpuExecutionProvider}) ||
      !optimizer_utils::CheckOutputEdges(graph, node, 1)) {
    return false;
  }

  // if Relu is followed by QuantizeLinear, it can be fused into QuantizeLinear potentially
  const auto& next_node = *node.OutputNodesBegin();
  if (!graph_utils::IsSupportedProvider(next_node, {kCpuExecutionProvider}) ||
      !QDQ::MatchQNode(next_node)) {
    return false;
  }

  return true;
}

Status ReluQuantFusion::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect, const logging::Logger&) const {
  Node& q_node = *graph.GetNode(node.OutputNodesBegin()->Index());

  std::vector<NodeArg*>& q_input_defs = q_node.MutableInputDefs();

  constexpr size_t q_input_cnt_required = 3;
  if (q_input_defs.size() != q_input_cnt_required) {
    std::cout << __FILE__ << ":" << __LINE__ << " cannot fuse" << std::endl;
    return Status::OK();
  }

  constexpr size_t s_idx = 1;
  constexpr size_t zp_idx = 2;

  const ONNX_NAMESPACE::TensorProto* s_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *q_input_defs[s_idx]) ||
      !graph.GetInitializedTensor(q_input_defs[s_idx]->Name(), s_tensor_proto)) {
    std::cout << __FILE__ << ":" << __LINE__ << " cannot fuse" << std::endl;
    return Status::OK();
  }
  Initializer s_init(*s_tensor_proto, graph.ModelPath());
  const float scale = s_init.data<float>()[0];

  const ONNX_NAMESPACE::TensorProto* zp_tensor_proto = nullptr;
  if (!graph_utils::NodeArgIsConstant(graph, *q_input_defs[zp_idx]) ||
      !graph.GetInitializedTensor(q_input_defs[zp_idx]->Name(), zp_tensor_proto)) {
    std::cout << __FILE__ << ":" << __LINE__ << " cannot fuse" << std::endl;
    return Status::OK();
  }
  Initializer zp_init(*zp_tensor_proto, graph.ModelPath());

  if (zp_init.size() != 1) {
    std::cout << __FILE__ << ":" << __LINE__ << " cannot fuse" << std::endl;
    return Status::OK();
  }

  auto can_fuse = [&]() {
    switch (zp_init.data_type()) {
      case ONNX_NAMESPACE::TensorProto_DataType_INT8: {
        auto zero_point = zp_init.data<int8_t>()[0];
        std::cout << __FILE__ << ":" << __LINE__ << " scale:" << scale << ", zp:" << int(zero_point) << std::endl;
        return QDQ::QuantizeDomain<int8_t>::MinUpper(scale, zero_point) >= 0;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT8: {
        auto zero_point = zp_init.data<uint8_t>()[0];
        std::cout << __FILE__ << ":" << __LINE__ << " scale:" << scale << ", zp:" << int(zero_point) << std::endl;
        return QDQ::QuantizeDomain<uint8_t>::MinUpper(scale, zero_point) >= 0;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_INT16: {
        auto zero_point = zp_init.data<int16_t>()[0];
        std::cout << __FILE__ << ":" << __LINE__ << " scale:" << scale << ", zp:" << int(zero_point) << std::endl;
        return QDQ::QuantizeDomain<int16_t>::MinUpper(scale, zero_point) >= 0;
      }
      case ONNX_NAMESPACE::TensorProto_DataType_UINT16: {
        auto zero_point = zp_init.data<uint16_t>()[0];
        std::cout << __FILE__ << ":" << __LINE__ << " scale:" << scale << ", zp:" << int(zero_point) << std::endl;
        return QDQ::QuantizeDomain<uint16_t>::MinUpper(scale, zero_point) >= 0;
      }
      default:
        return false;
    }
  };
  if (!can_fuse()) {
    std::cout << __FILE__ << ":" << __LINE__ << " cannot fuse" << std::endl;
    return Status::OK();
  }

  if (graph_utils::RemoveNode(graph, node)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  std::cout << __FILE__ << ":" << __LINE__ << " fused" << std::endl;
  return Status::OK();
}
}  // namespace onnxruntime
