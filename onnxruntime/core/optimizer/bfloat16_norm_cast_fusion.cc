// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/bfloat16_norm_cast_fusion.h"

#include <cmath>
#include <cstring>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace {

int32_t ElementType(const NodeArg& arg) {
  const auto* type = arg.TypeAsProto();
  return type != nullptr && type->has_tensor_type()
             ? type->tensor_type().elem_type()
             : ONNX_NAMESPACE::TensorProto::UNDEFINED;
}

bool IsCast(const Node& node, int32_t from, int32_t to) {
  if (node.OpType() != "Cast" || !node.Domain().empty() ||
      node.GetExecutionProviderType() != kCudaExecutionProvider ||
      node.InputDefs().size() != 1 || node.OutputDefs().size() != 1) {
    return false;
  }
  const auto* attr = graph_utils::GetNodeAttribute(node, "to");
  return attr != nullptr && attr->i() == to &&
         ElementType(*node.InputDefs()[0]) == from && ElementType(*node.OutputDefs()[0]) == to;
}

NodeArg* LosslessBFloat16Scale(Graph& graph, const NodeArg& scale) {
  const auto* tensor = graph_utils::GetConstantInitializer(graph, scale.Name());
  if (tensor == nullptr || tensor->data_type() != ONNX_NAMESPACE::TensorProto::FLOAT) {
    return nullptr;
  }

  // Use the graph-aware loader for file-backed and in-memory external initializers.
  Initializer original(graph, *tensor, graph.ModelPath(), true);
  const auto name = graph.GenerateNodeArgName(scale.Name() + "_bf16");
  Initializer converted(ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16, name, original.dims());
  for (size_t i = 0; i < original.size(); ++i) {
    const float value = original.data<float>()[i];
    const BFloat16 bf16(value);
    const float restored = bf16.ToFloat();
    if (!std::isfinite(value) || std::memcmp(&value, &restored, sizeof(float)) != 0) {
      return nullptr;
    }
    converted.data<BFloat16>()[i] = bf16;
  }

  ONNX_NAMESPACE::TensorProto converted_tensor;
  converted.ToProto(converted_tensor);
  graph.AddInitializedTensor(converted_tensor);
  auto type = *scale.TypeAsProto();
  type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto::BFLOAT16);
  return &graph.GetOrCreateNodeArg(name, &type);
}

}  // namespace

Status BFloat16NormCastFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                         const logging::Logger& logger) const {
  GraphViewer viewer(graph);
  for (const auto index : viewer.GetNodesInTopologicalOrder()) {
    auto* node = graph.GetNode(index);
    if (node == nullptr) {
      continue;
    }
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*node, "SimplifiedLayerNormalization", {1}) ||
        !graph_utils::IsSupportedProvider(*node, GetCompatibleExecutionProviders()) ||
        node->InputDefs().size() != 2 || node->OutputDefs().size() != 1 ||
        ElementType(*node->InputDefs()[1]) != ONNX_NAMESPACE::TensorProto::FLOAT ||
        ElementType(*node->OutputDefs()[0]) != ONNX_NAMESPACE::TensorProto::FLOAT) {
      continue;
    }
    const auto input_type = ElementType(*node->InputDefs()[0]);
    const auto* stash_type = graph_utils::GetNodeAttribute(*node, "stash_type");
    if ((input_type != ONNX_NAMESPACE::TensorProto::FLOAT &&
         input_type != ONNX_NAMESPACE::TensorProto::BFLOAT16) ||
        (stash_type != nullptr && stash_type->i() != ONNX_NAMESPACE::TensorProto::FLOAT)) {
      continue;
    }

    auto* leading = graph.GetMutableProducerNode(node->InputDefs()[0]->Name());
    if (leading != nullptr &&
        IsCast(*leading, ONNX_NAMESPACE::TensorProto::BFLOAT16, ONNX_NAMESPACE::TensorProto::FLOAT) &&
        leading->GetOutputEdgesCount() == 1 && graph_utils::CanRemoveNode(graph, *leading, logger)) {
      // Remove old edges before changing dtype; generic node bypass assumes identical input/output types.
      const auto input_edges = graph_utils::GraphEdge::GetNodeInputEdges(*leading);
      auto* input = leading->MutableInputDefs()[0];
      const auto leading_index = leading->Index();
      graph_utils::RemoveNodeOutputEdges(graph, *leading);
      graph.RemoveNode(leading_index);
      graph_utils::ReplaceNodeInput(*node, 0, *input);
      for (const auto& input_edge : input_edges) {
        graph.AddEdge(input_edge.src_node, node->Index(), input_edge.src_arg_index, 0);
      }
      graph.NotifyNodesRemoved(gsl::span<const NodeIndex>{&leading_index, 1});
      modified = true;
    }

    if (node->GetOutputEdgesCount() != 1 || graph_utils::IsGraphOutput(graph, node->OutputDefs()[0])) {
      continue;
    }
    const auto& edge = *node->OutputEdgesBegin();
    auto* trailing = graph.GetNode(edge.GetNode().Index());
    if (edge.GetSrcArgIndex() != 0 || edge.GetDstArgIndex() != 0 || trailing == nullptr ||
        !IsCast(*trailing, ONNX_NAMESPACE::TensorProto::FLOAT, ONNX_NAMESPACE::TensorProto::BFLOAT16)) {
      continue;
    }
    auto* scale = LosslessBFloat16Scale(graph, *node->InputDefs()[1]);
    if (scale == nullptr) {
      continue;
    }
    graph_utils::ReplaceNodeInput(*node, 1, *scale);
    graph_utils::FinalizeNodeFusion(graph, *node, *trailing);
    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
