// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/log1p_fusion.h"

#include <algorithm>
#include <array>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace {

bool IsSupportedType(const NodeArg& input, bool webgpu) {
  const auto* type = input.TypeAsProto();
  if (!type || !type->has_tensor_type()) return false;
  const int dtype = type->tensor_type().elem_type();
  return dtype == ONNX_NAMESPACE::TensorProto::FLOAT || dtype == ONNX_NAMESPACE::TensorProto::FLOAT16 ||
         (!webgpu && (dtype == ONNX_NAMESPACE::TensorProto::DOUBLE ||
                      dtype == ONNX_NAMESPACE::TensorProto::BFLOAT16));
}

bool IsOne(const Initializer& value) {
  if (value.size() != 1) return false;
  switch (value.data_type()) {
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      return *value.data<float>() == 1.0f;
    case ONNX_NAMESPACE::TensorProto::DOUBLE:
      return *value.data<double>() == 1.0;
    case ONNX_NAMESPACE::TensorProto::FLOAT16:
      return value.data<MLFloat16>()->ToFloat() == 1.0f;
    case ONNX_NAMESPACE::TensorProto::BFLOAT16:
      return value.data<BFloat16>()->ToFloat() == 1.0f;
    default:
      return false;
  }
}

}  // namespace

Status Log1pFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  const GraphViewer viewer(graph);
  for (const auto index : viewer.GetNodesInTopologicalOrder()) {
    Node* add = graph.GetNode(index);
    if (!add) continue;
    ORT_RETURN_IF_ERROR(Recurse(*add, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*add, "Add", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(*add, GetCompatibleExecutionProviders()) ||
        add->GetOutputEdgesCount() != 1 || graph.NodeProducesGraphOutput(*add)) {
      continue;
    }

    Node& log = *graph.GetNode(add->OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(log, "Log", {6, 13}) ||
        log.GetExecutionProviderType() != add->GetExecutionProviderType()) {
      continue;
    }

    for (size_t constant_index = 0; constant_index != 2; ++constant_index) {
      NodeArg* input = add->MutableInputDefs()[1 - constant_index];
      if (!IsSupportedType(*input, add->GetExecutionProviderType() == kWebGpuExecutionProvider)) continue;
      const auto* proto = graph_utils::GetConstantInitializer(graph, add->InputDefs()[constant_index]->Name());
      if (!proto || proto->data_type() != input->TypeAsProto()->tensor_type().elem_type()) continue;

      // Only scalar/singleton constants whose broadcasting cannot expand the input.
      if (proto->dims_size() != 0 &&
          (!input->Shape() || proto->dims_size() > input->Shape()->dim_size())) continue;
      if (!std::all_of(proto->dims().begin(), proto->dims().end(), [](int64_t dim) { return dim == 1; })) continue;
      const Initializer value(graph, *proto, graph.ModelPath(), true);
      if (!IsOne(value)) continue;

      Node& fused = graph.AddNode(graph.GenerateNodeName("Log1p"), "Log1p",
                                  "Log(Add(x, 1)) with the original input", {input},
                                  log.MutableOutputDefs(), nullptr, kMSDomain);
      fused.SetExecutionProviderType(add->GetExecutionProviderType());
      const std::array<std::reference_wrapper<Node>, 2> nodes{*add, log};
      graph_utils::FinalizeNodeFusion(graph, nodes, fused);
      modified = true;
      break;
    }
  }
  return Status::OK();
}

}  // namespace onnxruntime
