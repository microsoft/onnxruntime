// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/resize_argmax_fusion.h"

#include <cmath>
#include <limits>

#include "core/graph/graph_utils.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_node_proto_helper.h"

namespace onnxruntime {
namespace {

bool CanFuseResize(const Graph& graph, const Node& node) {
  const auto* shape = node.InputDefs()[0]->Shape();
  if (!shape || shape->dim_size() != 4 ||
      node.InputDefs()[0]->TypeAsProto()->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    return false;
  }
  const ProtoHelperNodeContext ctx(node);
  const OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
  const auto coordinates = attrs.GetAttrOrDefault("coordinate_transformation_mode", std::string("half_pixel"));
  if (attrs.GetAttrOrDefault("mode", std::string("nearest")) != "linear" ||
      (coordinates != "half_pixel" && coordinates != "align_corners" &&
       coordinates != "asymmetric" && coordinates != "pytorch_half_pixel") ||
      attrs.GetAttrOrDefault("antialias", int64_t{0}) != 0 || attrs.GetAttrOrDefault("exclude_outside", int64_t{0}) != 0 ||
      attrs.GetAttrOrDefault("keep_aspect_ratio_policy", std::string("stretch")) != "stretch" ||
      graph_utils::GetNodeAttribute(node, "axes") != nullptr) {
    return false;
  }

  // The shared Resize coordinate table uses 32-bit indices and row offsets.
  constexpr int64_t limit = std::numeric_limits<int32_t>::max();
  for (int i = 0; i < 4; ++i) {
    if (!shape->dim(i).has_dim_value() || shape->dim(i).dim_value() <= 0 || shape->dim(i).dim_value() > limit) {
      return false;
    }
  }
  if (shape->dim(2).dim_value() > limit / 2 || shape->dim(3).dim_value() > limit / 2 ||
      shape->dim(2).dim_value() * shape->dim(3).dim_value() > limit) {
    return false;
  }

  bool found = false;
  int64_t output[4]{};
  for (size_t index = 2; index < node.InputDefs().size(); ++index) {
    const auto* arg = node.InputDefs()[index];
    if (!arg->Exists()) continue;
    const auto* tensor = graph.GetConstantInitializer(arg->Name(), true);
    if (!tensor) return false;
    const Initializer value(graph, *tensor, graph.ModelPath(), true);
    if (value.size() == 0) continue;
    if (found || value.size() != 4 || value.dims().size() != 1) return false;
    found = true;
    for (int i = 0; i < 4; ++i) {
      if (index == 2) {
        const float scale = value.data<float>()[i];
        const double length = std::floor(scale * static_cast<float>(shape->dim(i).dim_value()));
        if (!std::isfinite(scale) || scale <= 0 || length < 1 || length > limit || (i < 2 && scale != 1)) {
          return false;
        }
        output[i] = static_cast<int64_t>(length);
      } else {
        output[i] = value.data<int64_t>()[i];
        if (output[i] <= 0 || output[i] > limit || (i < 2 && output[i] != shape->dim(i).dim_value())) {
          return false;
        }
      }
    }
  }
  return found && output[0] == shape->dim(0).dim_value() && output[1] == shape->dim(1).dim_value() &&
         output[2] + output[3] <= limit / 2;
}

}  // namespace

Status ResizeArgMaxFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                     const logging::Logger& logger) const {
  const GraphViewer viewer(graph);
  for (auto index : viewer.GetNodesInTopologicalOrder()) {
    auto* resize = graph.GetNode(index);
    if (!resize) continue;
    ORT_RETURN_IF_ERROR(Recurse(*resize, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*resize, "Resize", {11, 13, 18, 19}) ||
        !graph_utils::IsSupportedProvider(*resize, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, *resize, 1) || !CanFuseResize(graph, *resize)) {
      continue;
    }
    auto& argmax = *graph.GetNode(resize->OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(argmax, "ArgMax", {1, 11, 12, 13}) ||
        argmax.GetExecutionProviderType() != resize->GetExecutionProviderType()) {
      continue;
    }
    const ProtoHelperNodeContext ctx(argmax);
    const OpNodeProtoHelper<ProtoHelperNodeContext> attrs(&ctx);
    const auto axis = attrs.GetAttrOrDefault("axis", int64_t{0});
    const auto keepdims = attrs.GetAttrOrDefault("keepdims", int64_t{1});
    const auto last = attrs.GetAttrOrDefault("select_last_index", int64_t{0});
    if ((axis != 1 && axis != -3) || (keepdims != 0 && keepdims != 1) || (last != 0 && last != 1)) continue;

    auto inputs = resize->MutableInputDefs();
    inputs.erase(inputs.begin() + 1);  // ROI is not used by these coordinate modes.
    auto& fused = graph.AddNode(graph.GenerateNodeName("ResizeArgMax"), "ResizeArgMax",
                                "Fuse spatial linear Resize and channel ArgMax", inputs, {}, {}, kMSDomain);
    if (const auto* coordinates = graph_utils::GetNodeAttribute(*resize, "coordinate_transformation_mode")) {
      fused.AddAttribute("coordinate_transformation_mode", coordinates->s());
    }
    fused.AddAttribute("keepdims", keepdims);
    fused.AddAttribute("select_last_index", last);
    fused.SetExecutionProviderType(kCpuExecutionProvider);
    graph_utils::FinalizeNodeFusion(graph, {*resize, argmax}, fused);
    modified = true;
  }
  return Status::OK();
}

}  // namespace onnxruntime
