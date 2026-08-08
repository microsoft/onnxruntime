// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_ep_types.h"

#include <utility>
#include <vector>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"

namespace {
// ORT_API_VERSION in which OrtNodeFusionOptions::fused_node_hardware_device was introduced (ORT 1.29).
// Used to avoid reading that trailing field from EPs that were compiled against an older, smaller struct.
constexpr uint32_t kOrtNodeFusionOptionsHardwareDeviceMinApiVersion = 29;
}  // namespace

OrtEpGraphSupportInfo::OrtEpGraphSupportInfo(const onnxruntime::EpGraph& graph,
                                             const onnxruntime::IExecutionProvider::IKernelLookup& kernel_lookup)
    : ort_graph(graph), kernel_lookup{kernel_lookup} {}

onnxruntime::Status OrtEpGraphSupportInfo::AddNodesToFuse(gsl::span<const OrtNode* const> nodes,
                                                          const OrtNodeFusionOptions* optional_fusion_options) {
  std::vector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.reserve(nodes.size());

  for (const OrtNode* node : nodes) {
    const auto* ep_node = onnxruntime::EpNode::ToInternal(node);
    ORT_RETURN_IF(ep_node == nullptr, "Invalid OrtNode variant for use in OrtEpApi.");
    ep_nodes.push_back(ep_node);
  }

  // Copy the caller-provided fusion options field-by-field rather than with a single by-value struct copy.
  // A plugin EP compiled against an older ORT header may pass a pointer to a smaller OrtNodeFusionOptions; a
  // full-struct copy would read past the end of that object. Only copy a field once the EP's reported
  // ort_version_supported indicates the field exists in its struct layout.
  OrtNodeFusionOptions fusion_options{};
  if (optional_fusion_options != nullptr) {
    // Present since the struct was introduced (ORT 1.23).
    fusion_options.ort_version_supported = optional_fusion_options->ort_version_supported;
    fusion_options.drop_constant_initializers = optional_fusion_options->drop_constant_initializers;
    if (optional_fusion_options->ort_version_supported >= kOrtNodeFusionOptionsHardwareDeviceMinApiVersion) {
      fusion_options.fused_node_hardware_device = optional_fusion_options->fused_node_hardware_device;
    }
  }

  node_groupings.emplace_back(NodeGroupingKind::kFusedNode, std::move(ep_nodes), fusion_options);
  return onnxruntime::Status::OK();
}

onnxruntime::Status OrtEpGraphSupportInfo::AddSingleNode(const OrtNode* node) {
  std::vector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.push_back(onnxruntime::EpNode::ToInternal(node));
  node_groupings.emplace_back(NodeGroupingKind::kSingleAssignedNode, std::move(ep_nodes));
  return onnxruntime::Status::OK();
}
