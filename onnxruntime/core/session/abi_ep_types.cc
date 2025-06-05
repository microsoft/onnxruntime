// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_ep_types.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"

onnxruntime::Status OrtEpGraphSupportInfo::AddFusedNodes(gsl::span<const OrtNode* const> nodes,
                                                         gsl::span<const OrtHardwareDevice* const> hardware_devices) {
  std::vector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.reserve(nodes.size());
  for (const OrtNode* node : nodes) {
    const auto* ep_node = onnxruntime::EpNode::ToInternal(node);
    ORT_RETURN_IF(ep_node == nullptr, "Invalid OrtNode variant for use in OrtEpApi.");
    ep_nodes.push_back(ep_node);
  }

  onnxruntime::InlinedVector<const OrtHardwareDevice*> devices(hardware_devices.begin(), hardware_devices.end());
  node_groupings.push_back(NodeGrouping{NodeGroupingKind::kFusedNode, devices, ep_nodes});

  return onnxruntime::Status::OK();
}

onnxruntime::Status OrtEpGraphSupportInfo::AddSingleNode(const OrtNode* node, const OrtHardwareDevice* hardware_device) {
  std::vector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.push_back(onnxruntime::EpNode::ToInternal(node));

  onnxruntime::InlinedVector<const OrtHardwareDevice*> devices;
  devices.push_back(hardware_device);

  node_groupings.push_back(NodeGrouping{NodeGroupingKind::kSingleAssignedNode, devices, ep_nodes});

  return onnxruntime::Status::OK();
}
