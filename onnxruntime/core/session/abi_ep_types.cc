// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_ep_types.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"

onnxruntime::Status OrtEpGraphSupportInfo::AddSupportedNodes(const OrtHardwareDevice* hardware_device,
                                                             gsl::span<const OrtNode* const> nodes) {
  onnxruntime::InlinedVector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.reserve(nodes.size());
  for (const OrtNode* node : nodes) {
    const auto* ep_node = onnxruntime::EpNode::ToInternal(node);
    ORT_RETURN_IF(ep_node == nullptr, "Invalid OrtNode variant for use in OrtEpApi.");
    ep_nodes.push_back(ep_node);
  }
  node_groupings.push_back(NodeGrouping{hardware_device, ep_nodes});

  return onnxruntime::Status::OK();
}
