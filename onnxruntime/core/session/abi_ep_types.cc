// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_ep_types.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"

onnxruntime::Status OrtEpGraphSupportInfo::AddSubgraph(const char* name,
                                                       const OrtHardwareDevice* hardware_device,
                                                       gsl::span<const OrtNode* const> nodes) {
  onnxruntime::InlinedVector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.reserve(nodes.size());
  for (const OrtNode* node : nodes) {
    ORT_RETURN_IF(node->type != OrtNode::Type::kEpNode, "Invalid OrtNode variant for use in OrtEpApi.");
    ep_nodes.push_back(static_cast<const onnxruntime::EpNode*>(node));
  }
  subgraphs.push_back(Subgraph{name, hardware_device, ep_nodes});

  return onnxruntime::Status::OK();
}
