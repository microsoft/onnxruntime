// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_ep_types.h"

#include <utility>
#include <vector>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"

onnxruntime::Status OrtEpGraphSupportInfo::AddNodesToFuse(gsl::span<const OrtNode* const> nodes,
                                                          const OrtNodeFusionOptions* optional_fusion_options) {
  std::vector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.reserve(nodes.size());

  for (const OrtNode* node : nodes) {
    const auto* ep_node = onnxruntime::EpNode::ToInternal(node);
    ORT_RETURN_IF(ep_node == nullptr, "Invalid OrtNode variant for use in OrtEpApi.");
    ep_nodes.push_back(ep_node);
  }

  node_groupings.emplace_back(NodeGroupingKind::kFusedNode, std::move(ep_nodes),
                              optional_fusion_options != nullptr ? *optional_fusion_options : OrtNodeFusionOptions{});
  return onnxruntime::Status::OK();
}

onnxruntime::Status OrtEpGraphSupportInfo::AddSingleNode(const OrtNode* node) {
  std::vector<const onnxruntime::EpNode*> ep_nodes;
  ep_nodes.push_back(onnxruntime::EpNode::ToInternal(node));
  node_groupings.emplace_back(NodeGroupingKind::kSingleAssignedNode, std::move(ep_nodes));
  return onnxruntime::Status::OK();
}
