// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/abi_ep_types.h"

#include <utility>
#include <vector>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/graph/ep_api_types.h"
#include "core/session/abi_devices.h"

const OrtNodeFusionOptions* OrtEpGraphSupportInfo::NodeGrouping::TryGetNodeFusionOptions() const {
  return std::get_if<OrtNodeFusionOptions>(&variant_);
}

const onnxruntime::EpNode* OrtEpGraphSupportInfo::NodeGrouping::TryGetSingleNode() const {
  const onnxruntime::EpNode* const* node_ptr = std::get_if<const onnxruntime::EpNode*>(&variant_);
  return (node_ptr != nullptr) ? *node_ptr : nullptr;
}

void OrtEpGraphSupportInfo::AddNodesToFuse(const OrtNodeFusionOptions& node_fusion_options) {
  node_groupings.emplace_back(node_fusion_options);
}

void OrtEpGraphSupportInfo::AddSingleNode(const onnxruntime::EpNode& node) {
  node_groupings.emplace_back(&node);
}
