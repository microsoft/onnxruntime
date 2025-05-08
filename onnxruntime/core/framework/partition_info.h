// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include "core/common/inlined_containers_fwd.h"

struct OrtEpDevice;  // TODO: Move session/abi_devices.h to framework

namespace onnxruntime {
struct EpAssignedNode {
  std::string name;
  std::string op_type;
};

struct EpAssignedSubgraph {
  std::string ep_name;
  InlinedVector<std::string> op_types_storage;

  // Make returning these via the C API easer.
  InlinedVector<const char*> op_types;
  InlinedVector<size_t> op_type_counts;

  // EPs must set this in every ComputeCapability.
  const OrtEpDevice* ep_device;

  // Can be expensive to store metadata for every node in the partition.
  // Storing per-node info has to be explicitly enabled.
  std::vector<std::unique_ptr<EpAssignedNode>> nodes_storage;
  std::vector<const EpAssignedNode*> nodes;

  void SyncOpTypes() {
    // Update c-string op_types now that the InlinedVector will not be modified
    // and potentially reallocated.
    op_types.reserve(op_types_storage.size());
    for (const std::string& op_type : op_types_storage) {
      op_types.push_back(op_type.c_str());
    }
  }
};
}  // namespace onnxruntime
