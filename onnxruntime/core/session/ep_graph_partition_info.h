// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_map>
#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"

struct OrtHardwareDevice;

namespace onnxruntime {
struct EpAssignedNode {
  std::string name;
  std::string op_type;
};

struct EpAssignedSubgraph {
  EpAssignedSubgraph() = default;
  EpAssignedSubgraph(EpAssignedSubgraph&&) = default;
  EpAssignedSubgraph& operator=(EpAssignedSubgraph&&) = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(EpAssignedSubgraph);

  std::string ep_name;

  // Store in arrays (instead of map) to make returning these via the C API easer.
  InlinedVector<std::string> op_types_storage;
  InlinedVector<const char*> op_types;
  InlinedVector<size_t> op_type_counts;

  // The hardware device that runs the nodes in this subgraph.
  // Can be nullptr if the EP does not support autoEP.
  const OrtHardwareDevice* hardware_device = nullptr;

  // Can be expensive to store metadata for every node in the partition.
  // Should storing per-node info have to be explicitly enabled?
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
