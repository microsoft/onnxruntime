// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)
#include <string>
#include <vector>
#include "core/common/common.h"

/// <summary>
/// Contains information about a node assigned to an EP. This is the definition of an opaque struct in the C API.
/// </summary>
struct OrtEpAssignedNode {
  std::string name;
  std::string domain;
  std::string op_type;
};

/// <summary>
/// Contains information about a subgraph assigned to an EP by the session graph partitioner.
/// This is the definition of an opaque struct in the C API.
/// </summary>
struct OrtEpAssignedSubgraph {
  OrtEpAssignedSubgraph() = default;
  OrtEpAssignedSubgraph(OrtEpAssignedSubgraph&&) = default;
  OrtEpAssignedSubgraph& operator=(OrtEpAssignedSubgraph&&) = default;
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(OrtEpAssignedSubgraph);

  std::string ep_name;
  std::vector<std::unique_ptr<OrtEpAssignedNode>> nodes_storage;
  std::vector<const OrtEpAssignedNode*> nodes;
};
#endif  // !defined(ORT_MINIMAL_BUILD)
