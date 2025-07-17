// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include <memory>
#include <unordered_map>
#include <vector>
#include <optional>
#include <tuple>

#include "ort_api.h"
#include "qnn_ep.h"
#include "core/graph/abi_graph_types.h"

// Forward declaration of OrtNode
struct OrtNode;

// Forward declaration
namespace onnxruntime {
class QnnEp;

// Function to get QDQ node units from OrtGraph
std::pair<std::vector<std::unique_ptr<OrtNodeUnit>>, std::unordered_map<const OrtNode*, const OrtNodeUnit*>>
GetAllOrtNodeUnits(const OrtEp* this_ptr, const OrtGraph* graph, const logging::Logger& logger);


}  // namespace onnxruntime
