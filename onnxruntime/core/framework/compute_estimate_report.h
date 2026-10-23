// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "core/common/status.h"

namespace onnxruntime {
class Graph;

// Canonical shapes in the final executable graph only. Subgraphs have separate summaries;
// counts are never summed across control-flow paths into a per-run total.
Status BuildComputeEstimateReport(const Graph& graph, std::string& report);
Status WriteComputeEstimateReport(const Graph& graph, const std::string& utf8_path);

}  // namespace onnxruntime
#endif
