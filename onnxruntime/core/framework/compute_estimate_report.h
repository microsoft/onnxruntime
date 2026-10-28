// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <string>

#include "core/common/status.h"

namespace onnxruntime {
class Graph;
struct RuntimeWorkloadSnapshot;
class RuntimeWorkloadCollector;

// Canonical shapes in the final executable graph only. Subgraphs have separate summaries;
// counts are never summed across control-flow paths into a per-run total.
Status BuildComputeEstimateReport(const Graph& graph, std::string& report);
Status WriteComputeEstimateReport(const Graph& graph, const std::string& utf8_path);

// Counts only retained, successfully submitted invocations. Run outcome and collection/coverage
// completeness are separate; route metadata does not adjust the logical counting convention.
Status BuildComputeEstimateReport(const RuntimeWorkloadSnapshot& snapshot, std::string& report);
Status WriteComputeEstimateReport(const RuntimeWorkloadSnapshot& snapshot, const std::string& utf8_path);

// Writes <prefix>_<process-token>_session-<id>_run-<id>.json after the collector is closed.
// No directories are created. Failures are diagnostic and must not replace inference status.
Status ExportRuntimeComputeEstimateReport(const RuntimeWorkloadCollector& collector, const std::string& utf8_prefix);

}  // namespace onnxruntime
#endif
