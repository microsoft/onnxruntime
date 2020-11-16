// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/framework_common.h"
#include "core/framework/execution_provider.h"

#include <memory>
#include <string>
#include <vector>

namespace onnxruntime {
class Graph;

namespace test {

// return number of nodes in the Graph and any subgraphs that are assigned to the specified execution provider
int CountAssignedNodes(const Graph& current_graph, const std::string& ep_type);

// run the model using the CPU EP to get expected output, comparing to the output when the 'execution_provider'
// is enabled. requires that at least one node is assigned to 'execution_provider'
void RunAndVerifyOutputsWithEP(const ORTCHAR_T* model_path,
                               const char* log_id,
                               std::unique_ptr<IExecutionProvider> execution_provider,
                               const NameMLValMap& feeds);
}  // namespace test
}  // namespace onnxruntime
