// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name);
common::Status TransformGraphForPipeline(Graph& graph);

}  // namespace training
}  // namespace onnxruntime
