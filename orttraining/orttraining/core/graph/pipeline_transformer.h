// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"

namespace onnxruntime {
namespace training {

void GetPipelineSendOutput(const Graph& graph, std::string& loss_name);
common::Status TransformGraphForPipeline(
    Graph& graph,
    std::string& forward_waited_event_name,
    std::string& forward_recorded_event_name,
    std::string& backward_waited_event_name,
    std::string& backward_recorded_event_name);

}  // namespace training
}  // namespace onnxruntime
