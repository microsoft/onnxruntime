// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/constants.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class Log1pFusion final : public GraphTransformer {
 public:
  Log1pFusion() noexcept : GraphTransformer("Log1pFusion", {kCudaExecutionProvider, kWebGpuExecutionProvider}) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
