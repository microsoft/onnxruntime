// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_CUDA) && !defined(USE_ROCM)

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class AttentionFusion
Rewrite graph fusing attention subgraph to a single Attention node.
*/
class GpuOpsPrepack : public GraphTransformer {
 public:
  GpuOpsPrepack() noexcept
      : GraphTransformer("GpuOpsPrepack", InlinedHashSet<std::string_view>{onnxruntime::kCudaExecutionProvider}) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
};

}  // namespace onnxruntime

#endif  // USE_CUDA && !USE_ROCM
