// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Opt-in fusion for finite-valued inference. Zero overridable defaults can be
// represented with rank zero, so the kernel can skip inactive adapters.
class MatMulNBitsLoRAFusion final : public GraphTransformer {
 public:
  MatMulNBitsLoRAFusion() : GraphTransformer("MatMulNBitsLoRAFusion", {kWebGpuExecutionProvider}) {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
