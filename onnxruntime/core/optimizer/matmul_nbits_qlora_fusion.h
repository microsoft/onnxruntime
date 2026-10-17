// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class MatMulNBitsQLoRAFusion final : public GraphTransformer {
 public:
  MatMulNBitsQLoRAFusion() : GraphTransformer("MatMulNBitsQLoRAFusion", {kWebGpuExecutionProvider}) {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(MatMulNBitsQLoRAFusion);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
