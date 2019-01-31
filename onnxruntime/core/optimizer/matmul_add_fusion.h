// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class MatMulAddFusion : public onnxruntime::GraphTransformer {
 public:
  MatMulAddFusion() noexcept : onnxruntime::GraphTransformer("MatMulAddFusion", "Fusing MatMul and Add into Gemm") {}
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
