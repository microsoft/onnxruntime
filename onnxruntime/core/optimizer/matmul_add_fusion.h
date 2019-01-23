// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class MatMulAddFusion : public onnxruntime::GraphTransformer {
 public:
  MatMulAddFusion() noexcept : onnxruntime::GraphTransformer("MatMulAddFusion", "Fusing MatMul and Add into Gemm") {}
  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;
};

}  // namespace onnxruntime
