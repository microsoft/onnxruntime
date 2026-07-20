// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD)

#include <cstdint>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

// Fuses DequantizeLinear chains back into a single MatMulNBits contrib op.
//
// Supported patterns:
//   Pattern 1: DQ(3D, UINT4, axis=2) -> Reshape(2D) -> Transpose([1,0])
//              -> [optional Cast] -> MatMul/Gemm => MatMulNBits
//   Pattern 2: DQ(2D, UINT4, axis=0) -> MatMul/Gemm => MatMulNBits
//
// These patterns are produced when a quantized model goes through external
// toolchains that lower MatMulNBits to DQ + reshape/transpose + MatMul
// primitives, and then re-import the graph into ORT.
class DQMatMulNBitsFusion : public GraphTransformer {
 public:
  explicit DQMatMulNBitsFusion(
      int64_t accuracy_level = 4,
      const InlinedHashSet<std::string_view>& compatible_eps = {});

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;

  int64_t accuracy_level_;
};

}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
