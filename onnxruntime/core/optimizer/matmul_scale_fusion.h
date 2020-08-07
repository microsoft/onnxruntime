// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * Fuses MatMul with surrounding scales (multiplies or divides) by a constant
 * scalar into TransposeScaleMatMul.
 * 
 * For example, given matrices A and B and constant scalars t, u, and v:
 *   Mul(v, MatMul(Mul(t, A), Mul(u, B))
 *     -> TransposeScaleMatMul(A, B, alpha=t*u*v)
 *
 * Note: Since both leading and following scales may be fused into a single
 * scale, the order and number of mathematical operations may change. This may
 * yield different results with floating point calculations.
 */
class MatMulScaleFusion : public GraphTransformer {
 public:
  /**
   * Constructor.
   * @param compatible_execution_providers The compatible execution providers.
   * @param excluded_initializer_names Fusion will be skipped on scales by any
   *        of the named initializers.
   */
  MatMulScaleFusion(
      const std::unordered_set<std::string>& compatible_execution_providers = {},
      const std::unordered_set<std::string>& excluded_initializer_names = {})
      : GraphTransformer{"MatMulScaleFusion", compatible_execution_providers},
        excluded_initializer_names_{excluded_initializer_names} {
  }

 private:
  Status ApplyImpl(
      Graph& graph, bool& modified,
      int graph_level, const logging::Logger& logger) const override;

  const std::unordered_set<std::string> excluded_initializer_names_;
};

}  // namespace onnxruntime
